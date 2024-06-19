from scipy.spatial.distance import cdist
import logging, json
import pickle
import os
import sys
import numpy as np
import math, csv
import torch
from util.util import Timer
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# from utils.data_loaders.make_dataloader import *
# from misc.misc_utils import *
from datasets.test_dataset import TestMulRanDataset, TestKittiDataset
from models.MinkLoc3dv2.mink_params import make_sparse_tensor, CartesianQuantizer, PolarQuantizer
__all__ = ['evaluate_sequence_reg']


ALL_DATASETS = [
    TestKittiDataset, TestMulRanDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}

####  MulRan  ########################################################################
#####################################################################################
# Load poses
#####################################################################################

def load_poses_from_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data_poses = list(reader)

    transforms = []
    positions = []
    for cnt, line in enumerate(data_poses):
        line_f = [float(i) for i in line]
        P = np.vstack((np.reshape(line_f[1:], (3, 4)), [0, 0, 0, 1]))
        transforms.append(P)
        positions.append([P[0, 3], P[1, 3], P[2, 3]])
    return np.asarray(transforms), np.asarray(positions)


#####################################################################################
# Load timestamps
#####################################################################################

def load_timestamps_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data_poses = list(reader)
    data_poses_ts = np.asarray(
        [float(t)/1e9 for t in np.asarray(data_poses)[:, 0]])
    return data_poses_ts


def load_poses_timestamps_from_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data_poses = list(reader)

    timestamps = []
    positions = []
    for cnt, line in enumerate(data_poses):
        if cnt == 0:
            continue
        line_f = [float(i) for i in line]
        
        timestamps.append(line_f[0])
        positions.append([line_f[1],line_f[2],line_f[3]])
    return np.asarray(timestamps), np.asarray(positions)

####  Kitti  ########################################################################
# Load poses
#####################################################################################
def transfrom_cam2velo(Tcam):
    R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                  -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
                  ]).reshape(3, 3)
    t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    cam2velo = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    return Tcam @ cam2velo

def load_poses_from_txt(file_name):
    """
    Modified function from: https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    transforms = {}
    positions = []
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i != ""]
        withIdx = len(line_split) == 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        transforms[frame_idx] = transfrom_cam2velo(P)
        positions.append([P[0, 3], P[2, 3], P[1, 3]])
    return transforms, np.asarray(positions)
#####################################################################################
# Load timestamps
#####################################################################################
def load_timestamps(file_name):
    # file_name = data_dir + '/times.txt'
    file1 = open(file_name, 'r+')
    stimes_list = file1.readlines()
    s_exp_list = np.asarray([float(t[-4:-1]) for t in stimes_list])
    times_list = np.asarray([float(t[:-2]) for t in stimes_list])
    times_listn = [times_list[t] * (10**(s_exp_list[t]))
                   for t in range(len(times_list))]
    file1.close()
    return times_listn

def save_pickle(data_variable, file_name):
    dbfile2 = open(file_name, 'ab')
    pickle.dump(data_variable, dbfile2)
    dbfile2.close()
    logging.info(f'Finished saving: {file_name}')


def evaluate_sequence_reg(model, cfg):
    save_descriptors = cfg.eval_save_descriptors
    save_counts = cfg.eval_save_counts
    revisit_json_file = 'is_revisit_D-{}_T-{}.json'.format(
        int(cfg.revisit_criteria), int(cfg.skip_time))
    if 'Kitti' in cfg.eval_dataset:
        eval_seq = cfg.kitti_eval_seq
        cfg.kitti_data_split['test'] = [eval_seq]
        eval_seq = '%02d' % eval_seq
        sequence_path = cfg.kitti_dir + 'sequences/' + eval_seq + '/'
        _, positions_database = load_poses_from_txt(
            sequence_path + 'poses.txt')
        timestamps = load_timestamps(sequence_path + 'times.txt')
        revisit_json_dir = 'configs/kitti_tuples/'
        revisit_json = json.load(
            open(revisit_json_dir + revisit_json_file, "r"))
        is_revisit_list = revisit_json[eval_seq]

    elif 'MulRan' in cfg.eval_dataset:
        eval_seq = cfg.mulran_eval_seq
        cfg.mulran_data_split['test'] = [eval_seq]
        sequence_path = cfg.mulran_dir + eval_seq
        _, positions_database = load_poses_from_csv(
            sequence_path + '/scan_poses.csv')
        timestamps = load_timestamps_csv(sequence_path + '/scan_poses.csv')
        revisit_json_dir =  'configs/mulran_tuples/'
        revisit_json = json.load(
            open(revisit_json_dir + revisit_json_file, "r"))
        is_revisit_list = revisit_json[eval_seq]
    
    logging.info(f'Evaluating sequence {eval_seq} at {sequence_path}')
    thresholds = np.linspace(
        cfg.cd_thresh_min, cfg.cd_thresh_max, int(cfg.num_thresholds))
    
    Dataset = dataset_str_mapping[cfg.eval_dataset]
    dset = Dataset('test',
                   random_scale=False,
                   random_rotation=False,
                   random_occlusion=False,
                   config=cfg)
    
    test_loader = torch.utils.data.DataLoader(dset,
                                         batch_size=cfg.eval_batch_size,
                                         num_workers=cfg.test_num_workers,
                                         pin_memory=True)

    iterator = test_loader.__iter__()
    logging.info(f'len_dataloader {len(test_loader.dataset)}')
    logging.info(f'len_is_revisit_list {len(is_revisit_list)}')
    
    
    num_queries = len(positions_database)
    num_thresholds = len(thresholds)
    logging.info(f'num_queries {num_queries}')
    
    # Databases of previously visited/'seen' places.
    seen_poses, seen_descriptors, seen_feats = [], [], []

    # Store results of evaluation.
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    prep_timer, desc_timer, ret_timer = Timer(), Timer(), Timer()

    min_min_dist = 1.0
    max_min_dist = 0.0
    num_revisits = 0
    num_correct_loc = 0
    start_time = timestamps[0]
    error_statistics = []
    
    #######
    model_input = InputFactory(cfg)
    revisit_record = []
    for query_idx in range(num_queries):

        input_data,_ = iterator.next()
        prep_timer.tic()
       
        if not len(input_data) > 0:
            logging.info(f'Corrupt cloud id: {query_idx}')
            continue
        
        # feed_tensor = input_data.float().to("cuda")  # B 3 N
        feed_tensor = model_input(input_data)
        prep_timer.toc()
        desc_timer.tic()
        
        output_desc = model(feed_tensor, is_training = False)  # .squeeze()
        # print(output_desc.shape)
        
        desc_timer.toc()
        global_descriptor = output_desc.cpu().detach().numpy()
        
        global_descriptor = np.reshape(global_descriptor, (1, -1))
        query_pose = positions_database[query_idx]
        query_time = timestamps[query_idx]

        if len(global_descriptor) < 1:
            continue

        seen_descriptors.append(global_descriptor)
        seen_poses.append(query_pose)

        if (query_time - start_time - cfg.skip_time) < 0:
            continue

        # Build retrieval database using entries 30s prior to current query.
        tt = next(x[0] for x in enumerate(timestamps)
                  if x[1] > (query_time - cfg.skip_time))
        db_seen_descriptors = np.copy(seen_descriptors)
        db_seen_poses = np.copy(seen_poses)
        db_seen_poses = db_seen_poses[:tt+1]
        db_seen_descriptors = db_seen_descriptors[:tt+1]
        db_seen_descriptors = db_seen_descriptors.reshape(
            -1, np.shape(global_descriptor)[1])

        # Find top-1 candidate.
        nearest_idx = 0
        min_dist = math.inf

        ret_timer.tic()
        feat_dists = cdist(global_descriptor, db_seen_descriptors,
                           metric=cfg.eval_feature_distance).reshape(-1)
        min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)
        ret_timer.toc()

        place_candidate = seen_poses[nearest_idx]
        p_dist = np.linalg.norm(query_pose - place_candidate)

        # is_revisit = check_if_revisit(query_pose, db_seen_poses, cfg.revisit_criteria)
        is_revisit = is_revisit_list[query_idx]
        is_correct_loc = 0
        
        if is_revisit:
            num_revisits += 1
            if p_dist <= cfg.revisit_criteria:
                num_correct_loc += 1
                is_correct_loc = 1
            error_statistics.append(p_dist)
            logging.info(
                f'id: {query_idx} n_id: {nearest_idx} is_rev: {is_revisit} is_correct_loc: {is_correct_loc} min_dist: {min_dist} p_dist: {p_dist}')
            
            revisit_record.append([query_idx, nearest_idx, int(is_revisit), is_correct_loc, p_dist])


        if min_dist < min_min_dist:
            min_min_dist = min_dist
        if min_dist > max_min_dist:
            max_min_dist = min_dist

        # Evaluate top-1 candidate.
        for thres_idx in range(num_thresholds):
            threshold = thresholds[thres_idx]

            if(min_dist < threshold):  # Positive Prediction
                if p_dist <= cfg.revisit_criteria:
                    num_true_positive[thres_idx] += 1

                elif p_dist > cfg.not_revisit_criteria:
                    num_false_positive[thres_idx] += 1

            else:  # Negative Prediction
                if(is_revisit == 0):
                    num_true_negative[thres_idx] += 1
                else:
                    num_false_negative[thres_idx] += 1


    revisit_record = np.float64(np.array(revisit_record))
    revisit_record.tofile('/home/xy/xy/code/SemanticKitti/dataset/utils/model_1.bin')

    F1max = 0.0
    Precisions, Recalls = [], []
    if not save_descriptors:
        for ithThres in range(num_thresholds):
            nTrueNegative = num_true_negative[ithThres]
            nFalsePositive = num_false_positive[ithThres]
            nTruePositive = num_true_positive[ithThres]
            nFalseNegative = num_false_negative[ithThres]

            Precision = 0.0
            Recall = 0.0
            F1 = 0.0

            if nTruePositive > 0.0:
                Precision = nTruePositive / (nTruePositive + nFalsePositive)
                Recall = nTruePositive / (nTruePositive + nFalseNegative)

                F1 = 2 * Precision * Recall * (1/(Precision + Recall))

            if F1 > F1max:
                F1max = F1
                F1_TN = nTrueNegative
                F1_FP = nFalsePositive
                F1_TP = nTruePositive
                F1_FN = nFalseNegative
                F1_thresh_id = ithThres
            Precisions.append(Precision)
            Recalls.append(Recall)
        logging.info(f'len_dataloader {len(test_loader.dataset)}')
        logging.info(f'len_is_revisit_list {len(is_revisit_list)}')
    
        logging.info(f'num_revisits: {num_revisits}')
        logging.info(f'num_correct_loc: {num_correct_loc}')
        logging.info(
            f'percentage_correct_loc: {num_correct_loc*100.0/num_revisits}')
        logging.info(
            f'min_min_dist: {min_min_dist} max_min_dist: {max_min_dist}')
        logging.info(
            f'F1_TN: {F1_TN} F1_FP: {F1_FP} F1_TP: {F1_TP} F1_FN: {F1_FN}')
        logging.info(f'F1_thresh_id: {F1_thresh_id}')
        logging.info(f'F1max: {F1max}')

        # print('Recalls, Precisions', len(Recalls),type(Recalls), len(Precisions),type(Precisions))
        
        # print(len(error_statistics), eval_seq)
        # eval_seq_name = eval_seq.replace('/','_')
        # f1=open("eval/pr_curves/error/"+ cfg.backbone+"_" + eval_seq_name+"_error.txt","a")
        # for line in error_statistics:
        #     f1.write(str(line)+'\n')
        # f1.close()
        
    if not save_descriptors:
        logging.info('Average times per scan:')
        logging.info(
            f"--- Prep: {prep_timer.avg}s Desc: {desc_timer.avg}s Ret: {ret_timer.avg}s ---")
        logging.info('Average total time per scan:')
        logging.info(
            f"--- {prep_timer.avg + desc_timer.avg + ret_timer.avg}s ---")

    
    if save_counts:
        save_dir = os.path.join(os.path.dirname(
            __file__), 'pickles/', str(eval_seq))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_pickle(num_true_positive, save_dir + '/num_true_positive.pickle')
        save_pickle(num_false_positive, save_dir +
                    '/num_false_positive.pickle')
        save_pickle(num_true_negative, save_dir + '/num_true_negative.pickle')
        save_pickle(num_false_negative, save_dir +
                    '/num_false_negative.pickle')

    return F1max

       
class InputFactory:
    def __init__(self, cfg):
        if cfg.backbone in ['mink', 'mink_laws']:
            self.get_quantizer(cfg)
            self.collation_fn = self.get_sparse_input
        else:
            self.collation_fn = self.get_input
    
    def __call__(self, input_data):
        return self.collation_fn(input_data)

    def get_input(self, input_data):
        feed_tensor = input_data.float().to("cuda")  # B 3 N
        return feed_tensor
    
    def get_quantizer(self, cfg):
        if 'polar' in cfg.coordinates:
            # 3 quantization steps for polar coordinates: for sectors (in degrees), rings (in meters) and z coordinate (in meters)
            self.quantization_step = tuple([float(e) for e in cfg.quantization_step.split(',')])
            assert len(self.quantization_step) == 3, f'Expected 3 quantization steps: for sectors (degrees), rings (meters) and z coordinate (meters)'
            self.quantizer = PolarQuantizer(quant_step=self.quantization_step)
        elif 'cartesian' in cfg.coordinates:
            # Single quantization step for cartesian coordinates
            self.quantization_step = cfg.quantization_step
            self.quantizer = CartesianQuantizer(quant_step=self.quantization_step)
        
    def get_sparse_input(self, input_data):
        
        batch = make_sparse_tensor(input_data, self.quantizer, self.quantization_step)
        batch = {e: batch[e].to('cuda') for e in batch}
        return batch
   