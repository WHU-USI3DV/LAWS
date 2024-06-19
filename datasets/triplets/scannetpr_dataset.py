import os, sys
import torch
import random
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
import copy
import torchvision.transforms as T
import pickle
import math
import csv
from os.path import exists, join, isdir
EPSILON = 1e-8

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

# ScannetPR
class ScannetPRDataset(torch.utils.data.Dataset):
    """Class to handle Scannet dataset for Triple segmentation."""

    def __init__(self, dataset_folder, min_pointclouds_per_class=10):
        # Dataset folder
        self.path = dataset_folder
        self.data_path = os.path.join(self.path, 'scans')
        # Training or test set
        self.set = 'training'
        # Get a list of sequences
        data_split_path = os.path.join(self.path, "tools/Tasks/Benchmark")
        # Cloud names
        if self.set == 'training':
            scene_file_name = os.path.join(data_split_path, 'scannetv2_train.txt')
            self.all_scenes = np.sort(np.loadtxt(scene_file_name, dtype=np.str))
        elif self.set == 'validation':
            scene_file_name = os.path.join(data_split_path, 'scannetv2_val.txt')
            self.scenes = np.sort(np.loadtxt(scene_file_name, dtype=np.str))
            # self.clouds = [self.scenes[0]]  # test
        elif self.set == 'test':
            scene_file_name = os.path.join(data_split_path, 'scannetv2_test.txt')
            self.scenes = np.loadtxt(scene_file_name, dtype=np.str)
            self.scenes = self.scenes
        else:
            raise ValueError('Unsupport set type')
        
        self.min_pointclouds_per_class = min_pointclouds_per_class
        self.pcd_count = 0
        self.num_scene = 0
        # Potential like cloud segmentation?
        
        #####################
        # Prepare point cloud
        #####################
        self.fids = []          # list of list of actual frame id used to create pcd
        self.poses = []         # list of list of frame pose used to create pcd
        self.files = []         # list of list of pcd files created, pts in camera coordinate frame
        
        train_dict_file = 'cache/train_fict.pkl'
        train_ctrdict_file = 'cache/train_ctr_fict.pkl'
        if not os.path.exists(train_dict_file):
            self.split_training_sets()
        else:
            with open(train_dict_file, "rb") as f:
            # dict, key = scene string, val = list of filenames
                self.all_scene_train_files = pickle.load(f)  # 存储了所有场景中的训练文件
            with open(train_ctrdict_file, "rb") as f:
                self.all_scene_train_ctr = pickle.load(f)  # 存储了所有场景中的训练文件
        
    def __len__(self):
            """Return the number of classes within this group."""
            return len(self.classes_ids)

    def split_training_sets(self):
        logging.info('Initialization: split training sets')
        if not os.path.exists(self.data_path):
            raise ValueError('Missing input pcd folder:', self.data_path)
        
        # zero-meaned pcd file names for each pcd
        valid_pcd_file = os.path.join(self.path, 'VLAD_triplets', 'vlad_pcd.pkl')
        with open(valid_pcd_file, "rb") as f:
            # dict, key = scene string, val = list of filenames
            self.all_scene_pcds = pickle.load(f)
        all_pcd_ctr_file = os.path.join(self.path, 'VLAD_triplets', 'all_pcd_ctr.pickle')
        with open(all_pcd_ctr_file, "rb") as f:
            # dict, key = scene string, val = list of filenames
            self.all_pcd_ctr = pickle.load(f)
        
        train_dict, test_dict, train_ctr_dict, test_ctr_dict = {}, {}, {}, {}
        for i, scene in enumerate(self.all_scenes):
            scene_files = self.all_scene_pcds[scene]
            scene_ctrs = self.all_pcd_ctr[scene]
            index = list(range(0, len(scene_files)))
            random.shuffle(index)
            propotion = int(0.7*len(scene_files))
            train_index, test_index = index[:propotion], index[propotion:]
            
            # file
            train_set = [scene_files[i] for i in train_index]
            test_set = [scene_files[i] for i in test_index]
            
            train_dict[scene] = train_set
            test_dict[scene] = test_set

            # centers
            train_ctrs = scene_ctrs[train_index]
            test_ctrs = scene_ctrs[test_index]
            train_ctr_dict[scene] = train_ctrs
            test_ctr_dict[scene] = test_ctrs
            
        output_to_file(train_dict, 'cache/train_fict.pkl')
        output_to_file(test_dict, 'cache/test_fict.pkl')
        output_to_file(train_ctr_dict, 'cache/train_ctr_fict.pkl')
        output_to_file(test_ctr_dict, 'cache/test_ctr_fict.pkl')
   
    def arrange_curr_scans_info(self, incre):
        """
        generate the record file of information about scene, file, 
        world coordinates, orientation of sub point clouds 
        """ 
        record_count=0
        if os.path.exists(f'cache/cache_{incre}.csv'):
            logging.info('remove the cache csv file')
            os.remove(f'cache/cache_{incre}.csv')
        for i, scene in enumerate(self.scenes):
            record_count += len(self.all_scene_train_files[scene])
            print('{}%'.format(int(100*i/len(self.scenes))), flush=True, end='\r')
            # path to original ScanNet data
            scene_folder = os.path.join(self.data_path, scene)
            # path to processed ScanNet point cloud
            scene_pcd_path = os.path.join(scene_folder, 'input_pcd_0mean')
            if not os.path.exists(scene_pcd_path):
                raise ValueError('Missing scene folder:', scene_pcd_path)
            
            scene_ctrs, sub_infos = [],[]
            current_scene_ctrs = self.all_scene_train_ctr[scene]
            
            for subpcd_file, subpcd_ctrs in zip(self.all_scene_train_files[scene], current_scene_ctrs):
                
                actual_frame_id = int(subpcd_file[13:-8])
                # get pose of current frame
                pose = np.loadtxt(os.path.join(scene_folder, 'pose', str(actual_frame_id)+'.txt'))
                # double check if pose is lost
                chk_val = np.sum(pose)
                if np.isinf(chk_val) or np.isnan(chk_val):
                    raise ValueError('Invalid pose value for', scene_folder, actual_frame_id)
                # collect all the pcd centers of current scene 
                scene_ctrs.append(subpcd_ctrs)
                
                # calculate the pitch/roll/yaw of the current pcd
                sy = np.sqrt(pose[0,0]**2 + pose[1,0]**2)
                singular = sy < 1e-6
                if not singular:
                    x = math.atan2( pose[2,1], pose[2,2])
                    y = math.atan2(-pose[2,0], sy)
                    z = math.atan2( pose[1,0], pose[0,0])
                else:
                    x = math.atan2(-pose[1,2], pose[1,1])
                    y = math.atan2(-pose[2,0], sy)
                    z = 0
                roll = math.degrees(x)
                pitch = math.degrees(y)
                yaw = math.degrees(z)
                
                sub_info=[scene, actual_frame_id, subpcd_ctrs[0], subpcd_ctrs[1], subpcd_ctrs[2],roll,pitch,yaw]
                sub_infos.append(sub_info)
            
            scene_id = i+self.num_scene
            scene_id = np.array([scene_id]).tolist()
            scene_center = np.mean(np.array(scene_ctrs), axis=0).tolist()
            for sub_info in sub_infos:
                sub_info = scene_id + sub_info + scene_center
                with open(f'cache/cache_{incre}.csv', 'a', newline='') as csvfile:
                    posewriter = csv.writer(
                        csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    posewriter.writerow(sub_info)
        
        self.num_scene += i
        logging.info(f"The dataloader has received {self.num_scene} scenes")
    
    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.pointclouds_per_class[c]) for c in self.classes_ids])
       
    def get__class_id__group_id(self, x, y, scene_id, center_x, center_y, yaw):
        """Return class_id and group_id for a given point.
        The class_id is a triplet (tuple) of UTM_east, UTM_north and
        heading (e.g. (396520, 4983800,120)).
        The group_id represents the group to which the class belongs
        (e.g. (0, 1, 0)), and it is between (0, 0, 0) and (N, N, L).
        """
        # rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        # rounded_utm_north = int(utm_north // M * M)
        # rounded_heading = int(yaw // alpha * alpha)
        rounded_x = 2 if (x-center_x) > 0 else 0
        rounded_y = 1 if (y-center_y) > 0 else 0
        room_class = rounded_x+rounded_y
        scene_class= int(scene_id)
        # print(x-center_x, y-center_y, rounded_x, rounded_y, room_class, scene_class)
        class_id = (scene_class, room_class) # class_id goes from (0, 0) to (1200, 3)
        
        # group_id goes from 0 to 3
        group_id = (room_class)
        return class_id, group_id


class ScannetPRDatasetFT(ScannetPRDataset):
    """Class to handle Scannet dataset for Triple segmentation."""

    def __init__(self, dataset_folder, min_pointclouds_per_class=10):
        ScannetPRDataset.__init__(self, dataset_folder, min_pointclouds_per_class)
        
    def __len__(self):
            """Return the number of classes within this group."""
            return len(self.classes_ids)

    def update(self, incre, add_sequences):
        self.scenes = add_sequences
        self.arrange_curr_scans_info(incre)
        self.classes_per_group, self.pointclouds_per_class = self.update_class_and_group(incre, self.min_pointclouds_per_class)
    
    def update_class_and_group(self, incre, min_pointclouds_per_class):
        # df_train = pd.DataFrame(columns=['scene_id','file','x','y','center_x','center_y'])
        df_locations = pd.read_csv(f'cache/cache_{incre}.csv',
            sep=',',
            header=None, 
            names=['scene_id','scene','sub_id','x','y','z','roll','pitch','yaw','center_x','center_y','center_z'],
            lineterminator="\n")
        df_locations['sub_id'] =df_locations['scene']+'_'+df_locations['sub_id'].astype(str)+'_sub.bin'
        print("Number of training submaps: "+str(len(df_locations['sub_id'])))

        camera_heading = [(m.x, m.y, m.scene_id, m.center_x, m.center_y, m.yaw) for m in df_locations.iloc]
        camera_heading = np.array(camera_heading).astype(np.float)

        print("For each point cloud, get class and group to which it belongs")
        class_id__group_id = [self.get__class_id__group_id(*m)
                            for m in camera_heading]
        print("Group together point clouds belonging to the same class")
        pointclouds_per_class = defaultdict(list)
        for pc_info, (class_id, _) in zip(df_locations.iloc, class_id__group_id):
            
            pointclouds_per_class[class_id].append(os.path.join('scans', pc_info.scene, 'downsample', pc_info.sub_id))
       
        logging.debug("Group together classes belonging to the same group")
        # Classes_per_group is a dict where the key is group_id, and the value
        # is a list with the class_ids belonging to that group.
        classes_per_group = defaultdict(set)
        count=0
        for class_id, group_id in class_id__group_id:
            if class_id not in pointclouds_per_class:
                count+=1 
                continue  # Skip classes with too few images
            classes_per_group[group_id].add(class_id)
        print('We remove %d point cloud for they are so they are too rare to form a class'%(count))
        
        classes_per_group = [list(c) for c in classes_per_group.values()]
        return classes_per_group, pointclouds_per_class
        

class ScannetTripleDataset(torch.utils.data.Dataset):
    """Class to handle Scannet dataset for Triple segmentation."""

    def __init__(self, config, set='training'):

        ##########################
        # Parameters for the files
        ##########################

        # Dataset folder
        self.path = config["scannet_dir"]
        
        # # pcd without zero-meaning coordinates
        # self.input_pcd_path = join(self.path, 'scans', 'input_pcd')
        # pcd with zero-meaning coordinates
        self.data_path = join(self.path, 'scans')
        # self.input_pcd_path = join(self.path, 'scans', 'input_pcd_0mean')
        print("point cloud path:", self.data_path)

        # Type of task conducted on this dataset
        self.dataset_task = 'registration'

        # Training or test set
        self.set = set

        # Get a list of sequences
        # data_split_path = join(self.path, "test_files")
        data_split_path = join(self.path, "tools/Tasks/Benchmark")
        # data_split_path = join(self.path, "Tasks/Benchmark")
        # Cloud names
        if self.set == 'training':
            scene_file_name = join(data_split_path, 'scannetv2_train.txt')
            self.scenes = np.sort(np.loadtxt(scene_file_name, dtype=np.str_))
        elif self.set == 'validation':
            scene_file_name = join(data_split_path, 'scannetv2_val.txt')
            self.scenes = np.sort(np.loadtxt(scene_file_name, dtype=np.str_))
            # self.clouds = [self.scenes[0]]  # test
        elif self.set == 'test':
            scene_file_name = join(data_split_path, 'scannetv2_test.txt')
            # scene_file_name = join(data_split_path, 'scannetv2_test_val.txt')
            self.scenes = np.loadtxt(scene_file_name, dtype=np.str_)
            self.scenes = self.scenes
            # self.scenes = [self.scenes[0]]  # only test one scene
            # print((self.scenes))
        else:
            raise ValueError('Unsupport set type')

        # Parameters from config
        self.config = config

        # self.intrinsics = []    # list of Ks for every scene
        # self.nframes = []       # list of nums of frames for every scene
        self.fids = []          # list of list of actual frame id used to create pcd
        self.poses = []         # list of list of frame pose used to create pcd
        self.files = []         # list of list of pcd files created, pts in camera coordinate frame
        # training/val only
        # self.pos_thred = 2**2
        self.posIds = []        # list of dictionary of positive pcd example ids for training
        self.negIds = []        # list of dictionary of negative pcd example ids for training
        self.pcd_sizes = []     # list of list of pcd sizes
        # val/testing only
        self.class_proportions = None
        self.val_confs = []     # for validation

        # load sub cloud from the HD mesh w.r.t. cameras
        self.prepare_point_cloud()

        # get all_inds as 2D array
        # (index of the scene, index of the frame)
        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.fids)])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.fids])
        # NOTE 2nd is the index of frames NOT the actual frame id
        self.all_inds = np.vstack((seq_inds, frame_inds)).T 
        print(seq_inds.shape, frame_inds[106], self.all_inds.shape)
        
        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize frame potentials with random values
        self.potentials = torch.from_numpy(np.random.rand(self.all_inds.shape[0]) * 0.1 + 0.1)
        self.potentials.share_memory_()

        # If true, the same amount of frames is picked per class
        # SET FALSE HERE
        
        # shared epoch indices and classes (in case we want class balanced sampler)
        # if set == 'training':
        #     N = int(np.ceil(config.epoch_steps * self.batch_num * 1.1))
        # else:
        #     N = int(np.ceil(config.validation_size * self.batch_num * 1.1))
            
        self.num_pos_samples = config["TRAIN_POSITIVES_PER_QUERY"]    
        self.num_neg_samples = config["TRAIN_NEGATIVES_PER_QUERY"]
        # print(config.validation_size)
        # print(self.batch_num)
        # print('N = ', N)

        # # current epoch id
        # self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
        # # index generated this epoch to get the desired point cloud
        # # epoch should have length of at least epoch_steps * batch_num
        # # with values from 0 - all_inds.shape[0] (initialised as 0s)
        # self.epoch_inds = torch.from_numpy(np.zeros((N,), dtype=np.int64))
        # # self.epoch_labels = torch.from_numpy(np.zeros((N,), dtype=np.int32))
        # # print('\nPotential Info:')
        # # print(self.potentials.size())  # size of the total pcd, #_scene * #_frame
        # # print(self.epoch_i.size())     # counter, single value
        # # print(self.epoch_inds.size())  # total selected center, >= batch_num * epoch_step

        # self.epoch_i.share_memory_()
        # self.epoch_inds.share_memory_()

        return 

    def __len__(self):
        """
        Return the length of data here
        """
        # return 0
        return self.pcd_count

    def load_pc_file(self, filename):# returns Nx3 matrix
        try:
          pc = np.fromfile(filename, dtype=np.float64)
          if(pc.shape[0] != 4096*6):
              print("Error in pointcloud shape")
              return np.array([])
          pc = np.reshape(pc,(pc.shape[0]//6, 6))
        except Exception as e:
          logging.info(f"ERROR point cloud {filename} couldn't be opened, it might be corrupted.")
          raise e
        
        tensor_pc = T.functional.to_tensor(pc)
        assert tensor_pc.shape[-1] == 6, \
            f"Point cloud {filename} should have shape [4096,x] but has {tensor_pc.shape}."
        
        return tensor_pc.squeeze(0).float()
      
    def __getitem__(self, ind):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        # Current/query pcd indices
        s_ind, f_ind = self.all_inds[ind]
        if self.set in ['training', 'validation']:
            ## items should be generated here:
            # reference to pointnet_vlad: train_pointnetvlad.py
            # current pcd index:        s_ind & f_ind;
            # positive pcd indices:     [pos_s_inds, pos_f_inds]; default 2
            # negative pcd indices:     [neg_s_inds, neg_f_inds]; default 6 (4 in dh3d)
            # other negative pcd index: o_s_ind, o_f_ind.

            # check there are enough positive pcds
            num_pos_ids = len(self.posIds[s_ind][f_ind])
            if num_pos_ids < 2:
                print('Skip current pcd (', self.files[s_ind][f_ind].split('/')[-1], ') due to empty positives.')
                return []
            # print('Current pcd index:', s_ind, f_ind)
            # print('Positive lists:', self.posIds[s_ind][f_ind])
            # print('Negative lists:', self.negIds[s_ind][f_ind])

            # Positive pcd indices
            # pos_s_ind = s_ind
            # tmp_f_inds = np.random.randint(0, num_pos_ids, 2)
            # pos_f_inds = np.array(self.posIds[s_ind][f_ind])[tmp_f_inds]
            pos_f_inds = [np.random.choice( self.posIds[s_ind][f_ind] )] 
            # 2 positives, ensure not choose the same positive pcd
            while True:
                tmp = np.random.choice(self.posIds[s_ind][f_ind] ) 
                if tmp != pos_f_inds[0]:
                    pos_f_inds.append(tmp)
                    break
            pos_f_inds = np.array(pos_f_inds)
            # print(num_pos_ids, pos_s_ind, pos_f_inds)
            all_indices = [ (s_ind, f_ind), (s_ind, pos_f_inds[0]), (s_ind, pos_f_inds[1]) ]
            
            ## Add selection criterion based on pcd size due GPU memory limitation ##
            # check pcd sizes, choose 2x(9000+), 1x(7000-9000), 1x(5000-7000), 2x(5000-)
            count_XL = 0
            max_XL = 1
            count_L = 0
            max_L = 1
            count_M = 0
            max_M = 1
            # choose at most one from the same scene
            if len(self.negIds[s_ind][f_ind]) > 0:
                neg_s_inds = [s_ind]
                neg_f_inds = [np.random.choice(self.negIds[s_ind][f_ind])]
                # get the size of selected pcd
                tmp_size = int(self.pcd_sizes[neg_s_inds[-1]][neg_f_inds[-1]])
                # print(tmp_size)
                if tmp_size >= 9000:
                    count_XL += 1
                elif tmp_size >= 7000:
                    count_L += 1
                elif tmp_size > 5000:
                    count_M += 1
                else:
                    pass
            else:
                neg_s_inds = []
                neg_f_inds = []
            # continue select the rest negative pcds from other scenes
            while len(neg_s_inds) < self.num_neg_samples:
                tmp_neg_s = np.random.choice( len(self.scenes) )
                if self.scenes[s_ind][:9] != self.scenes[tmp_neg_s][:9]:
                    tmp_neg_f = np.random.randint(0, len(self.fids[tmp_neg_s]))
                    # check the pcd size
                    tmp_size = int(self.pcd_sizes[tmp_neg_s][tmp_neg_f])
                    if tmp_size >= 9000:
                        if count_XL == max_XL:
                            continue
                        count_XL += 1
                    elif tmp_size >= 7000:
                        if count_L == max_L:
                            continue
                        count_L += 1
                    elif tmp_size >= 5000:
                        if count_M == max_M:
                            continue
                        count_M += 1
                    else:
                        pass
                    # print(tmp_size)
                    neg_s_inds.append(tmp_neg_s)
                    neg_f_inds.append(tmp_neg_f)
            
            # print(neg_s_inds, neg_f_inds)
            for idx in range(self.num_neg_samples):
                all_indices.append( (neg_s_inds[idx], neg_f_inds[idx]) )
            
            # Other negative pcd index for quadruplet
            # find a pcd that is negative to all anchor, positives, and negatives
            neg_star_s_ind = -1
            while neg_star_s_ind < 0:
                tmp_neg = np.random.choice( len(self.scenes) )
                bMatched = False
                if self.scenes[s_ind][:9] != self.scenes[tmp_neg][:9]:
                    bMatched = True
                for neg_s_ind in neg_s_inds:
                    if self.scenes[s_ind][:9] != self.scenes[tmp_neg][:9]:
                        bMatched = True
                        break
                if not bMatched:
                    neg_star_s_ind = tmp_neg
            neg_star_f_ind = np.random.randint(0, len(self.fids[neg_star_s_ind]))
            all_indices.append( (neg_star_s_ind, neg_star_f_ind) )
            # print('All chosen indices: [query/current], [positive]*2, [negative]*4', all_indices)
            
        else:
            # in testing, only get the current index
            all_indices = [(s_ind, f_ind)]
        pointclouds = []
        for s_ind, f_ind in all_indices:
            #################
            # Load the points
            # NOTE all points are in camera 
            #      coordinate frame
            #################

            # print(s_ind, f_ind, len(self.files), len(self.files[s_ind]))
            current_file = self.files[s_ind][f_ind]
            # print('Loading: ', current_file)

            fname = current_file.replace('input_pcd_0mean','downsample').replace('ply','bin')
            
            pc = self.load_pc_file(fname)
            pointclouds.append(pc)
                
        return pointclouds

    def prepare_point_cloud(self):
        """
        generate sub point clouds from the complete
        reconstructed scene, using current pose and
        depth frame
        """

        if not exists(self.data_path):
            raise ValueError('Missing input pcd folder:', self.data_path)
        
        # Load pre-processed files [all tasks are stored in a single file]
        # positive and negative indices for each pcd
        vlad_pn_file = join(self.path, 'VLAD_triplets', 'vlad_pos_neg.pkl')
        with open(vlad_pn_file, "rb") as f:
            # dict, key = scene string, val = list of pairs of (list pos, list neg)
            all_scene_pos_neg = pickle.load(f)
        # zero-meaned pcd file names for each pcd
        valid_pcd_file = join(self.path, 'VLAD_triplets', 'vlad_pcd.pkl')
        with open(valid_pcd_file, "rb") as f:
            # dict, key = scene string, val = list of filenames
            all_scene_pcds = pickle.load(f)
        # num of pts in each pcd
        pcd_size_file = join(self.path, 'VLAD_triplets', 'pcd_size.pkl')
        with open(pcd_size_file, "rb") as f:
            # dict, key = scene string, val = list of point numbers
            dict_pcd_size = pickle.load(f)

        # Loop through all scenes to retrieve data for current task
        pcd_count = 0
        for i, scene in enumerate(self.scenes):
            pcd_count += len(all_scene_pcds[scene])
            print('{}%'.format(int(100*i/len(self.scenes))), flush=True, end='\r')
            # print('scene_id_name', i, scene, 'num_of_pcds', num_scene_pcds)
            # print('Processing:', scene, '(', i+1 , '/', len(self.scenes), ')') 
            # # print('  from', scene_folder)
            # # print('   to ', scene_pcd_path)
            
            # path to original ScanNet data
            scene_folder = join(self.data_path, scene)
            # path to processed ScanNet point cloud
            scene_pcd_path = join(scene_folder, 'input_pcd_0mean')
            if not exists(scene_pcd_path):
                raise ValueError('Missing scene folder:', scene_pcd_path)

            # get necessary data
            scene_files = []    # list of pcd file names for current scene
            scene_poses = []    # list of pcd poses for current scene
            scene_fids = []     # list of actual frame id used to generate the pcd
            all_posId = []      # list of positive pcd Indices
            all_negId = []      # list of negative pcd Indices
            for j, subpcd_file in enumerate(all_scene_pcds[scene]):
                actual_frame_id = int(subpcd_file[13:-8])
                # print('{}%'.format(int(100*j/num_scene_pcds)), flush=True, end='\r')
                frame_subpcd_file = join(scene_pcd_path, subpcd_file)

                # get pose of current frame
                pose = np.loadtxt(join(scene_folder, 'pose', str(actual_frame_id)+'.txt'))
                # double check if pose is lost
                chk_val = np.sum(pose)
                if np.isinf(chk_val) or np.isnan(chk_val):
                    raise ValueError('Invalid pose value for', scene_folder, actual_frame_id)

                # store file name info
                scene_files.append(frame_subpcd_file)
                scene_poses.append(pose)
                scene_fids.append(actual_frame_id)

                # store current pos and neg index
                # all_posId[j] = all_scene_pos_neg[scene][j][0]
                # all_negId[j] = all_scene_pos_neg[scene][j][1]
                all_posId.append(all_scene_pos_neg[scene][j][0])
                all_negId.append(all_scene_pos_neg[scene][j][1])

                # double check if the point cloud file exists
                if not exists(frame_subpcd_file):
                    raise ValueError('Missing subpcd file:', frame_subpcd_file)
            # print('100 %')

            self.files.append(scene_files)
            self.poses.append(scene_poses)
            self.fids.append(scene_fids)
            self.pcd_sizes.append(dict_pcd_size[scene])
            if self.set in ['training', 'validation']:
                self.posIds.append(all_posId)
                self.negIds.append(all_negId)
                
            # print(self.posIds)
        print('Total # of pcd:', pcd_count, )
        self.pcd_count = pcd_count
        print(len(self.files), len(self.poses), len(self.fids), len(self.posIds))
        print(len(self.files[0]), len(self.poses[0]), len(self.fids[0]), len(self.posIds[0]))
        print(len(self.files[0][0]), self.fids[0], len(self.posIds[0][0]), self.posIds[0][0])
        
        # print(self.files[0], self.poses[0], self.fids[0], self.posIds[0], self.negIds[0])
        # self.files 1201 存的是房间内的所有信息
        
    def parse_scene_info(self, filename):
        """ read information file with given filename

            Returns
            -------
            int 
                number of frames in the sequence
            list
                [height width fx fy cx cy].
        """
        K = []
        info_file = open(filename)
        for line in info_file:
            vals = line.strip().split(' ')

            if 'depth' in vals[0]:
                K.append(float(vals[2]))
            if vals[0] == 'numDepthFrames':
                nFrames = int(vals[2])
        return nFrames, K


