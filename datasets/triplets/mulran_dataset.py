import os
import sys
import glob
import random
import numpy as np
import logging
import json
import csv
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
# from utils.o3d_tools import *
from datasets.triplets.pointcloud_dataset import *

class MulRanDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from MulRan dataset. 
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        self.root = root = config["mulran_dir"]
        
        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing MulRanDataset")
        logging.info(f"Loading the subset {phase} from {root}")

        # sequences = config.mulran_data_split[phase]
        # for drive_id in sequences:
        #     inames = self.get_all_scan_ids(drive_id)
        #     for query_id, start_time in enumerate(inames):
        #         self.files.append((drive_id, query_id))

    def get_all_scan_ids(self, drive_id):
        sequence_path = self.root + drive_id + '/Downsample/'
        fnames = sorted(glob.glob(os.path.join(sequence_path, '*.bin')))
        assert len(
            fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    def get_velodyne_fn(self, drive_id, query_id):
        sequence_path = self.root + drive_id + '/Downsample/'
        fname = sorted(glob.glob(os.path.join(
            sequence_path, '*.bin')))[query_id]
        return fname
    
    def get_pointcloud_tensor(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, pc_id)
        pc = np.fromfile(fname, dtype=np.float64).reshape(-1, 3)
        
        if(pc.shape[0] != 4096) or (pc.shape[1] != 3):
            print("Error in pointcloud shape", pc.shape)
            return np.array([])
        return pc
    
    def __getitem__(self, idx):
        drive_id = self.files[idx][0]
        t0 = self.files[idx][1]
        xyz0_th = self.get_pointcloud_tensor(drive_id, t0)
        meta_info = {'drive': drive_id, 't0': t0}

        return (xyz0_th,
                meta_info)

class MulRanTupleDataset(MulRanDataset):
    r"""
    Generate tuples (anchor, positives, negatives) using distance
    Optional other_neg for quadruplet loss. 
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):
        self.root = root = config['mulran_dir']
        self.positives_per_query = config['TRAIN_POSITIVES_PER_QUERY']
        self.negatives_per_query = config['TRAIN_NEGATIVES_PER_QUERY']
        self.quadruplet = False
        if config["LOSS_FUNCTION"] == 'quadruplet':
            self.quadruplet = True
        
        MulRanDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing MulRanTupleDataset")
        logging.info(f"Loading the subset {phase} from {root}")

        sequences = ['DCC/DCC_01', 'DCC/DCC_02',
              'Riverside/Riverside_01', 'Riverside/Riverside_03']
        tuple_dir = os.path.join(os.path.dirname(
            __file__), '../config1/mulran_tuples/')
        self.dict_3m = json.load(open(tuple_dir + 'positive_sequence_D-3_T-0.json', "r"))
        self.dict_20m = json.load(
            open(tuple_dir + 'positive_sequence_D-20_T-0.json', "r"))
        self.mulran_seq_lens = {"DCC/DCC_01": 5542, "DCC/DCC_02": 7561, "DCC/DCC_03": 7479,
                            "KAIST/KAIST_01": 8226, "KAIST/KAIST_02": 8941, "KAIST/KAIST_03": 8629,
                            "Sejong/Sejong_01": 28779, "Sejong/Sejong_02": 27494, "Sejong/Sejong_03": 27215,
                            "Riverside/Riverside_01": 5537, "Riverside/Riverside_02": 8157, "Riverside/Riverside_03": 10476}
        
        for drive_id in sequences:
            sequence_path = self.root + drive_id + '/Downsample/'
            fnames = sorted(glob.glob(os.path.join(sequence_path, '*.bin')))
            assert len(
                fnames) > 0, f"Make sure that the path {root} has data {drive_id}"
            inames = sorted([int(os.path.split(fname)[-1][:-4])
                            for fname in fnames])

            for query_id, start_time in enumerate(inames):
                positives = self.get_positives(drive_id, query_id)
                negatives = self.get_negatives(drive_id, query_id)
                self.files.append((drive_id, query_id, positives, negatives))
               
    def get_positives(self, sq, index):
        assert sq in self.dict_3m.keys(), f"Error: Sequence {sq} not in json."
        sq_1 = self.dict_3m[sq]
        if str(int(index)) in sq_1:
            positives = sq_1[str(int(index))]
        else:
            positives = []
        return positives

    def get_negatives(self, sq, index):
        assert sq in self.dict_20m.keys(), f"Error: Sequence {sq} not in json."
        sq_2 = self.dict_20m[sq]
        all_ids = set(np.arange(self.mulran_seq_lens[sq]))
        neg_set_inv = sq_2[str(int(index))]
        neg_set = all_ids.difference(neg_set_inv)
        negatives = list(neg_set)
        if index in negatives:
            negatives.remove(index)
        return negatives

    def get_other_negative(self, drive_id, query_id, sel_positive_ids, sel_negative_ids):
        # Dissimillar to all pointclouds in triplet tuple.
        all_ids = range(self.mulran_seq_lens[str(drive_id)])
        neighbour_ids = sel_positive_ids
        for neg in sel_negative_ids:
            neg_postives_files = self.get_positives(drive_id, neg)
            for pos in neg_postives_files:
                neighbour_ids.append(pos)
        possible_negs = list(set(all_ids) - set(neighbour_ids))
        if query_id in possible_negs:
            possible_negs.remove(query_id)
        assert len(
            possible_negs) > 0, f"No other negatives for drive {drive_id} id {query_id}"
        other_neg_id = random.sample(possible_negs, 1)
        return other_neg_id[0]

    def __getitem__(self, idx):
        drive_id, query_id = self.files[idx][0], self.files[idx][1]
        positive_ids, negative_ids = self.files[idx][2], self.files[idx][3]

        sel_positive_ids = random.sample(
            positive_ids, self.positives_per_query)
        sel_negative_ids = random.sample(
            negative_ids, self.negatives_per_query)
        positives, negatives, other_neg = [], [], None

        query_th = self.get_pointcloud_tensor(drive_id, query_id)
        for sp_id in sel_positive_ids:
            positives.append(self.get_pointcloud_tensor(drive_id, sp_id))
        for sn_id in sel_negative_ids:
            negatives.append(self.get_pointcloud_tensor(drive_id, sn_id))

        meta_info = {'drive': drive_id, 'query_id': query_id}

        if not self.quadruplet:
            return (query_th,
                    positives,
                    negatives,
                    meta_info)
        else:  # For Quadruplet Loss
            other_neg_id = self.get_other_negative(
                drive_id, query_id, sel_positive_ids, sel_negative_ids)
            other_neg_th = self.get_pointcloud_tensor(drive_id, other_neg_id)
            return (query_th,
                    positives,
                    negatives,
                    other_neg_th,
                    meta_info)

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


