# Base dataset classes, inherited by dataset-specific classes
import os
import pickle
from typing import List
from typing import Dict
import torch
import numpy as np
from torch.utils.data import Dataset
import logging, json
import glob

class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position

class TrainingDataset(Dataset):
    def __init__(self, dataset_path, transform=None, set_transform=None):
        
        self.transform = transform
        self.set_transform = set_transform
        self.files = []
        self.root = dataset_path
        logging.info("Initializing MulRanTupleDataset")
        logging.info(f"Loading the data from {self.root}")
        
        sequences = ['DCC/DCC_01', 'DCC/DCC_02',
              'Riverside/Riverside_01', 'Riverside/Riverside_03']
        tuple_dir = os.path.join(os.path.dirname(
            __file__), '../../configs/mulran_tuples/')
        # tuple_dir = '/configs/mulran_tuples/'
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
                fnames) > 0, f"Make sure that the path {self.root} has data {drive_id}"
            inames = sorted([int(os.path.split(fname)[-1][:-4])
                            for fname in fnames])

            for query_id, start_time in enumerate(inames):
                positives = self.get_positives(drive_id, query_id)
                non_negatives = self.get_non_negatives(drive_id, query_id)
                self.files.append((drive_id, query_id, positives, non_negatives))
             
        print('{} queries in the dataset'.format(len(self.files)))
               
    def __len__(self):
        return len(self.files)
    
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
        # Load point cloud and apply transform
        drive_id, query_id = self.files[idx][0], self.files[idx][1]
        # positive_ids, non_negatives_ids = self.files[idx][2], self.files[idx][3]
        
        query = self.get_pointcloud_tensor(drive_id, query_id)
        
        return query, idx
    
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
      
    def get_non_negatives(self, sq, index):
        assert sq in self.dict_20m.keys(), f"Error: Sequence {sq} not in json."
        sq_3 = self.dict_20m[sq]
        if str(int(index)) in sq_3:
            non_negatives = sq_3[str(int(index))]
        else:
            non_negatives = []
        return non_negatives


class PointCloudLoader:
    def __init__(self):
        # remove_zero_points: remove points with all zero coordinates
        # remove_ground_plane: remove points on ground plane level and below
        # ground_plane_level: ground plane level
        self.remove_zero_points = True
        self.remove_ground_plane = True
        self.ground_plane_level = None
        self.set_properties()

    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level. Must be defined in inherited classes.
        raise NotImplementedError('set_properties must be defined in inherited classes')

    def __call__(self, file_pathname):
        # Reads the point cloud from a disk and preprocess (optional removal of zero points and points on the ground
        # plane and below
        # file_pathname: relative file path
        assert os.path.exists(file_pathname), f"Cannot open point cloud: {file_pathname}"
        pc = self.read_pc(file_pathname)
        assert pc.shape[1] == 3

        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            mask = pc[:, 2] > self.ground_plane_level
            pc = pc[mask]

        return pc

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        raise NotImplementedError("read_pc must be overloaded in an inheriting class")

class PNVTrainingDataset(TrainingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_loader = PNVPointCloudLoader()
