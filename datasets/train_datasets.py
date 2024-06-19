# Dataset for LAWS and CosPlace

import pandas as pd
import os
import torch
import random
import logging
import numpy as np
from glob import glob
import torchvision.transforms as T
from collections import defaultdict
import MinkowskiEngine as ME

from tqdm import tqdm
import math


def load_pc_file(base_path, filename):
    # returns Nx3 matrix
    pc = np.fromfile(os.path.join(base_path, filename), dtype=np.float64)
    
    if(pc.shape[0] != 4096*3):
        print("Error in pointcloud shape")
        return np.array([])
    pc = np.reshape(pc,(pc.shape[0]//3, 3))
    return pc

# Only Orthogonal Gird
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset_folder, M=10,  N=5, 
                 current_group=0, min_pointclouds_per_class=10):
        """
        Parameters (please check our paper for a clearer explanation of the parameters).
        ----------
        args : args for data augmentation
        dataset_folder : str, the path of the folder with the train images.
        M : int, the length of the side of each cell in meters.
        alpha : int, size of each class in degrees.
        N : int, distance (M-wise) between two classes of the same group.
        L : int, distance (alpha-wise) between two classes of the same group.
        current_group : int, which one of the groups to consider.
        min_pointclouds_per_class : int, minimum number of image in a class.
        """
        self.M = M
        self.N = N
        self.current_group = current_group
        self.dataset_folder = dataset_folder
        
       
        # dataset_name should be 'oxford'
        dataset_name = os.path.basename(args.dataset_folder)
        filename = f"cache/{dataset_name}_M{M}_N{N}_mipc{min_pointclouds_per_class}.torch"
        if not os.path.exists(filename):
            os.makedirs("cache", exist_ok=True)
            print(f"Cached dataset {filename} does not exist, I'll create it now.")
            self.initialize(dataset_folder, M, N, min_pointclouds_per_class, filename)
            
        elif current_group == 0:
            logging.info(f"Using cached dataset {filename}")
        
        classes_per_group, self.pointclouds_per_class = torch.load(filename)
        
        
        if current_group >= len(classes_per_group):
            raise ValueError(f"With this configuration there are only {len(classes_per_group)} " +
                             f"groups, therefore I can't create the {current_group}th group. " +
                             "You should reduce the number of groups in --groups_num")
        self.classes_ids = classes_per_group[current_group]

        print('Total group:', len(classes_per_group), 'the current_group is ', current_group, 'the group has:', len(self.classes_ids),' classes')
      
    def __getitem__(self, class_num):
        # This function takes as input the class_num instead of the index of
        # the image. This way each class is equally represented during training.
        
        class_id = self.classes_ids[class_num]
        # Pick a random image among those in this class.
        pc_path = random.choice(self.pointclouds_per_class[class_id])
        
        try:
            pc = load_pc_file(self.dataset_folder, pc_path)
        except Exception as e:
            logging.info(f"ERROR image {pc_path} couldn't be opened, it might be corrupted.")
            raise e
        
        # tensor_pc = T.functional.to_tensor(pc)
        tensor_pc = T.functional.to_tensor(pc).squeeze(0)
        
        assert tensor_pc.shape[-1] == 3, \
            f"Point cloud {pc_path} should have shape [4096,3] but has {tensor_pc.shape}."
        
        return tensor_pc, class_num, class_id
    
    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.pointclouds_per_class[c]) for c in self.classes_ids])
    
    def __len__(self):
        """Return the number of classes within this group."""
        return len(self.classes_ids)
    
    @staticmethod
    def read_utm_infos_Oxord(dataset_folder):
        print(f"Searching training images in {dataset_folder}")
        
        # Initialize pandas DataFrame
        runs_folder = "oxford/"
        filename = "pointcloud_locations_20m_10overlap.csv"
        pointcloud_fols = "/pointcloud_20m_10overlap/"
        x_width = 150
        y_width = 150
        p1 = [5735712.768124,620084.402381]
        p2 = [5735611.299219,620540.270327]
        p3 = [5735237.358209,620543.094379]
        p4 = [5734749.303802,619932.693364]
        p = [p1,p2,p3,p4]
        all_folders = sorted(os.listdir(os.path.join(dataset_folder,runs_folder)))
        folders = []
        # All runs are used for training (both full and partial)
        index_list = range(len(all_folders))
        print("Number of runs: "+str(len(index_list)))
        for index in index_list:
            folders.append(all_folders[index])
        print(folders)

        df_train = pd.DataFrame(columns=['file','northing','easting'])
        df_test = pd.DataFrame(columns=['file','northing','easting'])

        for folder in folders:
            df_locations = pd.read_csv(os.path.join(
                dataset_folder,runs_folder,folder,filename),sep=',')
            df_locations['timestamp'] = runs_folder+folder + \
                pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
            df_locations = df_locations.rename(columns={'timestamp':'file'})

            for index, row in df_locations.iterrows():
                if(TrainDataset.check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
                    df_test = df_test.append(row, ignore_index=True)
                else:
                    df_train = df_train.append(row, ignore_index=True)
        print("Number of training submaps: "+str(len(df_train['file'])))
        print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))
        return df_train
    

    @staticmethod
    def read_utm_infos(dataset_folder):
        if 'Mulran' in dataset_folder:
            # dataset_folder /home/xy/xy/Mulran
            # Initialize pandas DataFrame
            sequences = ['DCC/DCC_01', 'DCC/DCC_02',
                'Riverside/Riverside_01', 'Riverside/Riverside_03']
            df_train = pd.DataFrame(columns=['file','northing','easting'])
            df_test = pd.DataFrame(columns=['file','northing','easting'])
            for drive_id in sequences:
                df_locations = pd.read_csv(os.path.join(
                    dataset_folder, drive_id, 'scan_position.csv'),
                    sep=',',
                    header=None,
                    names=['file','1','2','3','northing','5','6','7','easting','9','10','11','height'])
                df_locations['file'] = drive_id + '/Downsample/' + df_locations['file'].astype(str)+'.bin'
                
                for index, row in df_locations.iterrows():
                    position = {'file':row['file'],'northing':row['northing'], 'easting':row['easting']}
                    df_train = df_train._append(position, ignore_index=True)    
            
            print("Number of training submaps: "+str(len(df_train['file'])))
            print(df_train)
            print("For each point cloud, get its UTM east, UTM north from its path")
        else:
            df_train = TrainDataset.read_utm_infos_Oxord(dataset_folder)    
        return df_train
    
        
    @staticmethod
    def initialize(dataset_folder, M, N, min_pointclouds_per_class, cache_name):
        
        print("For each image, get its UTM east, UTM north and heading from its path")
        df_train = TrainDataset.read_utm_infos(dataset_folder)
        # field 1 is UTM east, field 2 is UTM north, field 9 is heading
        utmeast_utmnorth_heading = [( m.northing, m.easting) for m in df_train.iloc]
        utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(np.float64)
        
        print("For each image, get class and group to which it belongs")
        class_id__group_id = [TrainDataset.get__class_id__group_id(*m, M, N)
                              for m in utmeast_utmnorth_heading]
        
        print("Group together images belonging to the same class")
        pointclouds_per_class = defaultdict(list)
        for pc_info, (class_id, _) in zip(df_train.iloc, class_id__group_id):
            pointclouds_per_class[class_id].append(pc_info.file)
            
        # pointclouds_per_class is a dict where the key is class_id, and the value
        # is a list with the paths of images within that class.
        pointclouds_per_class = {k: v for k, v in pointclouds_per_class.items() if len(v) >= min_pointclouds_per_class}
        logging.debug("Group together classes belonging to the same group")
        # Classes_per_group is a dict where the key is group_id, and the value
        # is a list with the class_ids belonging to that group.
        classes_per_group = defaultdict(set)
        count=0
        for class_id, group_id in class_id__group_id:
            if class_id not in pointclouds_per_class:
                count+=1  # 记录有多少点云被筛除了
                continue  # Skip classes with too few images
            classes_per_group[group_id].add(class_id)
        print(f"The number of {count} removed submap")
        classes_per_group = [list(c) for c in classes_per_group.values()]
        
        torch.save((classes_per_group, pointclouds_per_class), cache_name)
    
    @staticmethod
    def check_in_test_set(northing, easting, points, x_width, y_width):
        in_test_set = False
        for point in points:
            if(point[0]-x_width < northing and northing < point[0]+x_width and point[1]-y_width < easting and easting < point[1]+y_width):
                in_test_set = True
                break
        return in_test_set

    @staticmethod
    def get__class_id__group_id(utm_east, utm_north, M, N):
        """Return class_id and group_id for a given point.
            The class_id is a triplet (tuple) of UTM_east, UTM_north
             (e.g. (396520, 4983800)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1)), and it is between (0, 0) and (N, N).
        """
        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)
        
        class_id = (rounded_utm_east, rounded_utm_north)
        # group_id goes from (0, 0) to (N, N)
        group_id = (rounded_utm_east % (M * N) // M,
                    rounded_utm_north % (M * N) // M)
        return class_id, group_id

# Orthogonal Gird & Diagonal Gird
class TrainDatasetv2(TrainDataset):
    def __init__(self, args, dataset_folder, M=20,  N=5, 
                current_group=0, min_pointclouds_per_class=10):
        """
        Parameters (please check our paper for a clearer explanation of the parameters).
        ----------
        args : args for data augmentation
        dataset_folder : str, the path of the folder with the train images.
        M : int, the length of the side of each cell in meters.
        alpha : int, size of each class in degrees.
        N : int, distance (M-wise) between two classes of the same group.
        L : int, distance (alpha-wise) between two classes of the same group.
        current_group : int, which one of the groups to consider.
        min_pointclouds_per_class : int, minimum number of image in a class.
        """
        self.M = M
        self.N = N
        self.current_group = current_group
        self.dataset_folder = dataset_folder
        
        self.group_dict = {(0,0):0, (0,1):1, (1,0):2, (1,1):3, (-1):4, (0):5, (1):6, (2):7}
        self.pointclouds_per_class = defaultdict(list)
        self.classes_per_group = []
        # dataset_name should be 'oxford'
        
        dataset_name = os.path.basename(args.dataset_folder)
        cache_name = f"cache/{dataset_name}_M{M}_N{N}_mipc{min_pointclouds_per_class}_v2.torch"
        
        if not os.path.exists(cache_name):
            os.makedirs("cache", exist_ok=True)
            print(f"Cached dataset {cache_name} does not exist, I'll create it now.")
            self.update_class_group(dataset_folder, M, N, min_pointclouds_per_class, 'Ortho')
            print(len(self.classes_per_group[0]), len(self.classes_per_group[1]), len(self.classes_per_group[2]), len(self.classes_per_group[3]))
            self.update_class_group(dataset_folder, M, N, min_pointclouds_per_class, 'Tilt')
            print(len(self.classes_per_group[0]), len(self.classes_per_group[1]), len(self.classes_per_group[2]), len(self.classes_per_group[3]))
            print(len(self.classes_per_group[4]), len(self.classes_per_group[5]), len(self.classes_per_group[6]), len(self.classes_per_group[7]))
            
            torch.save((self.pointclouds_per_class, self.classes_per_group), cache_name)
        elif current_group == 0:
            logging.info(f"Using cached dataset {cache_name}")
        
        self.pointclouds_per_class, self.classes_per_group = torch.load(cache_name)
        
        if current_group >= len(self.classes_per_group):
            raise ValueError(f"With this configuration there are only {len(self.classes_per_group)} " +
                             f"groups, therefore I can't create the {current_group}th group. " +
                             "You should reduce the number of groups in --groups_num")
        self.classes_ids = self.classes_per_group[current_group]

        print('Total group:', len(self.classes_per_group), 'the current_group is ', current_group, 'the group has:', len(self.classes_ids),' classes')
    
    def update_class_group(self, dataset_folder, M, N, min_pointclouds_per_class, grid_mode):
        
        df_train = TrainDataset.read_utm_infos(dataset_folder)
        
        utmeast_utmnorth_heading = [( m.northing, m.easting) for m in df_train.iloc]
        utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(np.float64)
        
        # For each image, get class and group to which it belongs
        class_id__group_id = [self.get__class_id__group_id(*m, grid_mode, M, N)
                              for m in utmeast_utmnorth_heading]
        
        # Group together images belonging to the same class
        pointclouds_per_class = defaultdict(list)
        for pc_info, (class_id, _) in zip(df_train.iloc, class_id__group_id):
            pointclouds_per_class[class_id].append(pc_info.file)
            
        # pointclouds_per_class is a dict where the key is class_id, and the value
        # is a list with the paths of images within that class.
        count_fail, count = 0,0
        for k, v in pointclouds_per_class.items():
            count +=1
            if len(v) < min_pointclouds_per_class:
                count_fail+=1
        logging.info(f"{count_fail}/{count} classes dose not satisfied the standards")   
        pointclouds_per_class = {k: v for k, v in pointclouds_per_class.items() if len(v) >= min_pointclouds_per_class}
        
        # logging.debug("Group together classes belonging to the same group")
        
        for key, value in pointclouds_per_class.items():
            self.pointclouds_per_class[key].append(value)
            self.pointclouds_per_class[key] = self.pointclouds_per_class[key][0]
        
        added_classes_per_group = defaultdict(set)
        for class_id, group_id in class_id__group_id:
            if class_id not in self.pointclouds_per_class:
                continue  # Skip classes with too few images
            added_classes_per_group[group_id].add(class_id)
        # Convert classes_per_group to a list of lists.
        # Each sublist represents the classes within a group.
        
        if grid_mode == 'Ortho':
            for c in range(0,4):
                self.classes_per_group.append(list(added_classes_per_group[c]))
        elif grid_mode == 'Tilt':
            for c in range(4,8):
                self.classes_per_group.append(list(added_classes_per_group[c]))
     
    def get__class_id__group_id(self, utm_east, utm_north, grid_mode, M, N):
        
        if grid_mode == 'Ortho':
            """Return class_id and group_id for a given point.
                The class_id is a triplet (tuple) of UTM_east, UTM_north
                (e.g. (396520, 4983800)).
                The group_id represents the group to which the class belongs
                (e.g. (0, 1)), and it is between (0, 0) and (N, N).
            """
            M = int(M/N)   # high level gird -> low level gird  20->10
            rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
            rounded_utm_north = int(utm_north // M * M)
            
            class_id = (rounded_utm_east, rounded_utm_north)
            # group_id goes from (0, 0) to (N, N)
            group_id = (rounded_utm_east % (M * N) // M,
                        rounded_utm_north % (M * N) // M)
            group_id = self.group_dict[group_id]
            
            return class_id, group_id
        elif grid_mode == 'Tilt':
            rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
            rounded_utm_north = int(utm_north // M * M)
            
            corner = np.array([rounded_utm_east, rounded_utm_north])
            center = np.array([rounded_utm_east+(M/2), rounded_utm_north+(M/2)])
            location = np.array([utm_east, utm_north])
            v1 = corner- center
            v2 = location - center
            theta = CalAngle(v1, v2)
            rounded_theta = math.ceil(theta/90)
            
            if rounded_theta == -1:
                class_id = (int(rounded_utm_east + (M/2)), int(rounded_utm_north + ((3/4)* M)))
            elif rounded_theta == 0:
                class_id = (int(rounded_utm_east + (M/4)), int(rounded_utm_north + (M/2)))
            elif rounded_theta == 1:
                class_id = (int(rounded_utm_east + (M/2)), int(rounded_utm_north + (M/4)))
            elif rounded_theta == 2:
                class_id = (int(rounded_utm_east + ((3/4)* M)), int(rounded_utm_north + (M/2)))        
            # class_id = (rounded_utm_east, rounded_utm_north, rounded_theta)
            group_id = (rounded_theta)  # choice[-1,0,1,2]
            group_id = self.group_dict[group_id]
            return class_id, group_id
        else:
            raise ValueError('grid mode got a wrong setting. please check the config file')

# Only Diagonal Gird
class TrainDatasetv2_1(TrainDataset):
    def __init__(self, args, dataset_folder, M=10,  N=5, 
                 current_group=0, min_pointclouds_per_class=10):
        
        self.M = M
        self.N = N
        self.current_group = current_group
        self.dataset_folder = dataset_folder
        self.group_dict = {(0,0):0, (0,1):1, (1,0):2, (1,1):3, (-1):4, (0):5, (1):6, (2):7}
        
        # dataset_name should be 'oxford'
        dataset_name = os.path.basename(args.dataset_folder)
        filename = f"cache/{dataset_name}_M{M}_N{N}_mipc{min_pointclouds_per_class}_v21.torch"
        if not os.path.exists(filename):
            os.makedirs("cache", exist_ok=True)
            print(f"Cached dataset {filename} does not exist, I'll create it now.")
            self.initialize(dataset_folder, M, N, min_pointclouds_per_class, filename)
            
        elif current_group == 0:
            logging.info(f"Using cached dataset {filename}")
        
        classes_per_group, self.pointclouds_per_class = torch.load(filename)
        
        
        if current_group >= len(classes_per_group):
            raise ValueError(f"With this configuration there are only {len(classes_per_group)} " +
                             f"groups, therefore I can't create the {current_group}th group. " +
                             "You should reduce the number of groups in --groups_num")
        self.classes_ids = classes_per_group[current_group]

        print('Total group:', len(classes_per_group), 'the current_group is ', current_group, 'the group has:', len(self.classes_ids),' classes')
      
    def __len__(self):
        """Return the number of classes within this group."""
        return len(self.classes_ids)
    

       
    def initialize(self, dataset_folder, M, N, min_pointclouds_per_class, cache_name):
        
        print("For each image, get its UTM east, UTM north and heading from its path")
        df_train = TrainDataset.read_utm_infos(dataset_folder)
        # field 1 is UTM east, field 2 is UTM north, field 9 is heading
        utmeast_utmnorth_heading = [( m.northing, m.easting) for m in df_train.iloc]
        utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(np.float32)
        
        print("For each image, get class and group to which it belongs")
        class_id__group_id = [self.get__class_id__group_id(*m, M, N)
                              for m in utmeast_utmnorth_heading]
        
        print("Group together images belonging to the same class")
        pointclouds_per_class = defaultdict(list)
        for pc_info, (class_id, _) in zip(df_train.iloc, class_id__group_id):
            pointclouds_per_class[class_id].append(pc_info.file)
            
        # pointclouds_per_class is a dict where the key is class_id, and the value
        # is a list with the paths of images within that class.
        pointclouds_per_class = {k: v for k, v in pointclouds_per_class.items() if len(v) >= min_pointclouds_per_class}
        logging.debug("Group together classes belonging to the same group")
        # Classes_per_group is a dict where the key is group_id, and the value
        # is a list with the class_ids belonging to that group.
        classes_per_group = defaultdict(set)
        count=0
        for class_id, group_id in class_id__group_id:
            if class_id not in pointclouds_per_class:
                count+=1  # 记录有多少点云被筛除了
                continue  # Skip classes with too few images
            classes_per_group[group_id].add(class_id)
        print(f"The number of {count} removed submap")
        classes_per_group = [list(c) for c in classes_per_group.values()]
        
        torch.save((classes_per_group, pointclouds_per_class), cache_name)
    
    def get__class_id__group_id(self, utm_east, utm_north, M, N):
        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)
        
        corner = np.array([rounded_utm_east, rounded_utm_north])
        center = np.array([rounded_utm_east+(M/2), rounded_utm_north+(M/2)])
        location = np.array([utm_east, utm_north])
        v1 = corner- center
        v2 = location - center
        theta = CalAngle(v1, v2)
        rounded_theta = math.ceil(theta/90)
        
        if rounded_theta == -1:
            class_id = (int(rounded_utm_east + (M/2)), int(rounded_utm_north + ((3/4)* M)))
        elif rounded_theta == 0:
            class_id = (int(rounded_utm_east + (M/4)), int(rounded_utm_north + (M/2)))
        elif rounded_theta == 1:
            class_id = (int(rounded_utm_east + (M/2)), int(rounded_utm_north + (M/4)))
        elif rounded_theta == 2:
            class_id = (int(rounded_utm_east + ((3/4)* M)), int(rounded_utm_north + (M/2)))        
        # class_id = (rounded_utm_east, rounded_utm_north, rounded_theta)
        group_id = (rounded_theta)  # choice[-1,0,1,2]
        group_id = self.group_dict[group_id]
        return class_id, group_id

class ValidationDataset(TrainDatasetv2):
    def __init__(self, args, dataset_folder, transform=None, M=20, alpha=30, N=5, L=2,
                current_group=0, min_pointclouds_per_class=10):
        """
        Parameters (please check our paper for a clearer explanation of the parameters).
        ----------
        args : args for data augmentation
        dataset_folder : str, the path of the folder with the train images.
        M : int, the length of the side of each cell in meters.
        alpha : int, size of each class in degrees.
        N : int, distance (M-wise) between two classes of the same group.
        L : int, distance (alpha-wise) between two classes of the same group.
        current_group : int, which one of the groups to consider.
        min_pointclouds_per_class : int, minimum number of image in a class.
        """
        self.M = M
        self.alpha = alpha
        self.N = N
        self.L = L
        self.current_group = current_group
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.pointclouds_per_class = defaultdict(list)
        
        # dataset_name should be 'oxford'
        
        dataset_name = os.path.basename(args.dataset_folder)
        cache_name = f"cache/{dataset_name}_M{M}_N{N}_mipc{min_pointclouds_per_class}_val_test.torch"
        if not os.path.exists(cache_name):
            os.makedirs("cache", exist_ok=True)
            print(f"Cached dataset {cache_name} does not exist, I'll create it now.")
            self.update_class_group(dataset_folder, M, N, min_pointclouds_per_class)
            
            torch.save((self.pointclouds_per_class, self.classes_per_group), cache_name)
        elif current_group == 0:
            logging.info(f"Using cached dataset {cache_name}")
        
        self.pointclouds_per_class, self.classes_per_group = torch.load(cache_name)
        
        if current_group >= len(self.classes_per_group):
            raise ValueError(f"With this configuration there are only {len(self.classes_per_group)} " +
                             f"groups, therefore I can't create the {current_group}th group. " +
                             "You should reduce the number of groups in --groups_num")
        self.classes_ids = self.classes_per_group[current_group]

        print('Total group:', len(self.classes_per_group), 'the current_group is ', current_group, 'the group has:', len(self.classes_ids),' classes')

    def read_utm_infos_validation(self, dataset_folder):
        print(f"Searching training images in {dataset_folder}")
        
        # Initialize pandas DataFrame
        runs_folder = "oxford/"
        filename = "pointcloud_locations_20m.csv"
        pointcloud_fols = "/pointcloud_20m/"
        x_width = 150
        y_width = 150
        p1 = [5735712.768124,620084.402381]
        p2 = [5735611.299219,620540.270327]
        p3 = [5735237.358209,620543.094379]
        p4 = [5734749.303802,619932.693364]
        p = [p1,p2,p3,p4]
        all_folders = sorted(os.listdir(os.path.join(dataset_folder,runs_folder)))
        folders = []
        # All runs are used for training (both full and partial)
        index_list = [5,6,7,9,10,11,12,13,14,15,16,17,18,19,22,24,31,32,33,38,39,43,44]
        print("Number of runs: "+str(len(index_list)))
        for index in index_list:
            folders.append(all_folders[index])
        print(folders)

        df_train = pd.DataFrame(columns=['file','northing','easting'])
        df_test = pd.DataFrame(columns=['file','northing','easting'])

        for folder in folders:
            df_locations = pd.read_csv(os.path.join(
                dataset_folder,runs_folder,folder,filename),sep=',')
            df_locations['timestamp'] = runs_folder+folder + \
                pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
            df_locations = df_locations.rename(columns={'timestamp':'file'})

            for index, row in df_locations.iterrows():
                if(TrainDataset.check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
                    df_test = df_test.append(row, ignore_index=True)
                else:
                    df_train = df_train.append(row, ignore_index=True)
        print("Number of training submaps: "+str(len(df_train['file'])))
        print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))
        return df_test
      
    def update_class_group(self, dataset_folder, M, N, min_pointclouds_per_class):
        
        df_train = self.read_utm_infos_validation(dataset_folder)
        
        print("For each image, get its UTM east, UTM north and heading from its path")
        # field 1 is UTM east, field 2 is UTM north, field 9 is heading
        # utmeast_utmnorth_heading = [(m[1], m[2], m[9]) for m in images_metadatas]
        # utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(np.float)
        utmeast_utmnorth_heading = [( m.northing, m.easting) for m in df_train.iloc]
        utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(np.float32)
        
        # For each image, get class and group to which it belongs
        class_id__group_id = [self.get_class_id(*m, M, N)
                              for m in utmeast_utmnorth_heading]
        
        # Group together images belonging to the same class
        for pc_info, (class_id, _) in zip(df_train.iloc, class_id__group_id):
            self.pointclouds_per_class[class_id].append(pc_info.file)
            
        # pointclouds_per_class is a dict where the key is class_id, and the value
        # is a list with the paths of images within that class.
        count_fail, count = 0,0
        for k, v in self.pointclouds_per_class.items():
            count +=1
            if len(v) < min_pointclouds_per_class:
                count_fail+=1
        print(f"{count_fail}/{count} classes dose not satisfied the standards")  
        
        self.pointclouds_per_class = {k: v for k, v in self.pointclouds_per_class.items() if len(v) >= min_pointclouds_per_class}
        
        # logging.debug("Group together classes belonging to the same group")
        # added_classes_per_group is a dict where the key is group_id, and the value
        # is a list with the class_ids belonging to that group.
        classes_per_group = defaultdict(set)
        for class_id, group_id in class_id__group_id:
            if class_id not in self.pointclouds_per_class:
                continue  # Skip classes with too few images
            classes_per_group[group_id].add(class_id)
        # Convert classes_per_group to a list of lists.
        # Each sublist represents the classes within a group.
        self.classes_per_group = [list(c) for c in classes_per_group.values()]
        
    def get_class_id(self, utm_east, utm_north, M, N):
        """Return class_id and group_id for a given point.
            The class_id is a triplet (tuple) of UTM_east, UTM_north
             (e.g. (396520, 4983800)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1)), and it is between (0, 0) and (N, N).
        """
        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)
        
        class_id = (rounded_utm_east, rounded_utm_north)
        # group_id goes from (0, 0) to (N, N)
        group_id = (0, 0)
        return class_id, group_id

 
    def __getitem__(self, class_num):
        # This function takes as input the class_num instead of the index of
        # the image. This way each class is equally represented during training.
        
        class_id = self.classes_ids[class_num]
        # Pick a random image among those in this class.
        sample_num = 4 if len(self.pointclouds_per_class[class_id])>=4 else len(self.pointclouds_per_class[class_id])
        
        if len(self.pointclouds_per_class[class_id])< 4:
          print(len(self.pointclouds_per_class[class_id]))
          return None
        else:
          pc_path = random.sample(self.pointclouds_per_class[class_id], 4)
          # print(class_num, class_id, pc_path)  # 158 (5735270, 619980) oxford/2014-12-12-10-45-15/pointcloud_20m_10overlap/1418383384410661.bin
          
          tensor_pcs = []
          for file in pc_path:
              pc = load_pc_file(self.dataset_folder, file)
              # tensor_pc = T.functional.to_tensor(pc)
              assert pc.shape[-1] == 3, \
                  f"Point cloud {pc_path} should have shape [4096,3] but has {tensor_pc.shape}."
              tensor_pcs.append(pc)
          tensor_pcs = np.array(tensor_pcs)  
          tensor_pc = T.functional.to_tensor(tensor_pcs)
          
          return tensor_pcs, class_num
        
# Without Group
class TrainDatasetv0(TrainDataset):
    def __init__(self, args, dataset_folder, M=20,  N=5, 
                current_group=0,min_pointclouds_per_class=10):
        """
        Parameters (please check our paper for a clearer explanation of the parameters).
        ----------
        args : args for data augmentation
        dataset_folder : str, the path of the folder with the train images.
        M : int, the length of the side of each cell in meters.
        alpha : int, size of each class in degrees.
        N : int, distance (M-wise) between two classes of the same group.
        L : int, distance (alpha-wise) between two classes of the same group.
        current_group : int, which one of the groups to consider.
        min_pointclouds_per_class : int, minimum number of image in a class.
        """
        self.M = M
        self.N = N
        self.current_group = current_group
        self.dataset_folder = dataset_folder
        self.transform = None
        # dataset_name should be 'oxford'
        dataset_name = os.path.basename(args.dataset_folder)
        filename = f"cache/{dataset_name}_M{M}_N{N}_mipc{min_pointclouds_per_class}_all.torch"
        if not os.path.exists(filename):
            os.makedirs("cache", exist_ok=True)
            print(f"Cached dataset {filename} does not exist, I'll create it now.")
            self.initialize(dataset_folder, M, N, min_pointclouds_per_class, filename)
        elif current_group == 0:
            logging.info(f"Using cached dataset {filename}")
        
        classes_per_group, self.pointclouds_per_class = torch.load(filename)
       
        if current_group >= len(classes_per_group):
            raise ValueError(f"With this configuration there are only {len(classes_per_group)} " +
                             f"groups, therefore I can't create the {current_group}th group. " +
                             "You should reduce the number of groups in --groups_num")
        self.classes_ids = classes_per_group[current_group]

        print('Total group:', len(classes_per_group), 'the current_group is ', current_group, 'the group has:', len(self.classes_ids),' classes')
        
    def initialize(self, dataset_folder, M, N, min_pointclouds_per_class, cache_name):
        print(f"Searching training images in {dataset_folder}")
        
        # Initialize pandas DataFrame
        runs_folder = "oxford/"
        filename = "pointcloud_locations_20m_10overlap.csv"
        pointcloud_fols = "/pointcloud_20m_10overlap/"
        x_width = 150
        y_width = 150
        p1 = [5735712.768124,620084.402381]
        p2 = [5735611.299219,620540.270327]
        p3 = [5735237.358209,620543.094379]
        p4 = [5734749.303802,619932.693364]
        p = [p1,p2,p3,p4]
        all_folders = sorted(os.listdir(os.path.join(dataset_folder,runs_folder)))
        folders = []
        # All runs are used for training (both full and partial)
        index_list = range(len(all_folders))
        print("Number of runs: "+str(len(index_list)))
        for index in index_list:
            folders.append(all_folders[index])
        print(folders)

        df_train = pd.DataFrame(columns=['file','northing','easting'])
        df_test = pd.DataFrame(columns=['file','northing','easting'])

        for folder in folders:
            df_locations = pd.read_csv(os.path.join(
                dataset_folder,runs_folder,folder,filename),sep=',')
            df_locations['timestamp'] = runs_folder+folder + \
                pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
            df_locations = df_locations.rename(columns={'timestamp':'file'})

            for index, row in df_locations.iterrows():
                if(TrainDataset.check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
                    df_test = df_test.append(row, ignore_index=True)
                else:
                    df_train = df_train.append(row, ignore_index=True)
        print("Number of training submaps: "+str(len(df_train['file'])))
        print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))
        print("For each image, get its UTM east, UTM north and heading from its path")
        utmeast_utmnorth_heading = [( m.northing, m.easting) for m in df_train.iloc]
        utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(np.float32)
        
        print("For each image, get class and group to which it belongs")
        class_id__group_id = [self.get__class_id__group_id(*m, M, N)
                              for m in utmeast_utmnorth_heading]
        
        print("Group together images belonging to the same class")
        pointclouds_per_class = defaultdict(list)
        for pc_info, (class_id, _) in zip(df_train.iloc, class_id__group_id):
            pointclouds_per_class[class_id].append(pc_info.file)
        
        # pointclouds_per_class is a dict where the key is class_id, and the value
        # is a list with the paths of images within that class.
        pointclouds_per_class = {k: v for k, v in pointclouds_per_class.items() if len(v) >= min_pointclouds_per_class}
        
        logging.debug("Group together classes belonging to the same group")
        # Classes_per_group is a dict where the key is group_id, and the value
        # is a list with the class_ids belonging to that group.
        classes_per_group = defaultdict(set)
        count=0
        for class_id, group_id in class_id__group_id:
            if class_id not in pointclouds_per_class:
                count+=1  # 记录有多少点云被筛除了
                continue  # Skip classes with too few images
            classes_per_group[group_id].add(class_id)
        print('The number of removed submap: ',count)
        # Convert classes_per_group to a list of lists.
        # Each sublist represents the classes within a group.
        classes_per_group = [list(c) for c in classes_per_group.values()]
        
        torch.save((classes_per_group, pointclouds_per_class), cache_name)
    
    def get__class_id__group_id(self, utm_east, utm_north, M, N):
        
        """Return class_id and group_id for a given point.
            The class_id is a triplet (tuple) of UTM_east, UTM_north
            (e.g. (396520, 4983800)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1)), and it is between (0, 0) and (N, N).
        """
        M = int(M/N)   # high level gird -> low level gird  20->10
        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)
        
        class_id = (rounded_utm_east, rounded_utm_north)
        # group_id goes from (0, 0) to (N, N)
        group_id = (rounded_utm_east % (M * N) // M,
                    rounded_utm_north % (M * N) // M)
        group_id = (0,0)
        return class_id, group_id
 
def CalAngle(v1, v2):
        TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
        rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / TheNorm))
        # dot
        theta = np.rad2deg(np.arccos(np.dot(v1, v2) / TheNorm))
        
        if rho < 0:
            return - theta
        else:
            return theta
          
############################################################################################

# ScannetPR + CosPlace
class ScannetPRDataset(torch.utils.data.Dataset):
    """Class to handle Scannet dataset for Triple segmentation."""

    def __init__(self, args, dataset_folder, transform=None, M=10, N=4, current_group = 0, min_pointclouds_per_class=2):
        ##########################
        # Dataset folder
        self.path = dataset_folder
        self.data_path = os.path.join(self.path, 'scans')
        dataset_name = os.path.basename(args.dataset_folder)
        # The ground truth location of the subpcd is decided by 
        # world coordinate center of the subpcd, matched with v4.torch
        scene_sub_info = os.path.join(self.path,"scene_sub_infov3.csv")
        cache_name = f"cache/{dataset_name}_M{M}_N{N}_mipc{min_pointclouds_per_class}_v2.torch"
        if not os.path.exists(scene_sub_info):
            print(f"{scene_sub_info} does not exist, I'll create it now.")
            self.record_per_scan_info(scene_sub_info)
            self.class_and_group_initial(scene_sub_info, min_pointclouds_per_class, cache_name)
        elif not os.path.exists(cache_name):
            print(f"{scene_sub_info} already exist, now make label directly.")
            self.class_and_group_initial(scene_sub_info, min_pointclouds_per_class, cache_name)
        
        classes_per_group, self.pointclouds_per_class = torch.load(cache_name)
        
        if current_group >= len(classes_per_group):
            raise ValueError(f"With this configuration there are only {len(classes_per_group)} " +
                             f"groups, therefore I can't create the {current_group}th group. " +
                             "You should reduce the number of groups in --groups_num")
        self.classes_ids = classes_per_group[current_group]

        print('Total group:', len(classes_per_group), 'the current_group is ', current_group, 'the group has:', len(self.classes_ids),' classes')
      
      
    def __len__(self):
            """Return the number of classes within this group."""
            return len(self.classes_ids)

    def __getitem__(self, class_num):
        # This function takes as input the class_num instead of the index of
        # the image. This way each class is equally represented during training.
        
        class_id = self.classes_ids[class_num]
        # Pick a random image among those in this class.
        pc_path = random.choice(self.pointclouds_per_class[class_id])
        pc_path = os.path.join(self.data_path, pc_path).replace('ply','bin')
        try:
            pc = self.load_pc_file(pc_path)
        except Exception as e:
            logging.info(f"ERROR point cloud {pc_path} couldn't be opened, it might be corrupted.")
            raise e
        tensor_pc = T.functional.to_tensor(pc)
        assert tensor_pc.shape[1] == 4096, \
            f"Point cloud {pc_path} should have shape [4096,x] but has {tensor_pc.shape}."
        
        return tensor_pc, class_num, class_id
       
    def load_pc_file(self, filename):
        # returns Nx3 matrix
        pc = np.fromfile(filename, dtype=np.float64)
        
        if(pc.shape[0] != 4096*6):
            print("Error in pointcloud shape")
            return np.array([])
        pc = np.reshape(pc,(pc.shape[0]//6, 6))
        
        return pc

    def record_per_scan_info(self, scene_sub_info):
        """
        generate the record file of information about scene, file, 
        world coordinates, orientation of sub point clouds 
        """ 
        record_count=0
        for i, scene in enumerate(self.scenes):
            record_count += len(self.all_scene_pcds[scene])
            print('{}%'.format(int(100*i/len(self.scenes))), flush=True, end='\r')
            # path to original ScanNet data
            scene_folder = os.path.join(self.data_path, scene)
            # path to processed ScanNet point cloud
            scene_pcd_path = os.path.join(scene_folder, 'input_pcd_0mean')
            if not os.path.exists(scene_pcd_path):
                raise ValueError('Missing scene folder:', scene_pcd_path)
            
            world_p0s, sub_infos = [],[]
            for j, subpcd_file in enumerate(self.all_scene_pcds[scene]):
                actual_frame_id = int(subpcd_file[13:-8])
                # print('{}%'.format(int(100*j/num_scene_pcds)), flush=True, end='\r')
                frame_subpcd_file = os.path.join(scene_pcd_path, subpcd_file)
                if not os.path.exists(frame_subpcd_file):
                    raise ValueError('Missing subpcd file:', frame_subpcd_file)
                data = read_ply(frame_subpcd_file)
                sub_pts = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T # Nx3
                
                if sub_pts.shape[0] < 2:
                    raise ValueError("Empty Polygan Mesh !!!!")
                # Get center of the first frame in camera coordinates
                p0 = np.mean(sub_pts, axis=0)
                # get pose of current frame
                pose = np.loadtxt(os.path.join(scene_folder, 'pose', str(actual_frame_id)+'.txt'))
                # double check if pose is lost
                chk_val = np.sum(pose)
                if np.isinf(chk_val) or np.isnan(chk_val):
                    raise ValueError('Invalid pose value for', scene_folder, actual_frame_id)
                world_p0 = pose[:3, :3] @ p0 + pose[:3, 3]
                world_p0s.append(world_p0)
                sy = np.sqrt(pose[0,0]**2 + pose[1,0]**2)
                singular = sy < 1e-6;
                if not singular:
                    x = math.atan2( pose[2,1], pose[2,2]);
                    y = math.atan2(-pose[2,0], sy);
                    z = math.atan2( pose[1,0], pose[0,0]);
                else:
                    x = math.atan2(-pose[1,2], pose[1,1]);
                    y = math.atan2(-pose[2,0], sy);
                    z = 0;
                roll = math.degrees(x)
                pitch = math.degrees(y)
                yaw = math.degrees(z)
                sub_info=[scene, actual_frame_id, world_p0[0],world_p0[1],world_p0[2],roll,pitch,yaw]
                sub_infos.append(sub_info)
                
            scene_id = np.array([i]).tolist()
            scene_center = np.mean(np.array(world_p0s), axis=0).tolist()
            for sub_info in sub_infos:
                sub_info = scene_id + sub_info + scene_center
                with open(scene_sub_info, 'a', newline='') as csvfile:
                    posewriter = csv.writer(
                        csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    posewriter.writerow(sub_info)
        
        print('Total # of pcd:', record_count)
    
    def class_and_group_initial(self, scene_sub_info, min_pointclouds_per_class, cache_name):
        # df_train = pd.DataFrame(columns=['scene_id','file','x','y','center_x','center_y'])
        df_locations = pd.read_csv(scene_sub_info,
            sep=',',
            header=None, 
            names=['scene_id','scene','sub_id','x','y','z','roll','pitch','yaw','center_x','center_y','center_z'],
            lineterminator="\n")
        df_locations['sub_id'] =df_locations['scene']+'_'+df_locations['sub_id'].astype(str)+'_sub.ply'
        # scene=df_locations['scene'].to_numpy()
        
        print("For each point cloud, get its camera coordinate")
        print(len(df_locations))
        
        camera_heading = [(m.x, m.y, m.scene_id, m.center_x, m.center_y, m.yaw) for m in df_locations.iloc]
        camera_heading = np.array(camera_heading).astype(np.float32)

        print("For each point cloud, get class and group to which it belongs")
        
        class_id__group_id = [self.get__class_id__group_id(*m)
                            for m in camera_heading]
        
        print("Group together point clouds belonging to the same class")
        
        # get the align axis pointclouds_per_class
        pointclouds_per_class = defaultdict(list)
        for pc_info, (class_id, _) in zip(df_locations.iloc, class_id__group_id):
            pointclouds_per_class[class_id].append(os.path.join(pc_info.scene, 'downsample', pc_info.sub_id))
        # pointclouds_per_class is a dict where the key is class_id, and the value
        # is a list with the paths of images within that class.
        pointclouds_per_class = {k: v for k, v in pointclouds_per_class.items() if len(v) >= min_pointclouds_per_class}
        logging.debug("Group together classes belonging to the same group")
        # Classes_per_group is a dict where the key is group_id, and the value
        # is a list with the class_ids belonging to that group.
        classes_per_group = defaultdict(set)
        count=0
        for class_id, group_id in class_id__group_id:
            if class_id not in pointclouds_per_class:
                count+=1  # 记录有多少点云被筛除了
                continue  # Skip classes with too few images
            classes_per_group[group_id].add(class_id)
        print(f"The number of {count} removed submap")
        classes_per_group = [list(c) for c in classes_per_group.values()]
        
        torch.save((classes_per_group, pointclouds_per_class), cache_name)
        
        
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
        # get the align axis class id and group id, class_id(scene_num, [0-3])
        new_x = (x-center_x)
        new_y = (y-center_y)
        rounded_x = 2 if new_x > 0 else 0
        rounded_y = 1 if new_y > 0 else 0
        corner_class = rounded_x + rounded_y
        scene_class= int(scene_id)
        # print(x-center_x, y-center_y, rounded_x, rounded_y, room_class, scene_class)
        class_id = (scene_class, corner_class) # class_id goes from (0, 0) to (1200, 3)
        
        # group_id goes from 0 to 3
        group_id = (corner_class)
        
        return class_id, group_id
           
# ScannetPR + LAWS
class ScannetPRDatasetv2(ScannetPRDataset):
    def __init__(self, args, dataset_folder, transform=None, M=20,  N=5, 
                current_group=0, min_pointclouds_per_class=10):
        """
        Parameters (please check our paper for a clearer explanation of the parameters).
        ----------
        args : args for data augmentation
        dataset_folder : str, the path of the folder with the train images.
        M : int, the length of the side of each cell in meters.
        alpha : int, size of each class in degrees.
        N : int, distance (M-wise) between two classes of the same group.
        L : int, distance (alpha-wise) between two classes of the same group.
        current_group : int, which one of the groups to consider.
        min_pointclouds_per_class : int, minimum number of image in a class.
        """
        self.path = dataset_folder
        self.data_path = os.path.join(self.path, 'scans')
        self.current_group = current_group
        self.transform = transform
        self.pointclouds_per_class = defaultdict(list)
        self.classes_per_group = []
        
        dataset_name = os.path.basename(args.dataset_folder)
        scene_sub_info = os.path.join(self.path, "scene_sub_infov3.csv")
        cache_name = f"cache/{dataset_name}_M{M}_N{N}_mipc{min_pointclouds_per_class}_v2.torch"
        if not os.path.exists(cache_name):
            os.makedirs("cache", exist_ok=True)
            print(f"Cached dataset {cache_name} does not exist, I'll create it now.")
            self.update_class_group(scene_sub_info, min_pointclouds_per_class, 'Ortho')
            print(len(self.classes_per_group[0]), len(self.classes_per_group[1]), len(self.classes_per_group[2]), len(self.classes_per_group[3]))
            self.update_class_group(scene_sub_info, min_pointclouds_per_class, 'Tilt')
            print(len(self.classes_per_group[0]), len(self.classes_per_group[1]), len(self.classes_per_group[2]), len(self.classes_per_group[3]))
            print(len(self.classes_per_group[4]), len(self.classes_per_group[5]), len(self.classes_per_group[6]), len(self.classes_per_group[7]))
            
            torch.save((self.pointclouds_per_class, self.classes_per_group), cache_name)
        elif current_group == 0:
            logging.info(f"Using cached dataset {cache_name}")
        
        self.pointclouds_per_class, self.classes_per_group = torch.load(cache_name)
        
        if current_group >= len(self.classes_per_group):
            raise ValueError(f"With this configuration there are only {len(self.classes_per_group)} " +
                             f"groups, therefore I can't create the {current_group}th group. " +
                             "You should reduce the number of groups in --groups_num")
        self.classes_ids = self.classes_per_group[current_group]

        print('Total group:', len(self.classes_per_group), 'the current_group is ', current_group, 'the group has:', len(self.classes_ids),' classes')
    
    def update_class_group(self, scene_sub_info, min_pointclouds_per_class, grid_mode):
        # df_train = pd.DataFrame(columns=['scene_id','file','x','y','center_x','center_y'])
        df_locations = pd.read_csv(scene_sub_info,
            sep=',',
            header=None, 
            names=['scene_id','scene','sub_id','x','y','z','roll','pitch','yaw','center_x','center_y','center_z'],
            lineterminator="\n")
        df_locations['sub_id'] =df_locations['scene']+'_'+df_locations['sub_id'].astype(str)+'_sub.ply'
        # scene=df_locations['scene'].to_numpy()
        
        print("For each point cloud, get its camera coordinate")
        print(len(df_locations))
        
        camera_heading = [(m.x, m.y, m.scene_id, m.center_x, m.center_y) for m in df_locations.iloc]
        camera_heading = np.array(camera_heading).astype(np.float32)

        print("For each point cloud, get class and group to which it belongs")
        
        class_id__group_id = [self.get__class_id__group_id(*m, grid_mode)
                            for m in camera_heading]
        
        print("Group together point clouds belonging to the same class")
        
        
        # Group together point clouds belonging to the same class
        pointclouds_per_class = defaultdict(list)
        for pc_info, (class_id, _) in zip(df_locations.iloc, class_id__group_id):
            pointclouds_per_class[class_id].append(os.path.join(pc_info.scene, 'downsample', pc_info.sub_id))
        
        count_fail, count = 0,0
        for k, v in pointclouds_per_class.items():
            count +=1
            if len(v) < min_pointclouds_per_class:
                count_fail+=1
        logging.info(f"{count_fail}/{count} classes dose not satisfied the standards")   
        pointclouds_per_class = {k: v for k, v in pointclouds_per_class.items() if len(v) >= min_pointclouds_per_class}
        
        for key, value in pointclouds_per_class.items():
            self.pointclouds_per_class[key].append(value)
            self.pointclouds_per_class[key] = self.pointclouds_per_class[key][0]
        
        added_classes_per_group = defaultdict(set)
        for class_id, group_id in class_id__group_id:
            if class_id not in self.pointclouds_per_class:
                continue  # Skip classes with too few images
            added_classes_per_group[group_id].add(class_id)
        
        if grid_mode == 'Ortho':
            for c in range(0,4):
                self.classes_per_group.append(list(added_classes_per_group[c]))
        elif grid_mode == 'Tilt':
            for c in range(4,8):
                self.classes_per_group.append(list(added_classes_per_group[c]))
           
    def get__class_id__group_id(self, x, y, scene_id, center_x, center_y, grid_mode):
        if grid_mode == 'Ortho':
          new_x = (x-center_x)
          new_y = (y-center_y)
          rounded_x = 2 if new_x > 0 else 0
          rounded_y = 1 if new_y > 0 else 0
          corner_class = rounded_x + rounded_y
          scene_class= int(scene_id)
          # print(x-center_x, y-center_y, rounded_x, rounded_y, room_class, scene_class)
          class_id = (scene_class, corner_class) # class_id goes from (0, 0) to (1200, 3)
        
          # group_id goes from 0 to 3
          group_id = (corner_class)
          return class_id, group_id
        elif grid_mode == 'Tilt':
          # get the tilt axis class id and group id, class_id(scene_num, [0-3])
          # first rotate the axis by 45 degree
          theta = math.radians(45)
          new_x = (x-center_x)*math.cos(theta)+(y-center_y)*math.sin(theta)
          new_y = (y-center_y)*math.cos(theta)-(x-center_x)*math.sin(theta)
          rounded_x = 2 if new_x > 0 else 0
          rounded_y = 1 if new_y > 0 else 0
          corner_class = rounded_x + rounded_y + 4
          
          scene_class= int(scene_id)
          class_id = (scene_class, corner_class) # class_id goes from (0, 4) to (1200, 7)
          # group_id goes from 4 to 7
          group_id = (corner_class)
          return class_id, group_id
        else:
            raise ValueError('grid mode got a wrong setting. please check the config file')


        
# ScannetPR + LAWS + triplets
class ScannetPRDatasetv3(ScannetPRDatasetv2):
    def __init__(self, args, dataset_folder, transform=None, M=20,  N=5, 
                current_group=0, min_pointclouds_per_class=10):
        ScannetPRDatasetv2.__init__(self, args, dataset_folder, transform, M, N, current_group, min_pointclouds_per_class)
        self.min_pointclouds_per_class = min_pointclouds_per_class
    
    def __getitem__(self, class_num):
        # This function takes as input the class_num instead of the index of
        # the image. This way each class is equally represented during training.
        
        class_id = self.classes_ids[class_num]
        # Pick a random image among those in this class.
        # pc_path = random.choice(self.pointclouds_per_class[class_id])
        
        pc_paths = random.sample(self.pointclouds_per_class[class_id], self.min_pointclouds_per_class)
        
        tensor_pcs = []
        for pc_path in pc_paths:
            pc_path = os.path.join(self.data_path, pc_path).replace('ply','bin')
            pc = self.load_pc_file(pc_path)
            assert pc.shape[-1] == 6, \
                f"Point cloud {pc_path} should have shape [4096,3] but has {pc.shape}."
            tensor_pcs.append(pc) 
        tensor_pcs = np.array(tensor_pcs)       
        
        return tensor_pcs, class_num, class_id 
      

def make_collate_fn(dataset, quantizer, quantization_step):
  
  def collate_fn_rgb(data_list):
        # Constructs a batch object
        
        clouds = [e[0] for e in data_list]  #(B,S,4096,3)
        targets = [e[1] for e in data_list]  #(B)
        
        clouds, targets = torch.from_numpy(np.array(clouds)), torch.tensor(targets)
        
        S = clouds.shape[1]
        
        clouds=torch.flatten(clouds, start_dim=0, end_dim=1)
        targets = torch.flatten(targets.unsqueeze(-1).repeat(1,S))
       
        # Compute positives and negatives mask
        positives_mask = [[torch.eq(anc, e)  for e in targets]for anc in targets]
        negatives_mask = [[not torch.eq(anc, e)  for e in targets]for anc in targets]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)
        for i in range(len(positives_mask)):
          for j in range(len(positives_mask)):
            if i==j:
              positives_mask[i][j]=False
        
        # clouds [32,4096,6]
        # coords = [quantizer(e)[0] for e in clouds]

        batch_size, num_points = clouds.shape[0], clouds.shape[1]
        coords, feats, labels = [], [], []
        for cloud in clouds:
          xyz, rgb = cloud[:,:3].float(), cloud[:,3:].float()
          label = torch.ones((num_points, 1),dtype=torch.int32)
          # Quantize the input
          discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
              coordinates=xyz,
              features=rgb,
              labels=label,
              quantization_size=quantization_step,
              ignore_label=-100)
          coords.append(discrete_coords)
          feats.append(unique_feats)
          labels.append(unique_labels)
        coords_batch, feats_batch, labels_batch = [], [], []
        # Generate batched coordinates
        coords_batch = ME.utils.batched_coordinates(coords)
        # Concatenate all lists
        feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
        batch = {'coords': coords_batch, 'features': feats_batch}
    
        return batch, targets, positives_mask, negatives_mask
  
  def collate_fn_xyz(data_list):
        # Constructs a batch object
        
        clouds = [e[0] for e in data_list]  #(B,S,4096,3)
        labels = [e[1] for e in data_list]  #(B)
        
        clouds, labels = torch.from_numpy(np.array(clouds)), torch.tensor(labels)
        
        S = clouds.shape[1]
        
        clouds=torch.flatten(clouds, start_dim=0, end_dim=1)
        labels = torch.flatten(labels.unsqueeze(-1).repeat(1,S))
       
        # Compute positives and negatives mask
        positives_mask = [[torch.eq(anc, e)  for e in labels]for anc in labels]
        negatives_mask = [[not torch.eq(anc, e)  for e in labels]for anc in labels]
        # positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        # negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)
        for i in range(len(positives_mask)):
          for j in range(len(positives_mask)):
            if i==j:
              positives_mask[i][j]=False
        
        coords = [quantizer(e)[0] for e in clouds]

        coords = ME.utils.batched_coordinates(coords)
        # Assign a dummy feature equal to 1 to each point
        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        
        batch = {'coords': coords, 'features': feats, 'batch': clouds.type(torch.float32)}

                
        return batch, labels, positives_mask, negatives_mask
      
  
  if 'Scannet' in dataset:
    return collate_fn_rgb
  else: 
    return collate_fn_xyz
    

      