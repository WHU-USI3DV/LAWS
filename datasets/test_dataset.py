
import os
import numpy as np
from glob import glob
import torch
import logging
from util.util import Timer
import pickle
import pandas as pd


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):
        self.phase = phase
        self.files = []

        self.random_scale = random_scale
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.random_occlusion = random_occlusion
        self.random_rotation = random_rotation
        self.rotation_range = config.rotation_range
        if random_rotation:
            print('***********************Dataloader initialized with Random Rotation. ')
        if random_occlusion:
            print('***********************Dataloader initialized with Random Occlusion. ')
        if random_scale:
            print('***********************Dataloader initialized with Random Scale. ')

    def random_rotate(self, xyzr, r_angle=360, is_random=True, add_noise=True, rand_tr=False):
        # If is_random = True: Rotate about z-axis by random angle upto 'r_angle'.
        # Else: Rotate about z-axis by fixed angle 'r_angle'.
        r_angle = (np.pi/180) * r_angle
        if is_random:
            r_angle = r_angle*np.random.uniform()
        cos_angle = np.cos(r_angle)
        sin_angle = np.sin(r_angle)
        rot_matrix = np.array([[cos_angle, -sin_angle, 0],
                               [sin_angle, cos_angle, 0],
                               [0,             0,      1]])
        scan = xyzr[:, :3]
        int = xyzr[:, 3].reshape((-1, 1))
        augmented_scan = np.dot(scan, rot_matrix)

        if add_noise:
            n_sigma = 0.01  # Add gaussian noise
            noise = np.clip(n_sigma * np.random.randn(*
                            augmented_scan.shape), -0.03, 0.03)
            augmented_scan = augmented_scan + noise

        if rand_tr:
            tr_xy_max, tr_z_max = 1.5, 0.25
            tr_xy = np.clip(np.random.randn(1, 2), -tr_xy_max, tr_xy_max)
            tr_z = np.clip(0.1*np.random.randn(1, 1), -tr_z_max, tr_z_max)
            tr = np.hstack((tr_xy, tr_z))
            augmented_scan = augmented_scan + tr

        augmented_scan = np.hstack((augmented_scan, int))
        return augmented_scan.astype(np.float32)

    def occlude_scan(self, scan, angle=30):
        # Remove points within a sector of fixed angle (degrees) and random heading direction.
        thetas = (180/np.pi) * np.arctan2(scan[:, 1], scan[:, 0])
        heading = (180-angle/2)*np.random.uniform(-1, 1)
        occ_scan = np.vstack(
            (scan[thetas < (heading - angle/2)], scan[thetas > (heading + angle/2)]))
        return occ_scan.astype(np.float32)

    def pnv_preprocessing(self, xyzr, l=25):
        ind = np.argwhere(xyzr[:, 0] <= l).reshape(-1)
        xyzr = xyzr[ind]
        ind = np.argwhere(xyzr[:, 0] >= -l).reshape(-1)
        xyzr = xyzr[ind]
        ind = np.argwhere(xyzr[:, 1] <= l).reshape(-1)
        xyzr = xyzr[ind]
        ind = np.argwhere(xyzr[:, 1] >= -l).reshape(-1)
        xyzr = xyzr[ind]
        ind = np.argwhere(xyzr[:, 2] <= l).reshape(-1)
        xyzr = xyzr[ind]
        ind = np.argwhere(xyzr[:, 2] >= -l).reshape(-1)
        xyzr = xyzr[ind]

        vox_sz = 0.3
        while len(xyzr) > 4096:
            xyzr = downsample_point_cloud(xyzr, vox_sz)
            vox_sz += 0.01

        if xyzr.shape[0] >= 4096:
            ind = np.random.choice(xyzr.shape[0], 4096, replace=False)
            xyzr = xyzr[ind, :]
        else:
            ind = np.random.choice(xyzr.shape[0], 4096, replace=True)
            xyzr = xyzr[ind, :]
        mean = np.mean(xyzr, axis=0)
        pc = xyzr - mean
        scale = np.max(abs(pc))
        pc = pc/scale
        return pc

    def __len__(self):
        return len(self.files)

class TestMulRanDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from MulRan dataset. 
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        self.root = root = config.mulran_dir
        self.pnv_prep = config.pnv_preprocessing
        self.gp_rem = config.gp_rem
        self.int_norm = config.mulran_normalize_intensity

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing MulRanDataset")
        logging.info(f"Loading the subset {phase} from {root}")

        sequences = config.mulran_data_split[phase]
        
        for drive_id in sequences:
            inames = self.get_all_scan_ids(drive_id)
            for query_id, start_time in enumerate(inames):
                self.files.append((drive_id, query_id))
            
    def get_all_scan_ids(self, drive_id):
        sequence_path = self.root + drive_id + '/Downsample/'
        fnames = sorted(glob(os.path.join(sequence_path, '*.bin')))
        assert len(
            fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    def get_velodyne_fn(self, drive_id, query_id):
        sequence_path = self.root + drive_id + '/Downsample/'
        fname = sorted(glob(os.path.join(
            sequence_path, '*.bin')))[query_id]
        return fname

    def get_pointcloud_tensor(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, pc_id)
        pc = np.fromfile(fname, dtype=np.float64).reshape(-1, 3)
        
        if(pc.shape[0] != 4096):
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

class TestKittiDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from KITTI odometry dataset. 
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        self.root = root = config.kitti_dir
        self.gp_rem = config.gp_rem
        self.pnv_prep = config.pnv_preprocessing
        self.timer = Timer()

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing KittiDataset")
        logging.info(f"Loading the subset {phase} from {root}")
        if self.gp_rem:
            logging.info("Dataloader initialized with Ground Plane removal.")

        sequences = config.kitti_data_split[phase]
        for drive_id in sequences:
            drive_id = int(drive_id)
            inames = self.get_all_scan_ids(drive_id, is_sorted=True)
            for start_time in inames:
                self.files.append((drive_id, start_time))

    def get_all_scan_ids(self, drive_id, is_sorted=False):
        fnames = glob(
            self.root + '/sequences/%02d/downsampled/*.bin' % drive_id)
        assert len(
            fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        if is_sorted:
            return sorted(inames)
        return inames

    def get_velodyne_fn(self, drive, t):
        fname = self.root + '/sequences/%02d/downsampled/%06d.bin' % (drive, t)
        return fname
    
    def get_pointcloud_tensor(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, pc_id)
        
        xyz = np.fromfile(fname, dtype=np.float64).reshape(-1, 3)
        
        return xyz

    def __getitem__(self, idx):
        drive_id = self.files[idx][0]
        t0 = self.files[idx][1]

        xyz0_th = self.get_pointcloud_tensor(drive_id, t0)
        meta_info = {'drive': drive_id, 't0': t0}

        return (xyz0_th,
                meta_info)



class ScannetPRTestDataset(torch.utils.data.Dataset):
    """Class to handle Scannet dataset for Triple segmentation."""

    def __init__(self, config, dataset_folder, set='test'):
        # ScanNetDataset.__init__(self, 'ScannetPRTestDataset')

        ##########################
        # Parameters for the files
        ##########################

        # Dataset folder
        self.path = dataset_folder
        
        self.data_path = os.path.join(self.path, 'scans')
        # self.input_pcd_path = join(self.path, 'scans', 'input_pcd_0mean')
        print("point cloud path:", self.data_path)

        # Training or test set
        self.set = set

        # Get a list of sequences
        # data_split_path = join(self.path, "test_files")
        data_split_path = os.path.join(self.path, "tools/Tasks/Benchmark")
        # data_split_path = join(self.path, "Tasks/Benchmark")
        # Cloud names
        
        scene_file_name = os.path.join(data_split_path, 'scannetv2_test.txt')
        # scene_file_name = join(data_split_path, 'scannetv2_test_val.txt')
        self.scenes = np.loadtxt(scene_file_name, dtype=np.str_)
        

        # Parameters from config
        self.config = config

        #####################
        # Prepare point cloud
        #####################
        # self.intrinsics = []    # list of Ks for every scene
        # self.nframes = []       # list of nums of frames for every scene
        self.fids = []          # list of list of actual frame id used to create pcd
        self.poses = []         # list of list of frame pose used to create pcd
        self.files = []         # list of list of pcd files created, pts in camera coordinate frame
        self.frame_ctr = []
        # training/val only
        # self.pos_thred = 2**2
        self.posIds = []        # list of dictionary of positive pcd example ids for training
        self.negIds = []        # list of dictionary of negative pcd example ids for training
        self.pcd_sizes = []     # list of list of pcd sizes
        # val/testing only
        self.class_proportions = None
        self.val_confs = []     # for validation

        # Choose batch_num in_R and max_in_p depending on validation or training
        if self.set == 'training':
            self.batch_num = config.batch_num
            self.max_in_p = config.max_in_points
            self.in_R = config.in_radius
        else:
            # Loaded from training parameters
            self.batch_num = config.val_batch_num
            self.max_in_p = config.max_val_points
            self.in_R = config.val_radius
        self.pcd_count = 0
        # load sub cloud from the HD mesh w.r.t. cameras
        self.prepare_point_cloud()

        # get all_inds as 2D array
        # (index of the scene, index of the frame)
        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.fids)])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.fids])
        # NOTE 2nd is the index of frames NOT the actual frame id
        self.all_inds = np.vstack((seq_inds, frame_inds)).T 
        
    def __len__(self):
        """
        Return the length of data here
        """
        return self.pcd_count

    def __getitem__(self, idx):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """
        # Initiate concatanation lists
        
        # Current/query pcd indices
        s_ind, f_ind = self.all_inds[idx]
        current_file = self.files[s_ind][f_ind]
        current_file = current_file.replace('input_pcd_0mean','downsample')
        
        sub_pts, sub_rgb = self.load_pc_file(current_file)
        # sub_pts = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T # Nx3
        # sub_pts = sub_pts.astype(np.float32)
        if sub_pts.shape[0] < 2:
            raise ValueError("Empty Polygan Mesh !!!!")
        
        # Get center of the first frame in camera coordinates
        p0 = np.mean(sub_pts, axis=0)
        # Convert p0 to world coordinates
        # in case of Matrix x Vector, np.dot = np.matmul
        crnt_pose = self.poses[s_ind][f_ind]
        p0 = crnt_pose[:3, :3] @ p0 + crnt_pose[:3, 3]

        ###################
        # Concatenate batch
        ###################

        frame_inds = [s_ind, f_ind]
        frame_centers =p0
        world_crt = self.frame_ctr[s_ind][f_ind][1]
        input_list =[]
        
        input_list = [sub_pts, sub_rgb, frame_inds, frame_centers, world_crt]
        # input_list.append(sub_pts)
        # input_list.append(sub_rgb)
        # input_list.append(sub_lbls)
        # input_list.append(frame_inds)
        # input_list.append(frame_centers)
        return input_list

    def load_pc_file(self, filename):
        filename = filename.replace('ply','bin')
        # returns Nx3 matrix
        pc = np.fromfile(filename, dtype=np.float64)
        
        if(pc.shape[0] != 4096*6):
            print("Error in pointcloud shape")
            return np.array([])
        pc = np.reshape(pc,(pc.shape[0]//6, 6))
        
        return pc[:,:3], pc[:,3:]
    
    def prepare_point_cloud(self):
        """
        generate sub point clouds from the complete
        reconstructed scene, using current pose and
        depth frame
        """

        if not os.path.exists(self.data_path):
            raise ValueError('Missing input pcd folder:', self.data_path)
        
        # Load pre-processed files [all tasks are stored in a single file]
        # zero-meaned pcd file names for each pcd
        valid_pcd_file = os.path.join(self.path, 'VLAD_triplets', 'vlad_pcd.pkl')
        with open(valid_pcd_file, "rb") as f:
            # dict, key = scene string, val = list of filenames
            all_scene_pcds = pickle.load(f)
        # num of pts in each pcd
        pcd_size_file = os.path.join(self.path, 'VLAD_triplets', 'pcd_size.pkl')
        with open(pcd_size_file, "rb") as f:
            # dict, key = scene string, val = list of point numbers
            dict_pcd_size = pickle.load(f)
            
        pcd_centroids_file = os.path.join(self.path, 'VLAD_triplets', 'vlad_pcd_centroids.pkl')    
        with open(pcd_centroids_file, "rb") as f:
            # dict, key = scene string, val = list of point numbers
            pcd_centroids = pickle.load(f)
        # Loop through all scenes to retrieve data for current task
        k=0
        
        for i, scene in enumerate(self.scenes):
            
            self.pcd_count  += len(all_scene_pcds[scene])
            print('{}%'.format(int(100*i/len(self.scenes))), flush=True, end='\r')
            
            # path to original ScanNet data
            scene_folder = os.path.join(self.data_path, scene)
            # path to processed ScanNet point cloud
            scene_pcd_path = os.path.join(scene_folder, 'input_pcd_0mean')
            if not os.path.exists(scene_pcd_path):
                raise ValueError('Missing scene folder:', scene_pcd_path)

            # get necessary data
            scene_files = []    # list of pcd file names for current scene
            scene_poses = []    # list of pcd poses for current scene
            scene_fids = []     # list of actual frame id used to generate the pcd
            scene_ctr = []
            for j, subpcd_file in enumerate(all_scene_pcds[scene]):
                actual_frame_id = int(subpcd_file[13:-8])
                # print('{}%'.format(int(100*j/num_scene_pcds)), flush=True, end='\r')
                frame_subpcd_file = os.path.join(scene_pcd_path, subpcd_file)

                # get pose of current frame
                pose = np.loadtxt(os.path.join(scene_folder, 'pose', str(actual_frame_id)+'.txt'))
                # double check if pose is lost
                chk_val = np.sum(pose)
                if np.isinf(chk_val) or np.isnan(chk_val):
                    raise ValueError('Invalid pose value for', scene_folder, actual_frame_id)

                # store file name info
                scene_files.append(frame_subpcd_file)
                scene_poses.append(pose)
                scene_fids.append(actual_frame_id)
                scene_ctr.append(pcd_centroids[k])
                k+=1
                # double check if the point cloud file exists
                if not os.path.exists(frame_subpcd_file):
                    raise ValueError('Missing subpcd file:', frame_subpcd_file)
            # print('100 %')

            self.files.append(scene_files)
            self.poses.append(scene_poses)
            self.fids.append(scene_fids)
            self.pcd_sizes.append(dict_pcd_size[scene])
            self.frame_ctr.append(scene_ctr)
        
        print('Total # of pcd:', self.pcd_count )


class ScannetTripleCustomBatch:
    """
    Custom batch definition with memory pinning for ScannetTriple
    Originally a custom batch only has information of 1 point cloud
    """

    def __init__(self, input_list):
        # NOTE: points in camera coordinates
        #       centers in world coordinates
        sub_pts, sub_rgb, frame_inds, frame_centers, world_crt = input_list[0]
        
        self.points = sub_pts
        self.features =  sub_rgb
        self.frame_inds = frame_inds
        self.frame_centers = frame_centers
        self.world_crt = world_crt
        return

    # def pin_memory(self):
    #     """
    #     Manual pinning of the memory
    #     """

    #     self.points = [in_tensor.pin_memory() for in_tensor in self.points]
    #     self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
    #     self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
    #     self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
    #     self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
    #     self.features = self.features.pin_memory()
    #     self.labels = self.labels.pin_memory()
    #     self.scales = self.scales.pin_memory()
    #     self.rots = self.rots.pin_memory()
    #     self.frame_inds = self.frame_inds.pin_memory()
    #     self.frame_centers = self.frame_centers.pin_memory()

    #     return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.frame_inds = self.frame_inds.to(device)
        self.frame_centers = self.frame_centers.to(device)

        return self


def ScannetTripleCollate(batch_data):
    return ScannetTripleCustomBatch(batch_data)
