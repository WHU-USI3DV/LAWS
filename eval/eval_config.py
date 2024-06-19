import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


# Evaluation
eval_arg = add_argument_group('Eval')
eval_arg.add_argument('--eval_pipeline', type=str, default='LOGG3D')
eval_arg.add_argument('--kitti_eval_seq', type=int, default=8)
eval_arg.add_argument('--mulran_eval_seq', type=str,
                      default='Riverside/Riverside_02')
eval_arg.add_argument('--wild_eval_seq', type=str,
                      default='Karawatha/04')
eval_arg.add_argument('--checkpoint_name', type=str,
                      default='/kitti_10cm_loo/2021-09-14_20-28-22_3n24h_Kitti_v10_q29_10s8_263169.pth')
eval_arg.add_argument('--eval_batch_size', type=int, default=1)
eval_arg.add_argument('--test_num_workers', type=int, default=1)
eval_arg.add_argument("--eval_random_rotation", type=str2bool,
                      default=False, help="If random rotation. ")
eval_arg.add_argument("--eval_random_occlusion", type=str2bool,
                      default=False, help="If random occlusion. ")

eval_arg.add_argument("--revisit_criteria", default=3,
                      type=float, help="in meters")
eval_arg.add_argument("--not_revisit_criteria",
                      default=20, type=float, help="in meters")
eval_arg.add_argument("--skip_time", default=30, type=float, help="in seconds")
eval_arg.add_argument("--cd_thresh_min", default=0.001,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--cd_thresh_max", default=1.0,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--num_thresholds", default=1000, type=int,
                      help="Number of thresholds. Number of points on PR curve.")
eval_arg.add_argument("--radius", default=0.1, type=float,
                      help="waiting to be completed")
eval_arg.add_argument("--num_samples", default=64, type=int,
                      help="waiting to be completed.")
eval_arg.add_argument("--local_dim", default=256, type=int,
                      help="waiting to be completed.")
eval_arg.add_argument("--num_clusters", default=512, type=int,
                      help="waiting to be completed.")

# Minkloc parameters
model_arg = add_argument_group('MinkModel')
model_arg.add_argument("--planes", type=list, default=[64,128,64,32], help="_")
model_arg.add_argument("--layers", type=list, default=[1,1,1,1], help="_")
model_arg.add_argument("--num_top_down", type=int, default=2, help="_")
model_arg.add_argument("--conv0_kernel_size", type=int, default=5, help="_")
model_arg.add_argument("--feature_size", type=int, default=256, help="_")
model_arg.add_argument("--output_dim", type=int, default=256, help="_")
model_arg.add_argument("--block", type=str, default='ECABasicBlock', help="_")
model_arg.add_argument("--pooling", type=str, default='netvlad', help="_")
model_arg.add_argument("--coordinates", type=str, default='cartesian', help="_")
model_arg.add_argument("--quantization_step", type=float, default=0.01, help="_")
model_arg.add_argument("--normalize_embeddings", action='store_true')
# Dataset specific configurations
data_arg = add_argument_group('Data')
# KittiDataset #MulRanDataset
data_arg.add_argument('--eval_dataset', type=str, default='KittiDataset')
data_arg.add_argument('--collation_type', type=str,
                      default='default')  # default#sparcify_list
data_arg.add_argument("--eval_save_descriptors", type=str2bool, default=False)
data_arg.add_argument("--eval_save_counts", type=str2bool, default=False)
data_arg.add_argument("--gt_overlap", type=str2bool, default=False)
data_arg.add_argument('--num_points', type=int, default=80000)
data_arg.add_argument('--voxel_size', type=float, default=0.10)
data_arg.add_argument("--gp_rem", type=str2bool,
                      default=False, help="Remove ground plane.")
data_arg.add_argument('--eval_feature_distance', type=str,
                      default='cosine')  # cosine#euclidean
data_arg.add_argument("--pnv_preprocessing", type=str2bool,
                      default=False, help="Preprocessing in dataloader for PNV.")

data_arg.add_argument('--kitti_dir', type=str, default='/home/xy/xy/code/SemanticKitti/dataset/',
                      help="Path to the KITTI odometry dataset")
data_arg.add_argument('--kitti_data_split', type=dict, default={
    'train': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'val': [],
    'test': [0]
})

data_arg.add_argument('--mulran_dir', type=str,
                      default='/home/xy/xy/code/Mulran/', help="Path to the MulRan dataset")
data_arg.add_argument("--mulran_normalize_intensity", type=str2bool,
                      default=False, help="Normalize intensity return.")
data_arg.add_argument('--mulran_data_split', type=dict, default={
    'train': ['DCC/DCC_01', 'DCC/DCC_02',
              'Riverside/Riverside_01', 'Riverside/Riverside_03'],
    'val': [],
    'test': ['KAIST/KAIST_01']
})
data_arg.add_argument('--wild_dir', type=str,
                      default='/home/xy/xy/Datasets/Wild-Places-master/datastore/kni101/wild-places/', help="Path to the Wild Place dataset")
data_arg.add_argument('--wild_data_split', type=dict, default={
    'train': ['Karawatha/01', 'Karawatha/02',
              'Venman/01', 'Venman/02'],
    'val': [],
    'test': ['Karawatha/03', 'Karawatha/04',
              'Venman/03', 'Venman/04']
})

# data_arg.add_argument('--scannetpr_dir', type=str,
#                       default='/root/siton-tmp/data/ScannetPR', help="Path to the MulRan dataset")
data_arg.add_argument('--scannetpr_dir', type=str,
                      default='/home/xy/xy/code/Data/ScannetPR', help="Path to the MulRan dataset")
parser.add_argument("--use_rgb", action="store_true",
                        help="use Automatic Mixed Precision")                      
# Data loader configs
data_arg.add_argument('--train_phase', type=str, default="train")
data_arg.add_argument('--F1_thresh_id', type=int, default=100)
data_arg.add_argument('--val_phase', type=str, default="val")
data_arg.add_argument('--test_phase', type=str, default="test")
data_arg.add_argument('--use_random_rotation', type=str2bool, default=False)
data_arg.add_argument('--rotation_range', type=float, default=360)
data_arg.add_argument('--use_random_occlusion', type=str2bool, default=False)
data_arg.add_argument('--occlusion_angle', type=float, default=30)
data_arg.add_argument('--use_random_scale', type=str2bool, default=False)
data_arg.add_argument('--min_scale', type=float, default=0.8)
data_arg.add_argument('--max_scale', type=float, default=1.2)
data_arg.add_argument("--backbone", type=str, default="3d",
                        choices=["ppt","ppt_laws","pnv","pnv_laws","mink","mink_laws"])
data_arg.add_argument("--batch_size", type=int, default=32)
data_arg.add_argument("--groups_num", type=int, default=4)
data_arg.add_argument("--fc_output_dim", type=int, default=256)
data_arg.add_argument("--eval_mode", type=str, default='cp')
data_arg.add_argument("--input_channel", type=int, default=3)
data_arg.add_argument("--topN", type=int, default=5)
# # PPT config
data_arg.add_argument("--FEATURE_SIZE", type=list, default=[256,256,256,256],
                    help="path of the folder with train/val/test sets")
data_arg.add_argument("--MAX_SAMPLES", type=list, default=[64,256,1024,4096],
                    help="__Undetermined")
data_arg.add_argument("--CLUSTER_SIZE", type=list, default=[1,4,16,64],
                    help="__Undetermined")  
data_arg.add_argument("--OUTPUT_DIM", type=list, default=[256,256,256,256],
                    help="__Undetermined")  
data_arg.add_argument("--SAMPLING", type=list, default=[1024,256, 64,16],
                    help="__Undetermined")  
data_arg.add_argument("--KNN", type=list, default=[20,20,20,20],
                    help="__Undetermined")  
data_arg.add_argument("--GROUP", type=int, default=8,
                    help="__Undetermined")    
data_arg.add_argument("--AGGREGATION", type=str, default='spvlad',
                    help="__Undetermined")       
data_arg.add_argument("--GATING", type=bool, default= True,
                    help="__Undetermined")    


def get_config_eval():
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cfg = get_config_eval()
    dconfig = vars(cfg)
    print(dconfig)
