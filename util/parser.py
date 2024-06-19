
import os
import argparse


def parse_arguments(is_training=True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # CosPlace Groups parameters
    
    parser.add_argument("--M", type=int, default=20, help="_")
    parser.add_argument("--alpha", type=int, default=30, help="_")
    parser.add_argument("--N", type=int, default=2, help="_")
    parser.add_argument("--L", type=int, default=2, help="_")
    parser.add_argument("--groups_num", type=int, default=4, help="_")
    parser.add_argument("--min_images_per_class", type=int, default=2, help="_")
    parser.add_argument("--split", type=str, default="pos", help="_")
    parser.add_argument("--collate_fn", type=bool, default=False, help="_")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["mink", "mink_laws","pnv","pnv_laws","ppt","ppt_laws"], help="_")
    parser.add_argument("--input_channel", type=int, default=3,
                        help="Input dimension, xyz or xyzrgb")
    parser.add_argument("--fc_output_dim", type=int, default=256,
                        help="Output dimension of final fully connected layer")
    parser.add_argument("--num_clusters", type=int, default=512,
                        help="sample 512 points in the point cloud as clusters")   
    parser.add_argument("--radius", type=float, default=0.1,
                        help="radius of group")  
    parser.add_argument("--num_samples", type=int, default=128,
                        help="sample points in every cluster")    
    parser.add_argument("--local_dim", type=int, default=256,
                        help="the dimension of local feature")       
    parser.add_argument("--beta", type=float, default=0.2,
                        help="the weight of loss function 2")  
    parser.add_argument("--mode", type=str, default='mg',
                        help="the weight of loss function 2")  
    parser.add_argument("--config", type=str, default="config_baseline.txt",
                        help="the weight of loss function 2")   
    parser.add_argument("--model_config", type=str, default="minkloc3dv2.txt",
                        help="the weight of loss function 2")                                                                                          
    # # Point Transformer parameters
    # parser.add_argument("--num_point", type=int, default="4096",
    #                     help="input point numbers")
    # parser.add_argument("--nblocks", type=int, default=4,
    #                     help="_")            
    # parser.add_argument("--nneighbor", type=int, default=16,
    #                     help="_")                                       
    # parser.add_argument("--input_dim", type=int, default=3,
    #                     help="_")     
    # parser.add_argument("--num_class", type=int, default=20,
    #                     help="_")   
    # parser.add_argument("--transformer_dim", type=int, default=512,
    #                     help="_")                                              
    # Training parameters
    parser.add_argument("--augmentation_device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="on which device to run data augmentation")
    parser.add_argument("--batch_size", type=int, default=8, help="_")
    parser.add_argument("--epochs_num", type=int, default=10, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=10000, help="_")
    parser.add_argument("--lr", type=float, default=1e-5, help="_")
    parser.add_argument("--aggregators_lr", type=float, default=1e-3, help="_")
    parser.add_argument("--classifiers_lr", type=float, default=0.01, help="_")
    parser.add_argument("--trainer", type=str, default='TrainSMCL', help="_")
    parser.add_argument("--aug_mode", type=int, default=0, help="_")
    # Minkloc parameters
    parser.add_argument("--planes", type=list, default=[64,128,64,32], help="_")
    parser.add_argument("--layers", type=list, default=[1,1,1,1], help="_")
    parser.add_argument("--num_top_down", type=int, default=2, help="_")
    parser.add_argument("--conv0_kernel_size", type=int, default=5, help="_")
    parser.add_argument("--feature_size", type=int, default=256, help="_")
    parser.add_argument("--output_dim", type=int, default=256, help="_")
    parser.add_argument("--block", type=str, default='ECABasicBlock', help="_")
    parser.add_argument("--pooling", type=str, default='netvlad', help="_")
    parser.add_argument("--coordinates", type=str, default='cartesian', help="_")
    parser.add_argument("--quantization_step", type=float, default=0.01, help="_")
    parser.add_argument("--normalize_embeddings", action='store_true')
    parser.add_argument("--use_rgb", action='store_true')
    parser.add_argument("--dataloader", type=str, default="ScalableFTDataset",
                        help="how to train in a scalable way")  
    parser.add_argument("--k", type=int, default=4,
                        help="Increment every S round")
    parser.add_argument("--step", type=int, default=5,
                        help="The number of sequences of every stage")
    parser.add_argument("--scalable_setting", type=str, default="ft",
                        help="ft/jt/kd/dp")
    # Data augmentation
    parser.add_argument("--dataset", type=str, default="Oxford",
                        help="dataset type to train")                    
    # Validation / test parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (validating and testing)")
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance in meters for a prediction to be considered a positive")
    parser.add_argument("--evaluate_all_checkpoints", action='store_true')
    # Resume parameters
    parser.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth")
    parser.add_argument("--resume_model", type=str, default=None,
                        help="path to model to resume, e.g. logs/.../best_model.pth")
    parser.add_argument("--evaluate_model", type=str, default=None,
                        help="path to model to evaluation pth file")                    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--log_freq", type=int, default="3",
                        choices=["cuda", "cpu"], help="_")                    
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("--round_num", type=int, default=3, help="_")
    parser.add_argument("--num_workers", type=int, default=1, help="_")
    parser.add_argument("--margin", type=float, default=0.8, help="_")
    parser.add_argument("--m1", type=float, default=0.4, help="the left of margin")
    parser.add_argument("--m2", type=float, default=0.4, help="the right of margin")
    parser.add_argument("--schedulers_up", type=bool, default=False, help="_")
    parser.add_argument("--sim_threshold", type=float, default=0.35, help="_")
    parser.add_argument("--train_sequence", type=str, default='Karawatha', 
                        choices=["Karawatha","Venman","08","06"],
                        help="evaluation pickle file")
    parser.add_argument("--eval_sequence", type=str, default='oxf', 
                        choices=["oxf", "bus","uni","res","Karawatha","Venman","CSE3","CSE4","CSE5","DUC1","DUC2"],
                        help="evaluation pickle file")
    # Paths parameters
    parser.add_argument("--dataset_folder", type=str, default='oxford',
                        help="path of the folder with train/val/test sets")
    parser.add_argument("--dataset_folder_sub_1", type=str, default='murlan',
                        help="path of the folders with train/val/test sets")
    parser.add_argument("--dataset_folder_sub_2", type=str, default='murlan',
                        help="path of the folders with train/val/test sets")
    parser.add_argument("--dataset_folders", type=list, default='/home/xy/xy/code/Oxford/data/benchmark_datasets',
                        help="path of the folder with train/val/test sets")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="name of directory on which to save the logs, under logs/save_dir")
                      

    args = parser.parse_args()
    
    if args.dataset_folder == None:
        try:
            args.dataset_folder = os.environ['SF_XL_PROCESSED_FOLDER']
        except KeyError:
            raise Exception("You should set parameter --dataset_folder or export " +
                            "the SF_XL_PROCESSED_FOLDER environment variable as such \n" +
                            "export SF_XL_PROCESSED_FOLDER=/path/to/sf_xl/processed")
    if os.path.exists(args.dataset_folder_sub_1):
        if os.path.exists(args.dataset_folder_sub_2):
            args.dataset_folders.append(args.dataset_folder_sub_1)
            args.dataset_folders.append(args.dataset_folder_sub_2)
    if not os.path.exists(args.dataset_folder):
        if not os.path.exists(args.dataset_folders[0]) :
            raise FileNotFoundError(f"Folder {args.dataset_folder} does not exist")
    
    if is_training and os.path.exists(args.dataset_folder):
        args.train_set_folder = os.path.join(args.dataset_folder)
        if not os.path.exists(args.train_set_folder):
            raise FileNotFoundError(f"Folder {args.train_set_folder} does not exist")
        
        # args.val_set_folder = os.path.join(args.dataset_folder, "val")
        # if not os.path.exists(args.val_set_folder):
        #     raise FileNotFoundError(f"Folder {args.val_set_folder} does not exist")
    
    args.test_set_folder = os.path.join(args.dataset_folder, "test")
    # if not os.path.exists(args.test_set_folder):
    #     raise FileNotFoundError(f"Folder {args.test_set_folder} does not exist")
    
    return args

