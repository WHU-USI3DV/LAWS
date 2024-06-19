
#  Re-localization on Oxford RobotCar/Inhouse(uni/res/bus)
# python eval/relocalization.py --dataset_folder /home/xy/xy/code/Oxford/data/benchmark_datasets \
#                         --save_dir test/ \
#                         --backbone ppt_laws \
#                         --fc_output_dim 256 \
#                         --groups_num 8 \
#                         --batch_size 32 \
#                         --normalize_embeddings \
#                         --input_channel 1 \
#                         --eval_sequence oxf \
#                         --evaluate_model checkpoint_epoch_11.pth




#  Loop Closure on MulRan / KITTI

# eval_dataset TestMulRanDataset                 --eval_dataset TestKittiDataset \
# --mulran_eval_seq 'Riverside/Riverside_02' \   --kitti_eval_seq 8 \
# --skip_time 90                                   30

# Mulran sequences
# KAIST/KAIST_01
# DCC/DCC_03
# Riverside/Riverside_02

# KITTI sequences
# 0 2 5 6 7

python eval/loop_closure.py \
        --eval_dataset TestKittiDataset \
        --kitti_eval_seq 8 \
        --backbone ppt_laws \
        --eval_mode laws \
        --groups_num 8 \
        --input_channel 1 \
        --pooling netvlad \
        --normalize_embeddings \
        --F1_thresh_id 526 \
        --checkpoint_name 'logs/test/checkpoint_epoch_11.pth' \
        --eval_batch_size 1 \
        --skip_time 30

        

# python eval/indoor_localization.py --checkpoint_name 'logs/test/checkpoint_epoch_11.pth' \
#                                     --backbone ppt_laws \
#                                     --eval_mode laws \
#                                     --groups_num 8 \
#                                     --input_channel 3 \
#                                     --use_rgb \
#                                     --fc_output_dim 256 \
#                                     --batch_size 32 \
#                                     --normalize_embeddings
                  
