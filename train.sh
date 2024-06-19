# #!/bin/sh
# export PYTHONPATH=./


# exp_name=ppt_mulran_cosplace
# exp_dir=exp/${exp_name}
# mkdir -p ${exp_dir}

# trainer: BasicTrainer  BackboneExpansion
# dataset: Oxforf/Muran: TrainDataset  TrainDatasetv2  TrainDatasetv0 | ScannetPR: ScannetPRDataset, ScannetPRDatasetv2 --use_rgb \
# backbone:ppt    ppt_laws  --input_channel 3 \ |   pnv   pnv_laws --input_channel 3 \ | mink_laws --input_channel 1 \
# dataset_folder  /home/xy/xy/code/Oxford/data/benchmark_datasets | /home/xy/xy/code/Mulran | /home/xy/xy/code/Data/ScannetPR |

python train.py \
          --trainer BackboneExpansion \
          --dataset_folder /home/xy/xy/code/Oxford/data/benchmark_datasets \
          --dataset TrainDatasetv2 \
          --backbone pnv_laws \
          --save_dir pnv_test/ \
          --pooling netvlad \
          --normalize_embeddings \
          --input_channel 3 \
          --batch_size 16 \
          --iterations_per_epoch 10 \
          --epochs_num 16 \
          --fc_output_dim 256 \
          --lr 0.001 \
          --M 20 \
          --N 2 \
          --groups_num 8 \
          --min_images_per_class 2 \

