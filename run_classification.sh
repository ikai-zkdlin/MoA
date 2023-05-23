#!/usr/bin/env bash         \

for DATASET in svhn food-101 ucf101 caltech101 dtd fgvc_aircraft oxford_pets eurosat sun397 RESISC45 cifar10 cifar100 oxford_flowers 
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python3 -m torch.distributed.launch  --nproc_per_node=8 --use_env main_image.py \
    --batch_size 128  --cls_token   \
    --finetune  /YOUR_MODEL_PATH/mae_pretrain_vit_b.pth \
    --dist_eval --lr 1e-3 \
    --data_path  /YOUR_DATA_PATH/ \
    --output_dir /YOUR_OUTPUT_PATH/output_${DATASET}/ \
    --drop_path 0.0  --dataset $DATASET  --use_adapt
done