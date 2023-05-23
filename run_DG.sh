for DATASET in imagenetv2 imagenet-adversarial imagenet-sketch 
do
    for SEED in 0
    do
        for SHOT in 16
        do
            CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch  --nproc_per_node=1 --use_env main_image.py   --batch_size 128  --cls_token   \
            --finetune  /YOUR_MODEL_PATH/mae_pretrain_vit_b.pth --dist_eval --few_shot_seed $SEED --few_shot_shot $SHOT \
            --data_path  /YOUR_DATA_PATH/ \
            --source_dataset imagenet --source_data_path  /YOUR_DATA_PATH/ --dataset imagenet \
            --target_dataset $DATASET --target_data_path  /YOUR_DATA_PATH/  \
            --output_dir /YOUR_OUTPUT_PATH/  --drop_path 0.0  --use_adapt --head_num 2 --is_OOD_train
        done
    done
done