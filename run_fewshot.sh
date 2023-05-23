for DATASET in fgvc_aircraft food-101 oxford_flowers 
do
    for SEED in 0 1 2
    do
        for SHOT in 1 2 4 8 16
        do
             CUDA_VISIBLE_DEVICES=0  python3 -m torch.distributed.launch  --nproc_per_node=1 --use_env main_image.py   --batch_size 128  --cls_token   \
	        --finetune  /YOUR_MODEL_PATH/mae_pretrain_vit_b.pth \
            --is_fewshot_train --few_shot_seed $SEED --few_shot_shot $SHOT --lr 1e-3 --dist_eval \
            --source_dataset $DATASET --source_data_path  /YOUR_DATA_PATH/ \
            --target_dataset $DATASET --target_data_path  /YOUR_DATA_PATH/ \
            --output_dir /YOUR_OUTPUT_PATH/save-${DATASET}-${SEED}-${SHOT}/  \
            --drop_path 0.0 --use_adapt --dataset $DATASET
        done
    done
done