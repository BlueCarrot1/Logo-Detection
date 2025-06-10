#!/bin/bash

# 定义日志文件路径和保存路径的基本目录
LOG_DIR="./lora_lll/output_0603_mix"
SAVE_DIR="./lora_lll/output_0603_mix"


mkdir -p $SAVE_DIR/huawei_mcdonlad_logo_none_500_2

# Training with prompt ('logo', 'CAPTION_TO_PHRASE_GROUNDING') train size 50, lora_rank 2, repeat 1
START_TIME=$(date +%s)
nohup python ./lora_mix.py \
    --data_train_path_1 "./0525data/0525data/generate/records_huawei_train_500.json" \
    --data_train_image_path_1 "./0525data/0525data/generate/huawei" \
    --data_val_path_1 "./0525data/0525data/generate/records_huawei_validation_50.json" \
    --data_val_image_path_1 "./0525data/0525data/generate/huawei" \
    --data_train_path_2 "./0525data/0525data/generate/records_mcdonald_train_500.json" \
    --data_train_image_path_2 "./0525data/0525data/generate/mcdonald" \
    --data_val_path_2 "./0525data/0525data/generate/records_mcdonald_validation_50.json" \
    --data_val_image_path_2 "./0525data/0525data/generate/mcdonald" \
    --frozen_vision "True" \
    --save_path "$SAVE_DIR/huawei_mcdonlad_logo_none_500_2" \
    --checkpoint "./model/Florence-2-large-ft" \
    --epochs 31 \
    --learning_rate 5e-6 \
    --batch_size 2 \
    --num_workers 0 \
    --lora_rank 2 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --text_1 "" \
    --task_1 "huawei_logo" \
    --text_2 "" \
    --task_2 "mcdonald_logo" \
    > "$LOG_DIR/huawei_mcdonlad_logo_none_500_2/training.out" 2>&1 &
echo "Training process started with PID $! for huawei_mcdonlad_logo_none_500_2"

wait $PID
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Training process completed for huawei_mcdonlad_logo_none_500_2 in $ELAPSED_TIME seconds"
echo "Completed task"

