#!/bin/bash

# 定义日志文件路径
LOG_FILE="./Logo/lora_lll/output_0525.out"

# 删除旧的日志文件（如果存在）
if [ -f "$LOG_FILE" ]; then
    rm "$LOG_FILE"
fi

# 设置随机种子
SEED=111

# 定义所有参数
DATA_TRAIN_PATH="./Logo/0525data/0525data/generate/records_anker_train_50.json"
DATA_TRAIN_IMAGE_PATH="./Logo/0525data/0525data/generate/anker"
DATA_VAL_PATH="./Logo/0525data/0525data/generate/records_anker_validation_5.json"
DATA_VAL_IMAGE_PATH="./Logo/0525data/0525data/generate/anker"
FROZEN_VISION="True"
SAVE_PATH="./Logo/lora_lll/output_0525/<CAPTION_TO_PHRASE_GROUNDING>logo_50"
CHECKPOINT="./model/Florence-2-large-ft"
EPOCHS=30
LEARNING_RATE=5e-6
BATCH_SIZE=2
NUM_WORKERS=0
LORA_RANK=4
LORA_ALPHA=8
LORA_DROPOUT=0.05
TEXT='logo'
TASK='<CAPTION_TO_PHRASE_GROUNDING>'

# 运行 Python 脚本
nohup python ./Logo/lora.py \
    --data_train_path "$DATA_TRAIN_PATH" \
    --data_train_image_path "$DATA_TRAIN_IMAGE_PATH" \
    --data_val_path "$DATA_VAL_PATH" \
    --data_val_image_path "$DATA_VAL_IMAGE_PATH" \
    --frozen_vision "$FROZEN_VISION" \
    --save_path "$SAVE_PATH" \
    --checkpoint "$CHECKPOINT" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --text "$TEXT" \
    --task "$TASK" \
    > "$LOG_FILE" 2>&1 &

# 输出进程 ID
echo "Training process started with PID $!"