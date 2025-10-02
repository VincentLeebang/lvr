#!/bin/bash

# model configs
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
export WANDB_PROJECT="LVR-Qwen25-VL-3B-Vanilla-SFT"

RANDOM_SEED=42
DATA_PATH="/dockerx/groups/bangzheng/meta_data_vanilla_sft.json"
IMAGE_FOLDER="/dockerx/groups/bangzheng/images/"

# General training params
GLOBAL_BATCH_SIZE=256       # global_batch_size becomes irrelevant when use data packing
BATCH_PER_DEVICE=2            # if use data packing, BS should always be 1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# LLM-related params
LR=1e-5

MAX_TOKEN=5120
MIN_TOKEN=128


RUN_NAME="Vanilla-MaxVisToken${MAX_TOKEN}-MinVisToken${MIN_TOKEN}"
# ONLINE=True to enable online checkpointing with OCI
ONLINE=False
OUTPUT_DIR="vanilla_checkpoints/"



deepspeed /dockerx/bangzhli/projects/LVR-Finetune/src/train/train_sft.py \
    --run_name "$RUN_NAME" \
    --deepspeed /dockerx/bangzhli/projects/LVR-Finetune/scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --learning_rate $LR \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --online_checkpoint $ONLINE \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    --enable_data_packing False \
    --random_seed $RANDOM_SEED \
    # save_total_limit is for local storage only, no limit for online checkpointing