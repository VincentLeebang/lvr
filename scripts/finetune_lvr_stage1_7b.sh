#!/bin/bash

# model configs
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
export WANDB_PROJECT="LVR-Qwen25-VL-7B-SFT-STAGE-1-450k"

# Data Config
DATA_PACKING=True

LST=4096
MAX_INSTANCE_PER_BATCH=4
MAX_PACKED_TOKENS=$((MAX_INSTANCE_PER_BATCH * LST))


RANDOM_SEED=42
DATA_PATH="data/meta_data_lvr_sft_stage1.json"

# General training params
MAX_STEPS=2500
# GLOBAL_BATCH_SIZE=256       # global_batch_size becomes irrelevant when use data packing
BATCH_PER_DEVICE=1            # if use data packing, BS should always be 1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=8
# GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# LLM-related params
LR=1e-5
LVR_HEAD=False

# LVR-related params
LVR_LOSS_FCT=mse
LAMBDA_LVR=0.1

MAX_TOKEN=5120
MIN_TOKEN=128


RUN_NAME="Stage1_${LVR_LOSS_FCT}LVRLossLambda${LAMBDA_LVR}-MaxVisToken${MAX_TOKEN}-MinVisToken${MIN_TOKEN}"
# ONLINE=True to enable online checkpointing with OCI
ONLINE=False
OUTPUT_DIR="stage1_checkpoints/"


# if continue training, set checkpoint_name = checkpoint to continue;
# --checkpoint_name checkpoint-1400


deepspeed src/train/train_lvr.py \
    --run_name "$RUN_NAME" \
    --coconut True \
    --loss_lvr_fct $LVR_LOSS_FCT\
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --remove_unused_columns False \
    --lvr_head $LVR_HEAD \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --loss_lvr_lambda $LAMBDA_LVR \
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
    --enable_data_packing $DATA_PACKING \
    --max_packed_tokens $MAX_PACKED_TOKENS \
    --random_seed $RANDOM_SEED \
    --long_seq_threshold $LST \
    --max_instance_per_batch $MAX_INSTANCE_PER_BATCH \
    # save_total_limit is for local storage only, no limit for online checkpointing