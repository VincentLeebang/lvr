#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

STAGE1_STEPS=2500
CHKPT_PATH="stage1_checkpoints/Qwen2.5-VL-7B-Instruct/Stage1_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128/checkpoint-${STAGE1_STEPS}/"

# data configs
DATA_PATH="/dockerx/groups/bangzheng/data/virl39k.json"
IMAGE_FOLDER="/dockerx/groups/bangzheng/images/"
OUTPUT_DIR="stage2_checkpoints_39k/"

# training configs
FREEZE_VISION=True
FREEZE_MERGER=True
DECODING_STRATEGY="steps"
LVR_STEPS=8
LR=5e-6
TEMP=0.9
# Pretty sensitive to TEMP


# model configs
export WANDB_PROJECT="LVR-Qwen25-VL-7B-SFT-STAGE-2-39k"
RUN_NAME="Stage2_7B_decodingBy${DECODING_STRATEGY}_max${LVR_STEPS}lvrSteps_LR${LR}_TEMP${TEMP}_stage1Steps${STAGE1_STEPS}"

deepspeed src/train/train_grpo.py \
    --run_name "$RUN_NAME" \
    --deepspeed /dockerx/bangzhli/projects/LVR-Finetune/scripts/zero2.json \
    --online_checkpoint True \
    --checkpoint_name $CHKPT_PATH \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --freeze_vision_tower $FREEZE_VISION \
    --freeze_merger $FREEZE_MERGER \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --temperature $TEMP \
    --num_train_epochs 2 \
    --num_generations 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_completion_length 512 \
    --max_prompt_length 4096 \
    --image_min_pixels $((128 * 28 * 28)) \
    --image_max_pixels $((2560 * 28 * 28)) \
    --learning_rate $LR \
    --remove_unused_columns False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 50 \
    --dataloader_num_workers 8 \
    --decoding_strategy $DECODING_STRATEGY \
    --lvr_steps $LVR_STEPS \