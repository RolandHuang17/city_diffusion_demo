#!/bin/bash

# 设置环境变量
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="root/autodl-tmp/sddata/finetune/lora/city"
export HUB_MODEL_ID="city-lora"
# export DATASET_NAME="lambdalabs/pokemon-blip-captions"

export DATASET_NAME="/root/autodl-tmp/Proj/city_diffusion_demo/data"

# 运行训练脚本
accelerate launch --mixed_precision="fp16" train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=128 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="Totoro" \
  --seed=1337 \
  --validation_epochs 500