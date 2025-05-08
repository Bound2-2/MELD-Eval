#!/bin/bash

# 训练成对比较模型脚本
echo "开始训练成对比较模型..."

# 创建输出目录
OUTPUT_DIR="./models/pairwise_model"
mkdir -p $OUTPUT_DIR

# 设置可见GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 执行训练命令
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path ./Meta-Llama-3-8B-Instruct/ \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir ./data/ \
    --dataset MELD_pairwise \
    --cutoff_len 2048 \
    --learning_rate 0.0002 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir $OUTPUT_DIR \
    --fp16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.04 \
    --lora_target all

echo "成对比较模型训练完成！"