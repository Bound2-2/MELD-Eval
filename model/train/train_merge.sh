#!/bin/bash

# 合并LoRA权重到基础模型脚本
echo "开始合并模型权重..."

# 创建输出目录
mkdir -p ./models/merged/pointwise
mkdir -p ./models/merged/pairwise

# 合并逐点评分模型
echo "合并逐点评分模型..."
llamafactory-cli export \
    --model_name_or_path ./Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path ./models/pointwise_model \
    --template llama3 \
    --finetuning_type lora \
    --export_dir ./models/merged/pointwise \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False

# 合并成对比较模型
echo "合并成对比较模型..."
llamafactory-cli export \
    --model_name_or_path ./Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path ./models/pairwise_model \
    --template llama3 \
    --finetuning_type lora \
    --export_dir ./models/merged/pairwise \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False

echo "模型合并完成！"