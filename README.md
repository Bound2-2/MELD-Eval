模型训练详细流程
MELD 模型的训练过程基于 LLaMA-Factory 框架，主要分为以下几个阶段：环境准备、数据准备、模型训练、模型合并。以下是完整的流程，供希望复现我们工作的研究人员参考。

1. 环境准备
首先，克隆 LLaMA-Factory 仓库并创建环境：

bash
复制
编辑
# 克隆 LLaMA-Factory 仓库
git clone https://github.com/hiyouga/LLaMA-Factory.git

# 创建并激活 conda 环境
conda create -n llama_factory python=3.12
conda activate llama_factory

# 安装依赖
cd LLaMA-Factory
pip install -e '.[torch,metrics]'
2. 数据准备
下载 LLaMA-3-8B-Instruct 模型（例如从 Hugging Face）：

bash
复制
编辑
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
将我们准备的训练数据上传至：

bash
复制
编辑
./data/train_for_llama_factory/pairwise.json
./data/train_for_llama_factory/pointwise.json
编辑 data/dataset_info.json，添加如下数据集配置：

json
复制
编辑
"MELD_pairwise": {
  "file_name": "pairwise.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
},
"MELD_pointwise": {
  "file_name": "pointwise.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
3. 模型训练
分别使用以下脚本训练两个任务模型。

第四步：训练逐点评分模型
bash
复制
编辑
# 执行训练脚本
bash scripts/train_pointwise.sh
第五步：训练成对比较模型
bash
复制
编辑
# 执行训练脚本
bash scripts/train_pairwise.sh
4. 合并 LoRA 权重到基础模型
训练完成后，将 LoRA 权重合并为完整模型：

bash
复制
编辑
# 执行合并脚本
bash scripts/merge_models.sh
合并完成后，生成的完整模型将保存在以下目录中：

./models/merged/pointwise

./models/merged/pairwise

如需我补充三个脚本的具体内容 (train_pointwise.sh, train_pairwise.sh, merge_models.sh)，也可以继续提供。是否需要一并生成？
