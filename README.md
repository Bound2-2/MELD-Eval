## 模型训练详细流程

MELD 模型的训练过程利用 LLaMA-Factory 框架进行，主要分为以下几个步骤：准备环境、数据准备、模型训练、模型合并。以下是完整的训练流程，供希望复现我们工作的研究人员参考。

1.  **环境准备**

    首先，克隆 LLaMA-Factory 仓库并设置环境：

    ```bash
    # 克隆 LLaMA-Factory 仓库 
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    # 创建并激活 conda 环境
    conda create -n llama_factory python=3.12
    conda activate llama_factory
    # 安装 LLaMA-Factory
    cd LLaMA-Factory
    pip install -e '.[torch,metrics]'
    ```

2.  **获取基础模型**

    从 Hugging Face 下载 LLaMA-3-8B-Instruct 模型：

    ```bash
    git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
    ```

3.  **数据准备**

    在本仓库的 \`/data/train_for_llama_factory/\` 目录中，我们提供了两个关键训练数据集：

    * `pointwise.json`：用于逐点评分模型训练
    * `pairwise.json`：用于成对比较模型训练

    将这些文件复制到 LLaMA-Factory 的数据目录：

    ```bash
    cp /path/to/this/repo/data/train_for_llama_factory/pointwise.json /path/to/LLaMA-Factory/data/
    cp /path/to/this/repo/data/train_for_llama_factory/pairwise.json /path/to/LLaMA-Factory/data/
    ```

    然后，将数据集注册到 LLaMA-Factory 中。编辑 `llama_factory/data/dataset_info.json` 文件，添加以下内容：

    ```bash
    {
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
    }
    ```

4.  **训练逐点评分模型**

    ```bash
    # 执行训练脚本
    bash scripts/train_pointwise.sh
    ```

5.  **训练成对比较模型**

    ```bash
    # 执行训练脚本
    bash scripts/train_pairwise.sh
    ```

6.  **合并 LoRA 权重到基础模型**

    ```bash
    # 执行合并脚本
    bash scripts/merge_models.sh
   ```
