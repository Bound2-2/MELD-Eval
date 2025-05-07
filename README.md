## Model Training Pipeline

The MELD model is trained using the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework. The process includes the following main steps: environment setup, data preparation, model training, and LoRA weight merging. The full workflow is provided below for researchers who wish to reproduce our results.

---

1.  **Environment Setup**

    First, clone the LLaMA-Factory repository and configure the environment:

    ```bash
    # Clone the LLaMA-Factory repository
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    # Create and activate the conda environment
    conda create -n llama_factory python=3.12
    conda activate llama_factory
    # Install LLaMA-Factory
    cd LLaMA-Factory
    pip install -e '.[torch,metrics]'
    ```

---

2.  **Download the Base Model**

    Download the LLaMA-3-8B-Instruct model from Hugging Face:

    ```bash
    git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
    ```

---

3.  **Data Preparation**

    We provide two key training datasets in this repository under the `/data/train_for_llama_factory/` directory:

    * `pointwise.json`: for pointwise scoring model training  
    * `pairwise.json`: for pairwise comparison model training

    Copy these files to the `data/` directory of LLaMA-Factory:

    ```bash
    cp /path/to/this/repo/data/train_for_llama_factory/pointwise.json /path/to/LLaMA-Factory/data/
    cp /path/to/this/repo/data/train_for_llama_factory/pairwise.json /path/to/LLaMA-Factory/data/
    ```

    Then register the datasets in LLaMA-Factory by editing `llama_factory/data/dataset_info.json` and adding the following entries:

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

---

4.  **Train the Pointwise Scoring Model**

    ```bash
    # Run the training script
    bash scripts/train_pointwise.sh
    ```

---

5.  **Train the Pairwise Comparison Model**

    ```bash
    # Run the training script
    bash scripts/train_pairwise.sh
    ```

---

6.  **Merge LoRA Weights into the Base Model**

    ```bash
    # Run the merging script
    bash scripts/merge_models.sh
    ```
