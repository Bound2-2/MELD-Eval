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

4.  **Train the Pointwise Grading and Pairwise Comparison Model**

    ```bash
    # Run the training script
    bash model/train/train_pointwise.sh
     # Run the training script
    bash model/train/train_pairwise.sh
    ```

---

5.  **Merge LoRA Weights into the Base Model**
   
    ```bash
    # Run the merging script
    bash model/train/merge_models.sh
    ```


## Model Merging

After obtaining the trained pointwise scoring model and pairwise comparison model, we use the MergeKit tool to perform model merging, creating the final MELD evaluation model through different merging strategies.

1.  **Installing MergeKit**

First, install the MergeKit tool：

```bash
pip install mergekit
```
---

2.  **Preparing Merge Configuration Files**

We provide various merging strategy configuration files in the `data/merge/` directory, including:

* `dare.yaml` - Configuration based on DARE (Drop And Rescale) strategy.
* `linear.yaml` - Linear weighted merging configuration.
* `slerp.yaml` - Spherical linear interpolation merging configuration.
* `ties.yaml` - Configuration based on TIES method.

According to our experimental results, the DARE strategy provides the best performance while maintaining both evaluation capabilities.
---


### 3. Executing Model Merging

Use the following command to execute model merging:

```bash
# Using DARE strategy to merge models
mergekit-yaml data/merge/dare.yaml --out ./models/merged/MELD-8B

# Or try other merging strategies
# mergekit-yaml data/merge/linear.yaml --out ./models/merged/MELD-8B-linear
# mergekit-yaml data/merge/slerp.yaml --out ./models/merged/MELD-8B-slerp
# mergekit-yaml data/merge/ties.yaml --out ./models/merged/MELD-8B-ties
```

After merging is complete, the final MELD model will be saved in the `./models/merged/MELD-8B` directory.

