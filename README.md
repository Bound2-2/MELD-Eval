# ProjectName

A Fine-Grained Evaluation Framework for Language Models: Combining Pointwise Grading and Pairwise Comparison

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
 
## 目录

MELD-Eval/
├── README.md                      # Project overview and usage instructions
├── requirement.txt               # Python dependencies required for the project
├── data/                         # All datasets used for training and evaluation
│   ├── test/                     # Evaluation datasets
│   │   ├── bias/                 # For bias detection (position, content, length, etc.)
│   │   │   ├── LLMBar/           # Five sub-datasets for bias testing
│   │   │   └── MELD_length_bias_test.json  # Test set for analyzing length bias
│   │   ├── pairwise/             # Datasets for pairwise comparison evaluation
│   │   │   ├── MELD-Test.json
│   │   │   └── PandaLM-Test.json
│   │   └── pointwise/            # Datasets for pointwise scoring evaluation
│   │       ├── Auto-J-Test.json
│   │       ├── Feedback-Bech.json
│   │       └── MELD-Test.json
│   ├── train/                    # General-purpose training sets (raw format)
│   │   ├── pairwise.json
│   │   └── pointwise.json
│   └── train_for_llama_factory/ # Training sets formatted for LLaMA-Factory
│       ├── pairwise.json
│       └── pointwise.json
├── model/                        # Model training and merging configs/scripts
│   ├── merge/                    # Configuration files for model merging strategies
│   │   ├── dare.yaml
│   │   ├── linear.yaml
│   │   ├── slerp.yaml
│   │   └── ties.yaml
│   └── train/                    # Shell scripts to train individual and merged models
│       ├── train_merge.sh
│       ├── train_pairwise.sh
│       └── train_pointwise.sh
├── src/                          # Source code for evaluation and inference
│   ├── criteria/                 # Evaluation criteria definitions for 8 major task categories
│   │   ├── casual conversation.txt
│   │   ├── coding.txt
│   │   ├── math.txt
│   │   ├── nlp task.txt
│   │   ├── professional knowledge.txt
│   │   ├── reasoning.txt
│   │   ├── roleplay.txt
│   │   └── writing.txt
│   ├── eval/                     # Scripts for evaluating model predictions
│   │   ├── pairwise_eval.py
│   │   └── pointwise_eval.py
│   └── infer/                    # Inference scripts for generating model outputs
│       ├── pairwise_infer.py
│       └── pointwise_infer.py

## Model Training Pipeline

The MELD model is trained using the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework. The process includes the following main steps: environment setup, data preparation, model training, and LoRA weight merging. The full workflow is provided below for researchers who wish to reproduce our results.

---

1. **Environment Setup**
    
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

2. **Download the Base Model**
    
    Download the LLaMA-3-8B-Instruct model from Hugging Face:
    
    ```bash
    git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
    ```

---

3. **Data Preparation**
    
    We provide two key training datasets in this repository under the `/data/train_for_llama_factory/` directory:
    
    * `pointwise.json`: for pointwise scoring model training
    * `pairwise.json`: for pairwise comparison model training
    
    Copy these files to the `data/` directory of LLaMA-Factory:
    
    ```bash
    cp /path/to/this/repo/data/train_for_llama_factory/pointwise.json /path/to/LLaMA-Factory/data/
    cp /path/to/this/repo/data/train_for_llama_factory/pairwise.json /path/to/LLaMA-Factory/data/
    ```
    
    Then register the datasets in LLaMA-Factory by editing `llama_factory/data/dataset_info.json` and adding the following entries:
    
    ```json
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

4. **Train the Pointwise Grading and Pairwise Comparison Model**
    
    ```bash
    # Run the training script
    bash model/train/train_pointwise.sh
     # Run the training script
    bash model/train/train_pairwise.sh
    ```

---

5. **Merge LoRA Weights into the Base Model**
    
    ```bash
    # Run the merging script
    bash model/train/merge_models.sh
    ```

## Model Merging

After obtaining the trained pointwise scoring model and pairwise comparison model, we use the [MergeKit](https://github.com/arcee-ai/mergekit) to perform model merging, creating the final MELD evaluation model through different merging strategies.

---

1. **Installing MergeKit**

    First, install the MergeKit tool：

   ```bash
    pip install mergekit
   ```

---

2. **Preparing Merge Configuration Files**

    We provide various merging strategy configuration files in the `data/merge/` directory, including:

    * `dare.yaml` - Configuration based on DARE (Drop And Rescale) strategy.
    * `linear.yaml` - Linear weighted merging configuration.
    * `slerp.yaml` - Spherical linear interpolation merging configuration.
    * `ties.yaml` - Configuration based on TIES method.

    According to our experimental results, the DARE strategy provides the best performance while maintaining both evaluation capabilities.

---

3. **Executing Model Merging**

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

## Model quantization

Next, we'll quantize the merged model using [llama.cpp](https://github.com/ggml-org/llama.cpp).

---

1. **Environment Setup**
   
   First, install CMake:
   - For Linux:
    ```bash
     sudo apt update
     sudo apt install cmake
    ```
   - For Mac:
    ```bash
     brew install cmake
    ```
   Then, setting up llama.cpp Environment:
   ```bash
   # Create and Activate a Virtual Environment
   conda create -n llamacpp python=3.10 -y
   conda activate llamacpp
   ```
   ```bash
   # Install llama.cpp
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   ```
   ```bash
   #Install Dependencies
   pip install -r requirements.txt
   ```
   ```bash
   #Compile the llama.cpp Project
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```
---

2. **Convert the Model to GGUF Format**

   Next, we will convert the model stored in `./models/merged/MELD-8B` to the GGUF format.
   ```bash
   python convert_hf_to_gguf.py /Users/liyijie/dare-merge-judge-llama-3-8b-instruct --outfile /Users/liyijie/dare-merge-judge-llama-3-8b-instruct-gguf
   ```
---

3. **Quantize the Model**

   Use the following command to execute model quantizing:

   ```bash
   /Users/liyijie/llama.cpp/build/bin/llama-quantize /Users/liyijie/dare-merge-judge-llama-3-    8b-instruct-gguf /Users/liyijie/Q4_K-dare-merge-judge-llama-3-8b-instruct-gguf Q4_K
   ```

After 4-bit quantization, the MELD-8B model will be saved in the `./models/merged/MELD-8B directory.`

## Model Inference and Result Evaluation
---

1. **Model Setup**
    
    Before running the evaluation framework, you need to download and set up the required models. The framework uses several models including MELD-8B, LLaMA-3-8B-Instruct, [PandaLM-7B](https://github.com/WeOpenML/PandaLM), [Auto-J-13B](https://github.com/GAIR-NLP/auto-j), and [Prometheus-7B](https://github.com/prometheus-eval/prometheus-eval). After downloading, models should be saved to the following paths:
    
    ```
    MELD: /data/liyijie/models/base-model/meld-model/
    Llama3: /data/liyijie/models/base-model/llama-3-model/
    AutoJ: /data/liyijie/models/base-model/autoj-13b/
    Prometheus: /data/liyijie/models/base-model/prometheus-7b-v2.0
    PandaLM: /data/liyijie/models/base-model/PandaLM-7B-v1/
    ```

---

2. **Model Inference**
    
    - Pointwise Grading
    
    ```bash
    python evaluate_pointwise.py --model meld --input_file data.json --output_dir results/
    ```
   Input Format
    ```json
    [
      {
        "id": 1,
        "question_body": "Question text",
        "answer_body": "Response to evaluate",
        "label": "Optional human reference score",  
        "category": "Category" 
      }
    ]
    ```
    
    
    * Pariwise Comparison
    
    ```bash
    # Using original response order
    python evaluate_pairwise.py --model meld --input_file data.json --output_dir results/ --response_order original

    # Using swapped response order (for position bias analysis)
    python evaluate_pairwise.py --model meld --input_file data.json --output_dir results/ --response_order swapped
    ```
   Input Format
    ```json
    [
      {
        "id": 1,
        "question_body": "Question text",
        "answer1_body": "First response",
        "answer2_body": "Second response",
        "label": "Human reference label (A, B, or Tie)",  
        "category": "Category" 
      }
    ]
    ```
     Command-line Arguments
    - `--model`: Evaluation model to run (meld, llama3, autoj, prometheus, pandalm, all)
    - `--input_file`: Path to input JSON file
    - `--output_dir`: Directory for output files
    - `--criteria_dir`: Directory containing criteria files (required for MELD)
    - `--*_model_path`: Paths to respective evaluation models
    - `--seed`: Random seed for reproducibility
    
    Additional Pairwise Comparison Parameters
    - `--response_order`: Response evaluation order (original, swapped)
---
3. **Result Evaluation**
    
    - Pointwise grading
    
    ```bash
    python evaluate_pointwise.py --model all --input_file data.json --output_dir correlation_results.csv/
    ```
    - Pariwise comparison
    
    ```bash
   python analyze_correlation.py --input_dir results/ --output_csv metrics_summary.csv
    ```
---
