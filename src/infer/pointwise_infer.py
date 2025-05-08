import os
import re
import sys
import json
import argparse
import logging
import random
import torch
from tqdm import tqdm

# Set up logging
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

# AutoJ 常量和函数
PROMPT_INPUT_SYSTEM = '[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{input} [/INST]'
PROMPT_INPUT_WO_SYSTEM = "[INST] {input} [/INST]"
PROMPT_INPUT_FOR_SCENARIO_CLS = "Identify the scenario for the user's query, output 'default' if you are uncertain.\nQuery:\n{input}\nScenario:\n"

AUTOJ_PROMPT_SINGLE = """Write critiques for a submitted response on a given user's query, and grade the response:
  
[BEGIN DATA]
***
[Query]: {prompt}
***
[Response]: {response}
***
[END DATA]

Write critiques for this response. After that, you should give a final rating for the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]"."""

AUTOJ_PROMPT_PAIRWISE_TIE = """You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {prompt}
***
[Response 1]: {response}
***
[Response 2]: {response_another}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided."""

AUTOJ_PROTOCOL_MAPPING = {
    "pairwise_tie": AUTOJ_PROMPT_PAIRWISE_TIE,
    "single": AUTOJ_PROMPT_SINGLE,
}

def llama2_wrapper(usr_msg, sys_msg=None):
    if sys_msg is None:
        return PROMPT_INPUT_WO_SYSTEM.format(input=usr_msg)
    else:
        return PROMPT_INPUT_SYSTEM.format(input=usr_msg, system_message=sys_msg)

def build_autoj_input(prompt, resp1, resp2=None, protocol="single"):
    user_msg = AUTOJ_PROTOCOL_MAPPING[protocol].format(prompt=prompt, response=resp1, response_another=resp2)
    return llama2_wrapper(user_msg)

def seed_everything(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def extract_overall_score(input_string):
    """
    Extract the overall score number from a string.
    
    :param input_string: Input string containing score information
    :return: Extracted score as float, or "extraction failed"
    """
    # Try to match "Overall Score" pattern first (MELD format)
    match = re.search(r'Overall Score(?:\'|\")?:\s*([0-9]+(?:\.[0-9]+)?)', input_string)
    if match:
        return float(match.group(1))
    
    # Try to match single number (Llama3 format)
    match = re.search(r'\b([1-9]|10)\b', input_string)
    if match:
        return float(match.group(1))
    
    return "extraction failed"

def read_criteria(category, criteria_dir):
    """Read evaluation criteria from file based on category"""
    try:
        criteria_path = os.path.join(criteria_dir, f"{category}.txt")
        with open(criteria_path, 'r', encoding='utf-8') as file:
            criteria = file.read()
        return criteria
    except FileNotFoundError:
        logging.warning(f"No criteria found for {category}")
        return "casual conversation"

def write_result_to_jsonl(item, file, model_name):
    """统一输出格式为JSONL，确保包含score和result字段"""
    # 确保原始标签存储在score字段中
    if 'score' not in item and 'label' in item:
        item['score'] = item['label']
    
    # 添加评估模型标识
    item['model'] = model_name
    
    # 写入JSONL格式
    json.dump(item, file, ensure_ascii=False)
    file.write('\n')

class MELDEvaluator:
    """MELD model evaluator using vLLM"""
    
    def __init__(self, model_path, criteria_dir=None):
        self.model_path = model_path
        self.criteria_dir = criteria_dir
        self.llm = None
        
    def initialize_model(self):
        """Initialize the vLLM model"""
        try:
            from vllm import LLM, SamplingParams
            
            # Initialize the model only once
            if self.llm is None:
                logging.info(f"Loading MELD model from {self.model_path}")
                num_gpus = torch.cuda.device_count()
                self.llm = LLM(
                    model=self.model_path,
                    tensor_parallel_size=num_gpus,
                    gpu_memory_utilization=0.95,
                    max_model_len=4096
                )
                self.sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048)
            
            return True
        except ImportError:
            logging.error("vLLM module not found. Please install with: pip install vllm")
            return False
        except Exception as e:
            logging.error(f"Error initializing vLLM model: {e}")
            return False
    
    def call_model(self, prompt):
        """Call the vLLM model with the given prompt"""
        if not self.initialize_model():
            return "Error initializing model"
        
        try:
            # For chat models, wrap the prompt in a chat format
            messages = [{"role": "user", "content": prompt}]
            prompt_formatted = json.dumps(messages)
            
            outputs = self.llm.generate(prompt_formatted, self.sampling_params)
            if outputs and len(outputs) > 0 and len(outputs[0].outputs) > 0:
                # Try to parse as JSON first (for chat models)
                try:
                    response_json = json.loads(outputs[0].outputs[0].text)
                    if isinstance(response_json, list) and len(response_json) > 0:
                        for msg in response_json:
                            if msg.get("role") == "assistant":
                                return msg.get("content", "")
                    return outputs[0].outputs[0].text
                except json.JSONDecodeError:
                    # Not JSON, return as-is
                    return outputs[0].outputs[0].text
            return "No output generated"
        except Exception as e:
            logging.error(f"Model call error: {e}")
            return f"Error calling model: {str(e)}"
    
    def evaluate(self, input_data, output_path, start_index=0):
        """Evaluate responses using MELD criteria-based approach"""
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for index, data in enumerate(input_data[start_index:], start=start_index):
                instruction = data.get('question_body')
                response = data.get('answer_body')
                category = data.get("category", "general")
                
                # Get criteria if available
                criteria = ""
                if self.criteria_dir:
                    criteria = read_criteria(category, self.criteria_dir)
                
                prompt = f'''
                    You are assessing submitted response on a given user's query based on the criteria you have known and evaluating the quality of a response. Here is the data: 
                    [BEGIN DATA]
                    *** 
                    [Query]: {instruction}
                    *** 
                    [Response]: {response}
                    *** 
                    [END DATA]
                    You are given the criteria to craft good responses for this type of query from users: 
                    {category} 
                    The criteria are as follows: 
                    [Criteria start] 
                    {criteria}
                    [Criteria end]
                    Please follow the evaluation process below:
                    1.Review the response and the given criteria, evaluate the AI assistant's response from different dimensions, assigning a score of 1 to 10 for each dimension. For the scoring, return all your results in the following dictionary format (including the brackets), and ensure that your scores are integers: {{'Dimension 1': score, 'Dimension 2': score, ..., 'Overall Score': score}}, for example: {{'Factual Accuracy': 9, 'Meeting User Needs': 6, ..., 'Overall Score': 7}}.
                    2.Calculate the final score for the response. The final score is the average of the scores for each dimension. Round the result to the nearest integer.
                    3.Please Write detailed feedback. Based on the provided scoring criteria, write detailed evaluation feedback that strictly assesses the response quality rather than offering a general assessment. Ensure a comprehensive evaluation in line with the scoring criteria without breaking them down into points or making repetitive statements. Additionally, brainstorm to deliver thorough feedback that demonstrates the assessment thought process. Also, do not explicitly mention the scoring results in the detailed feedback as these have already been provided.
                    4. Please do not generate any additional openings, conclusions, or explanations.
                    The output format should be as follows: 
                    @@@Dimension Scores: {{'Dimension 1': score, 'Dimension 2': score, ..., 'Overall Score': score}}###Overall Score: {{score}}&&&Detailed Evaluation Feedback: {{evaluation content}}***
                '''
                
                model_answer = self.call_model(prompt)
                data['meld_response'] = model_answer
                result = extract_overall_score(model_answer)
                data['result'] = result
                
                # 使用统一的写入函数
                write_result_to_jsonl(data, output_file, "meld")
                
                logging.info(f"{index}: {result}")


class Llama3Evaluator:
    """Llama3 model evaluator using vLLM"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.llm = None
        
    def initialize_model(self):
        """Initialize the vLLM model"""
        try:
            from vllm import LLM, SamplingParams
            
            # Initialize the model only once
            if self.llm is None:
                logging.info(f"Loading Llama3 model from {self.model_path}")
                num_gpus = torch.cuda.device_count()
                self.llm = LLM(
                    model=self.model_path,
                    tensor_parallel_size=num_gpus,
                    gpu_memory_utilization=0.95,
                    max_model_len=4096
                )
                self.sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256)
            
            return True
        except ImportError:
            logging.error("vLLM module not found. Please install with: pip install vllm")
            return False
        except Exception as e:
            logging.error(f"Error initializing vLLM model: {e}")
            return False
    
    def call_model(self, prompt):
        """Call the vLLM model with the given prompt"""
        if not self.initialize_model():
            return "Error initializing model"
        
        try:
            # For chat models, wrap the prompt in a chat format
            messages = [{"role": "user", "content": prompt}]
            prompt_formatted = json.dumps(messages)
            
            outputs = self.llm.generate(prompt_formatted, self.sampling_params)
            if outputs and len(outputs) > 0 and len(outputs[0].outputs) > 0:
                # Try to parse as JSON first (for chat models)
                try:
                    response_json = json.loads(outputs[0].outputs[0].text)
                    if isinstance(response_json, list) and len(response_json) > 0:
                        for msg in response_json:
                            if msg.get("role") == "assistant":
                                return msg.get("content", "")
                    return outputs[0].outputs[0].text
                except json.JSONDecodeError:
                    # Not JSON, return as-is
                    return outputs[0].outputs[0].text
            return "No output generated"
        except Exception as e:
            logging.error(f"Model call error: {e}")
            return f"Error calling model: {str(e)}"
    
    def evaluate(self, input_data, output_path, start_index=0):
        """Evaluate responses using Llama3 simple scoring approach"""
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for index, data in enumerate(input_data[start_index:], start=start_index):
                instruction = data.get('question_body')
                response = data.get('answer_body')

                prompt = f'''
                Below are a response for a given task. The task is defined by the Instruction. Evaluate the response with an overall score on a scale of 1 to 10.

                [BEGIN DATA] 
                ### Instruction:
                {instruction}
                ### Response:
                {response}
                [END DATA]

                [your verdict here]:(The verdict should be an integer from 1 to 10, nothing else. Don't output any prefix here.)
                '''
                
                model_answer = self.call_model(prompt)
                data['llama3_response'] = model_answer
                
                # Extract score (should be just a number from 1-10)
                try:
                    score = float(model_answer.strip())
                except ValueError:
                    score = extract_overall_score(model_answer)
                
                data['result'] = score
                
                # 使用统一的写入函数
                write_result_to_jsonl(data, output_file, "llama3")
                
                logging.info(f"{index}: {score}")


class AutoJEvaluator:
    """AutoJ model evaluator using vLLM"""
    
    def __init__(self, model_path):
        self.model_path = model_path
    
    def extract_single_rating(self, score_output):
        """Extract rating from AutoJ output"""
        if "Rating: [[" in score_output:
            pos = score_output.rfind("Rating: [[")
            pos2 = score_output.find("]]", pos)
            if pos != -1 and pos2 != -1:
                return float(score_output[pos + len("Rating: [["):pos2].strip())
        return 0.0
    
    def evaluate(self, input_data, output_path):
        """Evaluate responses using AutoJ approach"""
        try:
            from vllm import LLM, SamplingParams
            
            # Initialize the model
            num_gpus = torch.cuda.device_count()
            llm = LLM(
                model=self.model_path,
                tensor_parallel_size=num_gpus,
                gpu_memory_utilization=0.95,
                max_model_len=4096
            )
            sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)
            
            # Open output file
            with open(output_path, 'w', encoding='utf-8') as f:
                # Process each input item
                for idx, item in enumerate(input_data):
                    prompt = item.get("question_body", "")
                    response = item.get("answer_body", "")
                    
                    # Build input using the integrated build_autoj_input function
                    input_text = build_autoj_input(prompt=prompt, resp1=response, resp2=None, protocol="single")
                    
                    # Generate model output
                    outputs = llm.generate(input_text, sampling_params)
                    
                    # Extract rating
                    rating = self.extract_single_rating(outputs[0].outputs[0].text)
                    
                    # Print result
                    logging.info(f"{idx}: Response Rating: {rating}")
                    
                    # Store result
                    item["autoj_response"] = outputs[0].outputs[0].text
                    item["result"] = rating
                    
                    # 使用统一的写入函数
                    write_result_to_jsonl(item, f, "autoj")
                
        except ImportError:
            logging.error("vLLM module not found. Please install with: pip install vllm")
        except Exception as e:
            logging.error(f"Error in AutoJ evaluation: {e}")


class PrometheusEvaluator:
    """Prometheus model evaluator"""
    
    def __init__(self, model_path):
        self.model_path = model_path
    
    def evaluate(self, input_data, output_path):
        """Evaluate responses using Prometheus approach"""
        try:
            from prometheus_eval.vllm import VLLM
            from prometheus_eval import PrometheusEval
            from prometheus_eval.prompts import RELATIVE_PROMPT_WO_REF, HELPFULNESS_RUBRIC
            
            # Initialize the model
            model = VLLM(model=self.model_path, dtype="float16")
            judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT_WO_REF)
            
            # Open output file
            with open(output_path, 'w', encoding='utf-8') as f:
                # Process each input item
                for idx, item in enumerate(input_data):
                    instruction = item.get("question_body", "")
                    response = item.get("answer_body", "")
                    
                    # Get evaluation
                    feedback, score = judge.single_absolute_grade(
                        instruction=instruction,
                        response=response,
                        rubric=HELPFULNESS_RUBRIC
                    )
                    
                    # Store results
                    item["prometheus_response"] = feedback
                    item["result"] = score
                    
                    # Print result
                    logging.info(f"{idx}: {score}")
                    
                    # 使用统一的写入函数
                    write_result_to_jsonl(item, f, "prometheus")
                
        except ImportError:
            logging.error("Prometheus modules not found. Please install prometheus_eval and its dependencies.")
        except Exception as e:
            logging.error(f"Error in Prometheus evaluation: {e}")


class PandaLMEvaluator:
    """PandaLM model evaluator - 直接集成到代码中"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pattern = None
        
    def initialize_model(self):
        """初始化PandaLM模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import transformers
            
            if self.model is None:
                logging.info(f"加载PandaLM模型: {self.model_path}")
                
                # 加载tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                except:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
                
                # 加载模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                
                # 添加特殊token
                if self.tokenizer.pad_token is None:
                    self._smart_tokenizer_and_embedding_resize(
                        special_tokens_dict=dict(pad_token="[PAD]"),
                        tokenizer=self.tokenizer,
                        model=self.model,
                    )
                
                self.tokenizer.add_special_tokens(
                    {
                        "eos_token": "</s>",
                        "bos_token": "</s>",
                        "unk_token": "</s>",
                    }
                )
                
                # 设置token IDs
                self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
                self.model.config.bos_token_id = 1
                self.model.config.eos_token_id = 2
                self.model.eval()
                
                # 编译模型（如果可能）
                if torch.__version__ >= "2" and sys.platform != "win32":
                    self.model = torch.compile(self.model)
                
                # 设置特殊字符正则表达式
                self.pattern = re.compile(
                    r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
                )
            
            return True
        except ImportError:
            logging.error("transformers模块未找到。请安装: pip install transformers")
            return False
        except Exception as e:
            logging.error(f"初始化PandaLM模型时出错: {e}")
            return False
    
    def _smart_tokenizer_and_embedding_resize(self, special_tokens_dict, tokenizer, model):
        """调整tokenizer和embedding大小"""
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
    def build_single_response_prompt(self, instruction, response):
        """构建评估单一响应的提示"""
        response = self.pattern.sub("", response.strip()).strip()
        prompt = (
            f"Below are a response for a given task. The task is defined by the Instruction. "
            f"Evaluate the response with an overall score on a scale of 1 to 10. The Evaluation should be an integer from 1 to 10, nothing else. Don't output any prefix here.)\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}\n\n"
            f"### Evaluation:\n"
        )
        return prompt
    
    def postprocess_output(self, text):
        """后处理输出，提取评分和其他详细信息"""
        try:
            evaluation_section = text.strip().split("### Evaluation:")[1].strip()
            score_match = re.search(r"\b([1-9]|10)\b", evaluation_section)
            score = int(score_match.group(1)) if score_match else None
            return {"score": score, "evaluation": evaluation_section}
        except:
            return {"score": None, "evaluation": "提取失败"}
    
    def evaluate(self, input_data, output_path, seed=42):
        """评估单一响应"""
        if not self.initialize_model():
            logging.error("无法初始化PandaLM模型")
            return
        
        from transformers import GenerationConfig
        
        # 设置随机种子
        seed_everything(seed)
        
        # 预处理输入
        prepared_inputs = []
        for item in tqdm(input_data, desc="准备输入"):
            prompt = self.build_single_response_prompt(
                instruction=item["question_body"],
                response=item["answer_body"]
            )
            prepared_inputs.append(self.tokenizer(prompt, return_tensors="pt", padding=True))
        
        # 生成输出
        generated_outputs = []
        for inputs in tqdm(prepared_inputs, desc="生成评估"):
            input_ids = inputs["input_ids"].to(self.model.device)
            
            generation_config = GenerationConfig(
                temperature=0,
                top_p=1,
                top_k=1,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.2,
                max_new_tokens=256
            )
            
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=256
                )
            
            for seq in generation_output.sequences:
                output = self.tokenizer.decode(seq)
                generated_outputs.append(output)
        
        # 处理结果并写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, (item, output) in enumerate(zip(input_data, generated_outputs)):
                evaluation = self.postprocess_output(output)
                
                # 更新结果
                item["pandalm_response"] = output
                item["result"] = evaluation["score"]
                
                # 使用统一的写入函数
                write_result_to_jsonl(item, f, "pandalm")
                
                logging.info(f"{idx}: {evaluation['score']}")


def main():
    parser = argparse.ArgumentParser(description="Combined LLM Evaluation Script")
    parser.add_argument("--model", type=str, choices=["meld", "llama3", "autoj", "prometheus", "pandalm", "all"], 
                        required=True, help="Which model to run for evaluation")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--criteria_dir", type=str, default=None, help="Directory containing criteria files (for MELD)")
    # Changed from API port to model paths for MELD and Llama3
    parser.add_argument("--meld_model_path", type=str, default="/data/liyijie/models/base-model/meld-model/", 
                       help="MELD model path for vLLM")
    parser.add_argument("--llama3_model_path", type=str, default="/data/liyijie/models/base-model/llama-3-model/", 
                       help="Llama3 model path for vLLM")
    parser.add_argument("--autoj_model_path", type=str, default="/data/liyijie/models/base-model/autoj-13b/", 
                        help="AutoJ model path")
    parser.add_argument("--prometheus_model_path", type=str, default="/data/liyijie/models/base-model/prometheus-7b-v2.0", 
                        help="Prometheus model path")
    parser.add_argument("--pandalm_model_path", type=str, default="/data/liyijie/models/base-model/PandaLM-7B-v1/", 
                        help="PandaLM model path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input data
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
            logging.info(f"Loaded {len(input_data)} evaluation items from {args.input_file}")
    except Exception as e:
        logging.error(f"Error loading input file: {e}")
        return
    
    # Determine which models to run
    models_to_run = []
    if args.model == "all":
        models_to_run = ["meld", "llama3", "autoj", "prometheus", "pandalm"]
    else:
        models_to_run = [args.model]
    
    # Run selected models
    for model in models_to_run:
        output_file = os.path.join(args.output_dir, f"{model}_result.jsonl")  # 文件扩展名改为jsonl
        logging.info(f"Running {model} evaluation, output to {output_file}")
        
        if model == "meld":
            evaluator = MELDEvaluator(args.meld_model_path, args.criteria_dir)
            evaluator.evaluate(input_data, output_file)
            
        elif model == "llama3":
            evaluator = Llama3Evaluator(args.llama3_model_path)
            evaluator.evaluate(input_data, output_file)
            
        elif model == "autoj":
            evaluator = AutoJEvaluator(args.autoj_model_path)
            evaluator.evaluate(input_data, output_file)
            
        elif model == "prometheus":
            evaluator = PrometheusEvaluator(args.prometheus_model_path)
            evaluator.evaluate(input_data, output_file)
            
        elif model == "pandalm":
            evaluator = PandaLMEvaluator(args.pandalm_model_path)
            evaluator.evaluate(input_data, output_file, args.seed)
    
    logging.info("All evaluations completed.")

if __name__ == "__main__":
    main()