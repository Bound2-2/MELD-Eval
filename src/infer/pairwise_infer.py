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

# AutoJ constants and functions
PROMPT_INPUT_SYSTEM = '[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{input} [/INST]'
PROMPT_INPUT_WO_SYSTEM = "[INST] {input} [/INST]"

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

def llama2_wrapper(usr_msg, sys_msg=None):
    if sys_msg is None:
        return PROMPT_INPUT_WO_SYSTEM.format(input=usr_msg)
    else:
        return PROMPT_INPUT_SYSTEM.format(input=usr_msg, system_message=sys_msg)

def build_autoj_input(prompt, resp1, resp2):
    user_msg = AUTOJ_PROMPT_PAIRWISE_TIE.format(prompt=prompt, response=resp1, response_another=resp2)
    return llama2_wrapper(user_msg)

def seed_everything(seed):
    """Set random seed to ensure reproducible results"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def extract_pairwise_result(raw_output):
    """
    Extract pairwise comparison result (A, B, or Tie)
    
    :param raw_output: Raw model output text
    :return: "A", "B", "Tie", or -1 if extraction fails
    """
    # Method 1: AutoJ format
    raw_output = raw_output.strip()
    pos = raw_output.rfind('final decision is ')
    if pos != -1:
        pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        if pred_rest.startswith('response 1'):
            return "A"
        elif pred_rest.startswith('response 2'):
            return "B"
        elif pred_rest.startswith('tie'):
            return "Tie"
    
    # Method 2: Llama3 format [[A]], [[B]], [[C]]
    match = re.search(r'\[\[([ABC])\]\]', raw_output)
    if match:
        result = match.group(1)
        if result == "C":  # In Llama3, C represents a tie
            return "Tie"
        return result
    
    # Method 3: Try to find "Final Result: A" format
    match = re.search(r'Final Result:\s*([AB]|Tie)', raw_output)
    if match:
        return match.group(1)
    
    return -1  # Unable to extract result

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
    """Standardize output format as JSONL, ensuring correct fields are included"""
    # Create new result object, preserving all fields from the original item
    result_item = item.copy()
    
    # Ensure unified field names are used
    if 'label' in result_item:
        # Keep the original label field
        pass
    elif 'score' in result_item:
        result_item['label'] = result_item['score']
    
    # Convert result field value to pred field
    if 'result' in result_item:
        result_item['pred'] = result_item['result']
        # Option to keep or remove the original result field
        # del result_item['result']
    
    # Add model name identifier
    result_item['model'] = model_name
    
    # Write in JSONL format
    json.dump(result_item, file, ensure_ascii=False)
    file.write('\n')

# Add helper function to get responses in specified order
def get_responses_in_order(data_item, order):
    """
    Return responses in the specified order
    
    :param data_item: Data item
    :param order: Order (original, swapped)
    :return: First response, second response
    """
    if order == "original":
        return data_item.get("answer1_body", ""), data_item.get("answer2_body", "")
    elif order == "swapped":
        return data_item.get("answer2_body", ""), data_item.get("answer1_body", "")
    else:
        logging.warning(f"Unknown order: {order}, using original order")
        return data_item.get("answer1_body", ""), data_item.get("answer2_body", "")

# Add function to apply correct mapping during result conversion
def map_result_based_on_order(result, order):
    """
    Map results based on evaluation order
    
    :param result: Model output result (A, B, Tie)
    :param order: Order (original, swapped)
    :return: Mapped result
    """
    if result == "Tie":
        return "Tie"
    
    if order == "original":
        return result
    elif order == "swapped":
        # Swap A and B
        if result == "A":
            return "B"
        elif result == "B":
            return "A"
    
    return result  # If order or result can't be determined, return original result

class MELDEvaluator:
    """MELD model evaluator using vLLM"""
    
    def __init__(self, model_path, criteria_dir=None):
        self.model_path = model_path
        self.criteria_dir = criteria_dir
        self.llm = None
        
    def initialize_model(self):
        """Initialize vLLM model"""
        try:
            from vllm import LLM, SamplingParams
            
            # Initialize model only once
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
        """Call vLLM model with given prompt"""
        if not self.initialize_model():
            return "Error initializing model"
        
        try:
            # For chat models, wrap prompt in chat format
            messages = [{"role": "user", "content": prompt}]
            prompt_formatted = json.dumps(messages)
            
            outputs = self.llm.generate(prompt_formatted, self.sampling_params)
            if outputs and len(outputs) > 0 and len(outputs[0].outputs) > 0:
                # First try to parse as JSON (for chat models)
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
    
    def evaluate(self, input_data, output_path, start_index=0, response_order="original"):
        """Evaluate paired responses using MELD criteria, supporting specified response order"""
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for index, data in enumerate(input_data[start_index:], start=start_index):
                instruction = data.get('question_body')
                
                # Get responses based on specified order
                response1, response2 = get_responses_in_order(data, response_order)
                category = data.get("category", "general")
                
                # Get evaluation criteria (if available)
                criteria = ""
                if self.criteria_dir:
                    criteria = read_criteria(category, self.criteria_dir)
                
                prompt = f'''
                    You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data: 
                    [BEGIN DATA]
                    *** 
                    [Query]: {instruction}
                    *** 
                    [Response 1]: {response1}
                    ***
                    [Response 2]: {response2}
                    *** 
                    [END DATA]
                    You are given the criteria to craft good responses for this type of query from users: 
                    {category} 
                    The criteria are as follows: 
                    [Criteria start] 
                    {criteria}
                    [Criteria end]
                    Please follow the evaluation process outlined below:
                    1. First, using the given scoring criteria, evaluate responses A and B from various dimensions, scoring each dimension from 1 to 10. In the answer section, return all your scoring results in the following dictionary format (including brackets), and ensure your scores are integers: {{'Dimension One': Score, 'Dimension Two': Score, ..., 'Overall Score': Score}}, e.g., {{'Factual Accuracy': 9, 'User Need Fulfillment': 6, ..., 'Overall Score': 7}}.
                    2. Calculate the final score for responses A and B separately. The final score is the average of the scores for each dimension. Round the result to the nearest integer.
                    3. Compare the final scores of response A and response B, and conclude which is better, or if they are tied.
                    4. Write detailed feedback explaining why A or B is better, focusing on aspects emphasized in the evaluation criteria. Additionally, brainstorm and provide a more detailed comparative feedback result. When writing feedback, compare responses A and B directly, mentioning their similarities and differences. Try to articulate a reasoning process that explores the commonalities and differences between the two responses, mentioning these reasons at the end.
                    5. Do not generate any additional introductions, conclusions, or explanations.
                    The output format should be as follows: "@@@{{response A: Scores per dimension: ['Dimension One': Score, 'Dimension Two': Score, ..., 'Overall Score': Score]}}@@@{{response B: Scores per dimension: ['Dimension One': Score, 'Dimension Two': Score, ..., 'Overall Score': Score]}}###Final Result: {{A or B or Tie}}&&&Detailed Evaluation Feedback: {{Evaluation Content}}***"
                '''
                
                model_answer = self.call_model(prompt)
                data['meld_response'] = model_answer
                
                # Extract result (A, B or Tie)
                raw_result = extract_pairwise_result(model_answer)
                
                # Map result based on evaluation order
                mapped_result = map_result_based_on_order(raw_result, response_order)
                
                data['raw_result'] = raw_result
                data['result'] = mapped_result
                data['response_order'] = response_order
                
                # Use standardized write function
                write_result_to_jsonl(data, output_file, "meld")
                
                logging.info(f"{index}: Raw={raw_result}, Mapped={mapped_result}, Order={response_order}")


class Llama3Evaluator:
    """Llama3 model evaluator using vLLM"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.llm = None
        
    def initialize_model(self):
        """Initialize vLLM model"""
        try:
            from vllm import LLM, SamplingParams
            
            # Initialize model only once
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
        """Call vLLM model with given prompt"""
        if not self.initialize_model():
            return "Error initializing model"
        
        try:
            # For chat models, wrap prompt in chat format
            messages = [{"role": "user", "content": prompt}]
            prompt_formatted = json.dumps(messages)
            
            outputs = self.llm.generate(prompt_formatted, self.sampling_params)
            if outputs and len(outputs) > 0 and len(outputs[0].outputs) > 0:
                # First try to parse as JSON (for chat models)
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
    
    def evaluate(self, input_data, output_path, start_index=0, response_order="original"):
        """Evaluate paired responses using Llama3, supporting specified response order"""
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for index, data in enumerate(input_data[start_index:], start=start_index):
                instruction = data.get('question_body')
                
                # Get responses based on specified order
                response1, response2 = get_responses_in_order(data, response_order)

                prompt = f'''
                Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. 
                Here are the instructions to assess and compare the two responses:
                Conclude your comparison by providing a final decision on which response is better, or they are tied. Output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie. Only output the final verdict, do not include any other content.
                Here is the data: 
                [BEGIN DATA] 
                *** 
                [User question]: {instruction}
                *** 
                [assistant A]: {response1}
                *** 
                [assistant B]: {response2}
                ***
                [END DATA] 

                [your verdict here]:(The verdict should be [[A]], [[B]], or [[C]], nothing else. Don't output any prefix here.)
                '''
                
                model_answer = self.call_model(prompt)
                data['llama3_response'] = model_answer
                
                # Extract pairwise result
                raw_result = extract_pairwise_result(model_answer)
                
                # Map result based on evaluation order
                mapped_result = map_result_based_on_order(raw_result, response_order)
                
                data['raw_result'] = raw_result
                data['result'] = mapped_result
                data['response_order'] = response_order
                
                # Use standardized write function
                write_result_to_jsonl(data, output_file, "llama3")
                
                logging.info(f"{index}: Raw={raw_result}, Mapped={mapped_result}, Order={response_order}")


class AutoJEvaluator:
    """AutoJ model evaluator using vLLM"""
    
    def __init__(self, model_path):
        self.model_path = model_path
    
    def extract_pairwise_rating(self, score_output):
        """Extract pairwise comparison result from AutoJ output"""
        return extract_pairwise_result(score_output)
    
    def evaluate(self, input_data, output_path, response_order="original"):
        """Evaluate paired responses using AutoJ method, supporting specified response order"""
        try:
            from vllm import LLM, SamplingParams
            
            # Initialize model
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
                    
                    # Get responses based on specified order
                    resp1, resp2 = get_responses_in_order(item, response_order)
                    
                    # Use integrated build_autoj_input function to construct input
                    input_text = build_autoj_input(prompt=prompt, resp1=resp1, resp2=resp2)
                    
                    # Generate model output
                    outputs = llm.generate(input_text, sampling_params)
                    judgment = outputs[0].outputs[0].text
                    
                    # Extract comparison result
                    raw_result = self.extract_pairwise_rating(judgment)
                    
                    # Map result based on evaluation order
                    mapped_result = map_result_based_on_order(raw_result, response_order)
                    
                    # Print result
                    logging.info(f"{idx}: Raw={raw_result}, Mapped={mapped_result}, Order={response_order}")
                    
                    # Store results
                    item["autoj_response"] = judgment
                    item["raw_result"] = raw_result
                    item["result"] = mapped_result
                    item["response_order"] = response_order
                    
                    # Use standardized write function
                    write_result_to_jsonl(item, f, "autoj")
                
        except ImportError:
            logging.error("vLLM module not found. Please install with: pip install vllm")
        except Exception as e:
            logging.error(f"Error in AutoJ evaluation: {e}")


class PrometheusEvaluator:
    """Prometheus model evaluator"""
    
    def __init__(self, model_path):
        self.model_path = model_path
    
    def evaluate(self, input_data, output_path, response_order="original"):
        """Evaluate paired responses using Prometheus method, supporting specified response order"""
        try:
            from prometheus_eval.vllm import VLLM
            from prometheus_eval import PrometheusEval
            from prometheus_eval.prompts import RELATIVE_PROMPT_WO_REF, HELPFULNESS_RUBRIC
            
            # Initialize model
            model = VLLM(model=self.model_path, dtype="float16")
            judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT_WO_REF)
            
            # Open output file
            with open(output_path, 'w', encoding='utf-8') as f:
                # Process each input item
                for idx, item in enumerate(input_data):
                    instruction = item.get("question_body", "")
                    
                    # Get responses based on specified order
                    response_A, response_B = get_responses_in_order(item, response_order)
                    
                    # Use model for evaluation
                    feedback, score = judge.single_relative_grade(
                        instruction=instruction,
                        response_A=response_A,
                        response_B=response_B,
                        rubric=HELPFULNESS_RUBRIC
                    )
                    
                    # Map Prometheus result to A/B/Tie format
                    if score > 0:
                        raw_result = "A"
                    elif score < 0:
                        raw_result = "B"
                    else:
                        raw_result = "Tie"
                    
                    # Map result based on evaluation order
                    mapped_result = map_result_based_on_order(raw_result, response_order)
                    
                    # Store results
                    item["prometheus_response"] = feedback
                    item["raw_result"] = raw_result
                    item["result"] = mapped_result
                    item["response_order"] = response_order
                    
                    # Print result
                    logging.info(f"{idx}: Raw={raw_result}, Mapped={mapped_result}, Order={response_order}")
                    
                    # Use standardized write function
                    write_result_to_jsonl(item, f, "prometheus")
                
        except ImportError:
            logging.error("Prometheus modules not found. Please install prometheus_eval and its dependencies.")
        except Exception as e:
            logging.error(f"Error in Prometheus evaluation: {e}")


class PandaLMEvaluator:
    """PandaLM model evaluator - directly integrated into code"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pattern = None
        self.prepared = []
        
    def initialize_model(self):
        """Initialize PandaLM model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import transformers
            
            if self.model is None:
                logging.info(f"Loading PandaLM model: {self.model_path}")
                
                # Load tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                except:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                
                # Add special tokens
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
                
                # Set token IDs
                self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
                self.model.config.bos_token_id = 1
                self.model.config.eos_token_id = 2
                self.model.eval()
                
                # Compile model (if possible)
                if torch.__version__ >= "2" and sys.platform != "win32":
                    self.model = torch.compile(self.model)
                
                # Set special character regex
                self.pattern = re.compile(
                    r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
                )
            
            return True
        except ImportError:
            logging.error("transformers module not found. Please install: pip install transformers")
            return False
        except Exception as e:
            logging.error(f"Error initializing PandaLM model: {e}")
            return False
    
    def _smart_tokenizer_and_embedding_resize(self, special_tokens_dict, tokenizer, model):
        """Adjust tokenizer and embedding size"""
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
    
    def build_pandalm_prompt(self, instruction, resp1, resp2):
        """Build PandaLM prompt"""
        if isinstance(resp1, bool):
            resp1 = str(resp1)
        if isinstance(resp2, bool):
            resp2 = str(resp2)
        
        resp1 = self.pattern.sub("", resp1.strip()).strip()
        resp2 = self.pattern.sub("", resp2.strip()).strip()
        
        rsp = f"### Response 1:\n{resp1}\n\n### Response 2:\n{resp2}"
        input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n{rsp}\n\n### Evaluation:\n"
        
        return input_sequence
    
    def parse_pandalm_response(self, text):
        """Parse PandaLM response"""
        sp = text.strip().split("\n")
        if sp[0] in ["1", "2"]:
            return int(sp[0])
        elif sp[0].lower() == "tie":
            return 0
        else:
            return 0
    
    def filter_special_token(self, text):
        """Filter special tokens"""
        return self.pattern.sub("", text.strip()).strip()
    
    def postprocess_output(self, text):
        """Post-process output"""
        try:
            text = text.strip().split("### Evaluation:")[1].strip()
            return self.filter_special_token(text)
        except:
            return text
    
    def evaluate(self, input_data, output_path, seed=42, response_order="original"):
        """Evaluate paired responses, supporting specified response order"""
        if not self.initialize_model():
            logging.error("Unable to initialize PandaLM model")
            return
        
        from transformers import GenerationConfig
        
        # Set random seed
        seed_everything(seed)
        
        # Preprocess input
        self.prepared = []
        for item in tqdm(input_data, desc="Preparing input"):
            # Get responses based on specified order
            resp1, resp2 = get_responses_in_order(item, response_order)
            
            prompt = self.build_pandalm_prompt(
                instruction=item["question_body"],
                resp1=resp1,
                resp2=resp2
            )
            self.prepared.append(self.tokenizer(prompt, return_tensors="pt", padding=True))
        
        # Generate output
        generated = []
        for idx in tqdm(range(len(self.prepared)), desc="Generating evaluations"):
            inputs = self.prepared[idx]
            input_ids = inputs["input_ids"].to(self.model.device)
            
            generation_config = GenerationConfig(
                temperature=0,
                top_p=1,
                top_k=1,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.2,
                max_new_tokens=512
            )
            
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=512
                )
            
            for j in range(len(generation_output.sequences)):
                s = generation_output.sequences[j]
                output = self.tokenizer.decode(s)
                resp = self.postprocess_output(output)
                resp = self.filter_special_token(resp)
                generated.append(resp)
        
        # Process results and write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, (item, output) in enumerate(zip(input_data, generated)):
                result = self.parse_pandalm_response(output)
                
                # Convert numerical results to A/B/Tie format
                if result == 1:
                    raw_result = "A"
                elif result == 2:
                    raw_result = "B"
                else:  # result == 0
                    raw_result = "Tie"
                
                # Map result based on evaluation order
                mapped_result = map_result_based_on_order(raw_result, response_order)
                
                # Store results
                item_copy = item.copy()
                item_copy["pandalm_response"] = output
                item_copy["raw_result"] = raw_result
                item_copy["result"] = mapped_result
                item_copy["response_order"] = response_order
                
                # Use standardized write function
                write_result_to_jsonl(item_copy, f, "pandalm")
                
                logging.info(f"{idx}: Raw={raw_result}, Mapped={mapped_result}, Order={response_order}")


def main():
   parser = argparse.ArgumentParser(description="Pairwise LLM response evaluation script")
   parser.add_argument("--model", type=str, choices=["meld", "llama3", "autoj", "prometheus", "pandalm", "all"], 
                       required=True, help="Evaluation model to run")
   parser.add_argument("--input_file", type=str, required=True, help="Input JSON file path")
   parser.add_argument("--output_dir", type=str, required=True, help="Output directory path")
   parser.add_argument("--criteria_dir", type=str, default="/Users/liyijie/Desktop/MELD/github仓库构建/src/criteria", help="Directory containing criteria files (required for MELD)")
   parser.add_argument("--meld_model_path", type=str, default="/data/liyijie/models/base-model/meld-model/", 
                      help="MELD model path")
   parser.add_argument("--llama3_model_path", type=str, default="/data/liyijie/models/base-model/llama-3-model/", 
                      help="Llama3 model path")
   parser.add_argument("--autoj_model_path", type=str, default="/data/liyijie/models/base-model/autoj-13b/", 
                       help="AutoJ model path")
   parser.add_argument("--prometheus_model_path", type=str, default="/data/liyijie/models/base-model/prometheus-7b-v2.0", 
                       help="Prometheus model path")
   parser.add_argument("--pandalm_model_path", type=str, default="/data/liyijie/models/base-model/PandaLM-7B-v1/", 
                       help="PandaLM model path")
   parser.add_argument("--seed", type=int, default=42, help="Random seed")
   # Add new parameter to control response order
   parser.add_argument("--response_order", type=str, choices=["original", "swapped"], default="original",
                      help="Response evaluation order: original (answer1->response1, answer2->response2) or swapped (answer2->response1, answer1->response2)")
   
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
       # Add order identifier to output filename
       order_suffix = "_swapped" if args.response_order == "swapped" else ""
       output_file = os.path.join(args.output_dir, f"{model}{order_suffix}_result.jsonl")  # Use jsonl extension
       logging.info(f"Running {model} evaluation using {args.response_order} order, output to {output_file}")
       
       if model == "meld":
           evaluator = MELDEvaluator(args.meld_model_path, args.criteria_dir)
           evaluator.evaluate(input_data, output_file, response_order=args.response_order)
           
       elif model == "llama3":
           evaluator = Llama3Evaluator(args.llama3_model_path)
           evaluator.evaluate(input_data, output_file, response_order=args.response_order)
           
       elif model == "autoj":
           evaluator = AutoJEvaluator(args.autoj_model_path)
           evaluator.evaluate(input_data, output_file, response_order=args.response_order)
           
       elif model == "prometheus":
           evaluator = PrometheusEvaluator(args.prometheus_model_path)
           evaluator.evaluate(input_data, output_file, response_order=args.response_order)
           
       elif model == "pandalm":
           evaluator = PandaLMEvaluator(args.pandalm_model_path)
           evaluator.evaluate(input_data, output_file, args.seed, response_order=args.response_order)
   
   logging.info("All evaluations completed")


if __name__ == "__main__":
   main()