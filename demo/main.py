import streamlit as st
import requests
import json
import re
import plotly.express as px
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bert_score
import pandas as pd
import plotly.graph_objects as go

###绘图###
def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {key: score.fmeasure for key, score in scores.items()}

def compute_bertscore(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], lang='en', model_type='bert-base-uncased', verbose=True)
    return F1.mean().item()

def generate_metrics_plot(rouge_scores, bleu_score, bert_score):
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'BERTScore']
    scores = [
        rouge_scores['rouge1'],
        rouge_scores['rouge2'],
        rouge_scores['rougeL'],
        bleu_score,
        bert_score,
    ]
    
    df = pd.DataFrame({
        'Metrics': metrics,
        'Scores': scores
    })
    
    fig = px.bar(df, x='Metrics', y='Scores', text='Scores',
                 color='Scores', color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        uniformtext_minsize=8, 
        uniformtext_mode='hide',
        xaxis_title='',  # 去掉x轴标题
        margin=dict(l=0, r=0, t=0, b=0),  # Reduce margins to use more space for the chart
        height=350,
    )
    return fig


def plot_scores_PAIRWISE(processed_response, for_chart):
    multi_dimension_score = {
                        'score_A': processed_response['score_A'],
                        'score_B': processed_response['score_B']
                    }
    metrics = list(multi_dimension_score['score_A'].keys())
    scores_A = [multi_dimension_score['score_A'][metric] for metric in metrics]
    scores_B = [multi_dimension_score['score_B'][metric] for metric in metrics]
    
    # 创建 Plotly 图表
    fig1 = go.Figure(data=[
        go.Bar(name='Score A', x=metrics, y=scores_A),
        go.Bar(name='Score B', x=metrics, y=scores_B)
    ])
    
    # 更新图表布局
    fig1.update_layout(
        barmode='group',
        yaxis_title='Scores',
        legend_title='Score Groups',
        margin=dict(l=10, r=10, t=30, b=10),
        height=400
    )

    # 只有在reference不为空且不是"N/A"时才生成参考对比图
    if for_chart["reference"] and for_chart["reference"] != "N/A":
        #第一张和reference比较图
        reference = for_chart["reference"]
        candidate1 = for_chart["answer1"]
        rouge_scores = compute_rouge(reference, candidate1)
        bleu_score = sacrebleu.corpus_bleu([candidate1], [[reference]]).score
        bert_score_val = compute_bertscore(reference, candidate1)
        fig2 = generate_metrics_plot(rouge_scores, bleu_score/10, bert_score_val)

        #第二张和reference比较图
        reference = for_chart["reference"]
        candidate1 = for_chart["answer2"]
        rouge_scores = compute_rouge(reference, candidate1)
        bleu_score = sacrebleu.corpus_bleu([candidate1], [[reference]]).score
        bert_score_val = compute_bertscore(reference, candidate1)
        fig3 = generate_metrics_plot(rouge_scores, bleu_score/10, bert_score_val)
        
        return fig1, fig2, fig3
    else:
        # 如果没有参考答案，只返回第一个图表
        return fig1, None, None
    

def plot_scores_POINTWISE(processed_response, for_chart):
    # 假设 processed_response 包含 'score_A' 或类似字段
    dimension_scores = processed_response["Dimension_Scores"]  # 使用 processed_response 作为输入数据
    
    metrics = list(dimension_scores.keys())
    scores = list(dimension_scores.values())

    # 创建 Plotly 图表
    fig1 = go.Figure(data=[
        go.Bar(x=metrics, y=scores)
    ])

    # 更新图表布局
    fig1.update_layout(
        yaxis_title='Score',
        xaxis=dict(type='category'),  # 确保x轴为类别类型
        yaxis=dict(range=[0, max(scores) + 1]),  # 可以根据需要调整y轴范围
        margin=dict(l=10, r=10, t=30, b=10),
        height=400
    )
    
    # 只有在reference不为空且不是"N/A"时才生成参考对比图
    if for_chart["reference"] and for_chart["reference"] != "N/A":
        #第一张和reference比较图
        reference = for_chart["reference"]
        candidate1 = for_chart["answer"]
        rouge_scores = compute_rouge(reference, candidate1)
        bleu_score = (sacrebleu.corpus_bleu([candidate1], [[reference]]).score)/10
        bert_score_val = compute_bertscore(reference, candidate1)
        fig2 = generate_metrics_plot(rouge_scores, bleu_score, bert_score_val)
        return fig1, fig2
    else:
        # 如果没有参考答案，只返回第一个图表
        return fig1, None


# Function to extract required parts from gpt_response
def extract_gpt_response_info_pairwise(gpt_response):
    # 添加错误处理，确保正则表达式匹配结果有效
    try:
        # 正则表达式模式用于提取各部分
        pattern_a = r"@@@(.*?)@@@"
        pattern_b = r"@@@(.*?)###"
        pattern_final_result = r"###(.*?)&&&"
        pattern_detailed_feedback = r"&&&Detailed Evaluation Feedback:(.*?)\*\*\*"
        
        # 使用正则表达式提取各部分内容
        match_a = re.search(pattern_a, gpt_response, re.DOTALL)
        match_b = re.search(pattern_b, gpt_response, re.DOTALL)
        match_final_result = re.search(pattern_final_result, gpt_response, re.DOTALL)
        match_detailed_feedback = re.search(pattern_detailed_feedback, gpt_response, re.DOTALL)
        
        # 初始化结果字典
        result = {}

        # 对 dict_A 和 dict_B 使用字符串解析（非标准JSON格式无法直接解析）
        dict_a_raw = match_a.group(1).strip() if match_a else ""
        dict_b_raw = match_b.group(1).strip() if match_b else ""
        
        # 将自定义的格式转换为键值对字典
        def parse_custom_format(raw_text):
            scores = {}
            # 匹配类似 'Key': value 的格式
            matches = re.findall(r"'(.*?)':\s*(\d+)", raw_text)
            for key, value in matches:
                scores[key] = int(value)
            return scores
        
        # 解析 dict_A 和 dict_B
        result['score_A'] = parse_custom_format(dict_a_raw)
        result['score_B'] = parse_custom_format(dict_b_raw)
        result['final_results'] = match_final_result.group(1).strip() if match_final_result else "No clear result"
        result['Detailed_Evaluation_Feedback'] = match_detailed_feedback.group(1).strip() if match_detailed_feedback else "No detailed feedback available"
        
        # 确保score_A和score_B至少有一个键值对
        if not result['score_A']:
            result['score_A'] = {'Overall': 5}
        if not result['score_B']:
            result['score_B'] = {'Overall': 5}
            
        return result
    except Exception as e:
        # 如果提取过程失败，返回默认结果
        st.error(f"Error parsing model response: {str(e)}")
        st.code(gpt_response, language="text")
        return {
            'score_A': {'Overall': 5},
            'score_B': {'Overall': 5},
            'final_results': "Error parsing results",
            'Detailed_Evaluation_Feedback': "Could not extract detailed feedback from model response."
        }

def extract_gpt_response_info_pointwise(gpt_response):
    try:
        # 正则表达式模式用于提取各部分
        pattern_dict_a = r"@@@Dimension Scores:\s*(\{.*?\})###"
        pattern_dict_b = r"###Overall Score:\s*(\d+)&&&"
        pattern_detailed_feedback = r"&&&Detailed Evaluation Feedback:(.*?)\*\*\*"

        # 使用正则表达式提取各部分内容
        match_dict_a = re.search(pattern_dict_a, gpt_response, re.DOTALL)
        match_dict_b = re.search(pattern_dict_b, gpt_response, re.DOTALL)
        match_detailed_feedback = re.search(pattern_detailed_feedback, gpt_response, re.DOTALL)

        # 初始化结果字典
        result = {}

        # 手动解析自定义格式的字典
        def parse_custom_format(raw_text):
            scores = {}
            # 匹配类似 'Key': value 的格式 (其中value是整数)
            matches = re.findall(r"'(.*?)':\s*(\d+)", raw_text)
            for key, value in matches:
                scores[key] = int(value)
            return scores

        # 解析字典A (Dimension Scores)
        dict_a_raw = match_dict_a.group(1).strip() if match_dict_a else ""
        dict_a = parse_custom_format(dict_a_raw)

        # 解析字典B (Overall Score)
        dict_b = {"Overall Score": int(match_dict_b.group(1).strip())} if match_dict_b else {"Overall Score": 5}

        # 解析详细反馈 (Detailed Evaluation Feedback)
        detailed_feedback = match_detailed_feedback.group(1).strip() if match_detailed_feedback else "No detailed feedback available"

        # 确保Dimension_Scores至少有一个键值对
        if not dict_a:
            dict_a = {'Overall': 5}
            
        # 将解析的内容存入结果字典
        result['Dimension_Scores'] = dict_a
        result['Overall_Score'] = dict_b
        result['Detailed_Evaluation_Feedback'] = detailed_feedback

        return result
    except Exception as e:
        # 如果提取过程失败，返回默认结果
        st.error(f"Error parsing model response: {str(e)}")
        st.code(gpt_response, language="text")
        return {
            'Dimension_Scores': {'Overall': 5},
            'Overall_Score': {"Overall Score": 5},
            'Detailed_Evaluation_Feedback': "Could not extract detailed feedback from model response."
        }


def read_criteria(scenario):
    """根据场景读取相应的评价标准文本文件"""
    try:
        with open(f'/root/autodl-tmp/demo/txt_criteria/{scenario}.txt', 'r', encoding='utf-8') as file:
            criteria = file.read()
        return criteria
    except FileNotFoundError:
        print(f"No criteria found for {scenario}")
        return "No specific criteria available for this scenario."

def user_selected_criteria(criteria_list):
    # 遍历列表，将每个元素转换为带有序号的格式
    formatted_criteria = [f"{i+1}. {criteria}" for i, criteria in enumerate(criteria_list)]
    # 将所有元素合并成一个字符串，每个元素占一行
    return "\n".join(formatted_criteria)

# 检查Ollama服务
def check_ollama_service():
    try:
        # 尝试一个简单的API调用
        response = requests.get("http://localhost:6006/api/version", timeout=2)
        return response.status_code == 200
    except:
        return False

# 直接调用Ollama API
def call_ollama_api(model_name, prompt, message_placeholder=None):
    """直接调用Ollama API并处理流式响应"""
    try:
        # 准备API请求
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True  # 使用流式响应
        }
        
        # 发送请求
        response = requests.post(
            "http://localhost:11434/api/chat",
            headers=headers,
            json=data,
            stream=True,
            timeout=300
        )
        
        # 检查响应状态
        if response.status_code != 200:
            return f"Error: API returned status code {response.status_code} - {response.text}"
        
        # 处理流式响应
        full_response = ""
        
        for line in response.iter_lines():
            if not line:
                continue
                
            try:
                # 解析每行JSON
                chunk = json.loads(line)
                
                # 提取内容
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    if content:
                        full_response += content
                        
                        # 如果提供了占位符，更新UI
                        if message_placeholder:
                            message_placeholder.markdown(full_response, unsafe_allow_html=True)
                
                # 检查是否完成
                if chunk.get('done', False):
                    break
                    
            except json.JSONDecodeError:
                continue
            
        return full_response
        
    except Exception as e:
        error_msg = f"API call error: {str(e)}"
        if message_placeholder:
            message_placeholder.error(error_msg)
        return error_msg

# App title
st.set_page_config(page_title="👨‍⚖️MELD",layout="wide")

# 使用session_state存储需要持久化的变量
if "model_name" not in st.session_state:
    st.session_state["model_name"] = "q4k_meld:latest"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "evaluation_mode" not in st.session_state:
    st.session_state.evaluation_mode = "PAIRWISE"

with st.sidebar:
    st.title('👨‍⚖️MELD')
    st.write('A Fine-Grained Evaluation Framework for Language Models: Combining Pointwise Grading and Pairwise Comparison.')
    
    # 添加评估模式选择
    st.subheader('Evaluation Mode')
    evaluation_mode = st.radio(
        "Select Evaluation Mode",
        ["PAIRWISE", "POINTWISE"],
        index=0 if st.session_state.evaluation_mode == "PAIRWISE" else 1,
        help="PAIRWISE compares two answers. POINTWISE evaluates a single answer."
    )
    st.session_state.evaluation_mode = evaluation_mode
    
    st.subheader('Model and parameters')
    # 使用固定模型
    st.write('Using model: q4k_meld:latest')
    st.session_state["model_name"] = "q4k_meld:latest"
    st.divider()

    temperature = st.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('max_length', min_value=1024, max_value=8192, value=1024, step=8)
    
    # 在侧边栏中添加模型验证按钮
    if st.button("Test Model Connection"):
        try:
            st.info("Testing connection...")
            test_prompt = "Hello, how are you?"
            test_response = call_ollama_api(st.session_state["model_name"], test_prompt)
            
            if isinstance(test_response, str) and test_response.startswith("Error"):
                st.error(test_response)
            else:
                st.success(f"Connection successful! Response: {test_response[:50]}...")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
    
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.clear()  # 清除 st.session_state 中的所有内容
    st.session_state['question_body'] = ""
    st.session_state['answer1_body'] = ""
    st.session_state['answer2_body'] = ""
    st.session_state['answer_body'] = ""
    st.session_state['reference'] = ""
    st.session_state["model_name"] = "q4k_meld:latest"
    st.session_state.evaluation_mode = "PAIRWISE"
    
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# 检查Ollama服务是否可用
if not check_ollama_service():
    st.error("⚠️ Ollama服务不可用。请在终端运行 'ollama serve' 启动服务后刷新此页面。")
    st.code("ollama serve", language="bash")
    st.stop()

st.title(f"MELD - {st.session_state.evaluation_mode} Evaluation")

if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False

uploaded_file = st.file_uploader(
        "",
        type=["json", "jsonl"],
        help="Scanned documents are not supported yet!",
    )

# 在这行代码后面立即添加新文件检测代码
if uploaded_file:
    file_name = uploaded_file.name
    
    # 在会话状态中跟踪上传的文件名
    if 'previous_file_name' not in st.session_state:
        st.session_state.previous_file_name = ""
    
    # 当上传新文件时重置处理标志
    if st.session_state.previous_file_name != file_name:
        st.session_state.file_processed = False
        st.session_state.previous_file_name = file_name
        st.info(f"New file detected：{file_name}")

if st.session_state.file_processed:
    if st.button("Process another file"):
        st.session_state.file_processed = False
        st.experimental_rerun()


# options = ['default', 'analyzing_general', 'asking_how_to_question', 'brainstorming', 'chitchat', 'classification_identification', 'code_correction_rewriting', 'code_generation', 'code_to_code_translation', 'counterfactual', 'creative_writing', 'data_analysis', 'explaining_code', 'explaining_general', 'functional_writing', 'information_extraction', 'instructional_rewriting', 'keywords_extraction', 'language_polishing', 'math_reasoning', 'open_question', 'paraphrasing', 'planning', 'question_generation', 'ranking', 'reading_comprehension', 'recommendation', 'roleplay', 'seeking_advice', 'solving_exam_question_with_math', 'solving_exam_question_without_math', 'text_correction', 'text_simplification', 'text_summarization', 'text_to_text_translation', 'title_generation', 'topic_modeling', 'value_judgement', 'verifying_fact', 'writing_advertisement', 'writing_cooking_recipe', 'writing_email', 'writing_job_application', 'writing_news_article', 'writing_personal_essay', 'writing_presentation_script', 'writing_product_description', 'writing_social_media_post', 'writing_song_lyrics']
options = ['Writing', 'Math', 'Reasoning', 'NLP Task', 'Coding', 'casual conversation', 'Professional Knowledge', 'Roleplay']

# 创建一个跟踪选择的变量
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = options[0]

def update_selection(option):
    st.session_state.selected_option = option

# 使用expander来组织显示，以便在需要时折叠和展开
with st.expander("Choose a category"):
    # 计算需要的行和列数量
    num_rows = len(options) // 4 + (1 if len(options) % 4 > 0 else 0)
    rows = [st.columns(4) for _ in range(num_rows)]
    option_index = 0

    for row in rows:
        for col in row:
            with col:
                # 仅在还有选项时显示单选按钮
                if option_index < len(options):
                    # 检查这个选项是否被选中
                    is_checked = st.radio(
                        "", [options[option_index]],
                        key=f"option_{option_index}",  # 确保每个单选按钮组的key不同
                        index=0 if st.session_state.selected_option == options[option_index] else None,
                        on_change=update_selection,
                        args=(options[option_index],)
                    )
                    option_index += 1

# 显示选中的选项
st.write("You selected:", st.session_state.selected_option)

# 定义维度评估复选框的选项
options_group_1 = [
    "User Satisfaction",
    "Accuracy",
    "Information Richness",
    "Text Quality",
    "Logical Coherence ",
    "Creativity",
    "Being Friendly",
    "Vivid",
    "Engagement",
    "Completeness"
]
options_group_2 = [
    "Relevance",
    "Clarity",
    "Code Correctness",
    "Completeness of Code",
    "Code Readability",
    "Input/Output Requirements",
    "Documentation",
    "Modularity",
    "Running Efficiency",
    "Structure"
]
options_group_3 = [
    "Clarity",
    "Relevance to Topic/Text",
    "Depth",
    "Coherence",
    "Originality",
    "Instruction Following",
    "Fluency",
    "Engagement",
    "Detail",
    "Creativity"
]
options_group_4 = [
    "Structure",
    "Conciseness",
    "Correctness (Math)",
    "Step-by-Step Explanation",
    "Depth of Understanding"
]

# 用于存储用户选择的选项，使用新的变量名 criteria_selected_option
if 'criteria_selected_option' not in st.session_state:
    st.session_state.criteria_selected_option = {
        "group_1": [],
        "group_2": [],
        "group_3": [],
        "group_4": []
    }

# 计算总的选项数量
total_selected = sum(len(st.session_state.criteria_selected_option[group]) for group in st.session_state.criteria_selected_option)

# 设置一个标志，超过 10 个选项时禁用复选框
disable_checkboxes = total_selected >= 10

# 创建一个包含 4 组复选框的横向排列
with st.expander("Select evaluation criteria"):
    # 使用 st.columns 创建 4 列布局
    cols = st.columns(4)

    # 在每列的顶部添加组名称
    with cols[0]:
        st.write("basic standard")
    with cols[1]:
        st.write("style")
    with cols[2]:
        st.write("content")
    with cols[3]:
        st.write("format")

# 获取所有组中的最短长度
min_length = min(
    len(options_group_1),
    len(options_group_2),
    len(options_group_3),
    len(options_group_4)
)

# 在每一列中放置复选框
for i in range(min_length):
    # 第 1 组复选框
    with cols[0]:
        option = options_group_1[i]
        checked = option in st.session_state.criteria_selected_option["group_1"]
        if st.checkbox(option, key=f"group_1_{option}", value=checked, disabled=not checked and disable_checkboxes):
            if option not in st.session_state.criteria_selected_option["group_1"]:
                st.session_state.criteria_selected_option["group_1"].append(option)
        else:
            if option in st.session_state.criteria_selected_option["group_1"]:
                st.session_state.criteria_selected_option["group_1"].remove(option)

    # 第 2 组复选框
    with cols[1]:
        option = options_group_2[i]
        checked = option in st.session_state.criteria_selected_option["group_2"]
        if st.checkbox(option, key=f"group_2_{option}", value=checked, disabled=not checked and disable_checkboxes):
            if option not in st.session_state.criteria_selected_option["group_2"]:
                st.session_state.criteria_selected_option["group_2"].append(option)
        else:
            if option in st.session_state.criteria_selected_option["group_2"]:
                st.session_state.criteria_selected_option["group_2"].remove(option)

    # 第 3 组复选框
    with cols[2]:
        option = options_group_3[i]
        checked = option in st.session_state.criteria_selected_option["group_3"]
        if st.checkbox(option, key=f"group_3_{option}", value=checked, disabled=not checked and disable_checkboxes):
            if option not in st.session_state.criteria_selected_option["group_3"]:
                st.session_state.criteria_selected_option["group_3"].append(option)
        else:
            if option in st.session_state.criteria_selected_option["group_3"]:
                st.session_state.criteria_selected_option["group_3"].remove(option)

    # 第 4 组复选框
    with cols[3]:
        option = options_group_4[i]
        checked = option in st.session_state.criteria_selected_option["group_4"]
        if st.checkbox(option, key=f"group_4_{option}", value=checked, disabled=not checked and disable_checkboxes):
            if option not in st.session_state.criteria_selected_option["group_4"]:
                st.session_state.criteria_selected_option["group_4"].append(option)
        else:
            if option in st.session_state.criteria_selected_option["group_4"]:
                st.session_state.criteria_selected_option["group_4"].remove(option)

# 输出选中的名称
selected_criteria = []
for group in st.session_state.criteria_selected_option:
    selected_criteria.extend(st.session_state.criteria_selected_option[group])

# 检查是否有选中的标准
if 0 < len(selected_criteria) < 5:
    st.warning("You must select either 0 or at least 5 criteria.")
    disable_other_operations = True
    st.stop() 
else:
    disable_other_operations = False

# 只有当选中 0 个或 5 个及以上选项时，才允许执行其他操作
if not disable_other_operations:
    # 继续执行其他操作
    if selected_criteria:
        st.write(f"You selected: {', '.join(selected_criteria)}")  # 输出格式为 "You selected: ..."
    else:
        st.write("You selected: No criteria selected.")

# PAIRWISE模式
if st.session_state.evaluation_mode == "PAIRWISE":
    # 文件上传处理部分
    if uploaded_file and not st.session_state.file_processed:
        try:
            # 重置文件指针到开始
            uploaded_file.seek(0)
            file_content = uploaded_file.read().decode('utf-8')  # 读取并解码为UTF-8格式的字符串
            
            # 添加调试信息
            st.write("File content sample (first 200 characters):")
            st.write(file_content[:200])
            
            # 尝试解析JSON数据
            try:
                data = json.loads(file_content)  # 解析JSON数据
            except json.JSONDecodeError as json_err:
                st.error(f"JSON parsing error: {str(json_err)}")
                st.error("Please upload a valid JSON file.")
                st.stop()
            
            # 确保数据是数组格式
            if not isinstance(data, list):
                st.error("The uploaded JSON must be in array format.")
                st.write(f"Current data type: {type(data)}")
                st.stop()
            
            if len(data) == 0:
                st.error("The uploaded JSON array is empty.")
                st.stop()
            
            # 检查每个元素是否包含必要的键
            missing_keys_items = []
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    st.error(f"Item {i+1} in the array is not a dictionary.")
                    st.stop()
                
                # 严格检查必需的字段
                required_keys = {'question_body', 'answer1_body', 'answer2_body'}
                if not all(key in item for key in required_keys):
                    missing_keys = required_keys - set(item.keys())
                    missing_keys_items.append((i, missing_keys))
            
            # 如果有任何项目缺少必要的键，显示错误并停止
            if missing_keys_items:
                st.error("The following items are missing required fields:")
                for i, missing in missing_keys_items:
                    st.write(f"Item {i+1}: Missing {', '.join(missing)}")
                st.error("Please ensure all items include 'question_body', 'answer1_body', and 'answer2_body' fields.")
                st.stop()
            
            # 数据验证通过，继续处理
            st.success(f"Successfully validated {len(data)} items, all items contain the required fields.")
            
            # 输出文件路径
            modified_file_path = 'critic_by_pairwise_data.json'
            
            # 处理上传的数据
            processed_items = []
            
            # 创建一个进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 使用一个容器来显示处理结果
            results_container = st.container()
            
            for i, item in enumerate(data):
                # 更新进度条和状态文本
                progress = (i) / len(data)
                progress_bar.progress(progress)
                status_text.text(f"Processing item {i+1}/{len(data)} ({int(progress*100)}%)")
                
                try:
                    with results_container.expander(f"Item {i+1}/{len(data)}", expanded=(i==0)):
                        st.write(f"Question: {item['question_body'][:100]}...")
                        
                        # 获取必须的键
                        question_body = item['question_body']
                        answer1_body = item['answer1_body']
                        answer2_body = item['answer2_body']
                        
                        # 检查是否有参考答案
                        reference = item.get('reference', "")
                        
                        # 选择适当的模板文件
                        if reference:
                            template_file = "/root/autodl-tmp/demo/prompt_template/PAIRWISE_WR.txt"
                        else:
                            template_file = "/root/autodl-tmp/demo/prompt_template/PAIRWISE_WOR.txt"
                        
                        # 尝试读取模板文件
                        try:
                            with open(template_file, "r") as file:
                                base_prompt = file.read()
                        except FileNotFoundError:
                            st.error(f"Template file {template_file} not found.")
                            # 尝试备用路径
                            backup_paths = [
                                f"./prompt_template/{os.path.basename(template_file)}",
                                f"../prompt_template/{os.path.basename(template_file)}",
                                f"prompt_template/{os.path.basename(template_file)}"
                            ]
                            template_found = False
                            for path in backup_paths:
                                try:
                                    with open(path, "r") as file:
                                        base_prompt = file.read()
                                        template_found = True
                                        st.success(f"Using backup template: {path}")
                                        break
                                except FileNotFoundError:
                                    continue
                            
                            if not template_found:
                                st.error("Could not find the necessary template file.")
                                continue  # Skip this item and move to the next one
                        
                        # 确定评价类别
                        scenario = item.get('category', st.session_state.selected_option)
                        
                        # 生成最终的prompt
                        if not selected_criteria:
                            final_prompt = base_prompt.format(
                                scenario = scenario,
                                criteria = read_criteria(scenario),
                                question_body = question_body,
                                answer1_body = answer1_body,
                                answer2_body = answer2_body,
                                reference = reference if reference else "N/A"
                            )
                        else:
                            final_prompt = base_prompt.format(
                                scenario = scenario,
                                criteria = user_selected_criteria(selected_criteria),
                                question_body = question_body,
                                answer1_body = answer1_body,
                                answer2_body = answer2_body,
                                reference = reference if reference else "N/A"
                            )
                        
                        # 使用占位符显示加载状态和响应
                        message_placeholder = st.empty()
                        
                        try:
                            with st.spinner(f'Evaluating answers...'):
                                # 使用API调用获取响应
                                full_response = call_ollama_api(
                                    st.session_state["model_name"], 
                                    final_prompt, 
                                    message_placeholder
                                )
                        except Exception as e:
                            st.error(f"Error connecting to Ollama: {str(e)}")
                            st.error("Please ensure Ollama service is running.")
                            item['processing_error'] = f"Connection error: {str(e)}"
                            processed_items.append(item)
                            continue  # 继续处理下一个项目
                        
                        message_placeholder.empty()
                        
                        try:
                            # 处理响应
                            processed_response = extract_gpt_response_info_pairwise(full_response)
                            
                            final_result = str(processed_response["final_results"]).replace("Final Result: ", "")
                            result_text = "🤝 It's a Tie!" if final_result == "Tie" else f"🏆 {final_result} Wins!"
                            
                            # 显示结果
                            col1, col2 = st.columns([1, 3])
                            
                            # 样式设置
                            common_style = "padding: 20px; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
                            background_color1 = "background-color: #f8f9fa;"
                            background_color2 = "background-color: #e9ecef;"
                            
                            with col1:
                                st.markdown(f"""
                                    <div style="{background_color1} {common_style}">
                                        <h2 style="color: #007BFF;">{result_text}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                    <div style="{background_color2} {common_style}">
                                        <h3 style="color: #6c757d;">Detailed Evaluation Feedback</h3>
                                        <p style="font-size: 16px; line-height: 1.6;">
                                            {processed_response["Detailed_Evaluation_Feedback"]}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                            
                            # 只展示简单的分数对比柱状图
                            try:
                                multi_dimension_score = {
                                    'score_A': processed_response['score_A'],
                                    'score_B': processed_response['score_B']
                                }
                                metrics = list(multi_dimension_score['score_A'].keys())
                                scores_A = [multi_dimension_score['score_A'][metric] for metric in metrics]
                                scores_B = [multi_dimension_score['score_B'][metric] for metric in metrics]
                                
                                # 创建 Plotly 图表
                                fig = go.Figure(data=[
                                    go.Bar(name='Answer A', x=metrics, y=scores_A),
                                    go.Bar(name='Answer B', x=metrics, y=scores_B)
                                ])
                                
                                # 更新图表布局
                                fig.update_layout(
                                    barmode='group',
                                    yaxis_title='Scores',
                                    legend_title='Answers',
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as chart_error:
                                st.error(f"Error creating score comparison chart: {str(chart_error)}")
                            
                            # 将结果添加到原数据中
                            item['model_critic'] = full_response
                            item['final_result'] = final_result
                            item['detailed_feedback'] = processed_response["Detailed_Evaluation_Feedback"]
                            item['score_A'] = processed_response.get('score_A', {})
                            item['score_B'] = processed_response.get('score_B', {})
                            
                            processed_items.append(item)
                            st.success(f"Successfully processed item {i+1}")
                            
                        except Exception as e:
                            st.error(f"Error processing item {i+1}: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc(), language="python")
                            
                            # 保存部分结果
                            item['model_critic'] = full_response if 'full_response' in locals() else "Error during processing"
                            item['processing_error'] = str(e)
                            processed_items.append(item)
                
                except Exception as item_error:
                    st.error(f"Unexpected error processing item {i+1}: {str(item_error)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
                    
                    # 记录错误但继续处理
                    item['processing_error'] = f"Processing error: {str(item_error)}"
                    processed_items.append(item)
            
            # 更新进度条为完成
            progress_bar.progress(1.0)
            status_text.text(f"Processed all {len(data)} items")
            
            # 保存处理后的数据
            try:
                with open(modified_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(processed_items, json_file, indent=4, ensure_ascii=False)
                st.success(f"All {len(processed_items)} items have been processed and saved to {modified_file_path}")
                
                # 设置处理完成标志
                st.session_state.file_processed = True
                
            except Exception as save_error:
                st.error(f"Error saving the processed results: {str(save_error)}")
                import traceback
                st.code(traceback.format_exc(), language="python")
        
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")

    # 文件处理完成后，提供下载按钮
    if st.session_state.file_processed:
        try:
            # 提供文件下载
            with open('critic_by_pairwise_data.json', 'rb') as f:
                download_clicked = st.download_button(
                    'Download Evaluation JSON File', 
                    f, 
                    file_name='critic_by_pairwise_data.json'
                )
                if download_clicked:
                    st.success("File downloaded successfully!")
                    # 重置标志以允许处理新文件
                    st.session_state.file_processed = False
                    st.rerun()  # 使用 st.rerun() 替代 st.experimental_rerun()
        except Exception as e:
            st.error(f"Error providing download: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")


# POINTWISE模式
else:
    # 文件上传处理部分
    if uploaded_file and not st.session_state.file_processed:
        try:
            # 重置文件指针到开始
            uploaded_file.seek(0)
            file_content = uploaded_file.read().decode('utf-8')  # 读取并解码为UTF-8格式的字符串
            
            # 添加调试信息
            st.write("File content sample (first 200 characters):")
            st.write(file_content[:200])
            
            # 尝试解析JSON数据
            try:
                data = json.loads(file_content)  # 解析JSON数据
            except json.JSONDecodeError as json_err:
                st.error(f"JSON parsing error: {str(json_err)}")
                st.error("Please upload a valid JSON file.")
                st.stop()
            
            # 确保数据是数组格式
            if not isinstance(data, list):
                st.error("The uploaded JSON must be in array format.")
                st.write(f"Current data type: {type(data)}")
                st.stop()
            
            if len(data) == 0:
                st.error("The uploaded JSON array is empty.")
                st.stop()
            
            # 检查每个元素是否包含必要的键
            missing_keys_items = []
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    st.error(f"Item {i+1} in the array is not a dictionary.")
                    st.stop()
                
                # 严格检查必需的字段
                required_keys = {'question_body', 'answer_body'}
                if not all(key in item for key in required_keys):
                    missing_keys = required_keys - set(item.keys())
                    missing_keys_items.append((i, missing_keys))
            
            # 如果有任何项目缺少必要的键，显示错误并停止
            if missing_keys_items:
                st.error("The following items are missing required fields:")
                for i, missing in missing_keys_items:
                    st.write(f"Item {i+1}: Missing {', '.join(missing)}")
                st.error("Please ensure all items include 'question_body' and 'answer_body' fields.")
                st.stop()
            
            # 数据验证通过，继续处理
            st.success(f"Successfully validated {len(data)} items, all items contain the required fields.")
            
            # 输出文件路径
            modified_file_path = 'critic_by_pointwise_data.json'
            
            # 处理上传的数据
            processed_items = []
            
            # 创建一个进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 使用一个容器来显示处理结果
            results_container = st.container()
            
            for i, item in enumerate(data):
                # 更新进度条和状态文本
                progress = (i) / len(data)
                progress_bar.progress(progress)
                status_text.text(f"Processing item {i+1}/{len(data)} ({int(progress*100)}%)")
                
                try:
                    with results_container.expander(f"Item {i+1}/{len(data)}", expanded=(i==0)):
                        st.write(f"Question: {item['question_body'][:100]}...")
                        
                        # 获取必须的键
                        question_body = item['question_body']
                        answer_body = item['answer_body']
                        
                        # 检查是否有参考答案
                        reference = item.get('reference', "")
                        
                        # 选择适当的模板文件
                        if reference:
                            template_file = "/root/autodl-tmp/demo/prompt_template/POINTWISE_WR.txt"
                        else:
                            template_file = "/root/autodl-tmp/demo/prompt_template/POINTWISE_WOR.txt"
                        
                        # 尝试读取模板文件
                        try:
                            with open(template_file, "r") as file:
                                base_prompt = file.read()
                        except FileNotFoundError:
                            st.error(f"Template file {template_file} not found.")
                            # 尝试备用路径
                            backup_paths = [
                                f"./prompt_template/{os.path.basename(template_file)}",
                                f"../prompt_template/{os.path.basename(template_file)}",
                                f"prompt_template/{os.path.basename(template_file)}"
                            ]
                            template_found = False
                            for path in backup_paths:
                                try:
                                    with open(path, "r") as file:
                                        base_prompt = file.read()
                                        template_found = True
                                        st.success(f"Using backup template: {path}")
                                        break
                                except FileNotFoundError:
                                    continue
                            
                            if not template_found:
                                st.error("Could not find the necessary template file.")
                                continue  # Skip this item and move to the next one
                        
                        # 确定评价类别
                        scenario = item.get('category', st.session_state.selected_option)
                        
                        # 生成最终的prompt
                        if not selected_criteria:
                            final_prompt = base_prompt.format(
                                scenario = scenario,
                                criteria = read_criteria(scenario),
                                question_body = question_body,
                                answer_body = answer_body,
                                reference = reference if reference else "N/A"
                            )
                        else:
                            final_prompt = base_prompt.format(
                                scenario = scenario,
                                criteria = user_selected_criteria(selected_criteria),
                                question_body = question_body,
                                answer_body = answer_body,
                                reference = reference if reference else "N/A"
                            )
                        
                        # 使用占位符显示加载状态和响应
                        message_placeholder = st.empty()
                        
                        try:
                            with st.spinner(f'Evaluating answer...'):
                                # 使用API调用获取响应
                                full_response = call_ollama_api(
                                    st.session_state["model_name"], 
                                    final_prompt, 
                                    message_placeholder
                                )
                        except Exception as e:
                            st.error(f"Error connecting to Ollama: {str(e)}")
                            st.error("Please ensure Ollama service is running.")
                            item['processing_error'] = f"Connection error: {str(e)}"
                            processed_items.append(item)
                            continue  # 继续处理下一个项目
                        
                        message_placeholder.empty()
                        
                        try:
                            # 处理响应
                            processed_response = extract_gpt_response_info_pointwise(full_response)
                            
                            overall_score = processed_response["Overall_Score"]["Overall Score"]
                            result_text = f'📝 Final Score: <span style="color: #FF4500;">{overall_score}/10</span>'
                            
                            # 显示结果
                            col1, col2 = st.columns([1, 3])
                            
                            # 样式设置
                            common_style = "padding: 20px; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
                            background_color1 = "background-color: #f8f9fa;"
                            background_color2 = "background-color: #e9ecef;"
                            
                            with col1:
                                st.markdown(f"""
                                    <div style="{background_color1} {common_style}">
                                        <h2 style="color: #007BFF;">{result_text}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                    <div style="{background_color2} {common_style}">
                                        <h3 style="color: #6c757d;">Detailed Evaluation Feedback</h3>
                                        <p style="font-size: 16px; line-height: 1.6;">
                                            {processed_response["Detailed_Evaluation_Feedback"]}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                            
                            # 只展示简单的维度分数图表
                            try:
                                dimension_scores = processed_response["Dimension_Scores"]
                                metrics = list(dimension_scores.keys())
                                scores = list(dimension_scores.values())
                                
                                # 创建 Plotly 图表
                                fig = go.Figure(data=[
                                    go.Bar(x=metrics, y=scores)
                                ])
                                
                                # 更新图表布局
                                fig.update_layout(
                                    yaxis_title='Score',
                                    xaxis=dict(type='category'),
                                    yaxis=dict(range=[0, max(scores) + 1]),
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as chart_error:
                                st.error(f"Error creating dimension score chart: {str(chart_error)}")
                            
                            # 将结果添加到原数据中
                            item['model_critic'] = full_response
                            item['overall_score'] = overall_score
                            item['dimension_scores'] = processed_response["Dimension_Scores"]
                            item['detailed_feedback'] = processed_response["Detailed_Evaluation_Feedback"]
                            
                            processed_items.append(item)
                            st.success(f"Successfully processed item {i+1}")
                            
                        except Exception as e:
                            st.error(f"Error processing item {i+1}: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc(), language="python")
                            
                            # 保存部分结果
                            item['model_critic'] = full_response if 'full_response' in locals() else "Error during processing"
                            item['processing_error'] = str(e)
                            processed_items.append(item)
                
                except Exception as item_error:
                    st.error(f"Unexpected error processing item {i+1}: {str(item_error)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
                    
                    # 记录错误但继续处理
                    item['processing_error'] = f"Processing error: {str(item_error)}"
                    processed_items.append(item)
            
            # 更新进度条为完成
            progress_bar.progress(1.0)
            status_text.text(f"Processed all {len(data)} items")
            
            # 保存处理后的数据
            try:
                with open(modified_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(processed_items, json_file, indent=4, ensure_ascii=False)
                st.success(f"All {len(processed_items)} items have been processed and saved to {modified_file_path}")
                
                # 设置处理完成标志
                st.session_state.file_processed = True
                
            except Exception as save_error:
                st.error(f"Error saving the processed results: {str(save_error)}")
                import traceback
                st.code(traceback.format_exc(), language="python")
        
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")

    # 文件处理完成后，提供下载按钮
    if st.session_state.file_processed:
        try:
            # 提供文件下载
            with open('critic_by_pointwise_data.json', 'rb') as f:
                download_clicked = st.download_button(
                    'Download Evaluation JSON File', 
                    f, 
                    file_name='critic_by_pointwise_data.json'
                )
                if download_clicked:
                    st.success("File downloaded successfully!")
                    # 重置标志以允许处理新文件
                    st.session_state.file_processed = False
                    st.rerun()  # 使用 st.rerun() 替代 st.experimental_rerun()
        except Exception as e:
            st.error(f"Error providing download: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")