import os
import server_config as server_config
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = server_config.CUDA_devices

import torch
import torch_npu
from flask import Flask, render_template, request
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import gc
import threading
import time
import requests
import json
import re
from torch_npu.contrib import transfer_to_npu

lock = threading.Lock()

app = Flask(__name__)

def Create_and_load_Model(checkpoint_path=server_config.checkpoint_path, tokenizer_path=server_config.tokenizer_path):
    try:
        # 从指定路径加载预训练的分词器，允许信任远程代码
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        raise Exception

    try:
        # 从指定路径加载预训练的模型，自动分配设备，并将模型设置为评估模式
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, device_map="auto").eval()

    except Exception as e:
        raise

    return model, tokenizer

def chat(query):
    with lock:
        generated_text = ""
        past_key_values = None
        current_length = 0
        max_length = server_config.max_output_tokens
        with torch.no_grad():
            # 调用模型的stream_chat方法，生成流式响应
            for response_text, _, past_key_values in global_Chat_Model.stream_chat(global_Tokenizer, query,
                                                                                    temperature=server_config.temperature,
                                                                                    past_key_values=past_key_values,
                                                                                    return_past_key_values=True):
                # 获取新生成的文本部分
                new_part = response_text[current_length:]
                # 检查生成的文本长度是否超过最大输出长度
                if len(generated_text) + len(new_part) > max_length:
                    # 计算还能添加的文本长度
                    remaining_length = max_length - len(generated_text)
                    # 截取新生成的文本部分
                    new_part = new_part[:remaining_length]
                    new_part += '...'
                    # 将截取后的文本添加到生成的文本中
                    generated_text += new_part
                    break
                # 将新生成的文本部分添加到生成的文本中
                generated_text += new_part
                # 更新当前生成的文本长度
                current_length = len(response_text)

            gc.collect()
            torch.cuda.empty_cache()

        return generated_text

def llm_call(sentence, verb, clause):
    query = "请结合叙实性动词，判断在主句语境下，假设句的真假值（真:T/假:F/不能确定:U）。仅返回结果对应英文符号。\n主句：{%s}\n叙实性动词：{%s}\n假设句：{%s}" % (
        sentence, verb, clause)
    TFU2str = {"T": "真", "F": "假", "U": "未知"}
    while True:
        content_response = chat(query).strip()
        if content_response not in TFU2str.keys():
            time.sleep(1)
            continue
        else:
            return TFU2str[content_response]

def outside_api_call(sentence, verb):
    api_key = server_config.api_key

    for i in range(3):
        try:
            prompt = """
            请识别 主句 中 命题态度动词 对应的 命题宾语从句(若识别失败返回{无命题宾语从句})，并判断 命题宾语从句 在 主句 语境下的真假值({真}/{假}/{不能确定})。若识别判断成功则回答 命题宾语从句(用{}包裹放第一行) 和 真假值(用{}包裹放第二行)，若识别失败则回答{无命题宾语从句}，均不做解释。\n\n主句：{%s}\n命题态度动词：{%s}
            """%(sentence, verb)
            
            prompt = prompt.strip()
            
            json_query_messages = [{"role": "user","content": prompt}]

            api_url = "https://api.deepseek.com/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "messages": json_query_messages,
                "model": "deepseek-chat",
                "stream": False,
                "temperature": 0
            }
            api_call_response = requests.post(api_url, headers=headers, data=json.dumps(data))
            json_response = api_call_response.json()
            content_response = json_response["choices"][0]["message"]["content"].strip()
            print(content_response)
            
            
            if content_response == "{无命题宾语从句}" or content_response == "无命题宾语从句":
                return "无", "无"
            else:
                match_c_c_tf = re.search(r"\{(.*?)\}\s*\n+\s*\{(.*?)\}", content_response, re.DOTALL)
                if match_c_c_tf:
                    c = match_c_c_tf.group(1).strip()
                    c_tf = match_c_c_tf.group(2).strip()
                    if c_tf in ["真","假","不能确定"]:
                        return c, c_tf
        except Exception as e:
            print(f"API调用失败: {e}")
            time.sleep(1)
            continue
        
    return "出错", "出错"
    

def init_server():
    global global_Chat_Model, global_Tokenizer
    try:
        global_Chat_Model, global_Tokenizer = Create_and_load_Model()
        query = "请结合叙实性动词，判断在主句语境下，假设句的真假值（真:T/假:F/不能确定:U）。仅返回结果对应英文符号。\n主句：{%s}\n叙实性动词：{%s}\n假设句：{%s}" % (
            '我知道你是人', '知道', '你是人')
        content_response = chat(query)
        print(content_response)
        pass
    except Exception as e:
        raise ValueError("模型加载出错")
    return app


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/show_list_factive_words')
def show_list_factive_words():
    path_factive_words = 'corpora/factive_verbs.xlsx'  # 修改为.xlsx
    word_type = request.args.get('word_type')  # 正叙实动词 反叙实动词 非叙实动词
    if word_type:
        df = pd.read_excel(path_factive_words)  # 修改为read_excel
        # df = df.drop(columns=['id'])  # df的列 '叙实性动词' '叙实性'
        if word_type != "总表":
            df = df[df['叙实性'] == word_type]
        count = len(df)
        head_html = """
            <p class="bold"> 检索到 %s 共 %d 个。</p>
            """ %(word_type, count)

        # 将 '叙实性动词' 列的值转换为超链接
        def create_link(verb):
            return f'<a href="/search_factive_verb?verb={verb}">{verb}</a>'
        df['叙实性动词'] = df['叙实性动词'].apply(create_link)

        table_html = df.to_html(classes='table table-striped', index=False, escape=False)
        return render_template('show_list_factive_words.html', head_html=head_html,table=table_html, word_type=word_type)

@app.route('/search_factive_verb', methods=['GET'])
def search_factive_verb():
    verb = request.args.get('verb')
    verb = verb.strip()
    filter_type = request.args.get('filter')

    main_html = ""
    head_html = ""

    if not verb:
        main_html = '<p style="color: red; font-weight: bold; text-align: center;">输入不可为空，请点击logo返回主页重新检索</p>'
    else:

        path_factive_verbs = f'corpora/factive_verbs.xlsx'  # 修改为.xlsx
        path_example_sentences = f'corpora/example_sentences/{verb}.xlsx'  # 修改为.xlsx

        df_verbs = pd.read_excel(path_factive_verbs)  # 修改为read_excel

        # id,叙实性动词,叙实性,叙实性概率,含真值宾语小句例句数量,含假值宾语小句例句数量,含真假未知宾语小句例句数量,含宾语小句例句数量,例句总量
        if verb in df_verbs['叙实性动词'].values:
            row = df_verbs[df_verbs['叙实性动词'] == verb]
            word_type = row['叙实性'].values[0]
            prob = row['叙实性概率'].values[0]
            count_T = int(row['含真值宾语小句例句数量'].values[0])
            count_F = int(row['含假值宾语小句例句数量'].values[0])
            count_U = int(row['含真假未知宾语小句例句数量'].values[0])
            count_valid_sents = int(row['含宾语小句例句数量'].values[0])
            count_sents = int(row['例句总量'].values[0])
            count_N = count_sents - count_valid_sents

            df_sents = pd.read_excel(path_example_sentences)  # 修改为read_excel
            # 过滤数据
            if filter_type == 'no_object_clause':
                # 假设无宾语小句的判断逻辑，这里需要根据实际情况修改
                df_sents = df_sents[df_sents['宾语小句'] == '无']
            elif filter_type == 'has_object_clause':
                df_sents = df_sents[df_sents['宾语小句'] != '无']
            elif filter_type == 'clause_true':
                df_sents = df_sents[df_sents['宾语小句真假性'] == '真']
            elif filter_type == 'clause_false':
                df_sents = df_sents[df_sents['宾语小句真假性'] == '假']
            elif filter_type == 'clause_unknown':
                df_sents = df_sents[df_sents['宾语小句真假性'] == '未知']

            # 在id列前添加单选框
            def add_radio_box(id_value):
                return f'<input type="radio" name="selected_row" value="{id_value}">{id_value}'
            df_sents['id'] = df_sents['id'].apply(add_radio_box)

            # 设置 escape=False 避免 HTML 标签被转义
            main_html = df_sents.to_html(classes='table table-striped', index=False, table_id='result-table', escape=False)

            # 为表格第二列添加宽度样式
            main_html = f"""
                <style>
                    #result-table td, #result-table th {{
                        text-align: center;
                        vertical-align: middle;
                    }}
                    #result-table td:nth-child(2), #result-table th:nth-child(2) {{
                        width: 2cm;
                    }}
                    #result-table td:nth-child(4), #result-table th:nth-child(4) {{
                        width: 4cm;
                    }}
                    #result-table td:nth-child(5), #result-table th:nth-child(5) {{
                        width: 2.5cm;
                    }}
                </style>
                {main_html}
            """

            type_TFU = {"正叙实动词": count_T, "反叙实动词": count_F, "非叙实动词": count_U}

            head_html = """
            <p class="title">%s 检索结果</p>
            <p class="bold">%s 的叙实性为 %s（%s 概率），典型例句有 %d 条，有 %d 条例句带宾语小句，其中宾语小句为真的有 %d 条，宾语小句为假的有 %d 条，宾语小句真假未知的有 %d 条。</p>
        
            """ % (verb, verb, word_type, prob, count_sents, count_valid_sents, count_T, count_F, count_U)

        else:
            main_html = '<p style="color: red; font-weight: bold; text-align: center;">该检索词 %s 为无叙实动词，请点击logo返回主页重新检索</p>' % (
                verb)

    return render_template('search_result.html', verb=verb, head_html=head_html, main_html=main_html)

@app.route('/judge_clause_value', methods=['GET'])
def judge_clause_value():
    sentence = request.args.get('sentence').strip()
    verb = request.args.get('verb').strip()
    clause = request.args.get('clause').strip()

    main_html = ""
    if not sentence or not verb or not clause:
        main_html = '<p style="color: red; font-weight: bold; text-align: center;">输入均不可为空，请重新输入</p>'
    elif verb not in sentence:
        main_html = """
        <p style="color: red; font-weight: bold; text-align: center;">叙实性动词框内词 %s 并未出现在主句中，请重新输入</p>
        """ % (verb)
    else:
        path_factive_verbs = f'corpora/factive_verbs.xlsx'  # 修改为.xlsx
        df_verbs = pd.read_excel(path_factive_verbs)  # 修改为read_excel
        if verb in df_verbs['叙实性动词'].values:
            result = llm_call(sentence, verb, clause)
            main_html = """
            <p>在主句 <span class="bold">%s</span> 语境下，由叙实性动词 <span class="bold">%s</span> 预设的宾语小句 <span class="bold">%s</span> 的真假性为</p>
            <p style="color: red; font-weight: bold; text-align: center; font-size: 25px;">%s</p>
            
            <p>判断结果由 微调的ChatGLM-3-6B 提供</p>
            """ % (sentence, verb, clause, result)
        else:
            main_html = '<p style="color: red; font-weight: bold; text-align: center;">叙实性动词输入框内的 %s 为无叙实动词，请重新输入</p>' % (
                verb)

    return render_template('judge_result.html', main_html=main_html, sentence=sentence, verb=verb, clause=clause)

@app.route('/rejudge', methods=['GET'])
def rejudge():
    sentence = request.args.get('sentence').strip()
    verb = request.args.get('verb').strip()
    api_llm = request.args.get('api_llm').strip()
    

    main_html = ""
    if not sentence or not verb:
        main_html = '<p style="color: red; font-weight: bold; text-align: center;">输入均不可为空，请重新输入</p>'
    elif verb not in sentence:
        main_html = """
        <p style="color: red; font-weight: bold; text-align: center;">叙实性动词框内词 %s 并未出现在主句中，请重新输入</p>
        """ % (verb)
    else:
        path_factive_verbs = f'corpora/factive_verbs.xlsx'  # 修改为.xlsx
        df_verbs = pd.read_excel(path_factive_verbs)  # 修改为read_excel
        if verb in df_verbs['叙实性动词'].values:
            c, c_tf = outside_api_call(sentence, verb)
            if c_tf == "出错":
                main_html = '<p style="color: red; font-weight: bold; text-align: center;">API调用失败，请稍后再试</p>'
            elif c == "无":
                main_html = """
                <p>在主句 <span class="bold">%s</span> 语境下，叙实性动词 <span class="bold">%s</span></p>
                <p style="color: red; font-weight: bold; text-align: center; font-size: 25px;">没有预设宾语小句</p>
                
                <p>判断结果由 在线%s 提供</p>
                """ % (sentence, verb, api_llm)
            else:
                main_html = """
                <p>
                在主句 <span class="bold">%s</span> 语境下，
                叙实性动词 <span class="bold">%s</span> 预设的宾语小句为
                </p>
                
                <p style="color: red; font-weight: bold; text-align: center; font-size: 25px;">%s</p>
                宾语小句对应的真假值为 <p style="color: red; font-weight: bold; text-align: center; font-size: 25px;">%s</p>
            
                <p>判断结果由 在线%s 提供</p>
                """ % (sentence, verb, c,c_tf,api_llm)
        else:
            main_html = '<p style="color: red; font-weight: bold; text-align: center;">叙实性动词输入框内的 %s 为无叙实动词，请重新输入</p>' % (
                verb)

    return render_template('rejudge.html', main_html=main_html, sentence=sentence, verb=verb)

if __name__ == '__main__':
    app = init_server()
    app.run(host=server_config.host, port=server_config.port, debug=False)