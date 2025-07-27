# /// script
# requires-python = ">=3.12"
# dependencies = ["langchain_openai","openai","pydantic","pandas","tqdm","openpyxl","requests"]
# ///
import random
import traceback
import numpy as np
import pandas as pd
import requests
import signal
import json
from tqdm import tqdm
import os
from llm import init_model

# 定义超超时跳过这次循环
class TimeOutError(Exception):
    pass

def handle(signum, frame):
    raise TimeOutError("硅基流动回应超时了！！")


def write_to_json_append(train_data_list, txt_path='task2/train_data/moreVariiesQA.txt'):
    # 转换新的训练集数据
    new_data = []
    print(f"train_data_list:{train_data_list}")
    for data in train_data_list:
        data = data
        if isinstance(data, str):
            continue
        # 替换掉数据不完整的：
        if "数据不完整" in data['q'] or "数据不完整" in data['a']:
            continue
        new_data.append({'instruction': data['q'].replace('\n', '').replace('\r', '').strip(), 'output': data['a'].replace('\n', '').replace('\r', '').strip()})

     # 追加写入到TXT文件
    with open(txt_path, 'a', encoding='utf-8') as f:
        for data in new_data:
            f.write(str(data) + '\n')

    # 打印合并后的记录数
    with open(txt_path, 'r', encoding='utf-8') as f:
        combined_data = f.readlines()

    print(f"已添加 {len(new_data)} 条新数据，文件现在包含 {len(combined_data)} 条记录。")

# 官方经我修改的
def call_llm(content: str):
    """
    调用大模型
    
    Args:
        content: 模型对话文本
    
    Returns:
        list: 问答对列表
    """
    # 调用大模型（硅基流动免费模型，推荐学习者自己申请）
    url = "https://api.siliconflow.cn/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen3-8B",
        "messages": [
            {
                "role": "user",
                "content": content  # 最终提示词，"/no_think"是关闭了qwen3的思考
            }
        ]
    }
    headers = {
        "Authorization": "Bearer sk-gojsqvvtdtbavxuibdkajzxugwnipdngoiqdlsjmlrsygtvi", # 替换自己的api token
        "Content-Type": "application/json"
    }
    resp = requests.request("POST", url, json=payload, headers=headers).json()
    
    try:
        # 提取API响应中的内容
        content = resp['choices'][0]['message']['content'].split('</think>')[-1]
        # 清理和格式化输出
        formatted_content = content.strip()
        return formatted_content
    except (KeyError, IndexError, AttributeError) as e:
        # 处理可能的错误
        print(f"Error extracting content from response: {e}")
        return None

# 我自己的接口
llm = init_model("qwen")
# 处理llm返回的结果
def parse_output(llm_result):
    return llm_result.content

if __name__ == "__main__":

    # 读取数据
    
    # 多元化问题的prompt
    prompt = '''你是一位专业的铁路客运信息助手，请根据提供的列车时刻表数据准确回答用户问题。
        
    # 列车时刻表数据
    {train_info}

    # （十分重要）回答要求（必须遵守，必须遵守，必须遵守，必须遵守）
    1. 必须结合所给的列车数据进行精确查询和分析，不要捏造不存在的信息
    2. 所有题目直接输出简要的结果（重要）
    3. 对于多条件筛选，确保满足所有条件
    4. 若数据缺失，请明确指出"数据不完整"
    5. 回答要简洁、准确
    7. 给出简要的答案，不要在答案中包含任何额外的解释或说明。不要写出“根据列车时刻表数据”
    8. 不要使用双引号，使用单引号
    9. 如果给你的信息不足以生成复杂的"q",那么就生成单字段的"q"
    10. 尽量避免输出"数据不完善"
    11. 不要提问列车的“运行”时间
    12. output字段一定要用双引号（重要）

    # 以下面的"q"字段为模版，综合选择最合适的1-3条，生成"a"字段的内容：
    
    1.单字段查询：如“Z152次列车的终到站是哪里？”
    2.多条件筛选：如“在综合候乘中心候车、发车时间晚于08:00的列车有哪些？”
    3.跨行计算：如“从兰州到北京西的列车中，哪趟运行时间最短？”
    4.时间推理：如“K420次列车在兰州站的停留时长是多久？”
    5.复杂时间范围过滤：如“在'综合候乘中心，到点时间介于06:00至08:00之间且站台为2的车次有哪些？”
    6.缺失数据处理：如“检票口为'5B'且开点时间缺失的车次有哪些？”
    7.复杂单位换算：如“K2095/8次列车的开点时间为05:51，若延误1小时15分钟，新的开点时间是几点？”

    # "a"样式的模版：
    回答：T308次列车在南昌站的到点时间是00:16:00
    回答：从成都西出发前往佳木斯，可以在23:40左右乘坐K4547/6次列车。
    回答：T308次列车在综合候乘中心，高架候车区西区候车
    回答：T308次列车检票口是1A号
    回答：T308次列车停靠在5号站台
    回答：T308次列车在南昌站停留了12分钟
    回答：D218次列车的始发站是兰州
    回答：D218次列车的终到站是上海
    回答：凌晨1点左右，从兰州开往上海没有直接符合条件的班次
    
    '''
    output_format = '''# 输出格式（输出成一行）
    {"instruction": "填写生成的问题","output": "问题答案"}
    ...
    '''  
    txt_path='./train_data/moreVariiesQA_1000lines.txt'
    data = pd.read_excel('./data/fill_NaN_data.xlsx')
    # 定义信号处理函数
    signal.signal(signal.SIGALRM, handler=handle)
    # 定义超时时间为30秒
    timeout = 100
    for i in range(300):
        try:
            # 随机选择1-10行数据, 经实验很难基本上输出不了稍微复杂的问题，绝大多数都是单字段问题
            # random_data = data.sample(n=random.randint(4,10), random_state=42)
            # 优化：选取连在一起的行，保证能输出更高质量的QA
            print(f"♻️这是第「{i}」次循环♻️")
            start_index = np.random.randint(0, len(data) - 8)
            random_rows = np.random.randint(3,5) # 这里不能调太大，容易超时
            random_data = data.iloc[start_index:start_index+random_rows]
            train_info = ""
            
            # 设置计时器
            signal.alarm(timeout)
            for i, row in random_data.iterrows():
                train_info += f"车次：{row['车次']}，候车区：{row['候车厅']}，检票口：{row['检票口']}，站台：{row['站台']},始发站：{row['始发站']}，终点站：{row['终到站']}，到点：{row['到点']}，开点：{row['开点']}\n"
            print("选取的train_info为：\n", train_info)
            llm_result = llm.invoke(prompt.format(train_info=train_info) + output_format)#call_llm(prompt.format(train_info=train_info) + output_format)
            # 未超时则取消计时器
            signal.alarm(0)
            
            print("llm_result为：\n", llm_result.content)
            with open(txt_path, 'a', encoding='utf-8') as f:
                for i in llm_result.content:
                    f.write(str(i))
                f.write("\n")
        except TimeOutError :
            print("等待大模型平台回复超时，跳过此次循环！")
            continue
