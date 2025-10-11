import json
import time
from wrappers.constants import MODEL_EXTRACT, CLIENT


def is_need_multi_step(query):
    """判断是否需要多次执行任务"""
    with open("multi_step/multi_step_instruction.json", "r", encoding="utf-8") as f:
        instructions = json.load(f).get("instructions", [])
    for instruction in instructions:
        if instruction["instruction"] == query:
            step1 = instruction.get("steps", [])[0]
            step2 = instruction.get("steps", [])[1]
            return True, step1, step2
    return False, None, None


def extract_info(query, log_history):
    """
    结合历史日志和当前OCR数据提取信息
    :param query: 用户指令
    :param log_history: 历史操作日志
    """
    client = CLIENT
    # 确保已经获取当前屏幕OCR识别结果
    time.sleep(5)

    try:
        with open("./user/information.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        ocr_data = data.get("ocr_records", {})
    except FileNotFoundError:
        # 文件不存在，我们可以设置ocr_data为一个空字典，并记录错误或忽略
        ocr_data = {}
        print("文件不存在")
    except json.JSONDecodeError:
        # 文件存在，但内容不是有效的JSON
        ocr_data = {}
        print("JSON解析错误")
    except Exception as e:
        # 其他可能的异常
        ocr_data = {}
        print(f"发生未知错误: {e}")

    # 构建系统提示词
    system_prompt = f"""
    你是一名专业的数据提取助手，需要结合以下信息完成任务：
    1. 用户当前任务：{query}
    2. 历史操作记录：{json.dumps(log_history, ensure_ascii=False)}
    3. 当前屏幕OCR识别结果：{json.dumps(ocr_data, ensure_ascii=False)}

    注意：
    1.只需要返回提取出所需的关键信息，不要返回其他的任何无关信息。
    2.只要根据用户当前任务的的前半部分需求来进行信息提取。
    3.返回内容时不要擅自加上引号
    
    请根据上述信息和注意的点，从OCR结果中精确提取所需信息。
    """

    response = client.chat.completions.create(
        model=MODEL_EXTRACT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请提取关键信息"},
        ],
    )

    # 解析并返回结果
    return response.choices[0].message.content
