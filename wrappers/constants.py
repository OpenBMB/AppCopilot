import os
import json
from openai import OpenAI
from jsonschema import Draft7Validator
from google.generativeai import types


# ---- utils functions for loading config -----
def get_schema():
    # get the current file path
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    ACTION_SCHEMA = json.load(
        open(os.path.join(current_dir, "schema_thought.json"), encoding="utf-8")
    )
    items = list(ACTION_SCHEMA.items())

    insert_index = 3
    items.insert(insert_index, ("required", ["thought"]))
    ACTION_SCHEMA = dict(items)

    # load extract schema:
    EXTRACT_SCHEMA = json.load(
        open(os.path.join(current_dir, "schema_for_extraction.json"), encoding="utf-8")
    )

    return ACTION_SCHEMA, EXTRACT_SCHEMA


def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)


# ----- model config -----
MODEL_EXTRACT = "deepseek-v3-250324"
ERROR_CALLING_LLM = "Error calling LLM"
MODEL_NOT_FOUND = "LLM not found"
END_POINT = "http://localhost:8001/v1/chat/completions"
PORTS = [8002, 8003, 8004]
CLIENT_API_KEY = "sk-y4nJGOZwMV7cYGgqEd3fF33321014fE0A2E1E51268E6470d"
CLIENT_BASE_URL = "https://yeysai.com/v1/"
CLIENT = OpenAI(
    api_key=CLIENT_API_KEY,
    base_url=CLIENT_BASE_URL,
)

# Default port for the Omni Parser model
OMNI_PORT = 7000

# ----- action schema -----
ACTION_SCHEMA, EXTRACT_SCHEMA = get_schema()
VALIDATOR = Draft7Validator(EXTRACT_SCHEMA)
# user_query_template_low = """The user query:  {user_request} 
#     Task progress (You have done the following operation on the current device): {history_actions}"""


# ---- several prompt -----
SYSTEM_PROMPT = f"""# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出
- 输出操作必须遵循Schema约束

# Schema
{json.dumps(ACTION_SCHEMA, indent=None, ensure_ascii=False, separators=(',', ':'))}"""

SYSTEM_PROMPT_UI_TARS = """
"You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n```\nThought: ...\nAction: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1, y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n\n## Note\n- Use Chinese in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.\n\n## User Instruction\nCould you help me set the image to Palette-Based?"
"""

SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (types.HarmBlockThreshold.BLOCK_NONE),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (types.HarmBlockThreshold.BLOCK_NONE),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}

# constants for CROSS DEVICE
# 支持的函数调用列表及其参数规范
# 用于定义设备间可调用的函数接口，确保通信时参数格式一致
SUPPORTED_FUNCTIONS = {
    "extract_keyword": {
        "description": "从设备2提取信息",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "设备2需要执行的操作步骤"},
                "extract_prompt": {
                    "type": "string",
                    "description": "视觉提取提示词，明确说明要提取什么信息",
                },
            },
            "required": ["task", "extract_prompt"],
        },
    }
}

TASK_SPLIT_PROMPT = """
# 任务拆分要求
将用户指令拆分为两台设备的操作描述，设备1和设备2是两台安卓手机，通过Socket通信。
输出格式必须是严格的JSON，**任何情况下不得得省略以下字段**：
- "device1_task": 设备1的完整操作描述（单句字符串）
- "device2_task": 设备2的完整操作描述（单句字符串）
- "dependency": 设备依赖关系（如"设备1需等待设备2发送关键词后再执行搜索操作"）
- "function_call": 设备1调用设备2的函数信息（**必须包含 "name" 和 "parameters" 两个键**）：
  - "name": 函数名称（字符串，如"extract_keyword"，不得为空）
  - "parameters": 函数参数（JSON对象，**必须包含 "task" 键**）：
    - "task": 详细描述设备2需要执行的操作（不得为空）
    - "extract_prompt": 明确的视觉提取提示词，例如："请从当前屏幕中提取视频标题，只需返回标题内容本身"

# 关键要求
1. 设备2的操作描述中必须包含明确的提取步骤和结果发送动作
2. 设备1的操作描述中必须包含接收设备2结果的等待环节
3. 两设备的操作描述需为连贯的单句，避免使用列表格式

# 示例
用户指令："设备1根据设备2的腾讯视频历史记录给设备2用户买礼物"
输出：
{{
  "device1_task": "打开淘宝APP进入首页，等待设备2发送的关键词，输入关键词后点击搜索并选择第一个商品加入购物车",
  "device2_task": "打开腾讯视频APP，点击底部'我的'进入个人中心，找到'历史播放'并点击进入，提取第一条视频的标题文本并将其作为关键词发送给设备1",
  "dependency": "设备1需等待设备2发送视频标题后再执行搜索操作",
  "function_call": {{
    "name": "extract_keyword",
    "parameters": {{
      "task": "打开腾讯视频APP，进入历史播放记录，提取第一条视频的标题",
      "extract_prompt": "请从历史记录列表中提取第一条视频的标题，只需返回标题文本"
     }}
  }}
}}

# 用户指令
{user_instruction}
"""

# Define available tasks
AVAILABLE_TASKS = {
    "answer_call": "接听电话",
    "hangup_call": "挂断电话",
    "make_call": "给妈妈打电话",
    "send_sms": "给12345发信息，内容为“你好”",
    "login_unicom": "登陆中国联通app",
    "book_broadband": "在中国联通app中预约宽带办理",
    "download_unicom": "下载中国电信app",
    "watch_video": "在bilibili上，看老番茄第2个视频，并点赞",
    "multi_step_task": "请帮我把bilibili的李子柒的前几个视频标题进行提取，并将发给微信联系人 File Transfer",
}