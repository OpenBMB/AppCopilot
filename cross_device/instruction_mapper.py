import json
import numpy as np
import os
import sys
from wrappers.cpm_wrapper import MiniCPMWrapper
from wrappers.constants import CLIENT, SUPPORTED_FUNCTIONS, TASK_SPLIT_PROMPT

current_dir = os.path.dirname(os.path.abspath(__file__))
outer_dir = os.path.dirname(current_dir)
sys.path.append(outer_dir)


class InstructionMapper:
    """
    负责将用户指令拆分为两台安卓设备的协同操作任务，通过调用llm，生成符合格式要求的设备任务描述、依赖关系及函数调用信息
    """

    def __init__(self):
        self.model = MiniCPMWrapper(
            model_name="AgentCPM-GUI", temperature=0.6, use_history=False
        )

    def split_task(self, user_instruction: str) -> dict:
        prompt = TASK_SPLIT_PROMPT.format(user_instruction=user_instruction)
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)  # 空图像
        messages = [
            {"role": "system", "content": TASK_SPLIT_PROMPT},
            {"role": "user", "content": user_instruction},
        ]
        response = CLIENT.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            temperature=0,
            top_p=1.0,
            n=1,
        ).model_dump()

        response = response["choices"][0]["message"]
        # response = self.model.predict_mm(prompt, [dummy_image])
        json_str = self._extract_json(response["content"])
        task_info = json.loads(json_str)
        cleaned_task_info = {}
        for key, value in task_info.items():
            cleaned_key = key.strip('\n "')
            cleaned_task_info[cleaned_key] = value

        required_fields = [
            "device1_tasks",
            "device2_tasks",
            "dependency",
            "function_call",
        ]
        for field in required_fields:
            if field not in cleaned_task_info:
                cleaned_task_info[field] = (
                    [] if "tasks" in field else "" if field == "dependency" else {}
                )

        return cleaned_task_info

    def _extract_json(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("模型输出不包含有效JSON")
        return text[start:end]
