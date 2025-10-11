import base64
import time
from typing import Any, Optional
import numpy as np
from PIL import Image
import requests
import json
from typing import Optional, Any
from wrappers.base_wrapper import LlmWrapper, MultimodalLlmWrapper
from wrappers.utils import array_to_jpeg_bytes
from wrappers.constants import (
    END_POINT,
    SYSTEM_PROMPT,
    EXTRACT_SCHEMA,
    ERROR_CALLING_LLM,
    VALIDATOR,
)
from omni_parser.paser import OmniParserClient
import log.log_recorder
import log.experience_pool
import log.log_replay
import user.user_manager
from user.user_manager import UserManager
import os
import re


class MiniCPMWrapper(LlmWrapper, MultimodalLlmWrapper):

    RETRY_WAITING_SECONDS = 20

    def __init__(
        self,
        model_name: str,
        max_retry: int = 3,
        temperature: float = 0.1,
        use_history: bool = False,
        history_size: int = 10,  # 最多保留最近 history_size 轮,
        enable_vision_parser: bool = False,
    ):
        if max_retry <= 0:
            max_retry = 3
            print("Max_retry must be positive. Reset it to 3")
        self.max_retry = min(max_retry, 5)
        self.temperature = temperature
        self.model = model_name

        # ---------- 新增 ----------
        self.use_history = use_history
        self.history_size = max(history_size, 1)
        # history 以「单条消息」为粒度： [{'role': .., 'content': ..}, ...]
        self.history: list[dict] = []

        # 初始化 OmniParser
        self.enable_vision_parser = enable_vision_parser
        if self.enable_vision_parser:
            self.omni_parser = OmniParserClient()
        else:
            self.omni_parser = None

    @staticmethod
    def resize_image_pil(image: np.ndarray, max_size=640) -> np.ndarray:
        img = Image.fromarray(image)
        width, height = img.size
        scale = min(max_size / max(width, height), 1.0)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return np.array(resized_img)

    @classmethod
    def encode_image(cls, image: np.ndarray) -> str:
        # small_image = cls.resize_image_pil(image)
        encoded = base64.b64encode(array_to_jpeg_bytes(image)).decode("utf-8")
        return encoded

    def _push_history(self, role: str, content: Any):
        """把一条消息写入历史，并自动裁剪长度。"""
        if not self.use_history:
            return
        self.history.append({"role": role, "content": content})
        # 每轮对话包含 user + assistant 两条消息
        max_msgs = self.history_size * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def clear_history(self):
        """外部可手动清空记忆。"""
        self.history.clear()

    def extract_and_validate_json(self, input_string):
        try:
            json_obj = json.loads(input_string)
            VALIDATOR.validate(json_obj, EXTRACT_SCHEMA)
            return json_obj
        except json.JSONDecodeError as e:
            print("Error, JSON is NOT valid.")
            return input_string
        except Exception as e:
            print(f"Error, JSON is NOT valid according to the schema.{input_string}", e)
            return input_string

    def predict(
        self,
        text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
        self,
        text_prompt: str,
        images: list[np.ndarray],
        model_choice="MiniCPM",
        last_result=None,  # 用于多步执行
    ) -> tuple[str, Optional[bool], Any]:
        assert len(images) == 1
        # ----------新增部分--------
        # 确保单例UserManager
        if not hasattr(self, "_user_manager"):
            self._user_manager = UserManager()
        user_manager = self._user_manager
        if not hasattr(self, "_last_recorded_input"):
            self._last_recorded_input = None

        # 获取或创建用户
        if not user_manager.current_user_id:
            user_id = user_manager.create_user({"username": "auto_created_user"})
            print(f"Created new user: {user_id}")

        if text_prompt != self._last_recorded_input:
            try:
                user_manager.record_user_interaction(
                    user_input=text_prompt,
                )
                self._last_recorded_input = text_prompt
            except Exception as e:
                print(f"用户交互记录失败: {str(e)}")
        user_dir = user_manager.get_user_dir(user_manager.current_user_id)
        with log.log_recorder.TaskLogger(
            log_dir=os.path.join(user_dir, "task_logs")
        ) as task_logger:
            task_logger.log_data["metadata"]["query"] = text_prompt
            # task_logger.log_data["metadata"]["ocr_result"] = ocr_results

        # -------- 构造 messages --------

        messages: list[dict] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            }
        ]

        # 1) 插入历史
        if self.use_history and self.history:
            messages.extend(self.history)

        # 2) 当前 user 消息

        # --------- 多部执行时加入上一步的结果 --------
        if last_result is not None:
            text_prompt += f"\n上一步提取的结果为：{last_result}"

        user_content = [
            {
                "type": "text",
                "text": f"<Question>{text_prompt}</Question>\n当前屏幕截图：(<image>./</image>)",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.encode_image(images[0])}"
                },
            },
        ]

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            # "max_tokens": 2048,
        }
        # 在payload中添加结果提取提示
        if "提取" in text_prompt or "获取" in text_prompt:
            payload["messages"][0]["content"] = SYSTEM_PROMPT + (
                "\n注意：如果你需要提取信息作为结果，请在JSON响应中包含RESULT字段。"
                '例如：{"RESULT": "海绵宝宝", ...}'
            )

        with open("payload.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        headers = {
            "Content-Type": "application/json",
        }
        json.dumps(payload, ensure_ascii=False, indent=2)

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        while counter > 0:
            try:
                response = requests.post(
                    END_POINT,
                    headers=headers,
                    json=payload,
                )
                if response.ok and "choices" in response.json():
                    assistant_msg = response.json()["choices"][0]["message"]
                    assistant_text = assistant_msg["content"]
                    result = None
                    if "RESULT" in assistant_text:
                        try:
                            result_match = re.search(
                                r'"RESULT":\s*"([^"]+)"', assistant_text
                            )
                            if result_match:
                                result = result_match.group(1)
                        except:
                            pass
                    action = self.extract_and_validate_json(assistant_text)
                    if "POINT" in action and self.enable_vision_parser:
                        original_point = action["POINT"]
                        new_point = self.omni_parser.fix_point(
                            images[0], original_point
                        )
                        action["POINT"] = new_point
                        assistant_text.replace(
                            f'"POINT": {original_point}', f'"POINT": {new_point}'
                        )

                    # --------新增部分--------
                    task_logger.record_step(
                        screenshot=Image.fromarray(images[0]),
                        action=action,
                        response=assistant_text,
                    )

                    # -------- 写回历史 --------
                    self._push_history("user", user_content)
                    self._push_history("assistant", assistant_msg["content"])

                    return assistant_text, None, response, action, result
                print(
                    "Error calling OpenAI API with error message: "
                    + response.json()["error"]["message"]
                )
                time.sleep(wait_seconds)
                wait_seconds *= 2
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Want to catch all exceptions happened during LLM calls.
                print(e)
                time.sleep(wait_seconds)
                wait_seconds *= 2
                counter -= 1
                print("Error calling LLM, will retry soon...")
                print(e)
                # 新增:记录失败状态
                task_logger.log_data["metadata"]["status"] = "failed"
        return ERROR_CALLING_LLM, None, None
