import base64
import time
from typing import Any, Optional
import numpy as np
from PIL import Image
import requests
import json
import logging
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from wrappers.base_wrapper import LlmWrapper, MultimodalLlmWrapper
from wrappers.utils import array_to_jpeg_bytes, action_majority_vote, post_to_port
from wrappers.constants import (
    END_POINT,
    SYSTEM_PROMPT,
    EXTRACT_SCHEMA,
    ERROR_CALLING_LLM,
    VALIDATOR,
    PORTS,
)


class ParallelMiniCPMWrapper(LlmWrapper, MultimodalLlmWrapper):

    RETRY_WAITING_SECONDS = 20

    def __init__(
        self,
        model_name: str,
        max_retry: int = 3,
        temperature: float = 0.1,
        use_history: bool = False,
        history_size: int = 10,  # 最多保留最近 history_size 轮
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
        self, text_prompt: str, images: list[np.ndarray], model_choice="MiniCPM"
    ) -> tuple[str, Optional[bool], Any]:
        assert len(images) == 1

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

        headers = {
            "Content-Type": "application/json",
        }
        json.dumps(payload, ensure_ascii=False, indent=2)

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        while counter > 0:
            try:
                with ThreadPoolExecutor(max_workers=len(PORTS)) as executor:
                    actions = []
                    results = executor.map(
                        lambda port: post_to_port(port, headers, payload), PORTS
                    )
                    for result in results:
                        response = result["response"]
                        if response.ok and "choices" in response.json():
                            assistant_msg = response.json()["choices"][0]["message"]
                            assistant_text = assistant_msg["content"]
                            action = self.extract_and_validate_json(assistant_text)
                            logging.info(f"One Action: {action}")
                            actions.append(action)
                    mv_action = action_majority_vote(actions)
                    self._push_history("user", user_content)
                    self._push_history("assistant", str(mv_action))
                    return assistant_text, None, response, mv_action
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Want to catch all exceptions happened during LLM calls.
                print(e)
                time.sleep(0.1)
                counter -= 1
                print("Error calling LLM, will retry soon...")
                print(e)
        return ERROR_CALLING_LLM, None, None
