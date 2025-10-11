import base64
import time
from typing import Any, Optional
import numpy as np
from PIL import Image
import json
from typing import Optional, Any
from wrappers.base_wrapper import LlmWrapper, MultimodalLlmWrapper
from wrappers.utils import array_to_jpeg_bytes
from wrappers.constants import (
    SYSTEM_PROMPT,
    EXTRACT_SCHEMA,
    ERROR_CALLING_LLM,
    VALIDATOR,
    CLIENT,
)


class OpenAILlmWrapper(LlmWrapper, MultimodalLlmWrapper):
    """Wrapper for OpenAI compatible models."""

    RETRY_WAITING_SECONDS = 20

    def __init__(
        self,
        model_name: str,
        max_retry: int = 3,
        temperature: float = 0.1,
        use_history: bool = False,
        history_size: int = 10,
    ):
        if max_retry <= 0:
            max_retry = 3
            print("Max_retry must be positive. Reset it to 3")
        self.max_retry = min(max_retry, 5)
        self.temperature = temperature
        self.model = model_name
        self.use_history = use_history
        self.history_size = max(history_size, 1)
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
        encoded = base64.b64encode(array_to_jpeg_bytes(image)).decode("utf-8")
        return encoded

    def _push_history(self, role: str, content: Any):
        """把一条消息写入历史，并自动裁剪长度。"""
        if not self.use_history:
            return
        self.history.append({"role": role, "content": content})
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

    def _prepare_messages(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> list[dict]:
        """Prepare messages for the LLM API call."""
        messages: list[dict] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            }
        ]

        if self.use_history and self.history:
            messages.extend(self.history)

        user_content = [
            {
                "type": "text",
                "text": f"<Question>{text_prompt}</Question>\n当前屏幕截图：(<image>./</image>)",
            },
        ]

        if images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encode_image(images[0])}"
                    },
                }
            )

        messages.append({"role": "user", "content": user_content})
        return messages

    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        assert len(images) == 1

        messages = self._prepare_messages(text_prompt, images)

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        while counter > 0:
            try:
                response = CLIENT.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    top_p=1.0,
                    n=1,
                ).model_dump()

                assistant_msg = response["choices"][0]["message"]
                assistant_text = assistant_msg["content"]
                action = self.extract_and_validate_json(assistant_text)

                self._push_history("user", messages[-1]["content"])
                self._push_history("assistant", assistant_msg["content"])

                return assistant_text, None, response, action

            except Exception as e:
                print(e)
                time.sleep(wait_seconds)
                wait_seconds *= 2
                counter -= 1
                print("Error calling LLM, will retry soon...")
                print(e)

        return ERROR_CALLING_LLM, None, None
