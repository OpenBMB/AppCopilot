import base64
import time
from typing import Any, Optional
import numpy as np
from PIL import Image
import json
from typing import Optional, Any
import requests
from wrappers.base_wrapper import LlmWrapper, MultimodalLlmWrapper
from wrappers.utils import array_to_jpeg_bytes
from wrappers.constants import (
    SYSTEM_PROMPT,
    EXTRACT_SCHEMA,
    ERROR_CALLING_LLM,
    VALIDATOR,
    CLIENT,
    END_POINT,
)


class UITarsLlmWrapper(LlmWrapper, MultimodalLlmWrapper):

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

    def extract_thought_and_action(self, input_string):
        parts = input_string.split("\n")
        thought = ""
        action = ""

        for part in parts:
            if part.startswith("Thought:"):
                thought = part[len("Thought:") :].strip()
            elif part.startswith("Action:"):
                action = part[len("Action:") :].strip()
                print(action)

        action = self.uitars2minicpm(action)
        print("thought:", thought)  # For debugging
        return action  # Return only the action dictionary

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
                "content": [{"type": "text", "text": "you are a helpful assistant"}],
            }
        ]
        text = (
            "You are a GUI agent. You are given a task and your action history, with screenshots. "
            "You need to perform the next action to complete the task. \n\n"
            "## Output Format\n\n"
            "Thought: ...\n"
            "Action: ...\n\n\n"
            "## Action Space\n"
            "click(start_box='<|box_start|>(x1,y1)<|box_end|>')\n"
            "long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')\n"
            "type(content='')\n"
            "scroll(direction='down or up or right or left')\n"
            # "open_app(app_name=\'\')\n"
            "press_back()\n"
            "press_home()\n"
            "wait()\n"
            "finished() # Submit the task regardless of whether it succeeds or fails.\n\n"
            "## Note\n"
            "- Use Chinese in Thought part.\n\n"
            "- Summarize your next action (with its target element) in one sentence in Thought part.\n\n"
            "## User Instruction\n" + text_prompt
        )

        if self.use_history and self.history:
            messages.extend(self.history)

        user_content = [
            {
                "type": "text",
                "text": text,
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

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }

        headers = {
            "Content-Type": "application/json",
        }

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
                    action = self.extract_thought_and_action(assistant_text)

                    self._push_history("user", messages[-1]["content"])
                    self._push_history("assistant", assistant_msg["content"])

                    return assistant_text, None, response, action
                print(
                    "Error calling OpenAI API with error message: "
                    + response.json()["error"]["message"]
                )
                time.sleep(wait_seconds)
                wait_seconds *= 2

            except Exception as e:
                print(e)
                time.sleep(wait_seconds)
                wait_seconds *= 2
                counter -= 1
                print("Error calling LLM, will retry soon...")
                print(e)

        return ERROR_CALLING_LLM, None, None

    def uitars2minicpm(self, action_str):
        """
        Convert the ui-tars action string to the minicpm schema format

        Args:
            action_str (str): like "click(start_box='<|box_start|>(558,925)<|box_end|>')"

        Returns:
            dict: new format action dictionary
        """
        result = {"STATUS": "continue"}

        # auxiliary function to extract coordinates
        def extract_coords(s):
            # directly find and extract the coordinates in the parentheses
            first_bracket = s.find("(")
            start = s.find("(", first_bracket + 1)
            end = s.find(")")
            if start != -1 and end != -1:
                coords_str = s[start + 1 : end].strip()  # extract the content in (x,y)
                x, y = coords_str.split(",")
                return [int(x), int(y)]
            raise ValueError(f"Cannot find coordinates in the string: {s}")

        if "click(" in action_str:
            result["POINT"] = extract_coords(action_str)

        elif "long_press(" in action_str:
            result["POINT"] = extract_coords(action_str)
            if "time='" in action_str:
                time = action_str.split("time='")[1].split("'")[0]
                result["duration"] = int(time) if time else 1000

        elif "type(" in action_str:
            content = action_str.split("content='")[1].split("'")[0]
            result["TYPE"] = content

        elif "scroll(" in action_str:
            direction = action_str.split("direction='")[1].split("'")[0]
            result["POINT"] = [500, 500]  # screen center point
            # need reverse direction
            if direction == "down":
                direction = "up"
            elif direction == "up":
                direction = "down"
            elif direction == "right":
                direction = "left"
            elif direction == "left":
                direction = "right"
            result["to"] = direction
        elif "press_back()" in action_str:
            result["PRESS"] = "BACK"

        elif "press_home()" in action_str:
            result["PRESS"] = "HOME"

        elif "wait()" in action_str:
            result["duration"] = 200

        elif "finished()" in action_str:
            result["STATUS"] = "finish"
        elif "open_app(app_name=" in action_str:
            result["OPEN_APP"] = action_str.split("app_name='")[1].split("'")[0]
        else:
            print(f"Error, invalid action: {action_str}")

        return result
