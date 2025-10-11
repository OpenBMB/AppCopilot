import base64
import time
from typing import Any, Optional
import numpy as np
from PIL import Image
import json
from typing import Optional, Any, List, Dict, Tuple
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
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from qwen_agent.tools.mobile_use import MobileUse
from qwen_agent.tools.mobile_use import ActionType, is_tap_action
from qwen_agent.tools.mobile_use import aitw_2_qwen2_5_action


class QwenVLWrapper(LlmWrapper, MultimodalLlmWrapper):
    RETRY_WAITING_SECONDS = 20
    user_query_template = """The user query:  {user_request} 
    Task progress (You have done the following operation on the current device): {history_actions}"""

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
        # if input_string == "":
        # raise ValueError("Error, empty output.")
        try:
            json_obj = json.loads(input_string)
            # validate JSON data against Schema
            jsonschema.validate(json_obj, ACTION_THOUGHT_SCHEMA)
            return json_obj
        except json.JSONDecodeError as e:
            print("Error, JSON is NOT valid.", input_string, "over")
            return input_string
        except Exception as e:
            print(
                "Error, JSON is NOT valid according to the schema.",
                input_string,
                "over",
            )
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
        user_content, messages, image_width, image_height = self._prepare_messages(
            text_prompt, images, []
        )
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
                response = requests.post(
                    END_POINT,
                    headers=headers,
                    json=payload,
                )
                if response.ok and "choices" in response.json():
                    assistant_msg = response.json()["choices"][0]["message"]
                    assistant_text = assistant_msg["content"]

                    action = self.qwen2_5_2_minicpm(
                        assistant_text, image_height, image_width
                    )
                    # action = self.extract_and_validate_json(assistant_text)

                    # -------- 写回历史 -------
                    self._push_history("user", user_content[0]["text"])
                    self._push_history("assistant", assistant_msg["content"])

                    return assistant_text, None, response, action
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
        return ERROR_CALLING_LLM, None, None

    def _prepare_messages(
        self, text_prompt: str, images: List[np.ndarray], history_actions: List[Dict]
    ) -> Tuple[List[Dict], int, int]:
        """准备发送给API的消息"""
        if not images:
            raise ValueError("At least one image is required for multimodal input")
        image = images[0]
        img_pil = Image.fromarray(image)
        image_width, image_height = img_pil.size
        history_actions_str = ""
        if history_actions:
            # from utils.qwen_mobile_tool import aitw_2_qwen2_5_action
            history_actions_str = "".join(
                [
                    f"Step {i+1}: {aitw_2_qwen2_5_action(action, image_height, image_width).strip()}; "
                    for i, action in enumerate(history_actions)
                ]
            )
        mobile_use = MobileUse(
            cfg={"display_width_px": image_width, "display_height_px": image_height}
        )
        prompt = NousFnCallPrompt()
        message = prompt.preprocess_fncall_messages(
            messages=[
                Message(
                    role="system",
                    content=[ContentItem(text="You are a helpful assistant.")],
                ),
                Message(
                    role="user",
                    content=[
                        ContentItem(
                            text=user_query_template_low.format(
                                user_request=text_prompt,
                                history_actions=history_actions_str,
                            )
                        ),
                        ContentItem(
                            image=f"data:image/jpeg;base64,{self.encode_image(image)}"
                        ),
                    ],
                ),
            ],
            functions=[mobile_use.function],
            lang=None,
        )

        messages = [msg.model_dump() for msg in message]
        messages = [{"role": "system", "content": messages[0]["content"][1]["text"]}]
        if self.use_history and self.history:
            messages.extend(self.history)

        user_content = [
            {
                "type": "text",
                "text": self.user_query_template.format(
                    user_request=text_prompt, history_actions=history_actions_str
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.encode_image(image)}"
                },
            },
        ]
        messages.append({"role": "user", "content": user_content})
        return user_content, messages, image_width, image_height

    def qwen2_5_2_minicpm(
        self, output_text: str, resized_height: int, resized_width: int
    ) -> dict:
        """
        Convert Qwen2.5's output to minicpm's output
        """
        action = json.loads(
            output_text.split("<tool_call>\n")[1].split("\n</tool_call>")[0]
        )
        qwen_action = action["arguments"]
        action_name = qwen_action["action"]
        # handle click action, long_press is directly processed as click because there is no corresponding action
        if action_name == "click":
            x, y = qwen_action["coordinate"]

            # normalize
            x = x / resized_width * 1000
            y = y / resized_height * 1000
            return {"POINT": [int(x), int(y)]}
        elif action_name == "long_press":
            x, y = qwen_action["coordinate"]
            x = x / resized_width * 1000
            y = y / resized_height * 1000
            time = qwen_action["time"]
            # convert time to milliseconds
            time = time * 1000
            return {"POINT": [int(x), int(y)], "duration": time}

        # handle swipe action
        elif action_name == "swipe":
            x1, y1 = qwen_action["coordinate"]
            x2, y2 = qwen_action["coordinate2"]
            x1 = x1 / resized_width * 1000
            y1 = y1 / resized_height * 1000
            x2 = x2 / resized_width * 1000
            y2 = y2 / resized_height * 1000
            # determine swipe direction based on start and end points
            if abs(x2 - x1) > abs(y2 - y1):  # horizontal swipe
                direction = "right" if x2 > x1 else "left"
            else:  # vertical swipe
                direction = "down" if y2 > y1 else "up"
            return {"POINT": [int(x1), int(y1)], "to": direction}

        # handle input text
        elif action_name == "type":
            return {"TYPE": qwen_action["text"]}

        # handle system button
        elif action_name == "system_button":
            button = qwen_action["button"]
            if button == "Back":
                return {"PRESS": "BACK"}
            elif button == "Home":
                return {"PRESS": "HOME"}
            elif button == "Enter":
                return {"PRESS": "ENTER"}

        # handle terminate action
        elif action_name == "terminate":
            return {"STATUS": "finish"}
        elif action_name == "wait":
            # convert time to milliseconds
            time = qwen_action["time"]
            time = time * 1000
            return {"duration": time}

        # for other actions (such as key,open, etc.), they may need to be ignored or specially processed
        # key wait cannot find corresponding action
        return {}

    def aitw_2_qwen2_5_action(
        aitw_action: dict, resized_height: int, resized_width: int
    ) -> str:
        """
        Convert AITW action to Qwen2.5 action format
        """
        ex_action_type = aitw_action["result_action_type"]
        qwen_action = {"name": "mobile_use", "arguments": {}}

        if ex_action_type == ActionType.DUAL_POINT:
            lift_yx = json.loads(aitw_action["result_lift_yx"])
            touch_yx = json.loads(aitw_action["result_touch_yx"])
            if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
                # Click action
                click_y, click_x = lift_yx[0], lift_yx[1]
                click_x = int(click_x * resized_width)
                click_y = int(click_y * resized_height)
                qwen_action["arguments"] = {
                    "action": "click",
                    "coordinate": [click_x, click_y],
                }
            else:
                # Swipe action
                qwen_action["arguments"] = {
                    "action": "swipe",
                    "coordinate": [
                        int(touch_yx[1] * resized_width),
                        int(touch_yx[0] * resized_height),
                    ],  # Start point
                    "coordinate2": [
                        int(lift_yx[1] * resized_width),
                        int(lift_yx[0] * resized_height),
                    ],  # End point
                }

        elif ex_action_type == ActionType.PRESS_BACK:
            button = "Back"
            qwen_action["arguments"] = {"action": "system_button", "button": button}

        elif ex_action_type == ActionType.PRESS_HOME:
            button = "Home"
            qwen_action["arguments"] = {"action": "system_button", "button": button}
        elif ex_action_type == ActionType.PRESS_ENTER:
            button = "Enter"
            qwen_action["arguments"] = {"action": "system_button", "button": button}
        elif ex_action_type == ActionType.TYPE:
            qwen_action["arguments"] = {
                "action": "type",
                "text": aitw_action["result_action_text"],
            }

        elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
            qwen_action["arguments"] = {"action": "terminate", "status": "success"}

        elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
            qwen_action["arguments"] = {"action": "terminate", "status": "failure"}
        elif ex_action_type == ActionType.LONG_POINT:
            qwen_action["arguments"] = {
                "action": "long_press",
                "coordinate": [
                    int(aitw_action["result_touch_yx"][1] * resized_width),
                    int(aitw_action["result_touch_yx"][0] * resized_height),
                ],
                "time": 2,
            }
        elif ex_action_type == ActionType.NO_ACTION:
            qwen_action["arguments"] = {"action": "wait", "time": 2}
        else:
            print("aitw_action:", aitw_action)
            raise NotImplementedError

        # Return formatted JSON string
        return json.dumps(qwen_action)


from typing import Union, Tuple, List

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool("mobile_use")
class MobileUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "click",
                    "long_press",
                    "swipe",
                    "type",
                    "system_button",
                    "open",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
                "type": "array",
            },
            "coordinate2": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=key`, `action=type`, and `action=open`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                "type": "number",
            },
            "button": {
                "description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`",
                "enum": [
                    "Back",
                    "Home",
                    "Menu",
                    "Enter",
                ],
                "type": "string",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action == "key":
            return self._key(params["text"])
        elif action == "click":
            return self._click(coordinate=params["coordinate"])
        elif action == "long_press":
            return self._long_press(
                coordinate=params["coordinate"], time=params["time"]
            )
        elif action == "swipe":
            return self._swipe(
                coordinate=params["coordinate"], coordinate2=params["coordinate2"]
            )
        elif action == "type":
            return self._type(params["text"])
        elif action == "system_button":
            return self._system_button(params["button"])
        elif action == "open":
            return self._open(params["text"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Unknown action: {action}")

    def _key(self, text: str):
        raise NotImplementedError()

    def _click(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _long_press(self, coordinate: Tuple[int, int], time: int):
        raise NotImplementedError()

    def _swipe(self, coordinate: Tuple[int, int], coordinate2: Tuple[int, int]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()

    def _system_button(self, button: str):
        raise NotImplementedError()

    def _open(self, text: str):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()
