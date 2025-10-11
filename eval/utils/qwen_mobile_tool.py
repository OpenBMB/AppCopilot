import os.path as osp
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from icecream import ic
import math
import argparse

from PIL import Image, ImageDraw, ImageFont, ImageColor
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from utils.evaluator import get_direction
from qwen_vl_utils import smart_resize
import json
from utils.action_utils import *
from utils.utils_qwen.agent_function_call import MobileUse
from IPython.display import display
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Only use the 4th GPU
args = type('Args', (), {})
import torch
torch.manual_seed(1)
def aitw_2_uitars(aitw_action: dict):
    """
    Convert AITW action to UITARS action format
    """
    ex_action_type = aitw_action['result_action_type']

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            # Click action
            click_y, click_x = lift_yx[0], lift_yx[1]
            click_x = int(click_x* 1000)
            click_y = int(click_y* 1000)
            return f"click(start_box=\'<|box_start|>({click_x},{click_y})<|box_end|>\')"
        else:
            # Swipe action
            touch_yx_new = {
                "x": touch_yx[1],
                "y": touch_yx[0]
            }
            lift_yx_new = {
                "x": lift_yx[1],
                "y": lift_yx[0]
            }
            direction = get_direction(touch_yx_new, lift_yx_new)
            return f"scroll(direction='{direction}')"
    
    elif ex_action_type == ActionType.PRESS_BACK:
        return f"press_back()"
    
    elif ex_action_type == ActionType.PRESS_HOME:
        return f"press_home()"
    
    elif ex_action_type == ActionType.PRESS_ENTER:
        return f"press_enter()"
    elif ex_action_type == ActionType.TYPE:
        return f"type(content='{aitw_action['result_action_text']}')"
    
    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        return f"finished()"
    
    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        return f"finished()"
    
    elif ex_action_type == ActionType.LONG_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        click_y, click_x = lift_yx[0], lift_yx[1]
        click_x = int(click_x* 1000)
        click_y = int(click_y* 1000)
        return f"long_press(start_box=\'<|box_start|>({click_x},{click_y})<|box_end|>\')"
    elif ex_action_type == ActionType.NO_ACTION:
        return f"wait()"
    elif ex_action_type == ActionType.OPEN_APP:
        return f"open(app_name='{aitw_action['result_action_app_name']}')"
    else:

        print('aitw_action:',aitw_action)
        raise NotImplementedError

    # Return formatted JSON string
    return json.dumps(qwen_action)
def aitz_2_qwen2_5(aitz_action: dict, resized_height: int, resized_width: int) -> str:
    """
    Convert AITZ action to Qwen2.5 action format
    
    Args:
        aitz_action (dict): AITZ format action, contains ACTION and ARGS
        resized_height (int): Screen height
        resized_width (int): Screen width
        
    Returns:
        str: Qwen2.5 format action string
    """
    aitz_action = json.loads(aitz_action)
    print(aitz_action)
    action_type = aitz_action["ACTION"]
    args = aitz_action["ARGS"]
    
    qwen_action = {}
    
    # Handle click action
    if action_type == "CLICK_ELEMENT":
        bbox = args["bbox"]
        # Calculate center point from bbox [x1, y1, x2, y2]
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        # Convert coordinates to screen coordinates
        center_x = int(center_x  * resized_width)
        center_y = int(center_y * resized_height)
        qwen_action = {
            "action": "click",
            "coordinate": [center_x, center_y]
        }
    
    # Handle swipe action
    elif action_type == "SCROLL":
        direction = args["direction"]
        # Set start and end points according to direction
        mid_x = resized_width // 2
        mid_y = resized_height // 2
        
        if direction == "up":
            # Swipe up from the middle of the screen (start at bottom, end at top)
            qwen_action = {
                "action": "swipe",
                "coordinate": [mid_x, mid_y + 300],
                "coordinate2": [mid_x, mid_y - 300]
            }
        elif direction == "down":
            # Swipe down from the middle of the screen
            qwen_action = {
                "action": "swipe",
                "coordinate": [mid_x, mid_y - 300],
                "coordinate2": [mid_x, mid_y + 300]
            }
        elif direction == "left":
            # Swipe left from the middle of the screen
            qwen_action = {
                "action": "swipe",
                "coordinate": [mid_x + 300, mid_y],
                "coordinate2": [mid_x - 300, mid_y]
            }
        elif direction == "right":
            # Swipe right from the middle of the screen
            qwen_action = {
                "action": "swipe",
                "coordinate": [mid_x - 300, mid_y],
                "coordinate2": [mid_x + 300, mid_y],
            }
    
    # Handle text input
    elif action_type == "INPUT":
        qwen_action = {
            "action": "type",
            "text": args["text"]
        }
    
    # Handle system buttons
    elif action_type == "PRESS BACK":
        qwen_action = {
            "action": "system_button",
            "button": "Back"
        }
    elif action_type == "PRESS HOME":
        qwen_action = {
            "action": "system_button",
            "button": "Home"
        }
    elif action_type == "PRESS ENTER":
        qwen_action = {
            "action": "system_button",
            "button": "Enter"
        }
    
    # Handle terminate action
    elif action_type == "STOP":
        qwen_action = {
            "action": "terminate",
            "status": args.get("task_status", "success")
        }
    
    # Build complete Qwen2.5 format output
    if qwen_action:
        return f'{{"name":"mobile_use","arguments":{json.dumps(qwen_action)}}}'
    else:
        return ""

def qwen2_5_2_aitz(output_text: str, resized_height: int, resized_width: int) -> str:
    """
    Convert Qwen2.5 output to AITZ output
    """
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
    qwen_action = action['arguments']
    action_name = qwen_action['action']
    # Handle click action, treat long_press as click because there is no corresponding action
    if action_name == "click" or action_name == "long_press":
        x, y = qwen_action["coordinate"]
        # Convert coordinates to bbox format [x1, y1, x2, y2]
        # Use 0.1 times the screen width and height as the click area
        # Normalize
        x = x/ resized_width
        y = y/ resized_height
        return {"ACTION": "CLICK_ELEMENT", "ARGS": {"bbox": [int((x-0.1)*999), int((y-0.1)*999), int((x+0.1)*999), int((y+0.1)*999)]}}
    
    # Handle swipe action
    elif action_name == "swipe":
        x1, y1 = qwen_action["coordinate"]
        x2, y2 = qwen_action["coordinate2"]
        # hack short swipe and should be click (copied from Qwen's evaluation logic)
        if np.linalg.norm([x2 - x1, y2 - y1]) <= 0.04:
            action_name = "click"
            x1=x1/ resized_width
            y1=y1/ resized_height
            x2=x2/ resized_width
            y2=y2/ resized_height
            return {"ACTION": "CLICK_ELEMENT", "ARGS": {"bbox": [int((x1-0.1)*999), int((y1-0.1)*999), int((x1+0.1)*999), int((y1+0.1)*999)]}}
        # Determine swipe direction based on start and end points
        if abs(x2 - x1) > abs(y2 - y1):  # Horizontal swipe
            direction = "right" if x2 > x1 else "left"
        else:  # Vertical swipe
            direction = "down" if y2 > y1 else "up"
        return {"ACTION": "SCROLL", "ARGS": {"direction": direction}}
    
    # Handle text input
    elif action_name == "type":
        return {"ACTION": "INPUT", "ARGS": {"text": qwen_action["text"]}}
    
    # Handle system buttons
    elif action_name == "system_button":
        button = qwen_action["button"]
        if button == "Back":
            return {"ACTION": "PRESS_BACK", "ARGS": {}}
        elif button == "Home":
            return {"ACTION": "PRESS_HOME", "ARGS": {}}
        elif button == "Enter":
            return {"ACTION": "PRESS_ENTER", "ARGS": {}}
    
    # Handle terminate action
    elif action_name == "terminate":
        return {"ACTION": "STOP", "ARGS": {"task_status": qwen_action["status"]}}
    
    # For other actions (such as key, wait, open, long_press, etc.), may need to ignore or handle specially
    # key, open, wait cannot find corresponding action, long_press is treated as click here
    return {"ACTION": "", "ARGS": {}}
    


model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model_path = "/home/test/test03/models/Qwen2.5-VL-7B-Instruct"
user_query_template = 'The user query:{user_request} (You have done the following operation on the current device):'
#model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
#processor = AutoProcessor.from_pretrained(model_path)
def get_qwen_response(user_query: str, screenshot: str, args=None, model_path: str = "/home/test/test03/models/Qwen2.5-VL-7B-Instruct") -> tuple:
    """
    Get response from Qwen model
    
    Args:
        user_query: User query text
        screenshot: Screenshot path
        model_path: Model path, default is the official model
        
    Returns:
        tuple: (response_text, status_code)
    """
    try:
        # Set default args
        if args is None:
            args = type('Args', (), {
                'greedy': False,
                'top_p': 0.01,
                'top_k': 1,
                'temperature': 0.01,
                'repetition_penalty': 1.0,
                'presence_penalty': 0.0,
                'out_seq_length': 1024,
                'seed': 1
            })
        
        # Build parameters using args
        generation_params = {
            'do_sample': not getattr(args, 'greedy', False),
            'top_p': getattr(args, 'top_p', 0.01),
            'top_k': getattr(args, 'top_k', 1),
            'temperature': getattr(args, 'temperature', 0.01),
            'repetition_penalty': getattr(args, 'repetition_penalty', 1.0),
            'presence_penalty': getattr(args, 'presence_penalty', 0.0),
            'max_new_tokens': getattr(args, 'out_seq_length', 1024),
            'seed': getattr(args, 'seed', 1)
        }
        
        # Handle image size
        dummy_image = Image.open(screenshot)
        #print(dummy_image.size)
        resized_height, resized_width = smart_resize(
            dummy_image.height,
            dummy_image.width,
            factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
            min_pixels=processor.image_processor.min_pixels,
            max_pixels=processor.image_processor.max_pixels,
        )
        #print(resized_height, resized_width)
        # Initialize mobile device interface
        mobile_use = MobileUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )

        # Build message
        message = NousFnCallPrompt.preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(text=user_query_template.format(user_request=user_query)),
                    ContentItem(image=f"file://{screenshot}")
                ]),
            ],
            functions=[mobile_use.function],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]
        
        # Handle input
        text = processor.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=True
        )
        print('text:',text)
        inputs = processor(
            text=[text], 
            images=[dummy_image], 
            padding=True, 
            return_tensors="pt"
        ).to('cuda')

        # If you need to set a random seed, set it before generate
        if hasattr(args, 'seed'):
            import torch
            torch.manual_seed(args.seed)

        # Call generate with correct parameters
        output_ids = model.generate(
            **inputs, 
            **generation_params
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        aitz_answer=qwen2_5_2_aitz(output_text,resized_height, resized_width)

        return json.dumps(aitz_answer), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return str(e), 500

user_query_template_history = '''The user query: {user_request}
Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.
After answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.
Task progress (You have done the following operation on the current device):
{history_actions}'''
def aitw_2_qwen2_5_action(aitw_action: dict, resized_height: int, resized_width: int) -> str:
    """
    Convert AITW action to Qwen2.5 action format
    """
    ex_action_type = aitw_action['result_action_type']
    qwen_action = {"name": "mobile_use", "arguments": {}}

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            # Click action
            click_y, click_x = lift_yx[0], lift_yx[1]
            click_x = int(click_x* resized_width)
            click_y = int(click_y* resized_height)
            qwen_action["arguments"] = {
                "action": "click",
                "coordinate": [click_x, click_y]
            }
        else:
            # Swipe action
            qwen_action["arguments"] = {
                "action": "swipe",
                "coordinate": [int(touch_yx[1]* resized_width), int(touch_yx[0]* resized_height)],  # Start point
                "coordinate2": [int(lift_yx[1]* resized_width), int(lift_yx[0]* resized_height)]    # End point
            }
    
    elif ex_action_type == ActionType.PRESS_BACK:
        button = "Back"
        qwen_action["arguments"] = {
            "action": "system_button",
            "button": button
        }
    
    elif ex_action_type == ActionType.PRESS_HOME:
        button = "Home"
        qwen_action["arguments"] = {
            "action": "system_button",
            "button": button
        }
    elif ex_action_type == ActionType.PRESS_ENTER:
        button = "Enter"
        qwen_action["arguments"] = {
            "action": "system_button",
            "button": button
        }
    elif ex_action_type == ActionType.TYPE:
        qwen_action["arguments"] = {
            "action": "type",
            "text": aitw_action['result_action_text']
        }
    
    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        qwen_action["arguments"] = {
            "action": "terminate",
            "status": "success"
        }
    
    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        qwen_action["arguments"] = {
            "action": "terminate",
            "status": "failure"
        }
    elif ex_action_type == ActionType.LONG_POINT:
        qwen_action["arguments"] = {
            "action": "long_press",
            "coordinate": [int(aitw_action['result_touch_yx'][1]* resized_width), int(aitw_action['result_touch_yx'][0]* resized_height)],
            "time": 2
        }
    elif ex_action_type == ActionType.NO_ACTION:
        qwen_action["arguments"] = {
            "action": "wait",
            "time": 2
        }
    else:
        print('aitw_action:',aitw_action)
        raise NotImplementedError

    # Return formatted JSON string
    return json.dumps(qwen_action)
def aitw_2_qwen2_5(aitw_action: dict, resized_height: int, resized_width: int) -> str:
    """
    Convert AITW action to Qwen2.5 prompt
    """
    aitw_action = json.loads(aitw_action)
    action=aitw_2_qwen2_5_action(aitw_action,resized_height, resized_width)
    thinking = f"<thinking>\n{aitw_action['coat_action_think']}\n</thinking>\n"
    action = f"<tool_call>\n{action}\n</tool_call>\n"
    result = f'<conclusion>\n"{aitw_action["coat_action_desc"]}"\n</conclusion>'
    return thinking + action + result
def get_qwen_response_history(user_query: str, screenshot: str, history_actions: list, model_path: str = "/home/test/test03/models/Qwen2.5-VL-7B-Instruct") -> tuple:
    """
    Get response from Qwen model
    
    Args:
        user_query: User query text
        screenshot: Screenshot path
        history_actions: History actions
        model_path: Model path, default is the official model
        
    Returns:
        tuple: (response_text, status_code)
    """
    #try:
    print('history_actions:',history_actions)
    # Handle image size
    dummy_image = Image.open(screenshot)
    #print(dummy_image.size)
    resized_height, resized_width = smart_resize(
        dummy_image.height,
        dummy_image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    #print('max_pixels:',processor.image_processor.max_pixels)
    # 12845056, exceeds 4096*3112 (4k resolution), should be enough for mobile resolution
    # Convert history_actions to Qwen2.5 format
    if history_actions:
        history_actions_str = "".join([f"Step {i+1}: {aitw_2_qwen2_5(action,resized_height, resized_width).replace('<tool_call>','').replace('</tool_call>','').strip()}; " for i, action in enumerate(history_actions)])
    else:
        history_actions_str = ""

    #print(resized_height, resized_width)
    # Initialize mobile device interface
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
    )

    # Build message
    message = NousFnCallPrompt.preprocess_fncall_messages(
        messages=[
            Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
            Message(role="user", content=[
                ContentItem(text=user_query_template_history.format(user_request=user_query,history_actions=history_actions_str)),
                ContentItem(image=f"file://{screenshot}")
            ]),
        ],
        functions=[mobile_use.function],
        lang=None,
    )
    message = [msg.model_dump() for msg in message]
    
    # Handle input
    text = processor.apply_chat_template(
        message, 
        tokenize=False, 
        add_generation_prompt=True
    )
    print('text:',text)
    inputs = processor(
        text=[text], 
        images=[dummy_image], 
        padding=True, 
        return_tensors="pt"
    ).to('cuda')

    # Modify generation_params definition
    generation_params = {
        # Replace 'greedy' with 'do_sample'
        'do_sample': not getattr(args, 'greedy', False),
        'top_p': getattr(args, 'top_p', 0.01),
        'top_k': getattr(args, 'top_k', 1),
        'temperature': getattr(args, 'temperature', 0.01),
        'repetition_penalty': getattr(args, 'repetition_penalty', 1.0),
        # 'presence_penalty' is not supported, can be removed
        # Replace 'out_seq_length' with 'max_new_tokens'
        # 'seed' is not directly supported, needs to be set externally
    }

    # If you need to set a random seed, set it before generate


    # Call generate with correct parameters
    output_ids = model.generate(
        **inputs, 
        max_new_tokens=getattr(args, 'out_seq_length', 2048),
        **generation_params
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )[0]
    aitz_answer=qwen2_5_2_aitz(output_text,resized_height, resized_width)

    return json.dumps(aitz_answer), 200
        
    #except Exception as e:
    #    print(f"Error: {str(e)}")
    #    return str(e), 500
# Example usage
if __name__ == "__main__":
    user_query = 'Open the file manager app and view the au_uu_SzH3yR2.mp3 file in MUSIC Folder'
    screenshot = "/home/test/test03/fuyikun/CoAT/data-example/GOOGLE_APPS-523638528775825151/GOOGLE_APPS-523638528775825151_0.png"
    response, state = get_qwen_response(user_query, screenshot)
    print(f"Response: {response}")
    print(f"State: {state}")
