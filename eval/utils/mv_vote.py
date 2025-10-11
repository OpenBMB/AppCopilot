import numpy as np
import io
from PIL import Image
import requests
import aiohttp
import asyncio
from collections import Counter
from typing import List, Dict, Any

# Function to perform majority voting on a list of actions
def clear_mv(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对 CLEAR 操作进行多数投票处理"""
    vote_result = {}
    clear_thoughts = []
    clear_actions = []
    for act in actions:
        if "CLEAR" in act:
            thought = act.get("thought", "")
            clear_action = act["CLEAR"]
            clear_actions.append(clear_action)
            clear_thoughts.append(thought)
    if clear_thoughts:
        if_clear = Counter(clear_actions)
        most_common_clear, _ = if_clear.most_common(1)[0]
        most_common_thought = next(thought for thought, action in zip(clear_thoughts, clear_actions) if action == most_common_clear)
        vote_result["CLEAR"] = most_common_clear
        if most_common_clear:
            vote_result["CLEAR"] = True
        else:
            vote_result["CLEAR"] = False
        if "thought" not in vote_result:
            vote_result["thought"] = most_common_thought    
        return vote_result


def status_mv(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对 STATUS 操作进行多数投票处理"""
    vote_result = {}
    status_thoughts = []
    for act in actions:
        if "STATUS" in act:
            thought = act.get("thought", "")
            status_action = act["STATUS"]
            status_thoughts.append((thought, status_action))
    if status_thoughts:
        # 统计 STATUS 操作的出现次数
        status_counter = Counter([pt[1] for pt in status_thoughts])
        most_common_status, _ = status_counter.most_common(1)[0]
        # 获取对应的 thought
        thought = next(pt[0] for pt in status_thoughts if pt[1] == most_common_status)
        vote_result["STATUS"] = most_common_status
        vote_result["thought"] = thought
        return vote_result

def type_mv(actions: str | None) -> list[str]:
    """对 TYPE 操作进行多数投票处理"""
    vote_result = {}
    type_thoughts = []
    for act in actions:
        if "TYPE" in act:
            thought = act.get("thought", "")
            type_action = act["TYPE"]
            type_thoughts.append((thought, type_action))
    if type_thoughts:
        # 统计 TYPE 操作的出现次数
        type_counter = Counter([pt[1] for pt in type_thoughts])
        most_common_type, _ = type_counter.most_common(1)[0]
        # 获取对应的 thought
        thought = next(pt[0] for pt in type_thoughts if pt[1] == most_common_type)
        vote_result["TYPE"] = most_common_type
        vote_result["thought"] = thought
        return vote_result

def press_mv(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对 PRESS 操作进行多数投票处理"""
    vote_result = {}
    press_thoughts = []
    for act in actions:
        if "PRESS" in act:
            thought = act.get("thought", "")
            press_action = act["PRESS"]
            press_thoughts.append((thought, press_action))
    if press_thoughts:
        # 统计 PRESS 操作的出现次数
        press_counter = Counter([pt[1] for pt in press_thoughts])
        most_common_press, _ = press_counter.most_common(1)[0]
        # 获取对应的 thought
        thought = next(pt[0] for pt in press_thoughts if pt[1] == most_common_press)
        vote_result["PRESS"] = most_common_press
        vote_result["thought"] = thought
        return vote_result

def point_mv(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对 POINT 操作进行多数投票处理"""
    vote_result = {}
    point_thoughts = []
    for act in actions:
        if "POINT" in act:
            point = tuple(act["POINT"])
            to = act["to"] if "to" in act else None
            duration = act["duration"] if "duration" in act else None
            thought = act.get("thought", "")
            point_thoughts.append((point, thought, to, duration))
    if point_thoughts:
        # 分别统计x和y坐标的出现次数
        x_coords = [pt[0][0] for pt in point_thoughts]  # 获取所有x坐标
        y_coords = [pt[0][1] for pt in point_thoughts]  # 获取所有y坐标

        x_avg = sum(x_coords) / len(x_coords)
        y_avg = sum(y_coords) / len(y_coords)
        
        min_distance = float('inf')
        closest_thought = ""
        closest_to = None

        for point, thought, to, duration in point_thoughts:
            x, y = point
            distance = ((x - x_avg) ** 2 + (y - y_avg) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closet_point = point
                closest_thought = thought
                closest_to = to
                closest_duration = duration
        vote_result["POINT"] = list(closet_point)
        if closest_to is not None:
            vote_result["to"] = closest_to
        if closest_duration is not None:
            vote_result["duration"] = closest_duration
        vote_result["thought"] = closest_thought
        return vote_result

def duration_mv(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对 duration 操作进行多数投票处理"""
    vote_result = {}
    durations = []
    duration_thoughts = []
    for act in actions:
        if "duration" in act:
            duration = act["duration"]
            thought = act.get("thought", "")
            durations.append(duration) 
            duration_thoughts.append((duration, thought))
    if duration_thoughts:
        avg_duration = sum(durations) / len(durations)
        closest_duration = min(durations, key=lambda x: abs(x - avg_duration))
        thought = next(pt[1] for pt in duration_thoughts if pt[0] == closest_duration)
        vote_result["duration"] = int(closest_duration)
        vote_result["thought"] = thought
        return vote_result
    
def get_action_type(action: Dict[str, Any]) -> str:
    """提取单条 action 的主类型"""
    for key in ["POINT", "PRESS", "TYPE", "CLEAR","STATUS","duration"]:    
        if key in action:
            return key
    return "UNKNOWN"

def action_majority_vote(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    vote_result = {}
    if not actions:
        return {}

    # 1. 找出每个 action 的主类型
    types = [get_action_type(act) for act in actions if get_action_type(act) != "UNKNOWN"]
    if not types:
        return {}

    # 2. 对类型做多数投票
    type_counter = Counter(types)
    main_type, _ = type_counter.most_common(1)[0]

    # 3. 调用对应的 Majority Vote 函数
    dispatch_table = {
        "POINT": point_mv,
        "PRESS": press_mv,
        "TYPE": type_mv,
        "CLEAR": clear_mv,
        "STATUS": status_mv,
        "duration": duration_mv
    }
    return dispatch_table[main_type](actions)