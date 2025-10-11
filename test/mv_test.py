from collections import Counter
from typing import List, Dict, Any
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrappers.utils import action_majority_vote

def test_action_majority_vote_with_points():
    actions = [
        {"POINT": [100, 200], "thought": "点击按钮1"},
        {"POINT": [102, 198], "thought": "点击按钮2"},
        {"POINT": [98, 201], "thought": "点击按钮3"}
    ]
    
    result = action_majority_vote(actions)
    
    assert "POINT" in result
    assert len(result["POINT"]) == 2
    assert abs(result["POINT"][0] - 100) <= 2  # x坐标约为100
    assert abs(result["POINT"][1] - 200) <= 2  # y坐标约为200
    assert "thought" in result

def test_action_majority_vote_with_to_coords():
    actions = [
        {
            "POINT": [100, 200],
            "thought": "拖动到新位置",
            "to": [300, 400]
        },
        {
            "POINT": [101, 201],
            "thought": "移动到位置",
            "to": [298, 402]
        }
    ]
    
    result = action_majority_vote(actions)
    
    assert "POINT" in result
    assert "to" in result
    assert len(result["to"]) == 2
    assert abs(result["to"][0] - 299) <= 2  # to的x坐标约为299
    assert abs(result["to"][1] - 401) <= 2  # to的y坐标约为401

def test_action_majority_vote_with_directions():
    actions = [
        {
            "POINT": [100, 200],
            "thought": "向下滚动",
            "to": "down"
        },
        {
            "POINT": [101, 201],
            "thought": "向下滚动页面",
            "to": "down"
        },
        {
            "POINT": [99, 199],
            "thought": "向上滚动",
            "to": "up"
        }
    ]
    
    result = action_majority_vote(actions)
    
    assert "POINT" in result
    assert "to" in result
    assert result["to"] == "down"  # 最常见的方向是down

def test_action_majority_vote_with_press():
    actions = [
        {"PRESS": "BACK", "thought": "返回上一页"},
        {"PRESS": "BACK", "thought": "回到上一级"},
        {"PRESS": "HOME", "thought": "回到主屏幕"}
    ]
    
    result = action_majority_vote(actions)
    
    assert "PRESS" in result
    assert result["PRESS"] == "BACK"  # 最常见的操作是返回

def test_action_majority_vote_with_type():
    actions = [
        {"TYPE": "Hello", "thought": "输入问候语"},
        {"TYPE": "Hello", "thought": "发送问候"},
        {"TYPE": "Hi", "thought": "打招呼"}
    ]
    
    result = action_majority_vote(actions)
    
    assert "TYPE" in result
    assert result["TYPE"] == "Hello"  # 最常见的输入是"Hello"

def test_action_majority_vote_with_clear_true():
    actions = [
        {"CLEAR": True, "thought": "清除输入框"},
        {"CLEAR": False, "thought": "清空内容"},
        {"CLEAR": False, "thought": "不清除"}
    ]
    
    result = action_majority_vote(actions)
    
    assert "CLEAR" in result
    assert result["CLEAR"] is False  # 最常见的操作是清除

def test_action_majority_vote_with_clear_false():
    actions = [
        {"CLEAR": True, "thought": "清除输入框"},
        {"CLEAR": True, "thought": "清空内容"},
        {"CLEAR": False, "thought": "不清除"}
    ]
    
    result = action_majority_vote(actions)
    
    assert "CLEAR" in result
    assert result["CLEAR"] is True  # 最常见的操作是清除

def test_action_majority_vote_with_mixed_actions():
    actions = [
        {"POINT": [100, 200], "to": "down","thought": "点击按钮1"},
        {"POINT": [102, 198], "to": "down","thought": "点击按钮2"},
        {"POINT": [110, 199], "to": [300, 400], "thought": "拖动到新位置"},
        {"POINT": [101, 200], "to": "down", "thought": "向下滚动"}
    ]
    
    result = action_majority_vote(actions)
    
    assert "POINT" in result
    assert len(result["POINT"]) == 2
    assert abs(result["POINT"][0] - 101) <= 2  # x坐标约为101
    assert abs(result["POINT"][1] - 199) <= 2  # y坐标约为199
    assert "to" in result
    assert result["to"] == "down"  # 最常见的方向是down

def test_action_majority_vote_with_durations():
    actions = [
        {"duration": 500, "thought": "点击按钮1"},
        {"duration": 600, "thought": "点击按钮2"},
        {"duration": 600, "thought": "点击按钮2"},
        {"duration": 550, "thought": "拖动到新位置"},
        {"duration": 530, "thought": "拖动到新位置"}
    ]
    
    result = action_majority_vote(actions)
    
    assert result["duration"] == 550  
    
def test_action_majority_vote_with_status():    
    actions = [
        {"STATUS": "finish", "thought": "任务完成"},
        {"STATUS": "continue", "thought": "继续进行"},
        {"STATUS": "continue", "thought": "继续进行"},
        {"STATUS": "need_feedback", "thought": "已完成"}
    ]
    
    result = action_majority_vote(actions)
    
    assert "STATUS" in result
    assert result["STATUS"] == "continue"  

def test_action_majority_vote_empty_actions():
    actions = []
    result = action_majority_vote(actions)
    assert result == {}

def test_action_majority_vote_no_points():
    actions = [
        {"to": "up"},
        {"to": "down"}
    ]
    result = action_majority_vote(actions)
    assert "POINT" not in result

if __name__ == "__main__":
    pytest.main(["-v", __file__])
