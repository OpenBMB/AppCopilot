import json


def generate_function_call(function_name: str, params: dict) -> str:
    """
    生成符合格式的函数调用指令（设备1调用设备2时使用）
    """
    call = {"action": "function_call", "function": function_name, "parameters": params}
    return json.dumps(call, ensure_ascii=False)


def parse_function_call(call_str: str) -> dict:
    """
    解析设备1发送的函数调用指令（设备2接收时使用）
    """
    try:
        return json.loads(call_str)
    except json.JSONDecodeError:

        return {"error": "invalid format"}


def generate_function_response(result: str, status: str = "success") -> str:
    """
    生成函数调用的响应结果（设备2返回结果给设备1时使用）
    """
    response = {"action": "function_response", "status": status, "result": result}
    return json.dumps(response, ensure_ascii=False)


def parse_function_response(response_str: str) -> dict:
    """
    解析设备2返回的函数响应结果（设备1接收时使用）
    """
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:

        return {"error": "invalid response"}
