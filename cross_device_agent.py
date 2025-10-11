import sys
import os
import time
import argparse
sys.path.append(os.getcwd())

from run_agent import GUITaskExecutor
from cross_device.socket_utils import DualSocket
from cross_device.instruction_mapper import InstructionMapper
from cross_device.function_call_utils import (
    generate_function_call,
    parse_function_call,
    generate_function_response,
    parse_function_response,
)
from adb_utils import setup_device, list_connected_devices, change_ui_settings
from PIL import Image
from cross_device.vision_extractor import VisionExtractor
from user.ocr_service import OCRService


class CrossDeviceCoordinator:
    """跨设备协同控制器

    用于协调两台安卓设备通过Socket通信完成用户指令，
    实现设备间的任务分配、函数调用和结果交互。
    """

    def __init__(
        self,
        device1_serial: str,
        device1_port: int,
        device2_serial: str,
        device2_port: int,
    ):
        self.device1 = setup_device(device1_serial)
        self.device2 = setup_device(device2_serial)
        self.socket1 = DualSocket(
            listen_port=device1_port,
            peer_port=device2_port,
            device_name=f"设备1({self.device1.serial})",
        )
        self.socket2 = DualSocket(
            listen_port=device2_port,
            peer_port=device1_port,
            device_name=f"设备2({self.device2.serial})",
        )
        self.mapper = InstructionMapper()
        self.ocr_service = OCRService()
        self.common_run_params = {
            "ocr_service": self.ocr_service,
            "enable_audio": False,
            "enable_vision_parser": False,
            "return_result": False,
        }
        self.enable_experience = False

    def _device2_function_handler(self, call_str: str) -> str:
        """设备2的函数调用处理函数

        接收设备1的调用指令，执行对应操作（如提取关键词），返回处理结果
        """
        call = parse_function_call(call_str)
        if "error" in call:
            return generate_function_response("无效调用格式", "error")

        if call["function"] == "extract_keyword":
            task = call["parameters"]["task"]
            task_executor = GUITaskExecutor(serial=self.device2.serial, **self.common_run_params)
            task_executor.run_task(query=task, enable_experience=self.enable_experience)
            screenshot = self.device2.screenshot()
            keyword = self._extract_info_from_screenshot(
                screenshot, call["parameters"]["extract_prompt"]
            )
            if keyword:
                return generate_function_response(keyword)
            return generate_function_response("提取关键词失败", "error")

    def _extract_info_from_screenshot(
        self, screenshot: Image.Image, prompt: str
    ) -> str:
        """从截图中提取信息

        使用视觉语言模型，根据提示词从截图中提取所需内容
        """
        from cross_device.vision_extractor import VisionExtractor

        extractor = VisionExtractor(model_name="gpt-4o")
        import numpy as np

        img_array = np.array(screenshot)
        result = extractor.query_image(img_array, prompt)
        return self._postprocess_extraction(result)

    def _postprocess_extraction(self, text: str) -> str:
        """提取结果后处理

        对模型返回的提取结果进行简单清洗（如正则匹配提取关键部分）
        """
        import re

        match = re.search(r"标题[:：]\s*(.+)", text)
        return match.group(1).strip() if match else text.strip()

    def _agent1_call_agent2(self, function_name: str, params: dict) -> str:
        """设备1调用设备2的函数

        生成函数调用指令，通过Socket发送给设备2，接收并返回处理结果
        """
        call_str = generate_function_call(function_name, params)
        self.socket2.start_server(handler=self._device2_function_handler)
        response_str = self.socket1.send_message(call_str)

        response = parse_function_response(response_str)
        if response["status"] == "success":
            return response["result"]
        else:
            return ""

    def start_workflow(self, user_instruction: str) -> None:
        """启动跨设备工作流程

        解析用户指令，分配任务给设备1和设备2，协调执行并处理结果
        """
        try:
            task_info = self.mapper.split_task(user_instruction)
            device1_task = task_info["device1_task"]
            function_call = task_info["function_call"]

            keyword = self._agent1_call_agent2(
                function_name=function_call["name"], params=function_call["parameters"]
            )
            if not keyword:
                print("获取关键词失败，终止流程")
                return

            print(f"从设备2获取到关键词: {keyword}")
            self._run_device1(device1_task, keyword)

        except Exception as e:
            print(f"流程执行出错：{e}")
        finally:
            self.socket1.stop()
            self.socket2.stop()

    def _run_device1(self, task: str, keyword: str) -> None:
        """执行设备1的任务

        将任务中的"等待关键词"替换为实际关键词，然后执行任务
        """
        new_task = task.replace("等待设备2发送的关键词", f"输入关键词：{keyword}")
        print(f"在设备1上执行: {new_task}")
        task_executor = GUITaskExecutor(serial=self.device1.serial, **self.common_run_params)
        task_executor.run_task(query=new_task, enable_experience=self.enable_experience)
        time.sleep(2)


def main():
    """命令行入口函数

    解析命令行参数，初始化设备协调器，启动工作流程
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="跨设备协同控制工具")
    parser.add_argument(
        "--device1-serial", type=str, default=None, help="设备1的ADB序列号（可选）"
    )
    parser.add_argument(
        "--device1-port", type=int, default=11001, help="设备1的通信端口（默认11001）"
    )
    parser.add_argument(
        "--device2-serial", type=str, default=None, help="设备2的ADB序列号（可选）"
    )
    parser.add_argument(
        "--device2-port", type=int, default=11002, help="设备2的通信端口（默认11002）"
    )
    parser.add_argument("--task", type=str, help="跨设备任务指令")
    parser.add_argument(
        "--list-devices", action="store_true", help="列出所有已连接的ADB设备"
    )
    args = parser.parse_args()

    if args.list_devices:
        devices = list_connected_devices()
        if not devices:
            print("没有找到已连接的ADB设备")
        else:
            print("已连接的ADB设备：")
            for i, device in enumerate(devices):
                print(f"{i+1}. {device}")
        return

    if not args.task:
        parser.error("请提供 --task 参数指定跨设备任务指令")

    devices = list_connected_devices()
    if not devices:
        raise RuntimeError("没有找到已连接的 ADB 设备")

    if args.device1_serial is None:
        if len(devices) >= 1:
            args.device1_serial = devices[0]
        else:
            raise RuntimeError("至少需要1台设备")

    if args.device2_serial is None:
        if len(devices) >= 2:
            args.device2_serial = devices[1]
        else:
            raise RuntimeError("至少需要2台设备")

    try:
        # 打开UI设置（如无障碍模式等）
        change_ui_settings(mode="open")
        coordinator = CrossDeviceCoordinator(
            device1_serial=args.device1_serial,
            device1_port=args.device1_port,
            device2_serial=args.device2_serial,
            device2_port=args.device2_port,
        )
        coordinator.start_workflow(args.task)
    finally:
        change_ui_settings(mode="close")


if __name__ == "__main__":
    main()
