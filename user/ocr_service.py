import threading
import time
import json
import os
import logging
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any
from PIL.Image import Image
import paddleocr
from paddleocr import PaddleOCR

class OCRService:
    def __init__(self):
        self.thread = None
        self.output_dir = "./user/ocr_output"
        self.ocr_data = []
        self.is_running = False
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_dir = "ocr_model\PP-OCRv5_server_det",
            text_recognition_model_dir = "ocr_model\PP-OCRv5_server_rec"
            )
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化JSON文件
        self.ensure_json_file()
        
    def ensure_json_file(self):
        """确保JSON文件存在且格式正确"""
        if not os.path.exists("./user/information.json"):
            with open("./user/information.json", "w", encoding="utf-8") as f:
                json.dump({"ocr_records": []}, f)
    
    def start(self, device_, interval=0.5):
        """启动OCR监控服务"""
        if self.is_running:
            logging.warning("OCRService is already running")
            return
            
        self.is_running = True
        
        # 启动监控线程
        self.thread = threading.Thread(
            target=self._monitor_loop,
            args=(device_, interval),
            daemon=True
        )
        self.thread.start()
        logging.info(f"OCRService started | Interval: {interval}s")
    
    def stop(self):
        """停止OCR服务并保存所有数据"""
        if not self.is_running:
            return
            
        time.sleep(10) # 让ocr再运行10秒

        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=3.0)
        
        self._save_data()
        logging.info("OCRService stopped")
    
    def _monitor_loop(self, device_, interval):
        """监控循环，定期执行OCR"""
        # 获取当前设备实例
        device = device_
        if not device:
            logging.warning("Device not available. Skipping OCR cycle.")
            time.sleep(interval)
        while self.is_running:
            try:
                # 执行OCR处理
                self.process_ocr(device)
                # 定期保存数据
                if len(self.ocr_data) >= 3:
                    self._save_data()
                
            except Exception as e:
                logging.error(f"OCR monitoring failed: {str(e)}")
            
            time.sleep(2)
    
    def process_ocr(self, device) -> Optional[List[Dict[str, Any]]]:
        """执行完整的OCR处理流程"""
        try:
            # 截图
            screenshot = device.screenshot(1120)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.output_dir, f"temp_{timestamp}.png")
            
            # 保存截图文件
            screenshot.save(file_path)
            
            # OCR处理
            ocr_results = ocr_process(self.ocr, file_path)

            # 创建记录
            record = {
                "timestamp": datetime.now().isoformat(),
                "image": file_path,
                "ocr_results": ocr_results
            }
            
            # 保存记录
            self.ocr_data.append(record)
            
            return ocr_results
        except Exception as e:
            logging.error(f"OCR processing failed: {str(e)}")
            return None

    def _save_data(self):
        """将缓存数据保存到JSON文件"""
        if not self.ocr_data:
            return
            
        data_to_save = self.ocr_data.copy()
        self.ocr_data = [] # 清空缓存
        
        try:
            # 读取现有数据
            with open("./user/information.json", "r", encoding="utf-8") as f:
                all_data = json.load(f)
            
            # 添加新数据
            all_data["ocr_records"].extend(data_to_save)
            
            # 写回文件
            with open("./user/information.json", "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save OCR data: {str(e)}")
        


def ocr_process(ocr, image_path: str) -> List[Dict[str, Any]]:
    """
    处理OCR识别结果，将其转换为字典列表格式
    
    Args:
        image_path: 图片文件路径或URL
        
    Returns:
        识别结果列表，每个元素是一个包含文本和位置信息的字典
    """
    try:
        # 使用PaddleOCR进行OCR识别
        result = ocr.predict(input=image_path)

        # 将结果转换为字典列表格式
        ocr_results = []
        for res in result:
            for i in range(len(res["rec_texts"])):
                item = {
                    "text": res["rec_texts"][i],
                    "box": [ 
                        [float(p[0]), float(p[1])] 
                        for p in res["rec_polys"][i]
                    ]
                }
                ocr_results.append(item)
        return ocr_results
        
    except Exception as e:
        logging.error(f"OCR processing failed: {str(e)}")
        return []

