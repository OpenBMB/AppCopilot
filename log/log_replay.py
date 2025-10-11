import os
import json
import logging
import time
import builtins
from PIL import Image
from pathlib import Path
from adb_utils import setup_device 

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("LogReplayer")

LOG_DIR = "xxxxx" 

def replay_log(log_dir_path):
    """重放日志记录的任务"""
    try:
        #加载日志文件
        log_dir_path = log_dir_path.replace("\\", "/")
        log_file = Path(log_dir_path) / "action_log.json"
        if not log_file.exists():
            logger.error(f"Log file not found: {log_file}")
            return False
        
        with open(log_file, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        
        #显示任务信息
        metadata = log_data["metadata"]
        logger.info(f"Replaying task: {metadata['query']}")
        logger.info(f"Start time: {metadata['start_time']}")
        logger.info(f"Steps to replay: {len(log_data['steps'])}")
        
        #准备设备
        device = setup_device()
        
        #逐步骤重放
        for step in log_data["steps"]:
            logger.info(f"Replaying step {step['step']}: Action={step['action']}")
            
            #执行原始操作
            device.step(step["action"])
            time.sleep(1.5)

            # output some information
            print(step["action"])
            
            #显示原始响应
            if "response" in step:
                logger.debug(f"Original response: {step['response']}")
        
        logger.info(f"Task completed successfully! Result: {metadata.get('result', '')}")
        return True
    
    except Exception as e:
        logger.error(f"Replay failed: {str(e)}")
        return False

if __name__ == "__main__":
    replay_log(LOG_DIR)