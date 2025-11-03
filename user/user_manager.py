import os
import json
import uuid
import re
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path

class UserManager:
    def __init__(self, storage_dir: str = "users_info"): 
        """
        初始化用户管理器
        
        Args:
            storage_dir: 用户数据存储目录
        """
        #获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.storage_dir = os.path.join(current_dir, storage_dir)
        self.current_user_id = None
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def get_user_dir(self, user_id: str) -> str:
        """获取用户目录（不含文件名）"""
        if not self._is_valid_user_id(user_id):
            raise ValueError("Invalid user_id")
        return os.path.join(self.storage_dir, user_id)
        
    def _get_user_filepath(self, user_id: str) -> str:
        """获取用户文件路径"""
        return str(self._resolve_user_file(user_id))

    def _is_valid_user_id(self, user_id: str) -> bool:
        if not isinstance(user_id, str) or not user_id:
            return False
        # Allow UUIDs and simple names: letters, numbers, - and _
        return re.fullmatch(r"[A-Za-z0-9_-]+", user_id) is not None

    def _resolve_user_file(self, user_id: str) -> Path:
        if not self._is_valid_user_id(user_id):
            raise ValueError(f"Invalid user_id: {user_id}")

        base = Path(self.storage_dir).resolve()
        user_dir = base.joinpath(user_id)
        filepath = user_dir.joinpath(f"{user_id}.json")

        # Resolve the final path and ensure it's within base
        try:
            resolved = filepath.resolve()
        except Exception as e:
            raise ValueError(f"Could not resolve path for user_id {user_id}: {e}") from e

        try:
            resolved.relative_to(base)
        except Exception:
            raise ValueError(f"Path traversal detected for user_id: {user_id}")

        return resolved
    
    
    def _load_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        加载单个用户数据
        
        Returns:
            用户数据字典，如果文件不存在返回None
        """
        filepath = self._get_user_filepath(user_id)
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载用户 {user_id} 失败: {str(e)}")
            return None
    
    def _save_user_data(self, user_id: str, data: Dict[str, Any]) -> bool:
        """
        保存用户数据
        
        Returns:
            是否保存成功
         """
        filepath = self._get_user_filepath(user_id)
        os.makedirs(os.path.dirname(filepath), exist_ok=True) 
        try:
            # 直接写入目标文件（非原子操作但更可靠）
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
            return True
        except Exception as e:
            logging.error(f"保存用户 {user_id} 失败: {str(e)}")
            return False

    def create_user(self, user_info: Dict[str, Any]) -> str:
        """
        创建新用户
        
        Args:
            user_info: 用户信息字典，至少包含username字段
            
        Returns:
            新创建的用户ID
        """
        user_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        user_data = {
            "user_id": user_id,
            "created_at": timestamp,
            "last_active": timestamp,
            "preferences": {},
            "history": [],
            **user_info  # 合并用户信息
        }
        
        if self._save_user_data(user_id, user_data):
            self.current_user_id = user_id
            return user_id
        raise RuntimeError("创建用户失败")
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取指定用户信息"""
        return self._load_user_data(user_id)
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新用户信息
        
        Args:
            user_id: 要更新的用户ID
            updates: 要更新的字段字典
            
        Returns:
            是否更新成功
        """
        user_data = self._load_user_data(user_id)
        if not user_data:
            return False
        
        user_data.update(updates)
        user_data["last_active"] = datetime.now().isoformat()
        return self._save_user_data(user_id, user_data)
    
    def delete_user(self, user_id: str) -> bool:
        """
        删除用户
        
        Args:
            user_id: 要删除的用户ID
            
        Returns:
            是否删除成功
        """
        try:
            os.remove(self._get_user_filepath(user_id))
            if self.current_user_id == user_id:
                self.current_user_id = None
            return True
        except FileNotFoundError:
            return True  # 文件不存在视为删除成功
        except Exception as e:
            logging.error(f"删除用户 {user_id} 失败: {str(e)}")
            return False
    
    def set_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """
        设置用户偏好
        
        Args:
            user_id: 用户ID
            key: 偏好键
            value: 偏好值
            
        Returns:
            是否设置成功
        """
        user_data = self._load_user_data(user_id)
        if not user_data:
            return False
        
        user_data["preferences"][key] = value
        user_data["last_active"] = datetime.now().isoformat()
        return self._save_user_data(user_id, user_data)
    
    def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        获取用户偏好
        
        Args:
            user_id: 用户ID
            key: 偏好键
            default: 默认值
            
        Returns:
            偏好值或默认值
        """
        user_data = self._load_user_data(user_id)
        if not user_data:
            return default
        return user_data["preferences"].get(key, default)
    
    def add_user_history(self, user_id: str, content: Dict[str, Any]) -> bool:
        """
        添加用户交互历史
        
        Args:
            user_id: 用户ID
            content: 交互内容
            
        Returns:
            是否添加成功
        """
        user_data = self._load_user_data(user_id)
        if not user_data:
            return False
        
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "content": content
        }
        
        user_data["history"].append(history_entry)
        user_data["last_active"] = history_entry["timestamp"]
        return self._save_user_data(user_id, user_data)
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取用户交互历史
        
        Args:
            user_id: 用户ID
            limit: 返回的历史记录条数限制
            
        Returns:
            用户历史记录列表
        """
        user_data = self._load_user_data(user_id)
        if not user_data:
            return []
        
        return user_data["history"][-limit:][::-1]  # 返回最新的记录在最前面
    
    def _record_user_interaction(
        self,
        user_input: str,
        ocr_results: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        统一记录用户交互历史
        
        Args:
            user_input: 用户输入文本
            ocr_results: OCR处理结果
            
        Returns:
            是否记录成功
        """
        if not self.current_user_id:
            return False
        
        content = {
            "user_input": user_input,
            "timestamp": datetime.now().isoformat()
        }
        
        if ocr_results:
            content["ocr_results"] = ocr_results
        
        return self.add_user_history(
            user_id=self.current_user_id,
            content=content
        )
    
    def record_user_interaction(self, user_input: str):
        """公共接口方法，封装对私有方法的访问"""
        self._record_user_interaction(user_input)

    def get_user_log_dir(self, user_id: str) -> str:
        """
        返回用户专属的日志目录路径
        """
        user_dir = os.path.join(self.storage_dir, user_id)
        log_dir = os.path.join(user_dir, "task_logs")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    def get_task_logger(self, user_id: str) -> 'TaskLogger':
        """
        为指定用户创建TaskLogger
        """
        from log.log_recorder import TaskLogger 
        return TaskLogger(log_dir=self.get_user_log_dir(user_id))
