import numpy as np
import base64
import requests
import hashlib
import json
from typing import List, Dict, Tuple
from wrappers.constants import OMNI_PORT
from wrappers.utils import array_to_jpeg_bytes


class OmniParserClient:
    def __init__(self, port: int = OMNI_PORT, record_history: bool = True, log_path: str = "omni_parser/fix_log.jsonl"):
        self.url = f"http://localhost:{port}/parse/base64"
        self.record_history = record_history
        self.log_path = log_path
        self.fixed_point_history: Dict[Tuple[str, Tuple[int, int]], Tuple[int, int]] = {}

    def encode_image(self, image: np.ndarray) -> str:
        """将 numpy 图像编码为 base64 字符串"""
        return base64.b64encode(array_to_jpeg_bytes(image)).decode("utf-8")

    def hash_image(self, image: np.ndarray) -> str:
        """对 numpy 图像计算 SHA256 哈希值"""
        return hashlib.sha256(array_to_jpeg_bytes(image)).hexdigest()

    def get_parsed_content(self, image: np.ndarray) -> List[Dict]:
        """调用远程 OmniParser 接口获取解析内容"""
        image_b64 = self.encode_image(image)
        payload = {"image_base64": image_b64}
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        result = response.json()
        parsed_content = result.get("parsed_content", [])
        for idx, item in enumerate(parsed_content):
            item["idx"] = idx
        return result.get("parsed_content", [])

    def check_in_bbox(self, point: Tuple[int, int], bbox: List[int], tolerance: int = 10, interativity: bool = False) -> bool:
        """判断点是否在某个 bbox 中，可设置容差"""
        if interativity is False:
            return False
        x, y = point
        x1, y1, x2, y2 = bbox
        if x1 < 1 or y1 < 1 or x2 < 1 or y2 < 1:
            x1 *= 1000
            y1 *= 1000
            x2 *= 1000
            y2 *= 1000
        x1, x2 = min(x1, x2) - tolerance, max(x1, x2) + tolerance
        y1, y2 = min(y1, y2) - tolerance, max(y1, y2) + tolerance
        return x1 <= x <= x2 and y1 <= y <= y2

    def check_in_bbox_all(self, parsed_content: List[Dict], point: Tuple[int, int]) -> List[bool]:
        """检查一个点是否在所有 bbox 中"""
        return [
            self.check_in_bbox(point, c.get("bbox"), interativity=c.get("interactivity", False))
            for c in parsed_content if c.get("bbox") is not None
        ]

    def _get_center(self, bbox: List[int]) -> Tuple[int, int]:
        """获取 bbox 的中心点（归一化到 1000 乘）"""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) * 500)
        center_y = int((y1 + y2) * 500)
        return center_x, center_y

    def get_bbox_points(self, parsed_content: List[Dict]) -> List[Tuple[int, int]]:
        """获取所有 bbox 的中心点"""
        return [
            self._get_center(c["bbox"])
            for c in parsed_content if "bbox" in c
        ]

    def get_interactive_points(self, parsed_content: List[Dict]) -> List[Tuple[int, int]]:
        """获取 interactivity=True 的 bbox 中心点"""
        return [
            self._get_center(c["bbox"])
            for c in parsed_content
            if c.get("interactivity") and "bbox" in c
        ]

    def find_best_bbox(self, candidates: List[Dict], point: Tuple[int, int]) -> Tuple[int, int]:
        """从候选 bbox 中寻找最匹配的中心点，考虑距离"""
        best_score = float("inf")
        best_center = [0, 0]
        for item in candidates:
            bbox = item.get("bbox")
            if not bbox:
                continue
            
            # 获取 bbox 的中心点和坐标
            x1, y1, x2, y2 = bbox
            center = self._get_center(bbox)

            # 如果 bbox 坐标小于 1，则乘以 1000
            if x1 < 1:
                x1, y1, x2, y2 = x1*1000, y1*1000, x2*1000, y2*1000
            
            # 计算目标点到 bbox 的距离
            dx = max(0, max(x1 - point[0], point[0] - x2))
            dy = max(0, max(y1 - point[1], point[1] - y2))
            edge_distance = (dx ** 2 + dy ** 2) ** 0.5
            # print(f"\033[91m{edge_distance}\033[0m")
            # print(f"\033[91midx:\033[0m", item.get("idx", "N/A"))
            # 计算中心点到目标点的距离
            distance = (center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2

            score = edge_distance
            if score < best_score:
                best_score = score
                best_center = center
        # 平均化中心点
        best_center = (point[0]+best_center[0])//2, (point[1]+best_center[1])//2
        return best_center

    def find_best_bbox_all(self, parsed_content: List[Dict], point: Tuple[int, int]) -> Tuple[int, int]:
        interactive = [c for c in parsed_content if c.get("bbox")]
        if interactive:
            return self.find_best_bbox(interactive, point)
        return self.find_best_bbox(parsed_content, point)

    def save_fix_log(self, image_hash: str, original: Tuple[int, int], fixed: Tuple[int, int]):
        """写入 JSONL 日志"""
        record = {
            "image_hash": image_hash,
            "original": original,
            "fixed": fixed
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def fix_point(self, image: np.ndarray, point: Tuple[int, int]) -> Tuple[int, int]:
        """主方法：若不在 bbox 中，则修正为最近点；记录历史"""
        image_hash = self.hash_image(image)
        key = image_hash

        # if key in self.fixed_point_history:
        #     exclude_points = self.fixed_point_history[key]

        parsed_content = self.get_parsed_content(image)

        checked = self.check_in_bbox_all(parsed_content, point)
        # print(f"Checked points: {checked}")
        if any(checked):
            self.fixed_point_history[key] = point
            return point

        print("\033[91mPoint not in any bounding box, finding best matched point...\033[0m")
        corrected = self.find_best_bbox_all(parsed_content, point)
        self.fixed_point_history[key] = corrected

        if self.record_history:
            self.save_fix_log(image_hash, point, corrected)
        print(f"Point {point} corrected to {corrected}")
        return corrected