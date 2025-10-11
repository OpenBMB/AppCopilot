import numpy as np
import os
import sys
import base64
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
outer_dir = os.path.dirname(current_dir)
sys.path.append(outer_dir)
from wrappers.constants import CLIENT
from PIL import Image
from io import BytesIO


class VisionExtractor:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model = model_name

    def query_image(self, image: np.ndarray, prompt: str) -> str:
        """
        向gpt-4o发送图像和提示词，获取视觉分析结果
        :param image: 输入图像（numpy数组格式）
        :param prompt: 提示词（描述需要分析的任务）
        :return: 模型返回的文本结果
        """
        try:
            img = Image.fromarray(image)
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=90)
            buffered.seek(0)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 调用vlm
            time.sleep(3)  # 避免429错误，增加请求间隔
            response = CLIENT.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.2,
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"处理图像时出错：{str(e)}"
