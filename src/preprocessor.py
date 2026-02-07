"""
图像预处理模块

处理图像的灰度化、二值化、形态学操作。
重点：连接断开的笔画，确保每个字是一个连通域。
"""

import cv2
import numpy as np

from src.config import PreprocessingConfig


class Preprocessor:
    """图像预处理器"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        执行预处理，返回二值图像

        关键：使用形态学操作连接断开的笔画
        """
        # 1. 灰度化
        gray = self._to_grayscale(image)

        # 2. 二值化
        binary = self._binarize(gray)

        # 3. 形态学操作连接笔画
        binary = self._connect_strokes(binary)

        return binary

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """转换为灰度图像"""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """二值化 - 使用Otsu"""
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    def _connect_strokes(self, binary: np.ndarray) -> np.ndarray:
        """
        使用形态学操作连接断开的笔画

        策略：先膨胀连接，再腐蚀恢复
        """
        kernel = np.ones((3, 3), np.uint8)

        # 膨胀2次连接断开的笔画
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # 腐蚀1次，部分恢复原始大小（但保持连接）
        result = cv2.erode(dilated, kernel, iterations=1)

        return result

    def get_grayscale(self, image: np.ndarray) -> np.ndarray:
        """获取灰度图像（用于提取字块图像）"""
        return self._to_grayscale(image)
