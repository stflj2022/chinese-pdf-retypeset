import cv2
import numpy as np
from src.config import PreprocessingConfig

class Preprocessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        预处理：灰度转二值，去噪
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 去噪
        if self.config.denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # 二值化
        # 使用Otsu自动阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        return binary
