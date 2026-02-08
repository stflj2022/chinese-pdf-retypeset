import numpy as np
from pathlib import Path
from typing import Callable, Optional
from pdf2image import convert_from_path
import cv2

from src.config import Config
from src.preprocessor import Preprocessor
from src.segmenter import Segmenter
from src.layouter import Layouter
from src.output import OutputGenerator

class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = Preprocessor(config.preprocessing)
        self.segmenter = Segmenter(config.segmentation)
        self.layouter = Layouter(config.layout)
        self.output_generator = OutputGenerator(config.output)
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        self.progress_callback = callback

    def _report_progress(self, current: int, total: int, message: str):
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def process(self, input_path: Path, output_path: Path):
        # 1. Load PDF
        self._report_progress(0, 100, "正在加载PDF...")
        try:
            images = convert_from_path(str(input_path), dpi=self.config.input.dpi)
        except Exception as e:
            # Fallback for images
            if input_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                images = [cv2.imread(str(input_path))]
                # Convert BGR to RGB for consistency if using PIL later, but we use CV2 internally
                # actually convert_from_path returns PIL images.
                # cv2.imread returns numpy.
                # Let's standardize on PIL -> Numpy
                img_pil = Image.open(input_path)
                images = [img_pil]
            else:
                raise e

        total_pages = len(images)
        vertical_chars = []
        horizontal_pages = []

        # 2. Process each page
        for i, img_pil in enumerate(images):
            self._report_progress(int((i / total_pages) * 50), 100, f"正在分析第 {i+1}/{total_pages} 页...")

            # Convert PIL to Numpy (Gray)
            img_np = np.array(img_pil)
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np

            # Preprocess
            binary = self.preprocessor.process(gray)

            # Detect orientation
            orientation = self.segmenter.detect_orientation(binary, gray)

            # Segment based on orientation
            if orientation == "vertical":
                # 竖版：收集所有字符
                chars = self.segmenter.segment_vertical(binary, gray)
                vertical_chars.extend(chars)
            else:
                # 横版：每页单独layout
                chars = self.segmenter.segment_horizontal(binary, gray)
                if chars:
                    page_output = self.layouter.layout(chars)
                    horizontal_pages.extend(page_output)

        # 3. Layout竖版字符
        output_pages = []
        if vertical_chars:
            self._report_progress(60, 100, f"正在重排 {len(vertical_chars)} 个竖版字符...")
            vertical_pages = self.layouter.layout(vertical_chars)
            output_pages.extend(vertical_pages)

        # 合并横版页面
        output_pages.extend(horizontal_pages)

        # 4. Save
        self._report_progress(90, 100, "正在保存文件...")
        self.output_generator.save(output_pages, output_path)

        self._report_progress(100, 100, "完成")
