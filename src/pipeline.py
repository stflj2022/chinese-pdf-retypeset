import numpy as np
from pathlib import Path
from typing import Callable, Optional
from pdf2image import convert_from_path
from PIL import Image
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

    def process(self, input_path: Path, output_path: Path, orientation: str = "auto"):
        """
        处理PDF文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            orientation: 页面方向，"auto"（自动检测每页）、"vertical"（竖版）或 "horizontal"（横版）
        """
        # 1. Load PDF
        self._report_progress(0, 100, "正在加载PDF...")
        try:
            images = convert_from_path(str(input_path), dpi=self.config.input.dpi)
        except Exception as e:
            # Fallback for images
            if input_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
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
        # 改为：记录每页的内容和方向，按原始顺序输出
        pages_content = []  # [(page_num, orientation, chars), ...]

        # 2. Process each page
        for i, img_pil in enumerate(images):
            self._report_progress(
                int((i / total_pages) * 50),
                100,
                f"正在分析第 {i + 1}/{total_pages} 页...",
            )

            # Convert PIL to Numpy (Gray)
            img_np = np.array(img_pil)
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np

            # Preprocess
            binary = self.preprocessor.process(gray)

            # 确定页面方向
            if orientation == "auto":
                page_orientation = self.segmenter._detect_orientation(binary, gray)
            else:
                page_orientation = orientation

            # Segment based on orientation
            if page_orientation == "vertical":
                # 竖版：收集字符（暂不layout）
                chars = self.segmenter.segment_vertical(binary, gray)
                pages_content.append((i, "vertical", chars))
            else:
                # 横版：收集字符（暂不layout）
                chars = self.segmenter.segment_horizontal(binary, gray)
                pages_content.append((i, "horizontal", chars))

        # 3. 按原始页码顺序处理和layout
        output_pages = []
        pending_vertical_chars = []

        for page_num, orient, chars in pages_content:
            if orient == "vertical":
                # 竖版：累积字符，等到遇到横版或最后再layout
                pending_vertical_chars.extend(chars)
                # 检查下一页是否是横版，如果是则先layout当前竖版
                next_page_is_horizontal = False
                for next_p, next_o, _ in pages_content[page_num + 1 :]:
                    if next_o == "horizontal" and next_p == page_num + 1:
                        next_page_is_horizontal = True
                        break
                    if next_o == "vertical":
                        break

                if next_page_is_horizontal or page_num == len(pages_content) - 1:
                    # 需要layout竖版字符了
                    if pending_vertical_chars:
                        self._report_progress(60, 100, f"正在重排竖版字符...")
                        vertical_pages = self.layouter.layout(pending_vertical_chars)
                        output_pages.extend(vertical_pages)
                        pending_vertical_chars = []
            else:
                # 横版：先layout累积的竖版字符（如果有），再layout当前横版
                if pending_vertical_chars:
                    self._report_progress(60, 100, f"正在重排竖版字符...")
                    vertical_pages = self.layouter.layout(pending_vertical_chars)
                    output_pages.extend(vertical_pages)
                    pending_vertical_chars = []

                if chars:
                    page_output = self.layouter.layout(chars)
                    output_pages.extend(page_output)

        # 处理剩余的竖版字符
        if pending_vertical_chars:
            self._report_progress(60, 100, f"正在重排竖版字符...")
            vertical_pages = self.layouter.layout(pending_vertical_chars)
            output_pages.extend(vertical_pages)

        # 4. Save
        self._report_progress(90, 100, "正在保存文件...")
        self.output_generator.save(output_pages, output_path)

        self._report_progress(100, 100, "完成")
