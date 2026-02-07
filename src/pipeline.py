"""
处理流水线模块

协调各模块完成完整的处理流程。
"""

from pathlib import Path
from typing import Callable, List, Optional, Union

import cv2
import numpy as np
from pdf2image import convert_from_path

from src.config import Config
from src.layouter import Layouter
from src.output import OutputGenerator
from src.preprocessor import Preprocessor
from src.segmenter import CharacterBox, Segmenter


class Pipeline:
    """处理流水线"""

    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = Preprocessor(config.preprocessing)
        self.segmenter = Segmenter(config.segmentation)
        self.layouter = Layouter(config.layout)
        self.output_generator = OutputGenerator(config.output, config.layout.output)

        self.progress_callback: Optional[Callable[[int, int, str], None]] = None

    def set_progress_callback(
        self, callback: Callable[[int, int, str], None]
    ) -> None:
        """设置进度回调"""
        self.progress_callback = callback

    def _report_progress(self, current: int, total: int, message: str) -> None:
        """报告进度"""
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> None:
        """处理输入文件并生成输出"""
        input_path = Path(input_path)
        output_path = Path(output_path)

        # 1. 加载输入
        self._report_progress(1, 5, "加载输入文件...")
        images = self._load_input(input_path)

        # 2. 处理每一页
        all_pages = []
        total_images = len(images)

        for i, image in enumerate(images):
            self._report_progress(
                2, 5, f"处理第 {i + 1}/{total_images} 页..."
            )
            pages = self._process_single_image(image)
            all_pages.extend(pages)

        # 3. 保存输出
        self._report_progress(5, 5, "保存输出文件...")
        self.output_generator.save(all_pages, output_path)

    def _load_input(self, input_path: Path) -> List[np.ndarray]:
        """加载输入文件"""
        suffix = input_path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(input_path)
        elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            return self._load_image(input_path)
        else:
            raise ValueError(f"Unsupported input format: {suffix}")

    def _load_pdf(self, pdf_path: Path) -> List[np.ndarray]:
        """加载PDF文件"""
        pil_images = convert_from_path(str(pdf_path), dpi=self.config.input.dpi)

        images = []
        for pil_image in pil_images:
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            np_image = np.array(pil_image)
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            images.append(np_image)

        return images

    def _load_image(self, image_path: Path) -> List[np.ndarray]:
        """加载图像文件"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return [image]

    def _process_single_image(self, image: np.ndarray) -> List[np.ndarray]:
        """处理单张图像"""
        # 1. 预处理
        gray = self.preprocessor.get_grayscale(image)
        binary = self.preprocessor.process(image)

        # 2. 根据布局模式选择分割方法
        direction = self.config.layout.direction

        if direction == "horizontal":
            characters = self.segmenter.segment(binary, gray)
        elif direction in ["vertical_rtl", "vertical_ltr"]:
            characters = self.segmenter.segment_vertical(binary, gray)
        elif direction == "auto":
            characters = self.segmenter.segment_auto(binary, gray)
        else:
            characters = self.segmenter.segment(binary, gray)

        if not characters:
            return [self.layouter._create_blank_page()]

        # 3. 布局（字符已经按阅读顺序排列）
        pages = self.layouter.layout(characters)

        return pages

    def process_preview(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        处理预览

        Returns:
            (原始预览, 处理后预览, 检测到的字符数)
        """
        gray = self.preprocessor.get_grayscale(image)
        binary = self.preprocessor.process(image)

        direction = self.config.layout.direction

        if direction == "horizontal":
            characters = self.segmenter.segment(binary, gray)
        elif direction in ["vertical_rtl", "vertical_ltr"]:
            characters = self.segmenter.segment_vertical(binary, gray)
        elif direction == "auto":
            characters = self.segmenter.segment_auto(binary, gray)
        else:
            characters = self.segmenter.segment(binary, gray)

        char_count = len(characters)

        if characters:
            pages = self.layouter.layout(characters)
            processed = pages[0] if pages else self.layouter._create_blank_page()
        else:
            processed = self.layouter._create_blank_page()

        original_preview = self.output_generator.get_preview(gray)
        processed_preview = self.output_generator.get_preview(processed)

        return original_preview, processed_preview, char_count

    def get_segmentation_preview(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """
        获取分割预览

        Returns:
            (标注了字符边界框的图像, 字符数量)
        """
        gray = self.preprocessor.get_grayscale(image)
        binary = self.preprocessor.process(image)

        direction = self.config.layout.direction

        if direction == "horizontal":
            characters = self.segmenter.segment(binary, gray)
        elif direction in ["vertical_rtl", "vertical_ltr"]:
            characters = self.segmenter.segment_vertical(binary, gray)
        elif direction == "auto":
            characters = self.segmenter.segment_auto(binary, gray)
        else:
            characters = self.segmenter.segment(binary, gray)

        # 在原图上绘制边界框
        preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 不同行/列用不同颜色
        colors = [
            (0, 255, 0),    # 绿
            (255, 0, 0),    # 蓝
            (0, 0, 255),    # 红
            (255, 255, 0),  # 青
            (255, 0, 255),  # 紫
            (0, 255, 255),  # 黄
        ]

        for char in characters:
            color = colors[char.line_id % len(colors)]
            cv2.rectangle(
                preview,
                (char.x, char.y),
                (char.right, char.bottom),
                color,
                1,
            )

        return preview, len(characters)
