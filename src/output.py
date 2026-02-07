"""
输出生成模块

将处理后的页面输出为PDF或图像文件。
优化文件大小。
"""

import io
from pathlib import Path
from typing import List, Union

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from src.config import OutputConfig, OutputLayoutConfig


class OutputGenerator:
    """输出生成器"""

    def __init__(self, config: OutputConfig, layout_config: OutputLayoutConfig):
        self.config = config
        self.layout_config = layout_config

    def save(
        self,
        pages: List[np.ndarray],
        output_path: Union[str, Path],
    ) -> None:
        """
        保存输出

        Args:
            pages: 页面图像列表（灰度）
            output_path: 输出路径
        """
        output_path = Path(output_path)

        if self.config.format == "pdf":
            self._save_as_pdf(pages, output_path)
        elif self.config.format == "images":
            self._save_as_images(pages, output_path)
        else:
            raise ValueError(f"Unsupported output format: {self.config.format}")

    def _save_as_pdf(self, pages: List[np.ndarray], output_path: Path) -> None:
        """保存为PDF - 优化文件大小"""
        if output_path.suffix.lower() != ".pdf":
            output_path = output_path.with_suffix(".pdf")

        doc = fitz.open()

        for page_image in pages:
            # 确保是灰度图
            if len(page_image.shape) == 3:
                page_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

            # 转换为PIL图像
            pil_image = Image.fromarray(page_image)

            # 压缩为JPEG（大幅减小文件大小）
            img_bytes = io.BytesIO()
            pil_image.save(
                img_bytes,
                format="JPEG",
                quality=self.config.quality,
                optimize=True,
            )
            img_bytes.seek(0)

            # 计算页面尺寸（点）
            dpi = self.layout_config.dpi
            width_pt = page_image.shape[1] * 72 / dpi
            height_pt = page_image.shape[0] * 72 / dpi

            # 创建新页面
            page = doc.new_page(width=width_pt, height=height_pt)

            # 插入图像
            rect = fitz.Rect(0, 0, width_pt, height_pt)
            page.insert_image(rect, stream=img_bytes.read())

        doc.save(str(output_path), garbage=4, deflate=True)
        doc.close()

    def _save_as_images(self, pages: List[np.ndarray], output_path: Path) -> None:
        """保存为图像文件"""
        output_dir = output_path.parent / output_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, page_image in enumerate(pages):
            image_path = output_dir / f"page_{i + 1:04d}.jpg"
            pil_image = Image.fromarray(page_image)
            pil_image.save(
                str(image_path),
                quality=self.config.quality,
                optimize=True,
            )

    def get_preview(self, page: np.ndarray, max_size: int = 800) -> np.ndarray:
        """获取预览图像"""
        h, w = page.shape[:2]
        scale = min(max_size / w, max_size / h)

        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            preview = cv2.resize(page, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            preview = page.copy()

        return preview
