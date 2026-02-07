"""
重新布局模块

将排序后的字符放大并重新布局到新页面。
"""

from typing import List, Tuple

import cv2
import numpy as np

from src.config import LayoutConfig, get_page_size
from src.segmenter import CharacterBox


class Layouter:
    """布局器"""

    def __init__(self, config: LayoutConfig):
        self.config = config
        self.output_config = config.output

    def layout(self, chars: List[CharacterBox]) -> List[np.ndarray]:
        """
        将字符放大并重新布局

        Args:
            chars: 排序后的字符列表

        Returns:
            页面图像列表
        """
        if not chars:
            return [self._create_blank_page()]

        # 获取页面尺寸
        page_width, page_height = get_page_size(
            self.output_config.page_size, self.output_config.dpi
        )

        # 计算可用区域
        margin = self.output_config.margin
        usable_width = page_width - 2 * margin
        usable_height = page_height - 2 * margin

        # 计算统一的字符大小（基于中位数）
        target_char_size = self._calculate_target_size(chars)

        pages = []
        current_page = self._create_blank_page()

        x = margin
        y = margin
        line_height = target_char_size
        current_line_id = chars[0].line_id if chars else -1

        for char in chars:
            # 检测换行（原文换行或超出宽度）
            need_newline = False

            if char.line_id != current_line_id and current_line_id != -1:
                need_newline = True
                current_line_id = char.line_id

            # 缩放字符
            scaled = self._scale_char(char.image, target_char_size)
            h, w = scaled.shape[:2]

            # 检查是否超出行宽
            if x + w > page_width - margin:
                need_newline = True

            if need_newline:
                x = margin
                y += line_height + self.output_config.line_spacing
                current_line_id = char.line_id

            # 检查是否需要换页
            if y + h > page_height - margin:
                pages.append(current_page)
                current_page = self._create_blank_page()
                x = margin
                y = margin

            # 放置字符（垂直居中对齐）
            y_offset = (line_height - h) // 2 if h < line_height else 0
            self._place_char(current_page, scaled, x, y + y_offset)

            x += w + self.output_config.char_spacing

        pages.append(current_page)
        return pages

    def _calculate_target_size(self, chars: List[CharacterBox]) -> int:
        """计算目标字符大小"""
        if not chars:
            return 50

        # 使用中位数高度
        heights = sorted([c.height for c in chars])
        median_h = heights[len(heights) // 2]

        # 应用放大倍数
        target = int(median_h * self.output_config.scale_factor)

        # 限制范围
        return max(30, min(200, target))

    def _scale_char(self, char_image: np.ndarray, target_size: int) -> np.ndarray:
        """
        缩放字符图像，保持宽高比

        Args:
            char_image: 字符图像（灰度）
            target_size: 目标高度

        Returns:
            缩放后的图像
        """
        h, w = char_image.shape[:2]

        if h == 0 or w == 0:
            return char_image

        # 简单策略：所有字符统一缩放到目标高度
        scale = target_size / h
        new_h = target_size
        new_w = max(1, int(w * scale))

        scaled = cv2.resize(
            char_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC
        )

        return scaled

    def _create_blank_page(self) -> np.ndarray:
        """创建空白页面"""
        page_width, page_height = get_page_size(
            self.output_config.page_size, self.output_config.dpi
        )

        # 白色背景
        page = np.ones((page_height, page_width), dtype=np.uint8) * 255

        return page

    def _place_char(
        self, page: np.ndarray, char_image: np.ndarray, x: int, y: int
    ) -> None:
        """将字符放置到页面上"""
        h, w = char_image.shape[:2]
        page_h, page_w = page.shape[:2]

        # 边界检查
        if x < 0 or y < 0:
            return
        if x + w > page_w:
            w = page_w - x
        if y + h > page_h:
            h = page_h - y
        if w <= 0 or h <= 0:
            return

        # 放置字符（使用最小值混合，保留深色像素）
        char_region = char_image[:h, :w]
        page[y:y+h, x:x+w] = np.minimum(page[y:y+h, x:x+w], char_region)
