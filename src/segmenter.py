"""
字符分割模块

使用形态学操作连接断开的笔画，然后按位置聚类成行。
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from src.config import SegmentationConfig


@dataclass
class CharacterBox:
    """字符/字块边界框"""
    image: np.ndarray
    x: int
    y: int
    width: int
    height: int
    line_id: int = -1
    char_id: int = -1

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


class Segmenter:
    """字块分割器"""

    def __init__(self, config: SegmentationConfig):
        self.config = config

    def segment(
        self,
        binary_image: np.ndarray,
        gray_image: np.ndarray,
    ) -> List[CharacterBox]:
        """
        横排文本分割

        1. 形态学操作连接断开的笔画
        2. 连通域分析提取字块
        3. 按y坐标聚类成行
        4. 每行内按x坐标排序
        """
        # 形态学处理
        processed = self._morphology_process(binary_image)

        # 提取字块
        blocks = self._extract_blocks(processed, gray_image)

        if not blocks:
            return []

        # 按y聚类成行
        lines = self._cluster_by_y(blocks)

        # 每行内按x排序，生成结果
        result = []
        for line_id, line_blocks in enumerate(lines):
            line_blocks.sort(key=lambda b: b.x)
            for char_id, block in enumerate(line_blocks):
                block.line_id = line_id
                block.char_id = char_id
                result.append(block)

        return result

    def segment_auto(
        self,
        binary_image: np.ndarray,
        gray_image: np.ndarray,
    ) -> List[CharacterBox]:
        """自动检测排版方向并分割"""
        processed = self._morphology_process(binary_image)
        blocks = self._extract_blocks(processed, gray_image)

        if not blocks:
            return []

        # 检测排版方向
        if self._is_vertical_layout(blocks):
            # 竖排
            columns = self._cluster_by_x(blocks)
            columns.sort(key=lambda col: -sum(b.center_x for b in col) / len(col))
            result = []
            for col_id, col_blocks in enumerate(columns):
                col_blocks.sort(key=lambda b: b.y)
                for char_id, block in enumerate(col_blocks):
                    block.line_id = col_id
                    block.char_id = char_id
                    result.append(block)
            return result
        else:
            # 横排
            lines = self._cluster_by_y(blocks)
            result = []
            for line_id, line_blocks in enumerate(lines):
                line_blocks.sort(key=lambda b: b.x)
                for char_id, block in enumerate(line_blocks):
                    block.line_id = line_id
                    block.char_id = char_id
                    result.append(block)
            return result

    def _is_vertical_layout(self, blocks: List[CharacterBox]) -> bool:
        """检测是否为竖排（保守策略，默认横排）"""
        if len(blocks) < 5:
            return False

        # 尝试两种聚类
        lines = self._cluster_by_y(blocks)
        columns = self._cluster_by_x(blocks)

        avg_per_line = len(blocks) / len(lines) if lines else 0
        avg_per_col = len(blocks) / len(columns) if columns else 0

        # 竖排特征：每列字数明显多于每行
        if avg_per_col > avg_per_line * 2 and len(columns) <= 10:
            return True
        return False

    def segment_vertical(
        self,
        binary_image: np.ndarray,
        gray_image: np.ndarray,
    ) -> List[CharacterBox]:
        """
        竖排文本分割（从右到左）

        1. 形态学操作
        2. 按x坐标聚类成列
        3. 列从右到左排序
        4. 每列内按y坐标排序
        """
        processed = self._morphology_process(binary_image)
        blocks = self._extract_blocks(processed, gray_image)

        if not blocks:
            return []

        # 按x聚类成列
        columns = self._cluster_by_x(blocks)

        # 列从右到左排序
        columns.sort(key=lambda col: -sum(b.center_x for b in col) / len(col))

        # 每列内按y排序
        result = []
        for col_id, col_blocks in enumerate(columns):
            col_blocks.sort(key=lambda b: b.y)
            for char_id, block in enumerate(col_blocks):
                block.line_id = col_id
                block.char_id = char_id
                result.append(block)

        return result

    def _morphology_process(self, binary: np.ndarray) -> np.ndarray:
        """
        形态学处理：连接断开的笔画

        使用较大的核进行膨胀-腐蚀，把同一个字的笔画连接起来
        """
        # 5x5核，膨胀2次，腐蚀2次
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=2)
        return eroded

    def _extract_blocks(
        self,
        binary: np.ndarray,
        gray: np.ndarray,
    ) -> List[CharacterBox]:
        """提取字块"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        blocks = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # 过滤太小的噪点（保留扁字如"一"）
            if area < 100:
                continue
            if w < 10 and h < 10:
                continue

            # 过滤太大的（可能是边框、大块污渍等）
            if w > binary.shape[1] * 0.5 or h > binary.shape[0] * 0.3:
                continue

            # 从原始灰度图提取字块图像
            block_image = gray[y:y+h, x:x+w].copy()

            blocks.append(CharacterBox(
                image=block_image,
                x=x,
                y=y,
                width=w,
                height=h,
            ))

        return blocks

    def _cluster_by_y(self, blocks: List[CharacterBox]) -> List[List[CharacterBox]]:
        """
        按y坐标聚类成行

        同一行的字块y中心应该接近
        """
        if not blocks:
            return []

        # 估计典型字块高度
        heights = [b.height for b in blocks]
        median_h = sorted(heights)[len(heights) // 2]

        # 容差：字块高度的60%
        tolerance = median_h * 0.6

        # 按y中心排序
        blocks_sorted = sorted(blocks, key=lambda b: b.center_y)

        lines = []
        current_line = [blocks_sorted[0]]
        current_y = blocks_sorted[0].center_y

        for block in blocks_sorted[1:]:
            if abs(block.center_y - current_y) <= tolerance:
                # 同一行
                current_line.append(block)
                # 更新行中心
                current_y = sum(b.center_y for b in current_line) / len(current_line)
            else:
                # 新行
                lines.append(current_line)
                current_line = [block]
                current_y = block.center_y

        lines.append(current_line)
        return lines

    def _cluster_by_x(self, blocks: List[CharacterBox]) -> List[List[CharacterBox]]:
        """按x坐标聚类成列"""
        if not blocks:
            return []

        widths = [b.width for b in blocks]
        median_w = sorted(widths)[len(widths) // 2]
        tolerance = median_w * 0.6

        blocks_sorted = sorted(blocks, key=lambda b: b.center_x)

        columns = []
        current_col = [blocks_sorted[0]]
        current_x = blocks_sorted[0].center_x

        for block in blocks_sorted[1:]:
            if abs(block.center_x - current_x) <= tolerance:
                current_col.append(block)
                current_x = sum(b.center_x for b in current_col) / len(current_col)
            else:
                columns.append(current_col)
                current_col = [block]
                current_x = block.center_x

        columns.append(current_col)
        return columns
