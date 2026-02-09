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
    text: str = ""  # 便于测试和调试
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
        orientation: str = "vertical",
    ) -> List[CharacterBox]:
        """
        根据指定方向分割文本

        Args:
            binary_image: 二值化图像
            gray_image: 灰度图像
            orientation: 页面方向，"vertical"（竖版）或 "horizontal"（横版）
        """
        if orientation == "horizontal":
            return self.segment_horizontal(binary_image, gray_image)
        else:
            return self.segment_vertical(binary_image, gray_image)

    def segment_vertical(
        self,
        binary_image: np.ndarray,
        gray_image: np.ndarray,
    ) -> List[CharacterBox]:
        """
        竖排文本分割（完全照抄原始竖版项目）

        1. 形态学操作连接断开的笔画
        2. 连通域分析提取字块
        3. 按x坐标聚类成列
        4. 列排序：从右到左
        5. 每列内按y排序：从上到下
        """
        # 形态学处理（竖版专用）
        processed = self._morphology_process_vertical(binary_image)

        # 提取字块（照抄原版 - 使用竖版参数，不去重）
        blocks = self._extract_blocks_for_vertical(processed, gray_image)

        if not blocks:
            return []

        # 按x聚类成列（照抄原版 - 使用竖版参数，tolerance=0.8）
        columns = self._cluster_by_x_vertical(blocks)

        # 排序：从右到左 (X 降序)
        columns.sort(key=lambda col: -self._avg_x(col))

        # 每列内按y排序，生成结果
        result = []
        for col_id, col_blocks in enumerate(columns):
            col_blocks.sort(key=lambda b: b.y)  # 从上到下
            for char_id, block in enumerate(col_blocks):
                block.line_id = col_id
                block.char_id = char_id
                result.append(block)

        return result

    def segment_horizontal(
        self,
        binary_image: np.ndarray,
        gray_image: np.ndarray,
    ) -> List[CharacterBox]:
        """
        横排文本分割（每行固定切成5条）

        1. 形态学操作连接断开的笔画
        2. 连通域分析提取字块
        3. 按y坐标聚类成行
        4. 每行固定切成5条（按长度平均分配）
        5. 每条内按x排序
        """
        # 形态学处理（横版专用）
        processed = self._morphology_process_horizontal(binary_image)

        # 提取字块
        blocks = self._extract_blocks(processed, gray_image)

        if not blocks:
            return []

        # 按y聚类成行
        rows = self._cluster_by_y(blocks)

        # 排序：从上到下 (Y 升序)
        rows.sort(key=lambda row: self._avg_y(row))

        # 每行固定切成5条
        result = []
        strips_per_row = 5

        for row_id, row_blocks in enumerate(rows):
            # 按X坐标排序
            row_blocks.sort(key=lambda b: b.x)

            if not row_blocks:
                continue

            # 计算行的总宽度
            row_min_x = min(b.x for b in row_blocks)
            row_max_x = max(b.x + b.width for b in row_blocks)
            row_width = row_max_x - row_min_x

            # 每条的宽度
            strip_width_target = row_width / strips_per_row

            # 按X位置分配到5条
            for strip_id in range(strips_per_row):
                # 计算这一条的X范围
                strip_start_x = row_min_x + strip_id * strip_width_target
                strip_end_x = row_min_x + (strip_id + 1) * strip_width_target

                # 找出属于这一条的字块（中心点在范围内）
                strip_blocks = [b for b in row_blocks if strip_start_x <= b.center_x < strip_end_x]

                # 最后一条包含所有剩余的字块
                if strip_id == strips_per_row - 1:
                    strip_blocks = [b for b in row_blocks if b.center_x >= strip_start_x]

                if not strip_blocks:
                    continue

                # 计算条的边界
                min_x = min(b.x for b in strip_blocks)
                max_x = max(b.x + b.width for b in strip_blocks)
                min_y = min(b.y for b in strip_blocks)
                max_y = max(b.y + b.height for b in strip_blocks)

                # 从原始灰度图提取条状图像
                strip_width = max_x - min_x
                strip_height = max_y - min_y
                strip_image = gray_image[min_y:max_y, min_x:max_x].copy()

                result.append(CharacterBox(
                    image=strip_image,
                    x=min_x,
                    y=min_y,
                    width=strip_width,
                    height=strip_height,
                    line_id=row_id,
                    char_id=strip_id,
                ))

        return result

    def _split_row_into_strips(self, row_blocks: List[CharacterBox]) -> List[List[CharacterBox]]:
        """将一行按X方向的间距分割成多条"""
        if not row_blocks:
            return []

        if len(row_blocks) == 1:
            return [row_blocks]

        # 计算平均字块宽度
        avg_width = sum(b.width for b in row_blocks) / len(row_blocks)

        # 间距阈值：如果两个字块之间的距离大于平均宽度的0.3倍，就分割
        # 降低阈值，让条更短，放大后不会超宽
        gap_threshold = avg_width * 0.3

        strips = []
        current_strip = [row_blocks[0]]

        for i in range(1, len(row_blocks)):
            prev_block = row_blocks[i-1]
            curr_block = row_blocks[i]

            # 计算间距
            gap = curr_block.x - (prev_block.x + prev_block.width)

            if gap > gap_threshold:
                # 间距太大，开始新的条
                strips.append(current_strip)
                current_strip = [curr_block]
            else:
                # 继续当前条
                current_strip.append(curr_block)

        strips.append(current_strip)
        return strips


    def _avg_x(self, blocks: List[CharacterBox]) -> float:
        if not blocks:
            return 0
        return sum(b.center_x for b in blocks) / len(blocks)

    def _avg_y(self, blocks: List[CharacterBox]) -> float:
        if not blocks:
            return 0
        return sum(b.center_y for b in blocks) / len(blocks)

    def _morphology_process_horizontal(self, binary: np.ndarray) -> np.ndarray:
        """
        形态学处理：膨胀-腐蚀（横版专用）
        使用温和参数，宁可字分开也不要过度合并导致缺失
        """
        # 4x4核，膨胀1次，腐蚀1次
        # 宁可某些字被分开（如"言"、"扑"），也不要过度合并导致字块缺失
        kernel = np.ones((4, 4), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        return eroded

    def _morphology_process_vertical(self, binary: np.ndarray) -> np.ndarray:
        """
        形态学处理：膨胀-腐蚀（竖版专用）
        使用原始参数，保持竖版稳定性
        """
        # 5x5核，膨胀2次，腐蚀2次（原始参数）
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=2)
        return eroded

    def _morphology_process(self, binary: np.ndarray) -> np.ndarray:
        """
        形态学处理：膨胀-腐蚀（已废弃，保留兼容性）
        """
        # 默认使用横版参数
        return self._morphology_process_horizontal(binary)

    def _extract_blocks(
        self,
        binary: np.ndarray,
        gray: np.ndarray,
    ) -> List[CharacterBox]:
        """提取字块（横版参数 - 更宽松以保留"一"、"三"）"""
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
            if area < 30:
                continue
            if w < 3 and h < 3:
                continue

            # 过滤太大的（可能是边框、图片、大块污渍等）
            # 加强过滤：宽度>30%或高度>20%或面积>10%都过滤
            page_area = binary.shape[0] * binary.shape[1]
            if w > binary.shape[1] * 0.3 or h > binary.shape[0] * 0.2 or area > page_area * 0.1:
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

        # 去重：移除重叠的字块（保留面积较大的）
        blocks = self._remove_overlapping_blocks(blocks)

        return blocks

    def _remove_overlapping_blocks(self, blocks: List[CharacterBox]) -> List[CharacterBox]:
        """移除重叠或相邻的重复字块"""
        if not blocks:
            return []

        # 按Y坐标排序，然后按X坐标排序
        blocks_sorted = sorted(blocks, key=lambda b: (b.center_y, b.center_x))

        result = []
        for block in blocks_sorted:
            # 检查是否与已保留的字块重叠或相邻
            is_duplicate = False
            for kept_block in result:
                if self._is_duplicate(block, kept_block):
                    is_duplicate = True
                    break

            if not is_duplicate:
                result.append(block)

        return result

    def _is_duplicate(self, block1: CharacterBox, block2: CharacterBox) -> bool:
        """判断两个字块是否重复（重叠或在同一行且相邻）"""
        # 首先检查是否在同一行（Y坐标接近）
        y_diff = abs(block1.center_y - block2.center_y)
        avg_height = (block1.height + block2.height) / 2

        # 如果不在同一行，不是重复
        if y_diff > avg_height * 0.5:
            return False

        # 检查是否都是扁字块（如"一"）
        is_flat1 = block1.width > block1.height * 3
        is_flat2 = block2.width > block2.height * 3

        # 如果都是扁字块，使用更严格的去重
        if is_flat1 and is_flat2:
            x_diff = abs(block1.center_x - block2.center_x)
            # 扁字块：如果X距离小于100像素，认为是重复
            if x_diff < 100:
                return True

        # 在同一行，检查X方向的距离
        x_diff = abs(block1.center_x - block2.center_x)
        avg_width = (block1.width + block2.width) / 2

        # 如果X方向距离很近（小于平均宽度的0.8倍），认为是重复
        if x_diff < avg_width * 0.8:
            return True

        # 否则检查是否有重叠
        x1 = max(block1.x, block2.x)
        y1 = max(block1.y, block2.y)
        x2 = min(block1.x + block1.width, block2.x + block2.width)
        y2 = min(block1.y + block1.height, block2.y + block2.height)

        if x2 <= x1 or y2 <= y1:
            return False  # 没有交集

        intersection = (x2 - x1) * (y2 - y1)
        area1 = block1.width * block1.height
        area2 = block2.width * block2.height

        # 如果交集占任一字块面积的50%以上，认为是重复
        if intersection / area1 > 0.5 or intersection / area2 > 0.5:
            return True

        return False

    def _extract_blocks_for_vertical(
        self,
        binary: np.ndarray,
        gray: np.ndarray,
    ) -> List[CharacterBox]:
        """提取字块（原始竖版项目，不去重）"""
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

            # 过滤太小的噪点
            if area < 50:
                continue
            if w < 5 and h < 5:
                continue

            # 过滤太大的（可能是边框、大块污渍等）
            if w > binary.shape[1] * 0.8 or h > binary.shape[0] * 0.8:
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

    def _extract_blocks_vertical(
        self,
        binary: np.ndarray,
        gray: np.ndarray,
    ) -> List[CharacterBox]:
        """提取字块（竖版参数）"""
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

            # 过滤太小的噪点（竖版参数）
            if area < 50:
                continue
            if w < 5 and h < 5:
                continue

            # 过滤太大的（竖版参数）
            if w > binary.shape[1] * 0.8 or h > binary.shape[0] * 0.8:
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

        # 去重：移除重叠的字块（保留面积较大的）
        blocks = self._remove_overlapping_blocks(blocks)

        return blocks

    def _cluster_by_x(self, blocks: List[CharacterBox]) -> List[List[CharacterBox]]:
        """按x坐标聚类成列（横版参数 - 完全照搬zip源码）"""
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


    def _cluster_by_x_vertical(self, blocks: List[CharacterBox]) -> List[List[CharacterBox]]:
        """按x坐标聚类成列（竖版参数）"""
        if not blocks:
            return []

        if len(blocks) < 3:
            return [blocks]

        # 估算列数和容差
        widths = [b.width for b in blocks]
        median_w = sorted(widths)[len(widths) // 2]

        # 容差：中位宽度的80%（竖版参数）
        tolerance = median_w * 0.8

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

        # 过滤掉只有极少字符的列（除非总字符也很少）
        if len(blocks) > 20:
             columns = [col for col in columns if len(col) >= 1]

        return columns

    def _cluster_by_y(self, blocks: List[CharacterBox]) -> List[List[CharacterBox]]:
        """按y坐标聚类成行"""
        if not blocks:
            return []

        if len(blocks) < 3:
            return [blocks]

        # 估算行数和容差
        heights = [b.height for b in blocks]
        median_h = sorted(heights)[len(heights) // 2]

        # 容差：中位高度的60%
        tolerance = median_h * 0.6

        blocks_sorted = sorted(blocks, key=lambda b: b.center_y)

        rows = []
        current_row = [blocks_sorted[0]]
        current_y = blocks_sorted[0].center_y

        for block in blocks_sorted[1:]:
            if abs(block.center_y - current_y) <= tolerance:
                current_row.append(block)
                current_y = sum(b.center_y for b in current_row) / len(current_row)
            else:
                rows.append(current_row)
                current_row = [block]
                current_y = block.center_y

        rows.append(current_row)

        # 过滤掉只有极少字符的行（除非总字符也很少）
        if len(blocks) > 20:
            rows = [row for row in rows if len(row) >= 1]

        return rows
