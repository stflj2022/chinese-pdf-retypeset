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
        orientation: str = "auto",
    ) -> List[CharacterBox]:
        """
        根据指定方向分割文本

        Args:
            binary_image: 二值化图像
            gray_image: 灰度图像
            orientation: 页面方向，"auto"（自动检测）、"vertical"（竖版）或 "horizontal"（横版）
        """
        if orientation == "auto":
            orientation = self._detect_orientation(binary_image, gray_image)

        if orientation == "horizontal":
            return self.segment_horizontal(binary_image, gray_image)
        else:
            return self.segment_vertical(binary_image, gray_image)

    def _detect_orientation(self, binary: np.ndarray, gray: np.ndarray) -> str:
        """
        自动检测页面方向

        策略：
        竖版：字符按列排列，有多个列，X方向变化大
        横版：字符按行排列，有多个行，Y方向变化大
        """
        # 简单提取字块
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        blocks = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area < 50 or w < 5 or h < 5:
                continue
            if w > binary.shape[1] * 0.8 or h > binary.shape[0] * 0.8:
                continue

            blocks.append({'x': x, 'y': y, 'w': w, 'h': h})

        if len(blocks) < 10:
            return "vertical"

        # 关键区别：
        # 竖版：字符按列排列，每列内Y变化大（从上到下），不同列X变化大（从右到左）
        # 横版：字符按行排列，每行内X变化大（从左到右），不同行Y变化大（从上到下）

        # 方法：统计Y坐标的离散程度
        # 竖版：相同X位置有多个不同Y（一列多字），Y的分布范围大
        # 横版：相同Y位置有多个不同X（一行多字），X的分布范围大

        x_coords = [b['x'] for b in blocks]
        y_coords = [b['y'] for b in blocks]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        # 竖版页面通常：Y_range > X_range（页面高度>宽度，且有列从上到下）
        # 横版页面通常：Y_range 也很大，但X_range更大（每行从左到右）

        # 更准确的方法：计算X和Y的"簇"数量
        # 竖版：X簇少（列数少），每簇内Y变化大
        # 横版：Y簇少（行数少），每簇内X变化大

        # 用简化的方法：看X坐标的分布
        # 竖版：X坐标集中在几个特定位置（列的位置）
        # 横版：X坐标分布较均匀（每行字从左到右）

        # 计算X坐标的分组数（用简单的距离聚类）
        sorted_x = sorted(x_coords)
        x_groups = 1
        gap_threshold = 50  # X间距大于50认为是不同列

        for i in range(1, len(sorted_x)):
            if sorted_x[i] - sorted_x[i-1] > gap_threshold:
                x_groups += 1

        # 计算Y坐标的分组数
        sorted_y = sorted(y_coords)
        y_groups = 1

        for i in range(1, len(sorted_y)):
            if sorted_y[i] - sorted_y[i-1] > gap_threshold:
                y_groups += 1

        # 竖版：X分组少（列数），Y分组多（每列的字在不同Y）
        # 横版：Y分组少（行数），X分组多（每行的字在不同X）

        # 实际上这个逻辑反了，改用另一种方法
        # 竖版：看X的唯一值数量（列数）vs 总字数
        # 横版：看Y的唯一值数量（行数）vs 总字数

        unique_x = len(set([round(x/20)*20 for x in x_coords]))  # 每20px为一组
        unique_y = len(set([round(y/20)*20 for y in y_coords]))  # 每20px为一组

        # 竖版：unique_x < unique_y（列数少于行位置数）
        # 横版：unique_x > unique_y（行位置数少于列分布）

        if unique_x < unique_y * 0.7:
            return "vertical"
        else:
            return "horizontal"

    def segment_vertical(
        self,
        binary_image: np.ndarray,
        gray_image: np.ndarray,
    ) -> List[CharacterBox]:
        """
        竖排文本分割（v29版 - 回退v11+过滤黑点）

        v11特征：顺序正确 ✅
        v29改进：添加黑点过滤
        """
        # v11：简单的水平膨胀
        kernel_h = np.ones((1, 5), np.uint8)
        dilated = cv2.dilate(binary_image, kernel_h, iterations=2)

        # 轻微腐蚀恢复形状
        kernel_v = np.ones((3, 1), np.uint8)
        eroded = cv2.erode(dilated, kernel_v, iterations=1)

        # 提取字块
        blocks = self._extract_blocks_for_vertical_v29(eroded, gray_image)

        if not blocks:
            return []

        # 按X聚类成列
        columns = self._cluster_by_x_vertical(blocks)

        # 排序：从右到左 (X 降序)
        columns.sort(key=lambda col: -self._avg_x(col))

        # 每列内按Y排序（从上到下）
        for col_blocks in columns:
            col_blocks.sort(key=lambda b: b.y)

        # 直接按列顺序输出
        result = []
        char_id = 0

        for col in columns:
            for block in col:
                block.line_id = 0
                block.char_id = char_id
                result.append(block)
                char_id += 1

        return result

    def _extract_blocks_for_vertical_v29(
        self,
        binary: np.ndarray,
        gray: np.ndarray,
    ) -> List[CharacterBox]:
        """提取字块（v29版：v11基础+过滤黑点）"""
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

            # 过滤太大的
            if w > binary.shape[1] * 0.8 or h > binary.shape[0] * 0.8:
                continue

            # 过滤疑似黑点（W=30-40, H<=20, 宽高比>2.0）
            aspect_ratio = w / h if h > 0 else 0
            if 30 <= w <= 40 and h <= 20 and aspect_ratio > 2.0:
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

    def _merge_split_by_diagnosis(self, blocks: List[CharacterBox], gray: np.ndarray) -> List[CharacterBox]:
        """
        基于v22诊断结果精确合并被切开的字（v24保守版）

        v22诊断发现被切开字的X坐标非常稳定：
        - 左块X：约200-205
        - 右块X：约282-284

        v24策略：只合并这两个特定X位置的块对，其他位置不动
        """
        if len(blocks) < 2:
            return blocks

        # 按Y排序，方便查找Y相近的块
        blocks_sorted = sorted(blocks, key=lambda b: b.y)

        merged = []
        used = set()

        for i, b1 in enumerate(blocks_sorted):
            if i in used:
                continue

            merged_with_b1 = None

            for j in range(i + 1, len(blocks_sorted)):
                if j in used:
                    continue

                b2 = blocks_sorted[j]

                # 检查Y是否相近（差距<5px）
                y_diff = abs(b1.y - b2.y)
                if y_diff >= 5:
                    break  # Y排序，后面差距更大

                # v24关键：只合并特定X位置的块
                # 左块X在195-210之间，右块X在275-290之间
                left_ok = (195 <= b1.x <= 210 and 275 <= b2.x <= 290) or \
                          (195 <= b2.x <= 210 and 275 <= b1.x <= 290)

                if not left_ok:
                    continue  # 不是特定位置，跳过

                # 计算X间距
                if b1.x < b2.x:
                    x_gap = b2.x - (b1.x + b1.width)
                else:
                    x_gap = b1.x - (b2.x + b2.width)

                # X间距在15-30px之间
                if 15 <= x_gap <= 30:
                    # 合并这两个块
                    used.add(i)
                    used.add(j)

                    min_x = min(b1.x, b2.x)
                    max_x = max(b1.x + b1.width, b2.x + b2.width)
                    min_y = min(b1.y, b2.y)
                    max_y = max(b1.y + b1.height, b2.y + b2.height)

                    merged_image = gray[min_y:max_y, min_x:max_x].copy()

                    merged_with_b1 = CharacterBox(
                        image=merged_image,
                        x=min_x,
                        y=min_y,
                        width=max_x - min_x,
                        height=max_y - min_y,
                    )
                    break

            if merged_with_b1:
                merged.append(merged_with_b1)
            else:
                merged.append(b1)

        return merged

    def _diagnose_split_characters(self, blocks: List[CharacterBox]) -> None:
        """
        诊断分析被切开字的特征（v22）

        输出：
        - Y坐标相同或相近的块对
        - 它们的X间距
        - 宽度、高度
        """
        if len(blocks) < 2:
            return

        # 按X和Y排序
        blocks_sorted = sorted(blocks, key=lambda b: (b.center_x, b.y))

        print("\n=== v22 诊断：被切开字分析 ===")

        # 计算中位高度和宽度
        heights = [b.height for b in blocks]
        widths = [b.width for b in blocks]
        median_h = sorted(heights)[len(heights) // 2]
        median_w = sorted(widths)[len(widths) // 2]

        print(f"中位高度: {median_h}, 中位宽度: {median_w}")

        # 查找Y坐标相近的块对
        found_pairs = []
        for i, b1 in enumerate(blocks_sorted):
            for j in range(i + 1, len(blocks_sorted)):
                b2 = blocks_sorted[j]

                # Y坐标相近（差异<5px）
                y_diff = abs(b1.y - b2.y)
                if y_diff < 5:
                    # X间距
                    if b1.x < b2.x:
                        x_gap = b2.x - (b1.x + b1.width)
                    else:
                        x_gap = b1.x - (b2.x + b2.width)

                    # 间距在合理范围内（0-30px）
                    if 0 <= x_gap <= 30:
                        found_pairs.append((b1, b2, x_gap, y_diff))

        # 输出前20对
        print(f"\n找到 {len(found_pairs)} 对可能被切开的字")
        print("\n前20对详细信息:")
        print("-" * 80)
        print(f"{'序号':<4} {'Y差距':<8} {'X间距':<8} {'块1信息':<30} {'块2信息':<30}")
        print("-" * 80)

        for i, (b1, b2, x_gap, y_diff) in enumerate(found_pairs[:20]):
            info1 = f"X={b1.x:4d} Y={b1.y:4d} W={b1.width:3d} H={b1.height:3d}"
            info2 = f"X={b2.x:4d} Y={b2.y:4d} W={b2.width:3d} H={b2.height:3d}"
            print(f"{i+1:<4} {y_diff:<8.1f} {x_gap:<8} {info1:<30} {info2:<30}")

        print("=" * 80 + "\n")

    def _merge_y_aligned_blocks(self, blocks: List[CharacterBox], gray: np.ndarray) -> List[CharacterBox]:
        """
        合并Y坐标精确对齐的相邻块（v19核心逻辑）

        针对被切开的字：宗、三、些、空、前、共、二、吉
        这些字的特征：
        1. 同一列内
        2. Y坐标完全相同或差异极小（<3px）
        3. X坐标相邻但有间隙（<15px）
        """
        if len(blocks) < 2:
            return blocks

        # 先按X宽松聚类成临时列
        temp_columns = self._cluster_by_x_vertical_loose(blocks)

        merged = []

        for col_blocks in temp_columns:
            if len(col_blocks) <= 1:
                merged.extend(col_blocks)
                continue

            # 按Y排序
            col_blocks.sort(key=lambda b: b.y)

            # 查找Y坐标完全匹配的块对
            col_merged = self._merge_exact_y_aligned(col_blocks, gray)
            merged.extend(col_merged)

        return merged

    def _merge_exact_y_aligned(self, col_blocks: List[CharacterBox], gray: np.ndarray) -> List[CharacterBox]:
        """
        在列内合并Y坐标完全相同的块
        """
        if len(col_blocks) < 2:
            return col_blocks

        result = []
        used = set()

        for i, b1 in enumerate(col_blocks):
            if i in used:
                continue

            merged_with_b1 = None

            for j, b2 in enumerate(col_blocks):
                if i >= j or j in used:
                    continue

                # 检查Y坐标是否完全匹配（差异<3px）
                y_diff = abs(b1.y - b2.y)
                if y_diff < 3:
                    # 检查X间隙
                    if b1.x < b2.x:
                        x_gap = b2.x - (b1.x + b1.width)
                    else:
                        x_gap = b1.x - (b2.x + b2.width)

                    # 间隙<15px，合并
                    if 0 <= x_gap < 15:
                        # 合并
                        used.add(i)
                        used.add(j)

                        min_x = min(b1.x, b2.x)
                        max_x = max(b1.x + b1.width, b2.x + b2.width)
                        min_y = min(b1.y, b2.y)
                        max_y = max(b1.y + b1.height, b2.y + b2.height)

                        merged_image = gray[min_y:max_y, min_x:max_x].copy()

                        merged_with_b1 = CharacterBox(
                            image=merged_image,
                            x=min_x,
                            y=min_y,
                            width=max_x - min_x,
                            height=max_y - min_y,
                        )
                        break

            if merged_with_b1:
                result.append(merged_with_b1)
            else:
                result.append(b1)

        return result

    def _reorganize_vertical_to_horizontal_v2(self, columns: List[List[CharacterBox]], start_line_id: int = 0) -> List[CharacterBox]:
        """
        将竖版列重组为横版行（v2逻辑）

        完全复制v2的行重组逻辑
        """
        # 收集所有字块
        all_blocks = [b for col in columns for b in col]

        # 计算中位字符高度
        median_height = sorted([b.height for b in all_blocks])[len(all_blocks)//2]

        # 行内Y容差：中位高度的80%（v2是80%）
        y_tolerance = median_height * 0.8

        print(f"[DEBUG v8] 中位字高: {median_height}, Y容差: {y_tolerance:.1f}")

        # 按Y坐标分组成行
        lines = []
        current_line = [all_blocks[0]]
        current_y = all_blocks[0].y

        for block in all_blocks[1:]:
            if abs(block.y - current_y) <= y_tolerance:
                current_line.append(block)
                current_y = sum(b.y for b in current_line) / len(current_line)
            else:
                lines.append(current_line)
                current_line = [block]
                current_y = block.y

        if current_line:
            lines.append(current_line)

        print(f"[DEBUG v8] 分成{len(lines)}行")

        # 每行内按X排序（从左到右）
        result = []
        for line_id, line_blocks in enumerate(lines):
            line_blocks.sort(key=lambda b: b.x)
            for char_id, block in enumerate(line_blocks):
                block.line_id = line_id
                block.char_id = char_id
                result.append(block)

        return result

    def _organize_vertical_simple(self, blocks: List[CharacterBox]) -> List[CharacterBox]:
        """
        v7极简版：直接按Y分行，每行内按X排序

        竖版PDF的特征：
        - 同一行的字，Y坐标相近（差异<字高）
        - 不同行的字，Y坐标差异较大

        转横版后：
        - 每行内按X从左到右排序
        - 行按Y从上到下排序
        """
        if not blocks:
            return []

        # 按Y坐标排序
        blocks.sort(key=lambda b: b.y)

        # 计算中位高度和Y容差
        heights = [b.height for b in blocks]
        median_height = sorted(heights)[len(heights) // 2]
        y_tolerance = median_height * 0.6  # 行内Y容差

        print(f"[DEBUG v7] 中位字高: {median_height}, Y容差: {y_tolerance:.1f}")

        # 按Y坐标分组成行
        rows = []
        current_row = [blocks[0]]
        current_y = blocks[0].y

        for block in blocks[1:]:
            if abs(block.y - current_y) <= y_tolerance:
                # 同一行
                current_row.append(block)
                # 更新当前行的Y中心
                current_y = sum(b.y for b in current_row) / len(current_row)
            else:
                # 新行
                rows.append(current_row)
                current_row = [block]
                current_y = block.y

        if current_row:
            rows.append(current_row)

        print(f"[DEBUG v7] 分成{len(rows)}行")
        for i, row in enumerate(rows[:5]):
            row_y = self._avg_y(row)
            print(f"[DEBUG v7] 行{i}: {len(row)}字, Y≈{row_y:.0f}")

        # 每行内按X排序（从左到右），然后输出
        result = []
        for row_idx, row_blocks in enumerate(rows):
            # 按X坐标排序
            row_blocks.sort(key=lambda b: b.x)

            # 更新line_id和char_id
            for char_idx, block in enumerate(row_blocks):
                block.line_id = row_idx
                block.char_id = char_idx
                result.append(block)

        return result

    def _cluster_by_y_for_vertical(self, blocks: List[CharacterBox]) -> List[List[CharacterBox]]:
        """
        按Y坐标分行（竖版专用v6）

        竖版中，同一行的字Y坐标应该相近
        使用较宽松的容差，因为不同列的字可能Y略有差异
        """
        if not blocks:
            return []

        # 按Y排序
        blocks.sort(key=lambda b: b.y)

        # 计算中位高度
        heights = [b.height for b in blocks]
        median_height = sorted(heights)[len(heights) // 2]

        # Y容差：中位高度的80%（宽松一些）
        y_tolerance = median_height * 0.8

        rows = []
        current_row = [blocks[0]]
        current_y = blocks[0].y

        for block in blocks[1:]:
            if abs(block.y - current_y) <= y_tolerance:
                current_row.append(block)
                current_y = sum(b.y for b in current_row) / len(current_row)
            else:
                rows.append(current_row)
                current_row = [block]
                current_y = block.y

        if current_row:
            rows.append(current_row)

        return rows

    def _merge_columns_from_rows(self, rows: List[List[CharacterBox]]) -> List[List[CharacterBox]]:
        """
        从各行的字块中合并出竖列（v6新函数）

        策略：
        1. 收集所有行中的字块
        2. 按X坐标分组成列
        3. 同一列的字X坐标应该相近

        关键：这次已经是按行分组后的结果，所以跨行的字如果X相近
        说明它们属于同一竖列
        """
        if not rows:
            return []

        # 收集所有字块
        all_blocks = [b for row in rows for b in row]

        # 按X坐标排序
        all_blocks.sort(key=lambda b: b.x)

        # 计算中位宽度
        widths = [b.width for b in all_blocks]
        median_width = sorted(widths)[len(widths) // 2]

        # X容差：中位宽度的60%
        x_tolerance = median_width * 0.6

        columns = []
        current_col = [all_blocks[0]]
        current_x = all_blocks[0].center_x

        for block in all_blocks[1:]:
            if abs(block.center_x - current_x) <= x_tolerance:
                current_col.append(block)
                current_x = sum(b.center_x for b in current_col) / len(current_col)
            else:
                columns.append(current_col)
                current_col = [block]
                current_x = block.center_x

        if current_col:
            columns.append(current_col)

        return columns

    def _output_vertical_as_horizontal(self, columns: List[List[CharacterBox]]) -> List[CharacterBox]:
        """
        将竖列输出为横版行（v6新函数）

        策略：
        1. 收集所有字块
        2. 按Y坐标分行
        3. 每行内按X排序（从左到右）
        """
        if not columns:
            return []

        # 收集所有字块
        all_blocks = [b for col in columns for b in col]

        # 计算Y容差
        heights = [b.height for b in all_blocks]
        median_height = sorted(heights)[len(heights) // 2]
        y_tolerance = median_height * 0.5

        # 按Y排序
        all_blocks.sort(key=lambda b: b.y)

        # 分组成行
        rows = []
        current_row = [all_blocks[0]]
        current_y = all_blocks[0].y

        for block in all_blocks[1:]:
            if abs(block.y - current_y) <= y_tolerance:
                current_row.append(block)
                current_y = sum(b.y for b in current_row) / len(current_row)
            else:
                rows.append(current_row)
                current_row = [block]
                current_y = block.y

        if current_row:
            rows.append(current_row)

        # 每行内按X排序，输出
        result = []
        for row_idx, row_blocks in enumerate(rows):
            row_blocks.sort(key=lambda b: b.x)
            for char_idx, block in enumerate(row_blocks):
                block.line_id = row_idx
                block.char_id = char_idx
                result.append(block)

        return result

    def _group_blocks_by_y(
        self,
        blocks: List[CharacterBox],
        block_height: int = 100,
    ) -> List[List[CharacterBox]]:
        """
        按Y坐标分块

        将页面按Y方向切成多个高度为block_height的条带
        每个条带内的字块可以被单独处理为竖版文本
        """
        if not blocks:
            return []

        # 按Y坐标排序
        blocks_sorted = sorted(blocks, key=lambda b: b.y)

        # 找出Y范围
        min_y = blocks_sorted[0].y
        max_y = max(b.y + b.height for b in blocks_sorted)

        # 计算块数
        num_blocks = int((max_y - min_y) / block_height) + 1

        # 初始化块
        y_blocks = [[] for _ in range(num_blocks)]

        # 分配字块到块
        for b in blocks_sorted:
            block_idx = int((b.y - min_y) / block_height)
            if 0 <= block_idx < num_blocks:
                y_blocks[block_idx].append(b)

        # 过滤空块
        return [block for block in y_blocks if block]

    def _reorganize_vertical_to_horizontal(
        self,
        columns: List[List[CharacterBox]],
        start_line_id: int = 0,
    ) -> List[CharacterBox]:
        """
        将竖版列重组为横版行（v5修复版）

        关键发现：v4的矩阵转置假设每列长度相同，但实际竖版PDF中：
        - 有的列只有几个字（如标题"序"）
        - 有的列有几十个字（如正文）
        - 直接用col[row_idx]会导致错位！

        v5新策略：基于Y坐标分行
        1. 收集所有字块，记录它们属于哪一列
        2. 按Y坐标相近的字块聚合成同一行
        3. 每行内按X排序（从左到右）

        这样处理列长度不一致的情况。
        """
        if not columns:
            return []

        # 给每个字块标记所属列（用于后续排序）
        all_blocks = []
        for col_idx, col in enumerate(columns):
            for block in col:
                # 保存列索引，以便后续按列顺序排序
                block._col_idx = col_idx
                all_blocks.append(block)

        if not all_blocks:
            return []

        # 计算中位字符高度，用于Y容差
        heights = [b.height for b in all_blocks]
        median_height = sorted(heights)[len(heights) // 2]
        y_tolerance = median_height * 0.5  # 行内Y差异不超过半个字高

        # 按Y坐标聚类成行
        all_blocks.sort(key=lambda b: b.y)  # 按Y排序

        rows = []
        current_row = [all_blocks[0]]
        current_y = all_blocks[0].y

        for block in all_blocks[1:]:
            if abs(block.y - current_y) <= y_tolerance:
                # 同一行
                current_row.append(block)
                # 更新当前行的Y中心
                current_y = sum(b.y for b in current_row) / len(current_row)
            else:
                # 新行
                rows.append(current_row)
                current_row = [block]
                current_y = block.y

        if current_row:
            rows.append(current_row)

        # 每行内按X排序（从左到右）
        result = []
        for row_idx, row_blocks in enumerate(rows):
            # 按X坐标排序（从左到右）
            row_blocks.sort(key=lambda b: b.x)

            # 更新line_id和char_id
            for char_idx, block in enumerate(row_blocks):
                block.line_id = start_line_id + row_idx
                block.char_id = char_idx
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
        # v27：先轻微腐蚀分离粘连，再水平膨胀连接
        kernel_small = np.ones((2, 2), np.uint8)
        binary_image = cv2.erode(binary_image, kernel_small, iterations=1)

        # 水平方向膨胀，用1x4核
        kernel_h = np.ones((1, 4), np.uint8)
        binary_image = cv2.dilate(binary_image, kernel_h, iterations=1)

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
        strips_per_row = 8  # 微调v9：从15减少到8，避免把单个字从中间切断

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
        v12稳定版：3x3核，膨胀2次，腐蚀1次
        """
        # 3x3核，膨胀2次，腐蚀1次（v12稳定参数）
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        return eroded

    def _merge_split_characters(self, blocks: List[CharacterBox], gray: np.ndarray) -> List[CharacterBox]:
        """
        合并被切开的字（v14新增）

        判断标准：两个字块如果Y相近且X相邻，可能是同一个字被切开
        例如："序"字被切成左右两部分
        """
        if len(blocks) < 2:
            return blocks

        # 按Y坐标排序
        blocks.sort(key=lambda b: b.y)

        # 计算中位高度和宽度
        heights = [b.height for b in blocks]
        widths = [b.width for b in blocks]
        median_h = sorted(heights)[len(heights) // 2]
        median_w = sorted(widths)[len(widths) // 2]

        # Y容差：中位高度的30%（同一行的字）
        y_tolerance = median_h * 0.3

        # X间距阈值：中位宽度的50%（同一字的两部分）
        x_gap_threshold = median_w * 0.8

        merged = []
        skip_indices = set()

        for i, block1 in enumerate(blocks):
            if i in skip_indices:
                continue

            best_match = None
            best_gap = float('inf')

            # 查找可能的合并对象
            for j, block2 in enumerate(blocks):
                if i == j or j in skip_indices:
                    continue

                # 检查Y是否相近（同一行）
                y_diff = abs(block1.y - block2.y)
                if y_diff > y_tolerance:
                    continue

                # 检查X是否相邻
                if block1.x < block2.x:  # block1在左，block2在右
                    gap = block2.x - (block1.x + block1.width)
                else:  # block1在右，block2在左
                    gap = block1.x - (block2.x + block2.width)

                # 间距太小，可能是同一个字
                if gap < x_gap_threshold and gap >= -5:  # 允许轻微重叠
                    if gap < best_gap:
                        best_gap = gap
                        best_match = (j, block2)

            # 如果找到匹配，合并
            if best_match:
                j, block2 = best_match
                skip_indices.add(j)

                # 合并两个字块
                min_x = min(block1.x, block2.x)
                max_x = max(block1.x + block1.width, block2.x + block2.width)
                min_y = min(block1.y, block2.y)
                max_y = max(block1.y + block1.height, block2.y + block2.height)

                merged_width = max_x - min_x
                merged_height = max_y - min_y

                # 从灰度图提取合并后的图像
                merged_image = gray[min_y:max_y, min_x:max_x].copy()

                merged.append(CharacterBox(
                    image=merged_image,
                    x=min_x,
                    y=min_y,
                    width=merged_width,
                    height=merged_height
                ))
            else:
                merged.append(block1)

        return merged

    def _morphology_process_vertical(self, binary: np.ndarray) -> np.ndarray:
        """
        形态学处理：膨胀-腐蚀（竖版专用v18 - 弱膨胀版本）

        v18策略：提供弱/强两个版本
        - 弱版本：保持列独立性，顺序正确
        - 强版本：连接被切开的字
        """
        return self._morphology_process_vertical_weak(binary)

    def _morphology_process_vertical_weak(self, binary: np.ndarray) -> np.ndarray:
        """
        弱膨胀版本（v21：增强水平连接）

        v20结果：共、空正常 ✅，宗、二、三、些、吉仍被切开

        v21策略：增强水平连接
        1. 第一阶段：2x2全方向核，膨胀1次（保持）
        2. 第二阶段：1x7水平核，膨胀3次（增强：1x5→1x7，2次→3次）
        """
        # 第一阶段：2x2全方向核，连接笔画间隙
        kernel_small = np.ones((2, 2), np.uint8)
        stage1 = cv2.dilate(binary, kernel_small, iterations=1)

        # 第二阶段：1x7水平核，膨胀3次（增强连接）
        kernel_h = np.ones((1, 7), np.uint8)
        dilated = cv2.dilate(stage1, kernel_h, iterations=3)

        # 轻微腐蚀恢复形状
        kernel_v = np.ones((3, 1), np.uint8)
        eroded = cv2.erode(dilated, kernel_v, iterations=1)

        return eroded

    def _morphology_process_vertical_strong(self, binary: np.ndarray) -> np.ndarray:
        """
        强膨胀版本（v18新增）
        - 1x7水平核
        - 膨胀4次
        - 更强水平连接，能连接被切开的字
        """
        kernel_h = np.ones((1, 7), np.uint8)
        dilated = cv2.dilate(binary, kernel_h, iterations=4)
        kernel_v = np.ones((3, 1), np.uint8)
        eroded = cv2.erode(dilated, kernel_v, iterations=1)
        return eroded

    def _merge_dual_extraction_results(
        self,
        blocks_weak: List[CharacterBox],
        blocks_strong: List[CharacterBox],
        gray: np.ndarray
    ) -> List[CharacterBox]:
        """
        合并弱膨胀和强膨胀的提取结果（v18核心逻辑）

        策略：
        1. 强膨胀结果中，如果一个块包含多个弱膨胀块，说明这些弱块应该合并
        2. 使用强膨胀块的位置，从灰度图提取完整图像
        3. 保持弱膨胀的顺序结构
        """
        if not blocks_weak:
            return blocks_strong
        if not blocks_strong:
            return blocks_weak

        result = []
        used_weak_indices = set()

        # 遍历强膨胀块
        for strong_block in blocks_strong:
            # 找出被这个强块包含的所有弱块
            contained_weak = []
            for i, weak_block in enumerate(blocks_weak):
                if i in used_weak_indices:
                    continue

                # 检查弱块是否在强块内（中心点判断）
                weak_center_x = weak_block.center_x
                weak_center_y = weak_block.center_y

                if (strong_block.x <= weak_center_x <= strong_block.x + strong_block.width and
                    strong_block.y <= weak_center_y <= strong_block.y + strong_block.height):
                    contained_weak.append((i, weak_block))

            if len(contained_weak) >= 2:
                # 强块包含多个弱块，说明这些弱块应该合并
                # 使用强块的位置
                merged_image = gray[
                    strong_block.y:strong_block.y + strong_block.height,
                    strong_block.x:strong_block.x + strong_block.width
                ].copy()

                result.append(CharacterBox(
                    image=merged_image,
                    x=strong_block.x,
                    y=strong_block.y,
                    width=strong_block.width,
                    height=strong_block.height,
                ))

                # 标记这些弱块已使用
                for i, _ in contained_weak:
                    used_weak_indices.add(i)
            elif len(contained_weak) == 1:
                # 强块只包含一个弱块，使用弱块（保持精度）
                _, weak_block = contained_weak[0]
                result.append(weak_block)
                used_weak_indices.add(contained_weak[0][0])

        # 添加未被强块包含的弱块
        for i, weak_block in enumerate(blocks_weak):
            if i not in used_weak_indices:
                result.append(weak_block)

        return result

    def _merge_split_characters_v17(self, blocks: List[CharacterBox], gray: np.ndarray) -> List[CharacterBox]:
        """
        合并被切开的字（v17优化版）

        v17策略：
        1. 先按X聚类成临时列（不改变原始顺序）
        2. 在每列内，查找Y相邻且大小相似的小块
        3. 合并这些小块

        关键：只在列内合并，不跨列，避免破坏顺序
        """
        if len(blocks) < 2:
            return blocks

        # 先按X聚类成临时列（使用较宽松的阈值）
        temp_columns = self._cluster_by_x_vertical_loose(blocks)

        merged_blocks = []

        for col_blocks in temp_columns:
            if len(col_blocks) == 1:
                merged_blocks.extend(col_blocks)
                continue

            # 在列内按Y排序
            col_blocks.sort(key=lambda b: b.y)

            # 在列内查找并合并相邻的小块
            col_merged = self._merge_adjacent_in_column(col_blocks, gray)
            merged_blocks.extend(col_merged)

        return merged_blocks

    def _cluster_by_x_vertical_loose(self, blocks: List[CharacterBox]) -> List[List[CharacterBox]]:
        """
        按X坐标聚类成列（宽松版，用于合并前的预处理）

        使用更宽松的阈值，确保被切开的部分被归为同一列
        """
        if not blocks:
            return []

        if len(blocks) < 3:
            return [blocks]

        # 按X坐标排序
        blocks_sorted = sorted(blocks, key=lambda b: b.center_x)

        # 使用较宽松的阈值（30px）
        threshold = 30

        groups = []
        current_group = [blocks_sorted[0]]
        current_x = blocks_sorted[0].center_x

        for block in blocks_sorted[1:]:
            if abs(block.center_x - current_x) <= threshold:
                current_group.append(block)
                current_x = sum(b.center_x for b in current_group) / len(current_group)
            else:
                groups.append(current_group)
                current_group = [block]
                current_x = block.center_x

        if current_group:
            groups.append(current_group)

        return groups

    def _merge_adjacent_in_column(self, col_blocks: List[CharacterBox], gray: np.ndarray) -> List[CharacterBox]:
        """
        在列内合并相邻的小块

        判断标准：
        1. Y坐标相邻（间距<中位高度的50%）
        2. 两个块都比较小（可能是被切开的部分）
        """
        if len(col_blocks) < 2:
            return col_blocks

        # 计算列内的中位高度
        heights = [b.height for b in col_blocks]
        median_h = sorted(heights)[len(heights) // 2]

        # Y间距阈值：中位高度的50%
        y_gap_threshold = median_h * 0.5

        # 小块阈值：宽度或高度小于中位值的70%
        widths = [b.width for b in col_blocks]
        median_w = sorted(widths)[len(widths) // 2]
        small_threshold = 0.7

        result = []
        used = set()

        for i, b1 in enumerate(col_blocks):
            if i in used:
                continue

            # 查找下一个块
            if i + 1 < len(col_blocks):
                b2 = col_blocks[i + 1]

                if (i + 1) not in used:
                    # 计算Y间距
                    y_gap = b2.y - (b1.y + b1.height)

                    # 检查是否相邻
                    if 0 <= y_gap <= y_gap_threshold:
                        # 检查是否都是小块（至少有一个是小块）
                        b1_small = b1.width < median_w * small_threshold or b1.height < median_h * small_threshold
                        b2_small = b2.width < median_w * small_threshold or b2.height < median_h * small_threshold

                        if b1_small or b2_small:
                            # 合并这两个块
                            used.add(i)
                            used.add(i + 1)

                            min_x = min(b1.x, b2.x)
                            max_x = max(b1.x + b1.width, b2.x + b2.width)
                            min_y = min(b1.y, b2.y)
                            max_y = max(b1.y + b1.height, b2.y + b2.height)

                            merged_image = gray[min_y:max_y, min_x:max_x].copy()

                            result.append(CharacterBox(
                                image=merged_image,
                                x=min_x,
                                y=min_y,
                                width=max_x - min_x,
                                height=max_y - min_y,
                            ))
                            continue

            # 没有合并，保留原块
            result.append(b1)

        return result

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
        """提取字块（横版参数 - v11：增加标点符号过滤）"""
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

            # 过滤太小的噪点（微调v10：降低面积阈值到20，保留"灭"、"清"、"以"等小字块）
            if area < 20:  # v7: 30 → v10: 20
                continue
            if w < 2 and h < 2:  # v7: 3 → v10: 2
                continue

            # 过滤太大的（可能是边框、图片、大块污渍等）
            # 加强过滤：宽度>30%或高度>20%或面积>10%都过滤
            page_area = binary.shape[0] * binary.shape[1]
            if w > binary.shape[1] * 0.3 or h > binary.shape[0] * 0.2 or area > page_area * 0.1:
                continue

            # v11新增：过滤标点符号（引号、书名号等）
            # 标点符号特征：宽高比极端（非常细长或非常扁平）
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 5 or aspect_ratio < 0.2:  # 宽高比>5或<0.2，可能是标点
                # 但如果是扁字（如"一"），保留
                if not (w > h * 3):  # 不是扁字
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

        # 分割放大后会超宽的字块
        blocks = self._split_oversized_blocks(blocks, gray)

        # 去重：移除重叠的字块（保留面积较大的）
        blocks = self._remove_overlapping_blocks(blocks)

        return blocks

    def _split_oversized_blocks(self, blocks: List[CharacterBox], gray: np.ndarray) -> List[CharacterBox]:
        """分割放大后会超宽的字块（如《唐诗鉴赏辞典》）"""
        if not blocks:
            return blocks

        scale_factor = 3.0  # 放大3倍
        page_width = gray.shape[1]
        result = []

        for block in blocks:
            # 计算放大后的宽度
            scaled_width = block.width * scale_factor

            # 如果放大后超过页面宽度的60%，分割
            if scaled_width > page_width * 0.6:
                # 按每字约50像素估算，分成多份
                num_splits = max(3, int(block.width / 50))
                split_blocks = self._split_block_multiple(block, gray, num_splits)
                if split_blocks:
                    result.extend(split_blocks)
                else:
                    result.append(block)
            else:
                result.append(block)

        return result

    def _split_block_multiple(self, block: CharacterBox, gray: np.ndarray, num_splits: int) -> List[CharacterBox]:
        """将字块等分成多份"""
        if num_splits < 2:
            return []

        split_width = block.width // num_splits
        result = []

        for i in range(num_splits):
            start_x = block.x + i * split_width
            end_x = start_x + split_width if i < num_splits - 1 else block.x + block.width

            width = end_x - start_x
            if width < 10:
                continue

            block_image = gray[block.y:block.y+block.height, start_x:end_x].copy()
            result.append(CharacterBox(
                image=block_image,
                x=start_x,
                y=block.y,
                width=width,
                height=block.height,
            ))

        return result

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
            # 扁字块：如果X距离小于20像素，认为是重复（大幅降低阈值）
            if x_diff < 20:
                return True

        # v12微调：移除激进的X距离检查，避免把"灭"、"引"、"清"等字的两部分误判为重复
        # 只检查真正的重叠，不检查相邻

        # 检查是否有重叠
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

    def _extract_blocks_for_vertical_v15(
        self,
        binary: np.ndarray,
        gray: np.ndarray,
    ) -> List[CharacterBox]:
        """
        提取字块（v15版：合并被切开的字）

        策略：
        1. 先提取所有字块
        2. 对每个字块，检查是否有相邻的小字块（可能是同一个字被切开）
        3. 如果有，合并它们
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # 先提取所有字块
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

            # 过滤太大的
            if w > binary.shape[1] * 0.8 or h > binary.shape[0] * 0.8:
                continue

            block_image = gray[y:y+h, x:x+w].copy()

            blocks.append(CharacterBox(
                image=block_image,
                x=x,
                y=y,
                width=w,
                height=h,
            ))

        if not blocks:
            return blocks

        # v15关键：合并被切开的字
        # 只合并明显是同一个字的小块：X相邻且Y相近
        blocks = self._merge_adjacent_small_blocks(blocks, gray)

        return blocks

    def _merge_adjacent_small_blocks(self, blocks: List[CharacterBox], gray: np.ndarray) -> List[CharacterBox]:
        """
        合并被切开的字（v15优化版）

        关键：只合并真正是同一个字的小块
        - 两个块都较小（宽度或高度小于中位值的80%）
        - X坐标相邻（间距小于10像素）
        - Y坐标相近（差异小于高度的30%）
        """
        if len(blocks) < 2:
            return blocks

        # 计算中位宽度和高度
        widths = [b.width for b in blocks]
        heights = [b.height for b in blocks]
        median_w = sorted(widths)[len(widths) // 2]
        median_h = sorted(heights)[len(heights) // 2]

        merged = []
        used = set()

        for i, b1 in enumerate(blocks):
            if i in used:
                continue

            # 只处理较小的块（可能是被切开的部分）
            if b1.width > median_w * 0.8 and b1.height > median_h * 0.8:
                merged.append(b1)
                continue

            # 查找可能的合并对象
            for j, b2 in enumerate(blocks):
                if i >= j or j in used:
                    continue

                # 只处理较小的块
                if b2.width > median_w * 0.8 and b2.height > median_h * 0.8:
                    continue

                # 检查Y是否相近
                y_diff = abs(b1.y - b2.y)
                if y_diff > min(b1.height, b2.height) * 0.4:
                    continue

                # 检查X是否相邻
                if b1.x < b2.x:
                    gap = b2.x - (b1.x + b1.width)
                else:
                    gap = b1.x - (b2.x + b2.width)

                # 间距很小（小于10像素）且没有重叠太多
                if -5 < gap < 12:
                    # 合并这两个块
                    used.add(i)
                    used.add(j)

                    min_x = min(b1.x, b2.x)
                    max_x = max(b1.x + b1.width, b2.x + b2.width)
                    min_y = min(b1.y, b2.y)
                    max_y = max(b1.y + b1.height, b2.y + b2.height)

                    merged_image = gray[min_y:max_y, min_x:max_x].copy()

                    merged.append(CharacterBox(
                        image=merged_image,
                        x=min_x,
                        y=min_y,
                        width=max_x - min_x,
                        height=max_y - min_y,
                    ))
                    break

            # 如果没有合并，添加原块
            if i not in used:
                merged.append(b1)

        return merged

    def _extract_blocks_for_vertical(
        self,
        binary: np.ndarray,
        gray: np.ndarray,
    ) -> List[CharacterBox]:
        """提取字块（v28：过滤疑似黑点的块）"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        print("\n=== v28 调试：过滤疑似黑点 ===")

        blocks = []
        all_blocks_info = []
        filtered_dots = []

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

            # 过滤太大的
            if w > binary.shape[1] * 0.8 or h > binary.shape[0] * 0.8:
                continue

            # 记录所有块信息
            aspect_ratio = w / h if h > 0 else 0

            block_info = {
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area, 'aspect': aspect_ratio
            }
            all_blocks_info.append(block_info)

            # v28：过滤疑似黑点
            # 特征：宽度较小(30-40)，高度很小(10-20)，宽高比>2
            is_dot = False
            if 30 <= w <= 40 and h <= 20 and aspect_ratio > 2.0:
                # 进一步检查：如果是每列Y最小的块，可能是黑点
                filtered_dots.append(block_info)
                is_dot = True

            if is_dot:
                print(f"过滤疑似黑点: X={x:4d} Y={y:4d} W={w:2d} H={h:2d} A={area:3d} 宽高比={aspect_ratio:.2f}")
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

        print(f"\n总块数(过滤后): {len(blocks)}")
        print(f"过滤的疑似黑点数: {len(filtered_dots)}")
        print("=" * 80 + "\n")

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
        """按x坐标聚类成列（竖版专用）

        竖版文本特征：
        - 同一列的字块X坐标非常接近（差异通常<20px）
        - 不同列之间有明显的间隔

        策略：
        1. 先用较小的阈值（20px）将字块按X粗分组
        2. 合并相邻的组（如果组间距离也较小）
        3. 过滤掉太小的组（可能是噪声）
        """
        if not blocks:
            return []

        if len(blocks) < 3:
            return [blocks]

        # 按X坐标排序
        blocks_sorted = sorted(blocks, key=lambda b: b.center_x)

        # 使用两阶段聚类
        # 第一阶段：用小阈值（15px）粗分组
        initial_threshold = 15  # 同一列的字块X差异通常<15px
        initial_groups = []
        current_group = [blocks_sorted[0]]
        current_x = blocks_sorted[0].center_x

        for block in blocks_sorted[1:]:
            if abs(block.center_x - current_x) <= initial_threshold:
                current_group.append(block)
                current_x = sum(b.center_x for b in current_group) / len(current_group)
            else:
                initial_groups.append(current_group)
                current_group = [block]
                current_x = block.center_x

        if current_group:
            initial_groups.append(current_group)

        # 第二阶段：合并相邻的组（如果组间距离较小）
        # 计算各组的中心X
        group_centers = [sum(b.center_x for b in g) / len(g) for g in initial_groups]
        group_gaps = []
        for i in range(1, len(group_centers)):
            group_gaps.append(group_centers[i] - group_centers[i-1])

        # 找出典型的列间间隔
        if group_gaps:
            group_gaps_sorted = sorted(group_gaps)
            # 使用中位数作为参考
            median_gap = group_gaps_sorted[len(group_gaps_sorted)//2]

            # 合并阈值：如果两组间隔小于典型间隔的60%，合并它们
            merge_threshold = median_gap * 0.6

            columns = []
            current_col = initial_groups[0]

            for i in range(1, len(initial_groups)):
                if group_gaps[i-1] < merge_threshold:
                    # 合并到当前列
                    current_col.extend(initial_groups[i])
                else:
                    columns.append(current_col)
                    current_col = initial_groups[i]

            if current_col:
                columns.append(current_col)
        else:
            columns = initial_groups

        # 过滤掉只有极少字符的列
        if len(blocks) > 20:
            columns = [col for col in columns if len(col) >= 2]

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
