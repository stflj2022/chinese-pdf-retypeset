"""
阅读顺序排序模块

支持横排和竖排文档的排序。
"""

from typing import List

from src.segmenter import CharacterBox, Segmenter


class Sorter:
    """排序器"""

    def __init__(self, layout_mode: str = "horizontal"):
        """
        Args:
            layout_mode: 布局模式
                - "horizontal": 横排（从左到右，从上到下）
                - "vertical_rtl": 竖排从右到左（古籍）
                - "vertical_ltr": 竖排从左到右
        """
        self.layout_mode = layout_mode

    def sort(
        self, chars: List[CharacterBox], segmenter: Segmenter
    ) -> List[CharacterBox]:
        """
        对字符按阅读顺序排序

        Args:
            chars: 字符列表
            segmenter: 分割器（用于行/列检测）

        Returns:
            排序后的字符列表
        """
        if not chars:
            return []

        if self.layout_mode == "horizontal":
            # 横排：检测行，每行从左到右
            lines = segmenter.detect_lines_horizontal(chars)
            return segmenter.flatten_to_reading_order(lines)

        elif self.layout_mode == "vertical_rtl":
            # 竖排从右到左：检测列，每列从上到下
            columns = segmenter.detect_columns_vertical(chars)
            return segmenter.flatten_to_reading_order(columns)

        elif self.layout_mode == "vertical_ltr":
            # 竖排从左到右：检测列，反转列顺序
            columns = segmenter.detect_columns_vertical(chars)
            columns.reverse()
            # 重新编号
            for col_id, col in enumerate(columns):
                for char in col:
                    char.line_id = col_id
            return segmenter.flatten_to_reading_order(columns)

        else:
            # 默认横排
            lines = segmenter.detect_lines_horizontal(chars)
            return segmenter.flatten_to_reading_order(lines)
