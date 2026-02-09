import cv2
import numpy as np
from typing import List, Tuple
from src.config import LayoutConfig
from src.segmenter import CharacterBox

class Layouter:
    def __init__(self, config: LayoutConfig):
        self.config = config
        self.page_width, self.page_height = self._get_page_size()
        
    def _get_page_size(self) -> Tuple[int, int]:
        dpi = self.config.output.dpi
        size_map = {
            "A4": (8.27, 11.69),
            "A5": (5.83, 8.27),
            "B5": (6.93, 9.84),
            "Letter": (8.5, 11.0)
        }
        w_inch, h_inch = size_map.get(self.config.output.page_size, (8.27, 11.69))
        return int(w_inch * dpi), int(h_inch * dpi)

    def layout(self, characters: List[CharacterBox]) -> List[np.ndarray]:
        if not characters:
            return []

        pages = []
        current_page = self._create_blank_page()

        margin_x = int(self.config.output.margin_x * self.config.output.dpi / 25.4)
        margin_y = int(self.config.output.margin_y * self.config.output.dpi / 25.4)

        cursor_x = margin_x
        cursor_y = margin_y

        # Char spacing and line spacing
        char_spacing = self.config.output.char_spacing
        line_spacing = self.config.output.line_spacing

        current_line_max_h = 0

        for char in characters:
            # Resize character image
            h, w = char.image.shape[:2]
            scale = self.config.output.scale_factor
            new_h, new_w = int(h * scale), int(w * scale)

            # If char is empty or weird
            if new_h <= 0 or new_w <= 0:
                continue

            resized_img = cv2.resize(char.image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Check if fits in line
            if cursor_x + new_w > self.page_width - margin_x:
                # New line
                cursor_x = margin_x
                cursor_y += current_line_max_h + line_spacing
                current_line_max_h = 0

                # Check if fits in page
                if cursor_y + new_h > self.page_height - margin_y:
                    # New page
                    pages.append(current_page)
                    current_page = self._create_blank_page()
                    cursor_y = margin_y

            # Place character
            # Ensure we don't go out of bounds (bottom edge)
            if cursor_y + new_h > self.page_height - margin_y:
                 pages.append(current_page)
                 current_page = self._create_blank_page()
                 cursor_y = margin_y
                 cursor_x = margin_x
                 current_line_max_h = 0

            # Copy image to page (handling boundaries just in case)
            h_place = min(new_h, self.page_height - cursor_y)
            w_place = min(new_w, self.page_width - cursor_x)

            if h_place > 0 and w_place > 0:
                # Invert if source is white-on-black?
                # Usually source is black text on white (gray 255).
                # If source is binary 0/255 (0=black), we can just copy.
                # If we want to make background white and text black:
                # We assume the input char image is proper (text is dark).
                # If binary image was inverted during processing, we might need to check.
                # Assuming standard gray image where text is dark.

                # We can do a direct copy or a mask copy.
                # Direct copy is simpler for now.
                # But to look good, we might want to multiply?
                # Let's just copy the pixel values.

                # If the background is white (255), we want to preserve it?
                # Actually, standard paste is fine.
                current_page[cursor_y:cursor_y+h_place, cursor_x:cursor_x+w_place] = resized_img[:h_place, :w_place]

            # Update cursor
            cursor_x += new_w + char_spacing
            current_line_max_h = max(current_line_max_h, new_h)
            
        pages.append(current_page)
        return pages

    def _create_blank_page(self) -> np.ndarray:
        # Create white image
        return np.full((self.page_height, self.page_width), 255, dtype=np.uint8)
