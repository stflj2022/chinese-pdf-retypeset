from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
from src.config import OutputConfig

class OutputGenerator:
    def __init__(self, config: OutputConfig):
        self.config = config

    def save(self, pages: List[np.ndarray], output_path: Path):
        if not pages:
            return

        # Convert numpy arrays (gray) to PIL Images
        pil_images = []
        for p in pages:
            img = Image.fromarray(p)
            pil_images.append(img)

        if self.config.format == "pdf":
            # Save as PDF
            if output_path.suffix.lower() != ".pdf":
                output_path = output_path.with_suffix(".pdf")
            
            pil_images[0].save(
                output_path,
                "PDF",
                resolution=100.0,
                save_all=True,
                append_images=pil_images[1:]
            )
        else:
            # Save as images
            output_dir = output_path.parent / output_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(pil_images):
                img.save(output_dir / f"page_{i+1:03d}.png")

