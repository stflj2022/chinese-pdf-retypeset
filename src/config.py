from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

@dataclass
class PreprocessingConfig:
    threshold: int = 127
    denoise: bool = True

@dataclass
class SegmentationConfig:
    # 垂直分割参数
    char_min_width: int = 5
    char_min_height: int = 5
    char_max_width_ratio: float = 0.8
    char_max_height_ratio: float = 0.8

@dataclass
class OutputLayoutConfig:
    page_size: str = "A4"  # A4, A5, B5, Letter
    dpi: int = 300
    margin_x: int = 20  # mm
    margin_y: int = 20  # mm
    char_spacing: int = 10  # pixels at 300dpi? Or relative?
    line_spacing: int = 20
    scale_factor: float = 1.5
    font_size: int = 0  # 0 means auto-scale from source image

@dataclass
class OutputConfig:
    format: Literal["pdf", "images"] = "pdf"
    output_dir: Path = field(default_factory=lambda: Path("output"))

@dataclass
class LayoutConfig:
    direction: str = "vertical_rtl"  # input layout
    output: OutputLayoutConfig = field(default_factory=OutputLayoutConfig)

@dataclass
class InputConfig:
    dpi: int = 300

@dataclass
class Config:
    input: InputConfig = field(default_factory=InputConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

def load_config(path: Path = None) -> Config:
    # Simplified loader for now
    return Config()
