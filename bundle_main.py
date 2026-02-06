#!/usr/bin/env python3
"""
PDF 重排工具 - 打包入口文件

这个文件用于 PyInstaller 打包，避免相对导入问题。
"""

import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# 导入并运行主程序
from main import main as _main
if __name__ == "__main__":
    _main()
