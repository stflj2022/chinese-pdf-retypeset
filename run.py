#!/usr/bin/env python3
"""
PDF 重排工具 - 入口文件
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入并运行
from src.main import main

if __name__ == "__main__":
    main()
