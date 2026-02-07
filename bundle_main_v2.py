#!/usr/bin/env python3
"""
PDF 重排工具 - 打包专用入口文件 v2

使用 __import__ 动态加载来避免相对导入问题。
"""

import sys
from pathlib import Path

# 添加当前目录到 Python 路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 设置包路径
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

# 动态加载主模块
import importlib.util
spec = importlib.util.spec_from_file_location("main", str(src_path / "main.py"))
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)

# 运行主函数
if __name__ == "__main__":
    main_module.main()
