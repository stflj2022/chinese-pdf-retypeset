"""
命令行入口

提供CLI接口和GUI启动入口。
"""

import argparse
import sys
from pathlib import Path

from src.config import Config, load_config
from src.pipeline import Pipeline


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="扫描版PDF智能重排工具（自动检测横版/竖版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用GUI
  python -m src.main --gui

  # 命令行处理（自动检测方向）
  python -m src.main input.pdf output.pdf

  # 使用自定义配置
  python -m src.main input.pdf output.pdf --config config.yaml

  # 调整参数
  python -m src.main input.pdf output.pdf --scale 1.5 --dpi 300
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="输入PDF或图像文件路径",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="输出文件路径",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="启动图形界面",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="配置文件路径",
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="字符放大倍数 (默认: 1.5)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        help="输入DPI (默认: 300)",
    )
    parser.add_argument(
        "--char-spacing",
        type=int,
        help="字符间距 (默认: 10)",
    )
    parser.add_argument(
        "--line-spacing",
        type=int,
        help="行间距 (默认: 20)",
    )
    parser.add_argument(
        "--page-size",
        choices=["A4", "A5", "Letter", "B5"],
        help="输出页面大小 (默认: A4)",
    )
    parser.add_argument(
        "--output-format",
        choices=["pdf", "images"],
        help="输出格式 (默认: pdf)",
    )

    args = parser.parse_args()

    # 启动GUI
    if args.gui:
        from src.gui import run_gui
        run_gui()
        return

    # 命令行模式需要输入和输出参数
    if not args.input or not args.output:
        # 如果没有参数，默认启动GUI
        if len(sys.argv) == 1:
            from src.gui import run_gui
            run_gui()
            return
            
        parser.print_help()
        print("\n错误: 命令行模式需要指定输入和输出文件路径")
        sys.exit(1)

    # 加载配置
    config = load_config(args.config)

    # 应用命令行参数
    if args.scale:
        config.layout.output.scale_factor = args.scale
    if args.dpi:
        config.input.dpi = args.dpi
        config.layout.output.dpi = args.dpi
    if args.char_spacing:
        config.layout.output.char_spacing = args.char_spacing
    if args.line_spacing:
        config.layout.output.line_spacing = args.line_spacing
    if args.page_size:
        config.layout.output.page_size = args.page_size
    if args.output_format:
        config.output.format = args.output_format

    # 创建流水线并处理
    pipeline = Pipeline(config)

    # 设置进度回调
    def progress_callback(current: int, total: int, message: str):
        print(f"[{current}/{total}] {message}")

    pipeline.set_progress_callback(progress_callback)

    # 处理
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    try:
        pipeline.process(input_path, output_path)
        print(f"处理完成: {output_path}")
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
