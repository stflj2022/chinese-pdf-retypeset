#!/bin/bash
# PDF智能重排工具启动脚本

cd "$(dirname "$0")"

# 检查依赖
if ! python3 -c "import cv2, numpy, pdf2image, PIL" 2>/dev/null; then
    echo "正在安装Python依赖..."
    pip3 install -r requirements.txt
fi

# 杀掉旧的GUI进程（如果存在）
pkill -f "python.*run.py" 2>/dev/null
sleep 0.5

# 启动GUI
echo "启动PDF智能重排工具..."
python3 run.py
