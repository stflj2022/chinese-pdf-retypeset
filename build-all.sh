#!/bin/bash
# 全平台打包脚本

set -e

PROJECT_NAME="chinese-pdf-retypeset"
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "v3.0")
BUILD_DIR="build-all"
DIST_DIR="$BUILD_DIR/dist"

echo "🚀 开始打包 $PROJECT_NAME $VERSION"
echo "================================"

# 清理旧构建
rm -rf "$BUILD_DIR"
mkdir -p "$DIST_DIR"

# 先创建dist目录（如果不存在）
mkdir -p dist

# 当前平台检测
OS=$(uname -s)
ARCH=$(uname -m)

echo "📦 当前平台: $OS $ARCH"

# Linux
if [[ "$OS" == "Linux" ]]; then
    echo "🐧 打包 Linux 版本..."

    # 1. PyInstaller 单文件
    pyinstaller pdf-retypeset.spec --clean

    # 2. 创建 AppImage（如果存在 appimage-builder）
    if command -v appimage-builder &> /dev/null; then
        echo "📦 创建 AppImage..."
        # 需要创建 AppImage 配置文件
        # appimage-builder --recipe AppImageBuilder.yml
    fi

    # 3. 创建便携版 tar.gz
    echo "📦 创建便携版..."
    cd dist
    tar -czf "$DIST_DIR/${PROJECT_NAME}-${VERSION}-${ARCH}-linux.tar.gz" pdf-retypeset
    cd ..

    echo "✅ Linux 打包完成"
    echo "   - 单文件: dist/pdf-retypeset"
    echo "   - 便携包: $DIST_DIR/${PROJECT_NAME}-${VERSION}-${ARCH}-linux.tar.gz"
fi

# macOS
if [[ "$OS" == "Darwin" ]]; then
    echo "🍎 打包 macOS 版本..."
    pyinstaller pdf-retypeset.spec --clean

    # 创建 .app bundle
    # 需要 macOS 特定配置

    cd dist
    tar -czf "$DIST_DIR/${PROJECT_NAME}-${VERSION}-${ARCH}-macos.tar.gz" pdf-retypeset
    cd ..

    echo "✅ macOS 打包完成"
fi

# Windows (Git Bash / MSYS2)
if [[ "$OS" == "MINGW"* ]] || [[ "$OS" == "MSYS"* ]]; then
    echo "🪟 打包 Windows 版本..."
    pyinstaller pdf-retypeset.spec --clean

    cd dist
    zip -r "$DIST_DIR/${PROJECT_NAME}-${VERSION}-windows.zip" pdf-retypeset.exe
    cd ..

    echo "✅ Windows 打包完成"
fi

echo ""
echo "🎉 所有构建完成！"
echo "📁 输出目录: $DIST_DIR"
ls -lh "$DIST_DIR"
