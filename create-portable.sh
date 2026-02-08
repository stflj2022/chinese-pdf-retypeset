#!/bin/bash
# 创建便携版启动包

set -e

VERSION="v3.2"
APP_NAME="chinese-pdf-retypeset"
BUILD_DIR="portable-release"

echo "🚀 创建便携版启动包..."

# 清理旧构建
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/$APP_NAME"

# 复制可执行文件
echo "📦 复制可执行文件..."
if [ -f "dist/$APP_NAME" ]; then
    cp "dist/$APP_NAME" "$BUILD_DIR/$APP_NAME/"
    chmod +x "$BUILD_DIR/$APP_NAME/$APP_NAME"
else
    echo "❌ 错误: 找不到 dist/$APP_NAME"
    echo "请先运行: pyinstaller pdf-retypeset.spec"
    exit 1
fi

# 复制图标
echo "🎨 复制图标..."
cp icon.png "$BUILD_DIR/$APP_NAME/"

# 创建.desktop启动器
echo "📝 创建启动器..."
cat > "$BUILD_DIR/$APP_NAME/$APP_NAME.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=中文PDF重排工具
Name[en]=Chinese PDF Retypeset
Comment=智能重排扫描版PDF
Comment[en]=Smart PDF Retypeset Tool
Exec=$PWD/chinese-pdf-retypeset
Icon=$PWD/icon.png
Terminal=false
Categories=Utility;Application;
Keywords=PDF;重排;排版;
StartupNotify=true
EOF

chmod +x "$BUILD_DIR/$APP_NAME/$APP_NAME.desktop"

# 创建启动脚本
cat > "$BUILD_DIR/$APP_NAME/启动工具.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
./chinese-pdf-retypeset
EOF
chmod +x "$BUILD_DIR/$APP_NAME/启动工具.sh"

# 创建README
cat > "$BUILD_DIR/$APP_NAME/README.txt" << EOF
中文PDF重排工具 $VERSION
===================

使用方法：
---------

方法1：双击 "启动工具.sh"
方法2：在终端运行：./chinese-pdf-retypeset
方法3：安装.desktop文件到桌面

安装图标：
---------
1. 复制 chinese-pdf-retypeset.desktop 到桌面
2. 右键 -> 允许启动
3. 双击即可启动

或者：
1. 复制到 ~/.local/share/applications/
2. 从应用菜单启动
EOF

# 打包
echo "📦 打包中..."
cd "$BUILD_DIR"
tar -czf "../${APP_NAME}-${VERSION}-portable-linux-x86_64.tar.gz" "$APP_NAME"
cd ..

# 清理
rm -rf "$BUILD_DIR"

echo "✅ 完成！"
echo "📁 文件: ${APP_NAME}-${VERSION}-portable-linux-x86_64.tar.gz"
ls -lh "${APP_NAME}-${VERSION}-portable-linux-x86_64.tar.gz"
