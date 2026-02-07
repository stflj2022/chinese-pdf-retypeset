#!/bin/bash
# 扩展交换空间到 50GB

set -e

echo "🔄 当前交换空间状态:"
free -h
echo ""

echo "📍 当前交换文件位置:"
grep swap /etc/fstab || echo "未找到交换配置"
echo ""

echo "🔧 开始创建 50GB 交换文件..."

# 1. 创建交换文件
echo "⏳ 创建 /swapfile50 (50GB)..."
sudo fallocate -l 50G /swapfile50 || sudo dd if=/dev/zero of=/swapfile50 bs=1G count=50 status=progress

# 2. 设置权限
echo "🔒 设置权限..."
sudo chmod 600 /swapfile50

# 3. 格式化为交换空间
echo "📝 格式化为交换空间..."
sudo mkswap /swapfile50

# 4. 启用交换文件
echo "✅ 启用交换文件..."
sudo swapon /swapfile50

# 5. 添加到 /etc/fstab 实现永久挂载
echo "💾 添加到 /etc/fstab..."
if ! grep -q "/swapfile50" /etc/fstab; then
    echo "/swapfile50 none swap sw 0 0" | sudo tee -a /etc/fstab
fi

echo ""
echo "✅ 完成！新的交换空间状态:"
free -h
echo ""

echo "📊 交换空间详情:"
cat /proc/swaps
echo ""

echo "🎉 交换空间已扩展到 50GB！"
