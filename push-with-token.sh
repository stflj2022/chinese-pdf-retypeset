#!/bin/bash
# GitHub Token 推送脚本

echo "========================================="
echo "  GitHub Token 推送脚本"
echo "========================================="
echo ""

cd /home/wu/桌面/pdf-retypeset

echo "请输入 GitHub Personal Access Token:"
echo "（访问 https://github.com/settings/tokens/new 创建）"
echo ""
read -s -p "Token: " TOKEN
echo ""

if [ -z "$TOKEN" ]; then
    echo "❌ Token 不能为空"
    exit 1
fi

# 使用 token 推送
echo ""
echo "🚀 正在推送到 GitHub..."
git remote set-url origin https://$TOKEN@github.com/stflj2022/-PDF-.git
git push -u origin main

# 恢复原始 URL
git remote set-url origin https://github.com/stflj2022/-PDF-.git

echo ""
echo "✅ 完成！"
