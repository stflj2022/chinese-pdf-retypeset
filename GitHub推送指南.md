# GitHub 推送指南

## 方法1：使用 GitHub CLI（推荐）

```bash
# 1. 安装 GitHub CLI
sudo apt install gh

# 2. 登录
gh auth login

# 按提示选择:
# - GitHub.com
# - HTTPS
# - Login with a web browser (会打开浏览器授权)

# 3. 推送
cd /home/wu/桌面/pdf-retypeset
git push -u origin main
```

---

## 方法2：使用 Personal Access Token

### 步骤1：创建 Token

1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 设置：
   - **Note**: pdf-retypeset
   - **Expiration**: 90 days 或 No expiration
   - **Scopes**: 勾选 `repo` (Full control of private repositories)
4. 点击 "Generate token"
5. **复制 token**（只显示一次！）

### 步骤2：使用 Token 推送

```bash
cd /home/wu/桌面/pdf-retypeset

# 方法A：直接使用（临时）
git push https://<TOKEN>@github.com/stflj2022/-PDF-.git main

# 方法B：配置 git（推荐）
git remote set-url origin https://<TOKEN>@github.com/stflj2022/-PDF-.git
git push -u origin main

# 方法C：使用 git credential helper（永久保存）
git config --global credential.helper store
git push -u origin main
# 输入用户名：stflj2022
# 输入密码：<粘贴 TOKEN>
```

---

## 方法3：配置 SSH Key

```bash
# 1. 生成 SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"
# 回车使用默认路径
# 可以设置密码或直接回车

# 2. 查看公钥
cat ~/.ssh/id_ed25519.pub

# 3. 添加到 GitHub
# 访问：https://github.com/settings/keys
# 点击 "New SSH key"
# 粘贴公钥内容
# 点击 "Add SSH key"

# 4. 切换到 SSH URL 并推送
cd /home/wu/桌面/pdf-retypeset
git remote set-url origin git@github.com:stflj2022/-PDF-.git
git push -u origin main
```

---

## 当前状态

✅ Git 仓库已初始化
✅ 代码已提交
❌ 推送到 GitHub 需要认证

**仓库地址**: https://github.com/stflj2022/-PDF-

---

## 推荐方案

**最简单**：使用 GitHub CLI (`gh auth login`)
**最通用**：使用 Personal Access Token
**最安全**：使用 SSH Key

选择一种方式完成认证后，运行：
```bash
cd /home/wu/桌面/pdf-retypeset
git push -u origin main
```
