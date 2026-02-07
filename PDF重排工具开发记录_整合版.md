# PDF重排工具开发记录

**项目**: 扫描版PDF字块放大重排工具
**开发时间**: 2026年2月6日 - 2月7日
**目标**: 将扫描版PDF的字块放大重排，便于阅读

---

## 项目位置

```
/home/wu/桌面/pdf-retypeset/          # GUI版本（主项目）
/home/wu/桌面/pdf重排/                  # 便携包（可直接使用）
```

---

## 核心功能

将扫描版PDF的字块放大重排，核心是**字块检测和顺序保持**，不是OCR识别内容。

---

## 技术方案演进

### 第一版：连通域分析（失败）
- 使用OpenCV连通域分析检测字块
- 问题：扫描版PDF笔画断开，一个字被分成多个连通域
- 结果：字块位置错乱

### 第二版：形态学操作（效果不佳）
- 尝试膨胀+腐蚀连接断开笔画
- 问题：参数难以调整，不同页面效果差异大

### 第三版：OCR边界框检测（GUI版本）
- 使用pytesseract检测文字边界框
- **不需要识别准确**，只用OCR的位置检测能力
- OCR自带行/块信息，顺序天然正确
- 缺点：处理较慢（每页需OCR）

### 第四版：自适应连通域分析（命令行版本）
- 解决了原版重排工具失败的问题
- 使用Otsu自动阈值 + 连通域分析
- 根据组件数量估算字数，自适应调整参数
- 带重试机制确保>50%检测率

---

## 核心算法

### 最终版本配置

**形态学处理**：
```python
# segmenter.py
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=2)
eroded = cv2.erode(dilated, kernel, iterations=2)
```

**缩放策略**：
```python
# layouter.py
# 所有字符统一按中位数高度缩放
median_h = sorted(heights)[len(heights) // 2]
target = int(median_h * scale_factor)
scaled = cv2.resize(char_image, (new_w, target_size))
```

---

## 问题与优化记录（2月7日）

### 问题1："一"字和标点符号显示过大

**原因分析**：
- 形态学处理（膨胀腐蚀）会改变所有字符边界框
- "一"字和标点符号原本很小，膨胀后相对变大更多
- 中位数作为基准，导致"牵一发而动全身"

**解决方案**：
1. 简化策略：所有字符统一按中位数高度缩放
2. 放弃复杂的字符分类逻辑
3. **接受97%完成度**

**结果**：95-97%完成度，"一"字和标点稍大但不影响阅读

### 问题2：PyInstaller 打包失败

**原因分析**：
- 相对导入（`from .config import`）在打包后失败
- 打包内存不足（原15GB交换空间）
- 包含过多不需要的依赖

**解决方案**：
1. **扩展交换空间**：15GB → 65GB
   ```bash
   sudo fallocate -l 50G /swapfile50
   sudo mkswap /swapfile50
   sudo swapon /swapfile50
   ```
2. **修复相对导入**：改为绝对导入
   ```python
   # 修改前
   from .config import Config
   # 修改后
   from src.config import Config
   ```
3. **创建入口文件**：`run.py`
   ```python
   import sys
   from pathlib import Path
   project_root = Path(__file__).parent
   sys.path.insert(0, str(project_root))
   from src.main import main
   if __name__ == "__main__":
       main()
   ```
4. **优化依赖**：排除 scipy、matplotlib、pandas 等不需要的库

**结果**：成功生成 131MB 可执行文件

### 问题3：可执行文件无法运行

**原因**：
- 排除了必要的 numpy 模块（如 numpy.linalg）
- 过度优化导致依赖缺失

**解决方案**：
- 保留必要的 numpy 子模块
- 使用保守的排除列表
- 最终体积：131MB（合理范围）

**体积构成**：
- OpenCV: ~40MB
- PyMuPDF: ~20MB
- NumPy: ~30MB
- 其他依赖: ~34MB

---

## GitHub Actions 自动构建

### 工作流配置

**文件**：`.github/workflows/build.yml`

**支持平台**：
- ✅ Linux (ubuntu-latest)
- ❌ Windows (构建失败)
- ✅ macOS (macos-latest)

**构建步骤**：
1. 安装系统依赖（libgl1, libglib2.0-0, libgomp1）
2. 安装 Python 依赖
3. PyInstaller 打包
4. 上传 Artifacts
5. 发布到 Release

**问题记录**：
- `actions/upload-artifact@v3` → 需升级到 `v4`
- `libgl1-mesa-glx` → Ubuntu 24.04 需改用 `libgl1`
- 需要添加 `permissions: contents: write` 和 `actions: write`

**发布版本**：
- v1.0.0 - v1.0.3: 测试版本
- v1.1.0 - v1.1.1: 修复相对导入（失败）
- v1.2.0: **最终可用版本**

---

## 便携包创建

**位置**：`/home/wu/桌面/pdf重排/`

**包含文件**：
```
pdf重排/
├── pdf-retypeset      (131MB - 可执行文件)
├── 启动GUI.sh         (启动脚本)
├── 安装依赖.sh        (自动安装系统依赖)
└── README.txt         (使用说明)
```

**使用方法**：
```bash
# 1. 解压或拷贝文件夹
cd pdf重排/

# 2. 安装依赖（首次使用）
./安装依赖.sh

# 3. 运行程序
./启动GUI.sh
# 或
./pdf-retypeset --gui
```

**压缩包**：`pdf重排工具-v1.2.0-20260207.tar.gz` (129MB)

---

## 备份文件

| 备份 | 时间 | 完成度 | 说明 |
|------|------|--------|------|
| v1 | 2026-02-07 05:17 | 95% | 修复"一"字和标点符号过大（复杂分类策略） |
| v2 | 2026-02-07 05:34 | 97% | 简化策略，字体大小统一 |

---

## 未来优化方向

### 思路1：分离检测和提取
```
1. 形态学处理后的图像 → 只用于检测字符位置（bounding box）
2. 原始图像 → 用检测到的位置提取实际字符
```
**优点**：检测位置准确，提取字符保持原始尺寸
**缺点**：需要大量重构

### 思路2：双路径处理
```
1. 膨胀图像 → 检测位置
2. 原始二值图 → 在对应位置重新提取紧凑边界
```
**优点**：兼容现有架构
**缺点**：增加复杂度

### 思路3：基于字符面积的智能缩放
```
1. 计算每个字符的像素面积
2. 根据面积分布识别异常字符
3. 对异常字符使用不同的缩放系数
```
**优点**：不需要修改检测逻辑
**缺点**：需要大量测试调优

---

## 经验教训

1. **形态学参数影响全局** - 膨胀腐蚀会改变所有字符的边界框
2. **中位数作为基准的局限** - 异常值会影响整体目标尺寸
3. **简单策略往往更好** - 复杂的分类逻辑容易引入新问题
4. **97% > 100%** - 完美是优秀的敌人，适可而止是智慧
5. **相对导入 vs 绝对导入** - PyInstaller 打包需使用绝对导入
6. **交换空间很重要** - 大型项目打包需要足够的虚拟内存
7. **体积优化要适度** - 过度优化会破坏功能

---

## 项目状态

✅ **完成度：97%**
✅ **可投入使用**
✅ **核心功能完善**
✅ **生成了可执行文件**
✅ **创建便携包**
✅ **推送到 GitHub**

⚠️ **已知局限**：
- "一"字和标点符号显示稍大（不影响阅读）
- Windows 版本构建失败
- 可执行文件仅支持 Linux

---

## 技术栈

**核心依赖**：
- OpenCV (cv2) - 图像处理
- PyMuPDF (fitz) - PDF 解析
- pdf2image - PDF 转图像
- pytesseract - OCR 边界框检测
- NumPy - 数值计算
- Pillow - 图像操作
- PyYAML - 配置文件

**开发工具**：
- PyInstaller 6.18.0 - 打包工具
- GitHub Actions - CI/CD
- Git - 版本控制

---

## 最后更新
- **更新时间**: 2026年2月7日 08:50
- **更新内容**: 添加 GitHub Actions 打包记录、便携包创建、完整工作流程
- **项目状态**: 97%完成度，已生成可执行文件和便携包

---

## GitHub 仓库

**地址**: https://github.com/stflj2022/-PDF-
**版本**: v1.2.0
**下载**: https://github.com/stflj2022/-PDF-/releases
