# PDF智能重排工具（横版+竖版）

**版本**: v2.0
**更新时间**: 2026-02-08
**项目位置**: `/home/wu/桌面/pdf_zhongjiban`

---

## 📋 项目简介

本工具整合了横版和竖版PDF重排功能，能够**自动检测每一页的方向**（横版或竖版），并使用相应的处理算法进行重排。

### 核心特性

✅ **自动方向检测** - 逐页检测PDF方向，支持混合方向的PDF
✅ **横版处理** - 支持横排文本（从左到右，从上到下）
✅ **竖版处理** - 支持竖排文本（从右到左，从上到下）
✅ **拖拽导入** - 支持直接拖拽PDF文件到界面
✅ **图形界面** - 友好的GUI界面，参数可调
✅ **参数保留** - 保留原项目精心调试的所有参数

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd ~/桌面/pdf_zhongjiban
pip install -r requirements.txt
```

**系统依赖**（如果尚未安装）：
```bash
sudo apt install poppler-utils
```

### 2. 启动程序

```bash
python3 run.py
```

或者使用命令行模式：
```bash
python3 -m src.main input.pdf output.pdf
```

### 3. 使用界面

1. **方式一：点击"浏览..."按钮选择PDF文件**
2. **方式二：直接拖拽PDF文件到"输入文件"输入框**
3. 输出路径会自动设置为 `{原文件名}_retypeset.pdf`
4. 调整参数（可选）
5. 点击"开始处理"

---

## 🎯 工作原理

### 自动方向检测算法

对每一页PDF，工具会：

1. **提取字块** - 使用形态学处理和连通域分析
2. **计算特征**：
   - 页面宽高比
   - 字块平均宽高比
   - X方向聚类质量（竖版特征）
   - Y方向聚类质量（横版特征）
3. **综合评分** - 根据多个特征计算横版/竖版得分
4. **选择算法** - 自动选择对应的分割算法

### 横版处理流程

1. 形态学处理连接断开的笔画
2. 连通域分析提取字块
3. **按Y坐标聚类成行**
4. 行排序：从上到下
5. 每行内按X排序：从左到右

### 竖版处理流程

1. 形态学处理连接断开的笔画
2. 连通域分析提取字块
3. **按X坐标聚类成列**
4. 列排序：从右到左
5. 每列内按Y排序：从上到下

---

## ⚙️ 核心参数说明

### 形态学处理参数（来自原始项目，不建议修改）

```python
# src/segmenter.py:244-248
kernel = np.ones((5, 5), np.uint8)  # 5x5核
dilated = cv2.dilate(binary, kernel, iterations=2)  # 膨胀2次
eroded = cv2.erode(dilated, kernel, iterations=2)   # 腐蚀2次
```

### 字块过滤参数（来自原始横版项目，不建议修改）

```python
# src/segmenter.py:270-277
area < 100  # 过滤面积小于100的噪点
w < 10 and h < 10  # 过滤太小的字块
w > binary.shape[1] * 0.5  # 过滤太宽的字块（可能是边框）
h > binary.shape[0] * 0.3  # 过滤太高的字块
```

### 聚类容差参数（来自原始横版项目，不建议修改）

```python
# src/segmenter.py:305 (竖版X聚类)
tolerance = median_w * 0.6  # 中位宽度的60%

# src/segmenter.py:343 (横版Y聚类)
tolerance = median_h * 0.6  # 中位高度的60%
```

### 方向检测算法（来自原始横版项目）

```python
# src/segmenter.py:187-190
# 竖排特征：每列字数明显多于每行，且列数不多
if avg_per_col > avg_per_row * 2 and len(columns) <= 10:
    return "vertical"
return "horizontal"
```

### 可调整的输出参数

在GUI界面中可以调整：

- **字符缩放** (scale_factor): 默认 1.5
- **DPI**: 默认 300
- **字间距** (char_spacing): 默认 10 像素
- **行间距** (line_spacing): 默认 20 像素
- **页面大小**: A4 / A5 / Letter / B5
- **输出格式**: PDF / 图片序列

---

## 📊 项目结构

```
/home/wu/桌面/pdf_zhongjiban/
├── src/
│   ├── __init__.py
│   ├── main.py           # 命令行入口
│   ├── gui.py            # GUI界面（支持拖拽）
│   ├── config.py         # 配置管理
│   ├── pipeline.py       # 处理流水线
│   ├── preprocessor.py   # 图像预处理
│   ├── segmenter.py      # 字符分割（横版+竖版+自动检测）
│   ├── layouter.py       # 页面排版
│   └── output.py         # 输出生成
├── run.py                # 启动脚本
├── requirements.txt      # Python依赖
└── README.md             # 本文档
```

---

## 🔧 高级用法

### 命令行模式

```bash
# 基本用法
python3 -m src.main input.pdf output.pdf

# 自定义参数
python3 -m src.main input.pdf output.pdf \
  --scale 1.5 \
  --dpi 300 \
  --char-spacing 10 \
  --line-spacing 20 \
  --page-size A4
```

### 强制指定方向（修改代码）

如果需要强制使用某个方向，可以修改 `src/pipeline.py:66`：

```python
# 强制使用竖版
chars = self.segmenter.segment_vertical(binary, gray)

# 强制使用横版
chars = self.segmenter.segment_horizontal(binary, gray)

# 自动检测（默认）
chars = self.segmenter.segment(binary, gray)
```

---

## 🐛 常见问题

### Q1: 拖拽功能不可用

**原因**: tkinterdnd2库未安装或不兼容

**解决**:
```bash
pip install tkinterdnd2
```

如果仍然不可用，可以使用"浏览..."按钮选择文件。

### Q2: 方向检测不准确

**原因**: 某些特殊排版可能导致误判

**解决**:
1. 检查PDF是否为扫描版（工具不支持文字版PDF）
2. 如果持续误判，可以强制指定方向（见"高级用法"）

### Q3: 字符识别不完整

**原因**: 形态学参数或过滤阈值不适合当前PDF

**解决**:
- 降低 `area` 阈值（src/segmenter.py:148）
- 调整形态学核大小和迭代次数（src/segmenter.py:124）

**警告**: 修改这些参数需要非常谨慎，建议先备份原文件！

### Q4: 处理速度慢

**原因**: PDF页数多或分辨率高

**解决**:
- 降低DPI（但会影响输出质量）
- 等待处理完成（处理时间取决于文件大小）

---

## 📦 依赖说明

### Python包

- `numpy>=1.21.0` - 数值计算
- `opencv-python>=4.5.0` - 图像处理
- `pdf2image>=1.16.0` - PDF转图像
- `Pillow>=9.0.0` - 图像处理
- `PyYAML>=6.0` - 配置文件解析
- `tkinterdnd2>=0.3.0` - 拖拽功能支持

### 系统依赖

- `poppler-utils` - PDF渲染（pdf2image需要）

---

## 🔄 版本历史

### v2.0 (2026-02-08)

- ✅ 整合横版和竖版处理功能
- ✅ 实现自动方向检测（逐页检测）
- ✅ 添加拖拽文件导入功能
- ✅ 保留原项目所有精心调试的参数
- ✅ 更新GUI标题和提示信息

### v1.0 (2026-02-07)

- 竖版PDF重排工具（原 pdf2竖版 项目）

---

## ⚠️ 重要提示

1. **参数谨慎修改**: 形态学参数、过滤阈值、聚类容差都是经过反复调试的，轻易修改可能导致识别效果下降
2. **逐页检测**: 工具会对每一页单独检测方向，支持混合方向的PDF
3. **扫描版PDF**: 工具专为扫描版PDF设计，不支持文字版PDF
4. **备份原文件**: 处理前建议备份原始PDF文件

---

## 📞 技术支持

### 问题反馈

如遇到问题，请提供：
1. 错误信息截图
2. 输入PDF文件（如果可以分享）
3. 处理日志
4. 系统环境信息

### 项目来源

- 横版处理: `/home/wu/桌面/pdf横版`
- 竖版处理: `/home/wu/桌面/pdf2竖版`
- 整合版本: `/home/wu/桌面/pdf_zhongjiban`

---

**祝使用愉快！** 🎉
