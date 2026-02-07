"""
图形用户界面模块

基于Tkinter的GUI，支持横排和竖排模式。
"""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

# 尝试导入拖拽支持
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

from src.config import Config, PAGE_SIZES
from src.pipeline import Pipeline


# 根据是否有拖拽支持选择基类
if HAS_DND:
    BaseApp = TkinterDnD.Tk
else:
    BaseApp = tk.Tk


class Application(BaseApp):
    """主应用程序窗口"""

    def __init__(self):
        super().__init__()

        self.title("扫描版PDF重排工具")
        self.geometry("1200x800")
        self.minsize(900, 700)

        # 配置
        self.config_obj = Config()
        # 默认使用横排模式和Otsu二值化
        self.config_obj.layout.direction = "horizontal"
        self.config_obj.preprocessing.binarization.method = "otsu"
        self.config_obj.preprocessing.denoise.kernel_size = 2

        self.pipeline: Optional[Pipeline] = None

        # 当前文件
        self.input_path: Optional[Path] = None
        self.current_image: Optional[np.ndarray] = None

        # 预览图像
        self.original_photo: Optional[ImageTk.PhotoImage] = None
        self.processed_photo: Optional[ImageTk.PhotoImage] = None

        # 创建界面
        self._create_widgets()
        self._create_menu()

        # 设置拖拽支持
        if HAS_DND:
            self._setup_dnd()

    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开...", command=self._open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="保存...", command=self._save_file, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.quit)

        self.bind("<Control-o>", lambda e: self._open_file())
        self.bind("<Control-s>", lambda e: self._save_file())

    def _create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 顶部：文件选择
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(file_frame, text="输入文件:").pack(side=tk.LEFT)
        self.file_entry = ttk.Entry(file_frame, width=60)
        self.file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="浏览...", command=self._open_file).pack(side=tk.LEFT)

        # 中部：预览区域
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 左侧：原始预览
        left_frame = ttk.LabelFrame(preview_frame, text="原始预览", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.original_canvas = tk.Canvas(left_frame, bg="gray90")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # 右侧：处理后预览
        right_frame = ttk.LabelFrame(preview_frame, text="重排预览", padding="5")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.processed_canvas = tk.Canvas(right_frame, bg="gray90")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)

        # 底部：设置和控制
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        # 设置区域
        settings_frame = ttk.LabelFrame(bottom_frame, text="设置", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # 第一行设置
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=2)

        # 布局模式
        ttk.Label(row1, text="布局模式:").pack(side=tk.LEFT, padx=5)
        self.layout_var = tk.StringVar(value="horizontal")
        layout_combo = ttk.Combobox(
            row1,
            textvariable=self.layout_var,
            values=["horizontal", "vertical_rtl", "vertical_ltr", "auto"],
            width=12,
            state="readonly",
        )
        layout_combo.pack(side=tk.LEFT, padx=5)
        layout_combo.bind("<<ComboboxSelected>>", self._on_settings_changed)

        # 布局模式说明
        self.layout_desc = ttk.Label(row1, text="(横排：从左到右，从上到下)", foreground="gray")
        self.layout_desc.pack(side=tk.LEFT, padx=5)

        # 放大倍数
        ttk.Label(row1, text="放大倍数:").pack(side=tk.LEFT, padx=(20, 5))
        self.scale_var = tk.StringVar(value="2.0")
        scale_combo = ttk.Combobox(
            row1,
            textvariable=self.scale_var,
            values=["1.5", "2.0", "2.5", "3.0"],
            width=8,
        )
        scale_combo.pack(side=tk.LEFT, padx=5)
        scale_combo.bind("<<ComboboxSelected>>", self._on_settings_changed)
        scale_combo.bind("<FocusOut>", self._on_settings_changed)

        # 页面大小
        ttk.Label(row1, text="页面大小:").pack(side=tk.LEFT, padx=(20, 5))
        self.page_size_var = tk.StringVar(value="A4")
        page_size_combo = ttk.Combobox(
            row1,
            textvariable=self.page_size_var,
            values=list(PAGE_SIZES.keys()),
            width=8,
            state="readonly",
        )
        page_size_combo.pack(side=tk.LEFT, padx=5)
        page_size_combo.bind("<<ComboboxSelected>>", self._on_settings_changed)

        # 第二行设置
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=2)

        # 字符间距
        ttk.Label(row2, text="字符间距:").pack(side=tk.LEFT, padx=5)
        self.char_spacing_var = tk.StringVar(value="5")
        char_spacing_entry = ttk.Entry(row2, textvariable=self.char_spacing_var, width=8)
        char_spacing_entry.pack(side=tk.LEFT, padx=5)
        char_spacing_entry.bind("<FocusOut>", self._on_settings_changed)

        # 行间距
        ttk.Label(row2, text="行间距:").pack(side=tk.LEFT, padx=(20, 5))
        self.line_spacing_var = tk.StringVar(value="15")
        line_spacing_entry = ttk.Entry(row2, textvariable=self.line_spacing_var, width=8)
        line_spacing_entry.pack(side=tk.LEFT, padx=5)
        line_spacing_entry.bind("<FocusOut>", self._on_settings_changed)

        # 页边距
        ttk.Label(row2, text="页边距:").pack(side=tk.LEFT, padx=(20, 5))
        self.margin_var = tk.StringVar(value="80")
        margin_entry = ttk.Entry(row2, textvariable=self.margin_var, width=8)
        margin_entry.pack(side=tk.LEFT, padx=5)
        margin_entry.bind("<FocusOut>", self._on_settings_changed)

        # 字符数量显示
        self.char_count_label = ttk.Label(row2, text="检测字符: -", foreground="blue")
        self.char_count_label.pack(side=tk.RIGHT, padx=20)

        # 控制区域
        control_frame = ttk.Frame(bottom_frame)
        control_frame.pack(fill=tk.X)

        # 处理按钮
        self.process_btn = ttk.Button(
            control_frame,
            text="开始处理",
            command=self._start_processing,
        )
        self.process_btn.pack(side=tk.LEFT)

        # 预览分割按钮
        self.preview_seg_btn = ttk.Button(
            control_frame,
            text="预览分割",
            command=self._preview_segmentation,
        )
        self.preview_seg_btn.pack(side=tk.LEFT, padx=10)

        # 进度条
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            control_frame,
            variable=self.progress_var,
            maximum=100,
            length=300,
        )
        self.progress_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # 进度标签
        self.progress_label = ttk.Label(control_frame, text="就绪")
        self.progress_label.pack(side=tk.LEFT, padx=5)

    def _update_layout_description(self):
        """更新布局模式说明"""
        mode = self.layout_var.get()
        descriptions = {
            "horizontal": "(横排：从左到右，从上到下)",
            "vertical_rtl": "(竖排：从右到左，从上到下 - 古籍)",
            "vertical_ltr": "(竖排：从左到右，从上到下)",
            "auto": "(自动检测每页排版方向)",
        }
        self.layout_desc.config(text=descriptions.get(mode, ""))

    def _setup_dnd(self):
        """设置拖拽支持"""
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self._on_drop)

    def _on_drop(self, event):
        """处理拖拽放下事件"""
        # 获取拖拽的文件路径
        filepath = event.data
        # 处理路径（可能带有花括号或空格）
        if filepath.startswith('{') and filepath.endswith('}'):
            filepath = filepath[1:-1]
        filepath = filepath.strip()

        # 检查文件扩展名
        path = Path(filepath)
        valid_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        if path.suffix.lower() in valid_extensions:
            self.input_path = path
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, str(self.input_path))
            self._load_preview()
        else:
            messagebox.showwarning("警告", f"不支持的文件格式: {path.suffix}")

    def _open_file(self):
        """打开文件"""
        filetypes = [
            ("PDF文件", "*.pdf"),
            ("图像文件", "*.png *.jpg *.jpeg *.tiff *.bmp"),
            ("所有文件", "*.*"),
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes)

        if filepath:
            self.input_path = Path(filepath)
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, str(self.input_path))
            self._load_preview()

    def _save_file(self):
        """保存文件"""
        if not self.input_path:
            messagebox.showwarning("警告", "请先打开一个文件")
            return

        filetypes = [("PDF文件", "*.pdf")]
        filepath = filedialog.asksaveasfilename(
            filetypes=filetypes,
            defaultextension=".pdf",
            initialdir=str(self.input_path.parent),
            initialfile=f"{self.input_path.stem}_重排",
        )

        if filepath:
            self._start_processing(Path(filepath))

    def _load_preview(self):
        """加载预览"""
        if not self.input_path:
            return

        try:
            self._update_config()
            self.pipeline = Pipeline(self.config_obj)

            # 加载第一页
            if self.input_path.suffix.lower() == ".pdf":
                from pdf2image import convert_from_path
                images = convert_from_path(
                    str(self.input_path),
                    dpi=self.config_obj.input.dpi,
                    first_page=1,
                    last_page=1,
                )
                if images:
                    pil_image = images[0]
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    self.current_image = np.array(pil_image)
                    self.current_image = cv2.cvtColor(
                        self.current_image, cv2.COLOR_RGB2BGR
                    )
            else:
                self.current_image = cv2.imread(str(self.input_path))

            if self.current_image is None:
                messagebox.showerror("错误", "无法加载文件")
                return

            self._update_preview()

        except Exception as e:
            messagebox.showerror("错误", f"加载文件失败: {e}")

    def _update_preview(self):
        """更新预览"""
        if self.current_image is None or self.pipeline is None:
            return

        try:
            # 获取预览
            original, processed, char_count = self.pipeline.process_preview(
                self.current_image
            )

            # 更新字符数量
            self.char_count_label.config(text=f"检测字符: {char_count}")

            # 转换为PIL图像
            original_pil = Image.fromarray(original)
            processed_pil = Image.fromarray(processed)

            # 调整大小以适应画布
            canvas_width = self.original_canvas.winfo_width()
            canvas_height = self.original_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                original_pil = self._fit_image(original_pil, canvas_width, canvas_height)
                processed_pil = self._fit_image(processed_pil, canvas_width, canvas_height)

            # 转换为PhotoImage
            self.original_photo = ImageTk.PhotoImage(original_pil)
            self.processed_photo = ImageTk.PhotoImage(processed_pil)

            # 显示在画布上
            self.original_canvas.delete("all")
            self.original_canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.original_photo,
                anchor=tk.CENTER,
            )

            self.processed_canvas.delete("all")
            self.processed_canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.processed_photo,
                anchor=tk.CENTER,
            )

        except Exception as e:
            print(f"预览更新失败: {e}")

    def _preview_segmentation(self):
        """预览分割结果"""
        if self.current_image is None or self.pipeline is None:
            messagebox.showwarning("警告", "请先打开一个文件")
            return

        try:
            preview, char_count = self.pipeline.get_segmentation_preview(
                self.current_image
            )

            # 更新字符数量
            self.char_count_label.config(text=f"检测字符: {char_count}")

            # 转换为PIL图像
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            preview_pil = Image.fromarray(preview_rgb)

            # 调整大小
            canvas_width = self.original_canvas.winfo_width()
            canvas_height = self.original_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                preview_pil = self._fit_image(preview_pil, canvas_width, canvas_height)

            # 显示
            self.original_photo = ImageTk.PhotoImage(preview_pil)
            self.original_canvas.delete("all")
            self.original_canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.original_photo,
                anchor=tk.CENTER,
            )

        except Exception as e:
            messagebox.showerror("错误", f"预览分割失败: {e}")

    def _fit_image(
        self, image: Image.Image, max_width: int, max_height: int
    ) -> Image.Image:
        """调整图像大小以适应指定区域"""
        width, height = image.size
        ratio = min(max_width / width, max_height / height)

        if ratio < 1:
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def _update_config(self):
        """从界面更新配置"""
        try:
            self.config_obj.layout.direction = self.layout_var.get()
            self.config_obj.layout.output.scale_factor = float(self.scale_var.get())
            self.config_obj.layout.output.char_spacing = int(self.char_spacing_var.get())
            self.config_obj.layout.output.line_spacing = int(self.line_spacing_var.get())
            self.config_obj.layout.output.margin = int(self.margin_var.get())
            self.config_obj.layout.output.page_size = self.page_size_var.get()
        except ValueError:
            pass

    def _on_settings_changed(self, event=None):
        """设置改变时更新预览"""
        self._update_layout_description()
        self._update_config()
        if self.pipeline:
            self.pipeline = Pipeline(self.config_obj)
        self._update_preview()

    def _start_processing(self, output_path: Optional[Path] = None):
        """开始处理"""
        if not self.input_path:
            messagebox.showwarning("警告", "请先打开一个文件")
            return

        if output_path is None:
            filetypes = [("PDF文件", "*.pdf")]
            filepath = filedialog.asksaveasfilename(
                filetypes=filetypes,
                defaultextension=".pdf",
                initialfile=f"{self.input_path.stem}_重排",
            )
            if not filepath:
                return
            output_path = Path(filepath)

        self.process_btn.config(state=tk.DISABLED)
        self.preview_seg_btn.config(state=tk.DISABLED)

        def process_thread():
            try:
                self._update_config()
                pipeline = Pipeline(self.config_obj)

                def progress_callback(current: int, total: int, message: str):
                    progress = (current / total) * 100
                    self.after(0, lambda: self._update_progress(progress, message))

                pipeline.set_progress_callback(progress_callback)
                pipeline.process(self.input_path, output_path)

                self.after(0, lambda: self._processing_complete(output_path))

            except Exception as e:
                self.after(0, lambda: self._processing_error(str(e)))

        thread = threading.Thread(target=process_thread)
        thread.start()

    def _update_progress(self, progress: float, message: str):
        """更新进度"""
        self.progress_var.set(progress)
        self.progress_label.config(text=message)

    def _processing_complete(self, output_path: Path):
        """处理完成"""
        self.process_btn.config(state=tk.NORMAL)
        self.preview_seg_btn.config(state=tk.NORMAL)
        self.progress_var.set(100)
        self.progress_label.config(text="完成")
        messagebox.showinfo("完成", f"文件已保存到:\n{output_path}")

    def _processing_error(self, error: str):
        """处理错误"""
        self.process_btn.config(state=tk.NORMAL)
        self.preview_seg_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.config(text="错误")
        messagebox.showerror("错误", f"处理失败:\n{error}")


def run_gui():
    """运行GUI"""
    app = Application()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
