import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from pathlib import Path
import threading
from src.config import Config, load_config
from src.pipeline import Pipeline

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF智能重排工具（横版+竖版）")
        self.root.geometry("600x500")

        self.config = load_config()
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪")

        self._create_widgets()
        self._setup_drag_drop()

    def _create_widgets(self):
        # File Selection
        frame_file = ttk.LabelFrame(self.root, text="文件选择（可拖拽文件到此处）", padding=10)
        frame_file.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_file, text="输入文件:").grid(row=0, column=0, sticky="w")
        self.input_entry = ttk.Entry(frame_file, textvariable=self.input_path, width=40)
        self.input_entry.grid(row=0, column=1, padx=5)
        ttk.Button(frame_file, text="浏览...", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(frame_file, text="输出文件:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame_file, textvariable=self.output_path, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(frame_file, text="浏览...", command=self._browse_output).grid(row=1, column=2)

        # Settings
        frame_settings = ttk.LabelFrame(self.root, text="排版设置", padding=10)
        frame_settings.pack(fill="both", expand=True, padx=10, pady=5)

        # Grid for settings
        settings = [
            ("字符缩放:", "scale_factor", float),
            ("DPI:", "dpi", int),
            ("字间距:", "char_spacing", int),
            ("行间距:", "line_spacing", int),
        ]

        self.entries = {}
        for i, (label, key, dtype) in enumerate(settings):
            ttk.Label(frame_settings, text=label).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(getattr(self.config.layout.output, key)))
            entry = ttk.Entry(frame_settings, textvariable=var, width=10)
            entry.grid(row=i, column=1, sticky="w", pady=2)
            self.entries[key] = (var, dtype)

        # Page Size
        ttk.Label(frame_settings, text="页面大小:").grid(row=len(settings), column=0, sticky="w", pady=2)
        self.page_size_var = tk.StringVar(value=self.config.layout.output.page_size)
        cb_page = ttk.Combobox(frame_settings, textvariable=self.page_size_var, values=["A4", "A5", "Letter", "B5"])
        cb_page.grid(row=len(settings), column=1, sticky="w", pady=2)

        # Output Format
        ttk.Label(frame_settings, text="输出格式:").grid(row=len(settings)+1, column=0, sticky="w", pady=2)
        self.format_var = tk.StringVar(value=self.config.output.format)
        cb_format = ttk.Combobox(frame_settings, textvariable=self.format_var, values=["pdf", "images"])
        cb_format.grid(row=len(settings)+1, column=1, sticky="w", pady=2)

        # Progress
        frame_progress = ttk.Frame(self.root, padding=10)
        frame_progress.pack(fill="x", padx=10)
        
        self.progress_bar = ttk.Progressbar(frame_progress, length=100, mode='determinate')
        self.progress_bar.pack(fill="x", pady=5)
        
        ttk.Label(frame_progress, textvariable=self.status_var).pack()

        # Action Buttons
        frame_actions = ttk.Frame(self.root, padding=10)
        frame_actions.pack(fill="x", padx=10)
        
        ttk.Button(frame_actions, text="开始处理", command=self._start_processing).pack(side="right")

    def _browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("Images", "*.jpg;*.png")])
        if filename:
            self.input_path.set(filename)
            # Auto set output
            p = Path(filename)
            self.output_path.set(str(p.parent / (p.stem + "_retypeset.pdf")))

    def _browse_output(self):
        filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if filename:
            self.output_path.set(filename)

    def _update_config(self):
        # Update config object from UI
        try:
            for key, (var, dtype) in self.entries.items():
                val = dtype(var.get())
                setattr(self.config.layout.output, key, val)
            
            # Input DPI needs to match output DPI for simplicity in this tool unless separate
            # The config has input.dpi and layout.output.dpi
            # We updated layout.output.dpi above. Let's sync input.dpi
            self.config.input.dpi = self.config.layout.output.dpi
            
            self.config.layout.output.page_size = self.page_size_var.get()
            self.config.output.format = self.format_var.get()
            
        except ValueError as e:
            messagebox.showerror("错误", f"无效的参数值: {e}")
            return False
        return True

    def _start_processing(self):
        input_file = self.input_path.get()
        output_file = self.output_path.get()

        if not input_file or not output_file:
            messagebox.showwarning("提示", "请选择输入和输出文件")
            return

        if not self._update_config():
            return

        # Disable UI
        # (Simplified: just setting flag)
        
        self.status_var.set("正在初始化...")
        self.progress_bar['value'] = 0
        
        threading.Thread(target=self._run_pipeline, args=(input_file, output_file)).start()

    def _run_pipeline(self, input_file, output_file):
        try:
            pipeline = Pipeline(self.config)
            
            def callback(current, total, msg):
                self.root.after(0, lambda: self._update_progress(current, total, msg))
                
            pipeline.set_progress_callback(callback)
            pipeline.process(Path(input_file), Path(output_file))
            
            self.root.after(0, lambda: messagebox.showinfo("完成", "处理成功！"))
            self.root.after(0, lambda: self.status_var.set("就绪"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"处理失败: {e}"))
            self.root.after(0, lambda: self.status_var.set("失败"))

    def _update_progress(self, current, total, msg):
        self.progress_bar['value'] = (current / total) * 100
        self.status_var.set(msg)

    def _setup_drag_drop(self):
        """设置拖拽功能"""
        try:
            # 尝试在输入框上启用拖拽
            self.input_entry.drop_target_register(DND_FILES)
            self.input_entry.dnd_bind('<<Drop>>', self._on_drop)
        except Exception as e:
            # 如果tkinterdnd2不可用，静默失败
            print(f"拖拽功能不可用: {e}")

    def _on_drop(self, event):
        """处理文件拖拽事件"""
        # 获取拖拽的文件路径
        files = self.root.tk.splitlist(event.data)
        if files:
            file_path = files[0]
            # 移除可能的花括号
            file_path = file_path.strip('{}')

            # 检查文件类型
            if file_path.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
                self.input_path.set(file_path)
                # 自动设置输出路径
                p = Path(file_path)
                self.output_path.set(str(p.parent / (p.stem + "_retypeset.pdf")))
            else:
                messagebox.showwarning("提示", "请拖拽PDF或图片文件（.pdf, .jpg, .png）")

def run_gui():
    try:
        # 尝试使用TkinterDnD支持拖拽
        root = TkinterDnD.Tk()
    except:
        # 如果不可用，使用标准Tk
        root = tk.Tk()
    app = App(root)
    root.mainloop()
