"""
Animal-10 图像分类可视化界面
- Tkinter GUI
- 选择图片 → 显示图片 → 预测动物类别
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import torch
from torchvision import transforms

# ─── 将项目根目录加入 sys.path ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Config, AnimalClassifier
from dataset import IMAGENET_MEAN, IMAGENET_STD


# ─── 预处理变换（与验证集一致） ───
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class AnimalPredictorApp:
    """动物分类预测 GUI 应用"""

    def __init__(self, root):
        self.root = root
        self.root.title("Animal-10 图像分类预测")
        self.root.geometry("520x600")
        self.root.resizable(False, False)

        # ─── 加载模型 ───
        self.model = None
        self.load_model()

        # ─── 当前图片路径 ───
        self.image_path = None
        self.photo = None  # 保持 PhotoImage 引用

        # ─── 构建 UI ───
        self._build_ui()

    def load_model(self):
        """加载训练好的模型"""
        try:
            config = Config()
            self.model = AnimalClassifier(config=config, use_pretrained=False)
            model_path = self.model.best_model_path
            if not os.path.exists(model_path):
                model_path = self.model.last_model_path
            if os.path.exists(model_path):
                self.model.load(model_path)
                self.model.eval()
                print(f"[INFO] 模型加载成功: {model_path}")
            else:
                print(f"[WARN] 未找到模型文件，请先训练模型")
                self.model = None
        except Exception as e:
            print(f"[ERROR] 加载模型失败: {e}")
            self.model = None

    def _build_ui(self):
        """构建用户界面"""
        # ─── 标题 ───
        title_label = tk.Label(
            self.root,
            text="Animal-10 图像分类预测",
            font=("Microsoft YaHei", 18, "bold"),
            fg="#2c3e50",
        )
        title_label.pack(pady=(20, 10))

        # ─── 模型状态 ───
        status_text = "✅ 模型已加载" if self.model else "❌ 模型未加载"
        status_color = "#27ae60" if self.model else "#e74c3c"
        self.status_label = tk.Label(
            self.root,
            text=status_text,
            font=("Microsoft YaHei", 10),
            fg=status_color,
        )
        self.status_label.pack(pady=(0, 10))

        # ─── 图片显示区域 ───
        self.image_frame = tk.Frame(
            self.root,
            width=200,
            height=200,
            bg="#ecf0f1",
            relief=tk.SUNKEN,
            bd=2,
        )
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(
            self.image_frame,
            text="请选择一张动物图片",
            font=("Microsoft YaHei", 12),
            bg="#ecf0f1",
            fg="#7f8c8d",
        )
        self.image_label.pack(expand=True, fill=tk.BOTH)

        # ─── 按钮区域 ───
        btn_frame = tk.Frame(self.root, bg="#f5f6fa")
        btn_frame.pack(pady=15)

        self.select_btn = tk.Button(
            btn_frame,
            text="📁 选择图片",
            font=("Microsoft YaHei", 12),
            bg="#3498db",
            fg="white",
            padx=25,
            pady=8,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.select_image,
        )
        self.select_btn.pack(side=tk.LEFT, padx=10)

        self.predict_btn = tk.Button(
            btn_frame,
            text="🔍 预测",
            font=("Microsoft YaHei", 12),
            bg="#2ecc71",
            fg="white",
            padx=25,
            pady=8,
            relief=tk.FLAT,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.predict,
        )
        self.predict_btn.pack(side=tk.LEFT, padx=10)

        # ─── 预测结果区域 ───
        self.result_frame = tk.Frame(
            self.root,
            bg="#f5f6fa",
            relief=tk.GROOVE,
            bd=2,
        )
        self.result_frame.pack(fill=tk.X, padx=40, pady=10)

        self.result_label = tk.Label(
            self.result_frame,
            text="等待预测...",
            font=("Microsoft YaHei", 14),
            bg="#f5f6fa",
            fg="#2c3e50",
            pady=10,
        )
        self.result_label.pack()

        self.confidence_label = tk.Label(
            self.result_frame,
            text="",
            font=("Microsoft YaHei", 11),
            bg="#f5f6fa",
            fg="#7f8c8d",
        )
        self.confidence_label.pack(pady=(5, 10))

    def select_image(self):
        """打开文件对话框选择图片"""
        file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_datasets", "test"),
            title="选择动物图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有文件", "*.*"),
            ],
        )
        if not file_path:
            return

        self.image_path = file_path
        self._display_image(file_path)
        self.predict_btn.config(state=tk.NORMAL)
        self.result_label.config(text="点击「预测」按钮识别动物")
        self.confidence_label.config(text="")

    def _display_image(self, path):
        """在 UI 中显示选中的图片"""
        try:
            # 打开并调整图片大小
            img = Image.open(path)
            img.thumbnail((180, 180), Image.LANCZOS)

            # 转换为 Tkinter 可显示的 PhotoImage
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(
                image=self.photo,
                text="",
                bg="#ecf0f1",
            )
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片:\n{e}")

    def predict(self):
        """调用模型进行预测"""
        if self.model is None:
            messagebox.showwarning("警告", "模型未加载，请先训练模型")
            return

        if self.image_path is None:
            return

        try:
            # ─── 预处理图片 ───
            image = Image.open(self.image_path).convert("RGB")
            input_tensor = VAL_TRANSFORM(image).unsqueeze(0)  # [1, 3, 224, 224]

            # ─── 推理 ───
            class_name, confidence = self.model.predict(input_tensor)

            # ─── 显示结果 ───
            # 中文映射（方便展示）
            cn_map = {
                "butterfly": "🦋 蝴蝶",
                "cat": "🐱 猫",
                "chicken": "🐔 鸡",
                "cow": "🐄 牛",
                "dog": "🐶 狗",
                "elephant": "🐘 大象",
                "horse": "🐴 马",
                "sheep": "🐑 羊",
                "spider": "🕷️ 蜘蛛",
                "squirrel": "🐿️ 松鼠",
            }
            display_name = cn_map.get(class_name, class_name)

            self.result_label.config(
                text=f"预测结果: {display_name}",
                fg="#2c3e50",
            )
            self.confidence_label.config(
                text=f"置信度: {confidence * 100:.2f}%",
                fg="#27ae60" if confidence > 0.5 else "#e67e22",
            )

        except Exception as e:
            messagebox.showerror("预测错误", f"预测过程中发生错误:\n{e}")
            import traceback
            traceback.print_exc()


def main():
    root = tk.Tk()
    app = AnimalPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()