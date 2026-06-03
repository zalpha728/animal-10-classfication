"""
图片预处理脚本
将 animal-10 数据集的图片调整为 ResNet-18 标准的 224×224 尺寸，
并将意大利语文件夹名改为英文，图片按顺序重命名。
"""

import os
from PIL import Image
from tqdm import tqdm

# 分类名称映射（意大利语 → 英文）
CATEGORY_MAP = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}

# 原始数据目录
RAW_DATA_DIR = "animals10_datasets"
# 输出数据目录
OUTPUT_DIR = "processed_datasets"

# ResNet-18 标准输入尺寸
IMG_SIZE = (224, 224)

# 支持的图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def process_dataset(subset_name):
    """
    处理指定子集（train 或 test）的所有图片。
    
    Args:
        subset_name: 子集名称，如 "train" 或 "test"
    """
    raw_subset_dir = os.path.join(RAW_DATA_DIR, subset_name)
    output_subset_dir = os.path.join(OUTPUT_DIR, subset_name)

    if not os.path.exists(raw_subset_dir):
        print(f"警告：目录 {raw_subset_dir} 不存在，跳过")
        return

    # 遍历每个分类文件夹
    for italian_name in os.listdir(raw_subset_dir):
        italian_dir = os.path.join(raw_subset_dir, italian_name)

        # 跳过非目录项
        if not os.path.isdir(italian_dir):
            continue

        # 获取英文分类名
        english_name = CATEGORY_MAP.get(italian_name, italian_name)
        output_class_dir = os.path.join(output_subset_dir, english_name)

        # 创建输出目录
        os.makedirs(output_class_dir, exist_ok=True)

        # 获取所有图片文件
        image_files = [
            f
            for f in os.listdir(italian_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]
        image_files.sort()  # 排序确保顺序一致

        # 使用 tqdm 显示进度
        for idx, filename in enumerate(tqdm(image_files, desc=f"{subset_name}/{english_name}")):
            input_path = os.path.join(italian_dir, filename)

            # 新文件名：类别_数字编号.jpg
            new_filename = f"{english_name}_{idx + 1:04d}.jpg"
            output_path = os.path.join(output_class_dir, new_filename)

            try:
                # 打开图片
                with Image.open(input_path) as img:
                    # 转换为 RGB（确保一致性）
                    img = img.convert("RGB")
                    # 调整尺寸为 224×224
                    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                    # 保存为 JPEG
                    img.save(output_path, "JPEG", quality=95)
            except Exception as e:
                print(f"处理文件 {input_path} 时出错: {e}")


def main():
    print("=" * 60)
    print("Animal-10 数据集预处理")
    print(f"输入目录: {RAW_DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"目标尺寸: {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    print("=" * 60)

    # 处理训练集
    print("\n处理训练集...")
    process_dataset("train")

    # 处理测试集
    print("\n处理测试集...")
    process_dataset("test")

    print("\n" + "=" * 60)
    print("预处理完成！")
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()