"""
数据加载与增强模块
- 使用 torchvision 的 ImageFolder 加载 processed_datasets
- 训练集：强增强策略
- 验证集：仅基本缩放裁剪 + 标准化
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 数据集路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "processed_datasets")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# 类别名称（按文件夹顺序）
CLASS_NAMES = [
    "butterfly", "cat", "chicken", "cow", "dog",
    "elephant", "horse", "sheep", "spider", "squirrel"
]

# ImageNet 标准化参数（ResNet 预训练模型要求）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    """
    训练集数据增强策略（强增强）
    每次训练时随机应用，增加数据多样性，防止过拟合
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """
    验证/测试集变换（不做随机增强）
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_datasets(train_transform=None, val_transform=None):
    """
    加载训练集和验证集

    参数:
        train_transform: 训练集变换，None 则使用默认增强
        val_transform:   验证集变换，None 则使用默认变换

    返回:
        train_dataset, val_dataset
    """
    if train_transform is None:
        train_transform = get_train_transforms()
    if val_transform is None:
        val_transform = get_val_transforms()

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=val_transform
    )

    return train_dataset, val_dataset


def get_dataloaders(
    batch_size=32,
    num_workers=0,
    pin_memory=False,
    train_transform=None,
    val_transform=None
):
    """
    获取训练集和验证集的 DataLoader

    参数:
        batch_size:   批大小（默认 32）
        num_workers:  数据加载线程数（CPU 环境建议 0）
        pin_memory:   是否使用锁页内存（CPU 环境无效）
        train_transform: 自定义训练变换
        val_transform:    自定义验证变换

    返回:
        train_loader, val_loader, class_names
    """
    train_dataset, val_dataset = get_datasets(train_transform, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ImageFolder 会按文件夹排序，验证类别名称顺序
    actual_class_names = train_dataset.classes
    print(f"类别数量: {len(actual_class_names)}")
    print(f"类别名称: {actual_class_names}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    return train_loader, val_loader, actual_class_names


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """反标准化：将张量转回可显示的图像范围 [0,1]"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def show_batch(images, labels, class_names, max_show=8):
    """显示一个 batch 的图像（用于验证增强效果）"""
    images = images[:max_show]
    labels = labels[:max_show]
    images = denormalize(images).clamp(0, 1)

    cols = min(4, max_show)
    rows = (max_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i in range(min(max_show, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        label = class_names[labels[i].item()]
        axes[i].imshow(img)
        axes[i].set_title(label)
        axes[i].axis("off")

    # 隐藏多余的子图
    for i in range(min(max_show, len(images)), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 测试数据加载
    print("=" * 60)
    print("测试数据加载与增强模块")
    print("=" * 60)

    # 加载数据
    train_loader, val_loader, class_names = get_dataloaders(
        batch_size=16,
        num_workers=0,
    )

    # 获取一个 batch 查看
    train_iter = iter(train_loader)
    images, labels = next(train_iter)

    print(f"\nBatch 形状: {images.shape}")        # [16, 3, 224, 224]
    print(f"标签形状: {labels.shape}")             # [16]
    print(f"标签值: {labels}")
    print(f"对应类别: {[class_names[l] for l in labels]}")

    # 显示增强后的图像（如果支持图形界面）
    try:
        show_batch(images, labels, class_names)
        print("\n[OK] 图像显示成功（请关闭图像窗口继续）")
    except Exception as e:
        print(f"\n[WARN] 无法显示图像（无图形界面）: {e}")

    print("\n[OK] 数据加载测试完成！")
