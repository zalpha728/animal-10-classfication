"""
训练入口脚本
- 加载数据集 → 创建模型 → 训练 → 保存模型参数
"""

import os
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import Config, AnimalClassifier


def plot_history(history, save_path=None):
    """绘制训练过程中的 loss 和 accuracy 曲线"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 曲线
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy 曲线
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] 训练曲线已保存至: {save_path}")

    plt.show()


def main():
    print("=" * 60)
    print("Animal-10 ResNet-18 模型训练")
    print("=" * 60)

    # ─── 1. 加载配置 ───
    config = Config()
    print(f"\n[配置]")
    print(f"  设备: {config.DEVICE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  学习率: {config.LEARNING_RATE}")
    print(f"  模型保存路径: {config.SAVE_DIR}")

    # ─── 2. 加载数据集 ───
    print(f"\n[数据集]")
    train_loader, val_loader, class_names = get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=0,
    )

    # ─── 3. 创建模型 ───
    print(f"\n[模型]")
    model = AnimalClassifier(config=config, use_pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # ─── 4. 训练 ───
    print(f"\n[开始训练]")
    history = model.fit(train_loader, val_loader)

    # ─── 5. 绘制训练曲线 ───
    print(f"\n[结果可视化]")
    curve_path = os.path.join(config.SAVE_DIR, "training_curve.png")
    plot_history(history, save_path=curve_path)

    # ─── 6. 打印总结 ───
    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"{'='*60}")
    print(f"  最佳验证准确率: {max(history['val_acc']):.2f}%")
    print(f"  最终验证准确率: {history['val_acc'][-1]:.2f}%")
    print(f"  最佳模型: {model.best_model_path}")
    print(f"  最后模型: {model.last_model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()