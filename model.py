"""
ResNet-18 动物分类模型
- 使用 torchvision 预训练 ResNet-18 + 残差连接
- 替换最后一层全连接层适配 10 分类
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import time


class Config:
    """模型与训练超参数配置"""

    # ─── 训练参数 ───
    EPOCHS = 25
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    STEP_SIZE = 7          # 每 STEP_SIZE 个 epoch 衰减一次学习率
    GAMMA = 0.1            # 学习率衰减因子

    # ─── 模型参数 ───
    NUM_CLASSES = 10
    IMG_SIZE = 224

    # ─── 路径 ───
    DATA_DIR = os.path.join(os.path.dirname(__file__), "processed_datasets")
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

    # ─── 设备 ───
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── 断点续训 ───
    RESUME = False
    RESUME_PATH = ""


class BasicBlock(nn.Module):
    """
    ResNet-18 的基本残差块（BasicBlock）
    包含两个 3×3 卷积层，通过跳跃连接（Shortcut）实现残差学习
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入输出维度不匹配，通过 downsample 调整维度
        if self.downsample is not None:
            identity = self.downsample(x)

        # ─── 残差连接（核心）：将输入加到输出上 ───
        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    手动实现的 ResNet-18 模型
    展示完整的残差连接结构
    """

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # 初始卷积层：7×7 → BatchNorm → ReLU → MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个残差层（stage），每个包含多个 BasicBlock
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # 全局平均池化 + 全连接分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 参数初始化
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """构建一个残差层（包含多个 BasicBlock）"""
        downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            # 1×1 卷积调整维度，使跳跃连接可以相加
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels * BasicBlock.expansion,
                                     out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Kaiming 正态初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class AnimalClassifier(nn.Module):
    """
    高级封装：使用 torchvision 预训练的 ResNet-18
    提供开箱即用的训练、验证、保存、加载功能
    """

    def __init__(self, config=None, use_pretrained=True):
        """
        初始化分类器

        参数:
            config: Config 实例，None 则使用默认配置
            use_pretrained: 是否加载 ImageNet 预训练权重
        """
        super(AnimalClassifier, self).__init__()
        self.config = config if config is not None else Config()

        # 加载预训练 ResNet-18
        from torchvision import models
        from torchvision.models import ResNet18_Weights
        if use_pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            print("[INFO] 已加载 ImageNet 预训练权重")
        else:
            self.backbone = models.resnet18(weights=None)
            print("[INFO] 使用随机初始化权重")

        # 获取 backbone 的输入特征维度
        in_features = self.backbone.fc.in_features

        # ─── 替换最后一层全连接层（迁移学习的关键步骤） ───
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, self.config.NUM_CLASSES),
        )

        # 保存训练历史
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        # 将模型移动到指定设备（CPU / CUDA）
        self.to(self.config.DEVICE)

        # 最佳模型路径
        self.best_model_path = os.path.join(self.config.SAVE_DIR, "best_model.pth")
        self.last_model_path = os.path.join(self.config.SAVE_DIR, "last_model.pth")

    def forward(self, x):
        return self.backbone(x)

    def train_one_epoch(self, train_loader, optimizer, criterion, epoch, config):
        """训练一个 epoch"""
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}] Train")
        for images, labels in pbar:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = self(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.2f}%"
            })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def validate(self, val_loader, criterion, config):
        """验证集评估"""
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for images, labels in pbar:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

                outputs = self(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader, config=None):
        """
        完整训练流程

        参数:
            train_loader: 训练 DataLoader
            val_loader:   验证 DataLoader
            config:       训练配置，None 则使用 self.config
        """
        if config is None:
            config = self.config

        # 创建保存目录
        os.makedirs(config.SAVE_DIR, exist_ok=True)

        # 损失函数 & 优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = StepLR(optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)

        # 断点续训
        start_epoch = 0
        best_val_acc = 0.0
        if config.RESUME and os.path.exists(config.RESUME_PATH):
            checkpoint = torch.load(config.RESUME_PATH, map_location=config.DEVICE)
            self.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            print(f"[INFO] 从 epoch {start_epoch} 恢复训练")

        print(f"\n{'='*60}")
        print(f"设备: {config.DEVICE}")
        print(f"训练配置: EPOCHS={config.EPOCHS}, BATCH_SIZE={config.BATCH_SIZE}, "
              f"LR={config.LEARNING_RATE}")
        print(f"{'='*60}\n")

        # 训练循环
        for epoch in range(start_epoch, config.EPOCHS):
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_one_epoch(
                train_loader, optimizer, criterion, epoch, config
            )

            # 验证
            val_loss, val_acc = self.validate(val_loader, criterion, config)

            # 学习率调整
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # 打印结果
            elapsed = time.time() - start_time
            print(f"\nEpoch [{epoch+1}/{config.EPOCHS}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s\n")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save(self.best_model_path, epoch, optimizer, best_val_acc)
                print(f"  >>> 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")

            # 保存最后一个 epoch 的模型
            self.save(self.last_model_path, epoch, optimizer, best_val_acc)

        print(f"\n{'='*60}")
        print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")
        print(f"最佳模型保存至: {self.best_model_path}")
        print(f"{'='*60}")

        return self.history

    def save(self, path, epoch, optimizer, best_val_acc):
        """保存模型检查点"""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        }, path)

    def load(self, path):
        """加载模型权重"""
        if not os.path.exists(path):
            print(f"[WARN] 未找到模型文件: {path}")
            return False

        checkpoint = torch.load(path, map_location=self.config.DEVICE)
        self.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        best_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"[INFO] 已加载模型: {path}")
        print(f"       上次训练 epoch: {epoch+1}, 最佳验证准确率: {best_acc:.2f}%")
        return True

    def predict(self, image_tensor):
        """
        单张图像推理

        参数:
            image_tensor: 经过预处理的图像张量 [1, 3, 224, 224]

        返回:
            class_name, confidence
        """
        self.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.config.DEVICE)
            outputs = self(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        from dataset import CLASS_NAMES
        class_name = CLASS_NAMES[predicted.item()]
        confidence = confidence.item()

        return class_name, confidence


if __name__ == "__main__":
    # ─── 测试模型结构 ───
    print("=" * 60)
    print("测试 ResNet-18 模型结构")
    print("=" * 60)

    from dataset import get_dataloaders

    # 1. 测试手动实现的 ResNet18
    print("\n1. 测试手动 ResNet18...")
    manual_model = ResNet18(num_classes=10)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = manual_model(dummy_input)
    print(f"   输入形状: {dummy_input.shape}")
    print(f"   输出形状: {output.shape}")  # 应为 [4, 10]
    print("   [OK] 手动 ResNet18 前向传播正常")

    # 2. 测试预训练分类器
    print("\n2. 测试预训练 AnimalClassifier...")
    config = Config()
    classifier = AnimalClassifier(config=config, use_pretrained=True)
    output = classifier(dummy_input)
    print(f"   输入形状: {dummy_input.shape}")
    print(f"   输出形状: {output.shape}")  # 应为 [4, 10]
    print("   [OK] 预训练分类器前向传播正常")

    # 3. 统计模型参数量
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"\n   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")

    # 4. 测试数据加载 + 模型推理（可选）
    print("\n3. 测试 DataLoader + 模型推理...")
    train_loader, val_loader, class_names = get_dataloaders(batch_size=4)

    images, labels = next(iter(train_loader))
    outputs = classifier(images)
    _, predicted = torch.max(outputs, 1)
    print(f"   真实标签: {[class_names[l] for l in labels]}")
    print(f"   预测结果: {[class_names[p] for p in predicted]}")
    print("   [OK] 完整流水线测试通过")

    print(f"\n{'='*60}")
    print("所有测试通过！")
    print(f"{'='*60}")