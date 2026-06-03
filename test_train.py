"""快速测试训练流程（1个 epoch）"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from model import Config
from dataset import get_dataloaders
from model import AnimalClassifier

config = Config()
config.EPOCHS = 1  # 只跑 1 个 epoch 测试

train_loader, val_loader, class_names = get_dataloaders(batch_size=config.BATCH_SIZE)

model = AnimalClassifier(config=config, use_pretrained=True)
history = model.fit(train_loader, val_loader)

print("\n测试通过！模型已保存至:")
print(f"  {model.best_model_path}")
print(f"  {model.last_model_path}")