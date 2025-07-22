import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np # 您之前的代码环境很可能已经有numpy
import os
import matplotlib.pyplot as plt # 用于最后的绘图（如果需要）
import matplotlib
matplotlib.use('Agg') # 确保在没有图形界面的服务器上也能保存图片


print("--- Step 1: 库导入成功 ---")

# ==============================================================================
# Part 1: 加载新模型和数据
# ==============================================================================
MODEL_PATH = 'garbage_best_model.pth' 
VAL_PATH = 'split_garbage_dataset/val'

if not os.path.exists(MODEL_PATH) or not os.path.exists(VAL_PATH):
    print(f"错误：找不到模型 '{MODEL_PATH}' 或验证集 '{VAL_PATH}'。")
    exit()

# 动态获取类别信息
temp_dataset = ImageFolder(root=VAL_PATH)
NUM_CLASSES = len(temp_dataset.classes)
CLASS_NAMES = temp_dataset.classes

# 定义模型结构
model = models.resnet34(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)

# 加载模型权重
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print(f"--- 模型已从 '{MODEL_PATH}' 加载，类别为: {CLASS_NAMES} ---")

# 准备验证数据
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_dataset = ImageFolder(root=VAL_PATH, transform=val_transforms)
valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

print("--- Step 2: 评估数据准备完毕 ---")

# ==============================================================================
# Part 2: 在验证集上进行评估 (无新依赖版)
# ==============================================================================
print("--- Step 3: 开始在验证集上进行评估 ---")

# 初始化用于统计的变量
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in valloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# --- Part 3.1: 计算总体准确率 ---
correct_predictions = np.sum(np.array(all_predictions) == np.array(all_labels))
total_samples = len(all_labels)
accuracy = 100 * correct_predictions / total_samples
print(f'\n模型在 {total_samples} 张验证图片上的总体准确率是: {accuracy:.2f} %')


# --- Part 3.2: 分类别计算准确率 (和您原来代码逻辑一致) ---
print("\n--- 分类别准确率 ---")
# 创建一个字典来存放每个类别的预测情况
correct_pred_per_class = {classname: 0 for classname in CLASS_NAMES}
total_pred_per_class = {classname: 0 for classname in CLASS_NAMES}

for label, prediction in zip(all_labels, all_predictions):
    if label == prediction:
        correct_pred_per_class[CLASS_NAMES[label]] += 1
    total_pred_per_class[CLASS_NAMES[label]] += 1

for classname, correct_count in correct_pred_per_class.items():
    # 避免除以0的错误
    total_count = total_pred_per_class[classname]
    class_accuracy = 100 * float(correct_count) / total_count if total_count > 0 else 0
    print(f"类别 '{classname:12s}' 的准确率是: {class_accuracy:.2f} % ({correct_count}/{total_count})")


# --- Part 3.3: 手动计算并打印混淆矩阵 (无新依赖) ---
print("\n--- 文本版混淆矩阵 ---")
# 初始化一个 N x N 的零矩阵，N是类别数
confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

for label, prediction in zip(all_labels, all_predictions):
    confusion_matrix[label, prediction] += 1

# 打印混淆矩阵
# 打印表头 (预测标签)
header = " " * 15 + " ".join([f"{name[:6]:>6}" for name in CLASS_NAMES])
print(header)
print("-" * len(header))

# 打印每一行 (真实标签)
for i, class_name in enumerate(CLASS_NAMES):
    row_str = f"{class_name[:12]:<12s} | "
    row_str += " ".join([f"{count:>6}" for count in confusion_matrix[i]])
    print(row_str)

print("\n说明：行代表真实标签，列代表预测标签。")
print("例如，第一行第二列的数字表示有多少个真实的'bukehuishou'被错误地预测成了'kehuishou'。")

print("\n--- 评估结束! ---")