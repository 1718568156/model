import torch
import torch.nn as nn
import os
import torch.optim as optim
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm 
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.models as models
print("--- Step 1: 库导入成功 ---")


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] )


batch_size = 32

epochs = 20

train_path = 'split_garbage_dataset/train'
val_path = 'split_garbage_dataset/val'

train_dataset = ImageFolder(root=train_path, transform=train_transforms)
val_dataset = ImageFolder(root=val_path, transform=val_transforms)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

classes = train_dataset.classes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))
model.to(device)

print(f"--- Step 3: 模型定义完毕，将在 {device} 上运行 ---")

train_counts = [0] * len(train_dataset.classes)
for _, label_idx in train_dataset.samples:#第一个赋值路径，第二个赋值类别
    train_counts[label_idx] += 1

total_samples = float(sum(train_counts))
class_weights = [total_samples / count for count in train_counts]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

print(f"计算出的类别权重: {class_weights_tensor}")

print("--- Step 4: 损失函数和优化器定义完毕 ---")

def evaluate(model , dataloader , device):
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad(): 
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

print("--- Step 5: 开始训练 ---")
total_start_time = time.time() 
best_accuracy = 0.0
PATH = './garbage_best_model.pth'
history = {'train_loss': [], 'test_accuracy': []}#两个字典

for epoch in range(epochs): 
    model.train() 
    running_loss = 0.0
    progress_bar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch {epoch + 1}/{epochs}")#进度条

    for i, data in progress_bar:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()#防止累加
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
       
    #评估阶段
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(trainloader)}')
    scheduler.step()#更新学习率
    test_accuracy = evaluate(model, valloader, device)
    history['train_loss'].append(running_loss / len(trainloader))
    history['test_accuracy'].append(test_accuracy)
    print(f"\nEpoch {epoch+1} Summary | Train Loss: {running_loss / len(trainloader):.4f} | Test Accuracy: {test_accuracy:.2f}%")
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        print(f"🎉 New best accuracy! Saving model to '{PATH}'")
        torch.save(model.state_dict(), PATH)

total_end_time = time.time()
total_duration = total_end_time - total_start_time
print(f'训练全部完成！总耗时: {total_duration:.2f} 秒')
print("--- 训练结束! ---")
#画图
print("--- Step 7: 绘制训练曲线 ---")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['test_accuracy'], label='Test Accuracy', color='orange')
plt.title('Test Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_curvesa.png')
print("--- 训练曲线图已保存到 training_curves.png ---")
plt.show()