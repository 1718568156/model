import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm 
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR


print("--- Step 1: 库导入成功 ---")



transform = transforms.Compose(
    [
    transforms.RandomHorizontalFlip(), # 50%的概率水平翻转图片
    transforms.RandomCrop(32, padding=4), #填充加裁剪
    transforms.ToTensor(),#归一化
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])#训练集

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])#测试集

# 定义每个批次加载的图片数量
batch_size = 128
#轮次
epochs = 40

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)#shuffle打乱数据，防止过拟合

# 准备测试数据集（过程同上，只是 train=False）
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)

# 定义10个类别的名称，方便后续查看
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

print("--- Step 2: 数据集准备完毕 ---")

#改版定义
class net(nn.Module):
    def __init__(self):
         super().__init__()
         self.conv1=nn.Sequential(
         nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
         nn.BatchNorm2d(32),
         nn.ReLU(inplace=True),#inplace会破坏当前层的数据，从而节省内存
         nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
         nn.BatchNorm2d(64),
         nn.ReLU(inplace=True),
         # 最大池化层，将尺寸减半
         nn.MaxPool2d(kernel_size=2, stride=2)


         )
         self.conv_block2 = nn.Sequential(
         # 通道数从64增加到128
         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
         nn.BatchNorm2d(128),
         nn.ReLU(inplace=True),
            
         # 保持128通道，进一步提取特征
         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
         nn.BatchNorm2d(128),
         nn.ReLU(inplace=True),
            
         # 再次最大池化，尺寸减半
         nn.MaxPool2d(kernel_size=2, stride=2) # 尺寸: 16x16 -> 8x8
        )
         self.classifier = nn.Sequential(
         nn.Flatten(),#铺平
            # 全连接层。输入维度根据上面计算得出：128个通道 * 8x8的特征图
         nn.Linear(128 * 8 * 8, 1024),
         nn.ReLU(inplace=True),
            
            # 加入Dropout层，以50%的概率随机丢弃神经元，防止过拟合
         nn.Dropout(0.5),
            
         nn.Linear(1024, 512),
         nn.ReLU(inplace=True),
         nn.Dropout(0.5),
            
            # 最终输出层，10个类别
         nn.Linear(512, 10)
        )
    def forward(self, x):
        # 定义数据流：先通过两个卷积块，再通过分类器
        x = self.conv1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x
        
net=net()
# 将网络送到GPU上（如果GPU可用）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

print(f"--- Step 3: 模型定义完毕，将在 {device} 上运行 ---")

# 损失函数：使用交叉熵损失
criterion = nn.CrossEntropyLoss()

# 优化器：使用带动量的随机梯度下降(SGD)
optimizer = optim.Adam(net.parameters(), lr=0.001)#momentum惯性变量

#学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

print("--- Step 4: 损失函数和优化器定义完毕 ---")

#新加一个模块 评估模块

def evaluate(net , dataloader , device):#评估准确率
    net.eval() # 切换到评估模式 当切换到这一层时某些网络层会关闭功能
    correct = 0#正确图片
    total = 0#所有图片
    with torch.no_grad(): # 评估期间不需要计算梯度，不需要保留一些数据用于反向传播
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)#来次前向传播
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


print("--- Step 5: 开始训练 ---")
total_start_time = time.time() # 记录总的开始时间
best_accuracy = 0.0
PATH = './cifar_best_model.pth'
history = {'train_loss': [], 'test_accuracy': []}#两个字典

for epoch in range(epochs): 
    net.train() 
    running_loss = 0.0
    progress_bar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch {epoch + 1}/{epochs}")

    for i, data in progress_bar:
        # 1. 获取一个批次的数据，并送到GPU
        # data 是一个包含 [inputs, labels] 的列表
        inputs, labels = data[0].to(device), data[1].to(device)

        # 2. 梯度清零
        optimizer.zero_grad()#防止累加

        # 3. 核心三步：前向传播 -> 计算损失 -> 反向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 4. 更新权重
        optimizer.step()

        # 5. 打印训练过程中的信息
        running_loss += loss.item()
       
    #评估阶段
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(trainloader)}')
    scheduler.step()#更新学习率
    test_accuracy = evaluate(net, testloader, device)
    history['train_loss'].append(running_loss / len(trainloader))
    history['test_accuracy'].append(test_accuracy)
    print(f"\nEpoch {epoch+1} Summary | Train Loss: {running_loss / len(trainloader):.4f} | Test Accuracy: {test_accuracy:.2f}%")
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        print(f"🎉 New best accuracy! Saving model to '{PATH}'")
        torch.save(net.state_dict(), PATH)


    


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
plt.savefig('training_curves.png')
print("--- 训练曲线图已保存到 training_curves.png ---")
plt.show()