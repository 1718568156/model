import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F



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
# 定义模型保存的路径
PATH = './cifar_best_model.pth'

# 加载已保存的模型权重
net.load_state_dict(torch.load(PATH))

# 同样，把模型送到GPU（如果可用）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

print(f"--- : 模型已从 {PATH} 加载，并送至 {device} ---")


# ==============================================================================
# Part 3: 准备测试数据
# ==============================================================================
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

batch_size = 64

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print("--- Step 3: 测试数据准备完毕 ---")



print("--- Step 4: 开始在测试集上评估 ---")
correct = 0
total = 0

# 我们不需要计算梯度，所以用 torch.no_grad() 来提高效率
with torch.no_grad():
    net.eval()
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        
        # 1. 运行模型进行预测
        outputs = net(images)
        
        # 2. torch.max会返回最大值和它的索引。我们只需要索引（也就是预测的类别）
        _, predicted = torch.max(outputs.data, 1)
        
        # 3. 统计总数和正确预测的数量
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算并打印总体准确率
accuracy = 100 * correct / total
print(f'模型在10000张测试图片上的准确率是: {accuracy:.2f} %')



print("--- Step 5: 开始分门别类地评估 ---")
# 准备一个字典来存放每个类别的预测情况
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        
        # 遍历这个批次里的每一张图片
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# 打印出每个类别的准确率
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"类别 '{classname:5s}' 的准确率是: {accuracy:.1f} %")

print("--- 评估结束! ---")