import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, jsonify, request,render_template
import torchvision.transforms as transforms

#net
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
        
app = Flask(__name__)
#模型
path = "./cifar_best_model.pth"

net=net()

#加载权重
net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

net.eval()
#数据预处理
transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

#图片类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

#路由

allowfiles = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowfiles

@app.route('/')
def home():
#     """这个函数负责提供我们的主网页"""
    return render_template('index.html')

@app.route("/predict",methods=['post'])
def main():
    #判断是否发东西了
    if 'image' not in request.files:
        return jsonify({'error': 'no file part in the request'}), 400
    
    file=request.files['image']

#发了不选
    if file.filename == '' :
        return jsonify({'error': ' please to select image'})
    #是否符合文件类型
    if not allowed_file(file.filename):
        return jsonify({'error': '文件格式不支持，请使用 png, jpg, jpeg'}), 400
    
    try:
        #读取图片文件流并转换为Pillow图像对象
        input_image = Image.open(file.stream).convert('RGB')
        tensor_image = transform(input_image).unsqueeze(0)

        with torch.no_grad():
            output_image = net(tensor_image)

        _, predicted = torch.max(output_image, 1)

        # e. 根据索引从类别列表中找到对应的类别名字
        out = classes[predicted.item()]

        # f. 准备一个成功的JSON响应
        return jsonify({'predicted_class': out})

    except Exception as e:
        # 如果在 try 的过程中发生任何错误，返回一个服务器内部错误
        return jsonify({'error': f'预测过程中发生错误: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


   
    

