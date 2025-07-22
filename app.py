import torch
import torch.nn as nn
import torchvision.models as models # <--- 1. 新增导入
from PIL import Image
from flask import Flask, jsonify, request, render_template
import torchvision.transforms as transforms
import io # <--- 2. 建议导入，处理字节流更稳定

MODEL_PATH = 'training/garbage_best_model.pth' 

CLASS_NAMES = ['bukehuishou', 'kehuishou'] 
NUM_CLASSES = len(CLASS_NAMES)

model = models.resnet34(pretrained=False) # pretrained=False, 我们要加载自己的权重
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval() 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
        
app = Flask(__name__)

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
            output_image = model(tensor_image)

        _, predicted = torch.max(output_image, 1)

        # e. 根据索引从类别列表中找到对应的类别名字
        out = CLASS_NAMES[predicted.item()]

        return jsonify({'predicted_class': out})

    except Exception as e:
        
        return jsonify({'error': f'预测过程中发生错误: {str(e)}'}), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

