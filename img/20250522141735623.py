""""
加载预先保存的图片向量，与 query.jpg 提取的向量进行比对，（可进行进一步更改使得图片格式不仅局限于jpg文件）
返回最相似的三个图片路径及距离。
无需每次遍历images文件夹，仅加载一次 .npz 数据。

测试API方式：
1.在pycharm中运行本程序，
2.在cmd中输入：
curl -X POST -F "file=@E:/TOTOTO/Python_project/query.jpg" http://localhost:5000/search
请注意2中引号中@后是query图片所在的文件地址，且注意斜杠是除号
"""
import os
from typing import Any
#下面两行是暂时跳过dll错误，找不到了有空再重新配置
#dll错误貌似与import的包没有关系，仍需排查
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#先不要乱动那个解释器的虚拟环境和当前项目关联了，起码能跑
import numpy as np
import faiss
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import ResNet50_Weights

# API需要的flask框架
from flask import Flask, request, jsonify,session
from flask_cors import CORS
from werkzeug.utils import secure_filename  # 确保导入安全文件名处理
import json
import timedelta
# from sql import db

# 1.模型
# 方式 A：预训练权重
#model = models.resnet50(weights=True).eval()# 定义resnet50模型，这是pytorch中的一个模型库
# 其中ResNet-50 是一个经典的深度卷积神经网络，有 50 层深度，常用于图像分类任务
# 或者 方式 B：显式指定权重（需 torchvision >= 0.13）
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()

# 将模型（如 ResNet-50）的最后一层移除，并用剩余部分构建一个新的特征提取器feature_extractor
feature_extractor = nn.Sequential(*list(model.children())[:-1])
# 原始模型（如 ResNet-50）最后一层通常是全连接层（nn.Linear），用于分类任务。移除后，模型仅保留特征提取部分。

# 2.预处理：定义一个预处理流程（此时不会执行任何操作），实际使用时才会执行预处理
preprocess = transforms.Compose([ # 对所有图像进行统一处理
    transforms.Resize(256),# 将图像的短边缩放到 256 像素，长边按比例缩放，保持图像宽高比不变。
    transforms.CenterCrop(224), # 从图像中心裁剪出一个 224x224 像素的正方形区域（能否改进？例如保留更大的区域）
    transforms.ToTensor(), # 将数据转化为pytorch格式可用
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 归一化 即对每个像素值先减去均值mean，再除以标准差std
    # 归一化目的：将数据分布调整到均值为0、标准差为1的标准正态分布，加速模型收敛并提升性能。
])

#func：读照片进来并处理
def extract_vector(image_path: object) -> Any:
    img = Image.open(image_path).convert('RGB') # 打开图并转RGB
    x = preprocess(img).unsqueeze(0)  # 对图像进行预处理，最终结果是[1,3,224,224]
    with torch.no_grad(): # 禁用梯度运算（为了快）
        feat = feature_extractor(x).squeeze()  # 放进模型中并去除最后一层，结果为[2048]
    vec = feat.cpu().numpy() # 将张量从 GPU 移动到 CPU，并转换为 NumPy 数组（如果在GPU运行则是必要的）
    vec /= (np.linalg.norm(vec) + 1e-6)# 对特征向量进行 L2 归一化，分母即计算向量的欧几里得距离（防止分母为0才加的1e-6）
    return vec.astype('float32')# 返回了一个归一化的特征向量

# func：读照片并搜索函数
""""
def search_topk(query_path, k=3): # k是返回结果数量（最接近的k个）
    qv = extract_vector(query_path) # 读照片进来处理
    D, I = index.search(np.expand_dims(qv, axis=0), k) # 在faiss索引中搜索最相似的k个向量（用欧几里得距离搜索）
    return [(paths[i], float(D[0][j])) for j, i in enumerate(I[0])] # 将faiss返回的索引位置和距离转换为具体的图像路径和数值
"""
def search_topk(query_path, index, paths, k=3): # k是返回结果数量（最接近的k个）
    qv = extract_vector(query_path) # 读照片进来处理
    D, I = index.search(np.expand_dims(qv, axis=0), k) # 在faiss索引中搜索最相似的k个向量（用欧几里得距离搜索）
    return [(paths[i], float(D[0][j])) for j, i in enumerate(I[0])] # 将faiss返回的索引位置和距离转换为具体的图像路径和数值

#API接口配置：
app = Flask(__name__)
#数据库中有 app = Flask(__name__)时 直接使用数据库中的app
# app = db.app
CORS(app, supports_credentials=True)
#app = db.app
app.secret_key="key not base63"
app.config['SESSION_COOKIE_NAME'] = 'session_key'

@app.route('/search', methods=['GET','POST'])# API接口：提交查询招聘，终端使用的是POST方法
def search():
    # 加载预计算的特征
    data = np.load('image_features.npz', allow_pickle=True)  # 加载训练结果.npz文件
    paths = data['paths']  # 从文件中得出图片路径
    vectors = data['vectors']  # 从文件中得出图片向量
    # 3.构建Faiss索引
    d = 2048  # 特征向量的维度（ResNet-50提取的特征向量维度为 2048）
    index = faiss.IndexFlatL2(d)  # 我也不知道，反正就是创造一个索引对象进行相似性搜索
    index.add(vectors)
    # 4.提取query特征并搜索
    #query = './query.jpg'
    #results = search_topk(query, k=3)  # 读照片并搜索
    # 5.输出结果
    #print("Top-3 similar images:")
    #for p, dist in results:
        #print(f"{p}\t距离: {dist:.4f}")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # 获取安全的文件名并提取扩展名
    original_filename = secure_filename(file.filename)
    _, ext = os.path.splitext(original_filename)
    # 生成动态文件名（保留原扩展名）
    query_filename = f"query{ext}"
    query_path = os.path.join('.', query_filename)
    file.save(query_path)
    # 执行搜索
    try:
        results = search_topk(query_path, index, paths, k=3)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    # 构建响应
    response = [{'path': p, 'distance': d} for p, d in results]
    return jsonify({'results': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

""""
下边是非API的控制台输出，用于验证查找算法是否正确
#运行示例
if __name__ == '__main__':
    # 加载预计算的特征
    data = np.load('image_features.npz', allow_pickle=True)# 加载训练结果.npz文件
    paths = data['paths'] # 从文件中得出图片路径
    vectors = data['vectors'] # 从文件中得出图片向量

    # 3.构建Faiss索引
    d = 2048  # 特征向量的维度（ResNet-50提取的特征向量维度为 2048）
    index = faiss.IndexFlatL2(d)  # 我也不知道，反正就是创造一个索引对象进行相似性搜索
    index.add(vectors)

    # 4.提取query特征并搜索
    query = './query.jpg'
    results = search_topk(query, k=3)  # 读照片并搜索

    # 5.输出结果
    print("Top-3 similar images:")
    for p, dist in results:
        print(f"{p}\t距离: {dist:.4f}")
"""