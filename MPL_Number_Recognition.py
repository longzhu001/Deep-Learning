# -*- coding: utf-8 -*-
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils import np_utils

__time__ = '2018/10/10 21:06'
__author__ = 'Mr.DONG'
__File__ = 'Number_Recognition.py'
__Software__ = 'PyCharm'


'''
多层感知器模型

数字图片识别   
    预测结果:  是途中的手写数字识别0~9
    
图像信息会被保存到每位 由 0 ~ 255 的数字构成的 28 x 28 的矩阵中
    因此 多层感知器模型的
    输入层的神经元个数是 784(28x28) 
    隐藏层 - 同样也构建了一个包含 784个神经元的隐藏层
    输入层和隐藏层的激活函数都才有 ReLU
    输出层 包含 10 个神经元 ,激活函数才有 softmax

数据处理 : 
    60000张图片模型
    单独使用10000张图像来评估模型的准确度

'''

'''
数据由于 SSL 证书的验证问题,无法直接在网上下载, 只能先下载数据包,通过 numpy.load这个方法识别npz的格式数据
然后数据里面分好的 测试数据和训练数据
第一种
f = np.load(mnist.npz)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
第二种
(X_train,y_train),(X_validation,y_validation)=mnist.load_data()
'''
# 从 Keras 中导入 mnist 数据集
path = 'mnist.npz'
f = np.load(path)
X_train, y_train = f['x_train'], f['y_train']
X_validation, y_validation = f['x_test'], f['y_test']

# 显示 4 张手写的数字图片
# 第一张
plt.subplot(221)
plt.imshow(X_train[0],cmap=plt.get_cmap('gray'))
# 第二张
plt.subplot(222)
plt.imshow(X_train[1],cmap=plt.get_cmap('gray'))
# 第三张
plt.subplot(223)
plt.imshow(X_train[2],cmap=plt.get_cmap('gray'))
# 第四张
plt.subplot(224)
plt.imshow(X_train[3],cmap=plt.get_cmap('gray'))

plt.show()

#设置随机种子
seed  =7
np.random.seed(seed)

num_pixels = X_train.shape[1] * X_train.shape[2]
print(num_pixels)

X_train = X_train.reshape(X_train.shape[0],num_pixels).astype('float32')
X_validation = X_validation.reshape(X_validation.shape[0],num_pixels).astype('float32')

# 格式化数据 0 ~ 1
X_train = X_train / 255
X_validation = X_validation / 255

'''
one-hot 独热编码 
    获取的原始特征，必须对每一特征分别进行归一化 
    one hot编码是将类别变量转换为机器学习算法易于利用的一种形式的过程
    使用one-hot编码，将离散特征的取值扩展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点
    将离散特征通过one-hot编码映射到欧式空间，是因为，在回归，分类，聚类等机器学习算法中，特征之间距离的计算或相似度的计算是非常重要的，而我们常用的距离或相似度的计算都是在欧式空间的相似度计算，计算余弦相似性，基于的就是欧式空间
'''
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_validation.shape[1]
print(num_classes)

#  定义基准 多层感知器模型(MLP)
def create_model():
    # 创建模型
    model = Sequential()
    # 输入层
    model.add(Dense(units=num_pixels,input_dim=num_pixels,kernel_initializer='normal',activation='relu'))
    # 输出层
    model.add(Dense(units=num_classes,kernel_initializer='normal',activation='softmax'))
    # 编译模型
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model = create_model()
model.fit(X_train,y_train,epochs=10,batch_size=200)

score = model.evaluate(X_validation,y_validation)
print('MPL : %.2f%%' %(score[1]*100))

'''
    MPL : 83.60%
'''












