# -*- coding: utf-8 -*-

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.utils import np_utils

__time__ = '2018/10/10 22:48'
__author__ = 'Mr.DONG'
__File__ = 'Complex_CNN.py'
__Software__ = 'PyCharm'

'''
发现问题 : 默认是tf  所以数据维度千万不要错了
    图片维序类型为 th 时（dim_ordering='th'）： 输入数据格式为[samples][channels][rows][cols]；
    图片维序类型为 tf 时（dim_ordering='tf'）： 输入数据格式为[samples][rows][cols][channels]；


复杂卷积神经网络
网络拓扑结构 : 
    卷积层 : 30个特征图,感受野大小 5x5
    采样因子 : pool_size 为 2x2 的池化层
    卷积层,具有15个特征图,感受野大小为 3x3
    采样因子 : pool_size 为 2x2的池化层
    Dropout概论Wie20% 的Dropout层
    Flatten层
    具有128个神经元和ReLU激活函数的全连接层
    具有50个神经元和ReLU激活函数的全连接层
    输出层
'''
import numpy as np

#设置随机种子
seed  =7
np.random.seed(seed)

# 从 Keras 中导入 mnist 数据集
path = 'mnist.npz'
f = np.load(path)
X_train, y_train = f['x_train'], f['y_train']
X_validation, y_validation = f['x_test'], f['y_test']

X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_validation = X_validation.reshape(X_validation.shape[0],28,28,1).astype('float32')

# 格式化数据 0 ~ 1
X_train = X_train / 255
X_validation = X_validation / 255

y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)


def create_model():
    model = Sequential()
    # 第一个隐藏层 为 Conv2D 的卷积层 , 使用 5x5 的视野
    model.add(Conv2D(30,(5,5),input_shape=(28,28,1),activation='relu',padding="valid"))
    # 定义一个采用最大值的MaxPooling2D的池化层
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 正则化层
    model.add(Dropout(0.2))
    # 转换为一位数据的Flatten层 , 输出便于标准的全连接层的处理
    model.add(Flatten())
    # 128个神经元的全连接层
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=50,activation='relu'))
    # 输出层有10个神经元
    model.add(Dense(units=10,activation='softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model = create_model()
model.fit(X_train,y_train,epochs=10,batch_size=200,verbose=2)
score = model.evaluate(X_validation,y_validation,verbose=0)
print('CNN_Small: %.2ff%%' %(score[1] * 100 ))









