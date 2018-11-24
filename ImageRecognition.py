# -*- coding: utf-8 -*-

from tensorflow.python.keras import backend
backend.set_image_data_format('channels_first')

import pickle
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import np_utils

__time__ = '2018/10/11 19:57'
__author__ = 'Mr.DONG'
__File__ = 'ImageRecognition.py'
__Software__ = 'PyCharm'
'''
简单卷积神经网络 CNN
网络拓扑结构 :  两个卷积层,一个池化层,一个Flatten层 和 一个全连接层 
    1.卷积层  32个特征图, 感受野大小 3x3
    2.Dropout层,概率为20% 
    3.卷积层 , 32个特征图 , 感受野大小 3x3
    4.Dropout层 , 为 20% 的概率
    5.采样因子, pool_size 为 2x2
    6.Flatten 层
    7. 具有512个神经元和 ReLU激活函数为全连接层
    8.Dropout 层概率为 50% 
    9.具有 10 个神经元的输出层,激活函数为softmax

图像识别实力 : CIFAR-10
数据集  : 
    CIFAR100 小图片分类数据库
该数据库具有50,000个32*32的彩色图片作为训练集，10,000个图片作为测试集。图片一共有100个类别，每个类别有600张图片。这100个类别又分为20个大类。
使用方法
from keras.datasets import cifar100
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
参数
label_mode：为‘fine’或‘coarse’之一，控制标签的精细度，‘fine’获得的标签是100个小类的标签，‘coarse’获得的标签是大类的标签
返回值
两个Tuple,(X_train, y_train), (X_test, y_test)，其中
X_train和X_test：是形如（nb_samples, 3, 32, 32）的RGB三通道图像数据，数据类型是无符号8位整形（uint8）
y_train和y_test：是形如（nb_samples,）标签数据，标签的范围是0~9
'''
import numpy as np
from matplotlib import pyplot as plt


def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

dataTrain= load_file('cifar-10-batches-py/data_batch_1')
dataTest= load_file('cifar-10-batches-py/test_batch')
x_train=dataTrain['data']
y_train=dataTrain['labels']
x_validation=dataTest['data']
y_validation=dataTest['labels']

seed = 7
np.random.seed(seed)

# 格式化数据到 0 ~ 1
x_train = x_train.reshape(x_train.shape[0],3,32,32).astype('float32')
x_validation = x_validation.reshape(x_validation.shape[0],3,32,32).astype('float32')
x_train = x_train / 255.0
x_validation = x_validation / 255.0

# 进行 ont - hot编码

y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_class = y_train.shape[1]
print(num_class)
def create_model(epochs=25):
    model = Sequential()
    model.add(Conv2D(32,(3,3),input_shape=(3,32,32),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=512,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(units=10,activation='softmax'))
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

epochs = 25
model =  create_model(epochs)
model.fit(x=x_train,y=y_train,epochs=epochs,batch_size=32,verbose=2)
scores =  model.evaluate(x=x_validation,y=y_validation,verbose=0)
print('Accuracy :%.2f%%'% (scores[1] * 100 ))



