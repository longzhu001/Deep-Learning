# -*- coding: utf-8 -*-
import pickle

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import np_utils

__time__ = '2018/10/11 22:01'
__author__ = 'Mr.DONG'
__File__ = 'ImageRecognition1.py'
__Software__ = 'PyCharm'
'''
大型卷积神经网络 CNN
网络拓扑结构 :  两个卷积层,一个池化层,一个Flatten层 和 一个全连接层 
    1.卷积层  32个特征图, 感受野大小 3x3
    2.Dropout层,概率为20% 
    3.卷积层 , 32个特征图 , 感受野大小 3x3
    4.采样因子, pool_size 为 2x2
    5.卷积层 , 64个特征图 , 感受野大小 3x3
    6.Dropout层 , 为 20% 的概率
    7.卷积层 , 64个特征图 , 感受野大小 3x3
    8.采样因子, pool_size 为 2x2
    9.卷积层 , 128个特征图 , 感受野大小 3x3
    10.Dropout层,概率为20% 
    11.卷积层 , 128个特征图 , 感受野大小 3x3
    12.采样因子, pool_size 为 2x2
    13.Flatten 层
    14.Dropout层,概率为20% 
    15.具有1024个神经元和 ReLU激活函数为全连接层
    16.Dropout层,概率为20% 
    17.具有512个神经元和 ReLU激活函数为全连接层
    18.Dropout层,概率为20% 
    19.具有 10 个神经元的输出层,激活函数为softmax

'''
import numpy as np

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
x_train = x_train.reshape(x_train.shape[0],32,32,3).astype('float32')
x_validation = x_validation.reshape(x_validation.shape[0],32,32,3).astype('float32')
x_train = x_train / 255.0
x_validation = x_validation / 255.0

# 进行 ont - hot编码

y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_class = y_train.shape[1]

def create_model(epochs=25):
    model = Sequential()
    model.add(Conv2D(32,(3,3),input_shape=(32,32,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(units=1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(units=512,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
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








