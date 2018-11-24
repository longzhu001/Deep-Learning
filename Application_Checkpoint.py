#!/usr/bin/env python
# encoding: utf-8
'''
@author: Mr_DONG
@software: PyCharm
@file: Application_Checkpoint.py
@time: 2018/10/9 14:58

神经网络的检查点  :  应用程序检查点
      是长时间运行进行的容错技术,  是在系统故障的情况下, 对系统状态快照保存的一种方法

      当训练深度学习的模型时 :
         可以利用检查点来捕获模型的权重,可以基于当前的权重进行预测,也可以使检查点保存的权重值继续训练模型

      ModeCheckpoint 回调类 可以定义模型权重值检查点的位置, 文件的名称,以及在什么情况下创建模型的检查点

      其中保存的文件, 也是通过  HDF5 格式文件来保存
'''
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils import to_categorical

'''
                检查点跟踪神经网络模型
        每当检查点 检查到模型性能提高时, 使用检查点保存输出模型的权重值 ,并且保存此文件

'''

from sklearn import datasets
import numpy as np

dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target
Y_labels = to_categorical(Y,num_classes=3)
seed = 7
np.random.seed(seed)

def create_model(optimizer='rmsprop',init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = create_model()

# 设置检查点
filepath = 'weights-improvment-{epoch : 02d}-{val_acc:.2f}.h5'
checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_acc',verbose=1,save_weights_only=True,mode='max')
callable_list = [checkpoint]
model.fit(x,Y_labels,validation_split=0.2,epochs=200,batch_size=5,verbose=0,callbacks=callable_list)

'''
                自动保存最优模型
        简单的策略就是将模型权重保存在同一个文件中,当且仅当模型的准确度提高时, 才会将权重更新保存到文件中
        当模型性能提高时,输出的权重文件, 就会覆盖上一次的结果 , 一直到保存到最好的模型为止
'''

# 设置检查点
filepath = 'weights.best.h5'
checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_acc',verbose=1,save_weights_only=True,mode='max')
callable_list = [checkpoint]
model.fit(x,Y_labels,validation_split=0.2,epochs=200,batch_size=5,verbose=0,callbacks=callable_list)


'''
                从检查点导入模型
        使用ModelCheckpoint 训练模型的过程中,通过检查点保存了模型的权重, 当训练模型时意外停止, 可以从自动保存的检查点加载和使用检查点时保存的模型
        假定 神经网络的拓扑结构是已知的, 又保存了模型的权重值, 就可以在模型训练前,序列化JSON格式或YAML格式的文件 , 以确保可以方便的回复网络的拓扑结构
            # 加载权重,恢复模型
            filepath = 'weights.best.h5'
            model.load_weights(filepath=filepath)
            就在返回模型的时候,加载权重 ,其余的不变
'''

dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target
Y_labels = to_categorical(Y,num_classes=3)
seed = 7
np.random.seed(seed)

def create_model(optimizer='rmsprop',init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

    # 加载权重,恢复模型
    filepath = 'weights.best.h5'
    model.load_weights(filepath=filepath)

    return model

model = create_model()

