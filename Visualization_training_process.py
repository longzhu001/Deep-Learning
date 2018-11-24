#!/usr/bin/env python
# encoding: utf-8
'''
@author: Mr_DONG
@software: PyCharm
@file: Visualization_training_process.py
@time: 2018/10/9 15:51

模型训练过程的可视化


'''



from sklearn import datasets
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils import to_categorical
from matplotlib import pyplot as plt

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

# 构建模型
model = create_model()

history  = model.fit(x,Y_labels,validation_split=0.2,epochs=200,batch_size=5,verbose=0)

#评估模型
scores = model.evaluate(x,Y_labels,verbose=0)

# History 列表
print(history.history.keys())

# accuracy 的历史
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
# 曲线注释标签
plt.legend(['train','validation'],loc='upper left')
plt.show()

# loss 的历史
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
# 曲线注释标签
plt.legend(['train','validation'],loc='upper left')
plt.show()


'''

结果 : 
    acc : 93.33%
    dict_keys(['val_loss','val_acc','loss','acc'])
    
'''










