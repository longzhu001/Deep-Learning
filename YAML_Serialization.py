#!/usr/bin/env python
# encoding: utf-8
'''
@author: Mr_DONG
@software: PyCharm
@file: YAML_Serialization.py
@time: 2018/10/9 13:52

是 JSON 的另外一个 标记语言 ,强调的是以  数据为中心, 而不是以标签为重点
是一种 能够被电脑 识别的直观的数据序列格式

'''

from sklearn import datasets
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import model_from_yaml
from tensorflow.python.keras.utils import to_categorical
import numpy as np


datasets = datasets.load_iris()

x = datasets.data
Y = datasets.target

Y_lables = to_categorical(Y,new_classes=3)

seed = 7
np.random.seed(seed)

def create_model(optimizer='rmsprop',init='glorot_uniform'):
    model =Sequential()
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))

    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return model


model = create_model()
model.fit(x,Y_lables,epochs=200,batch_size=5,verbose=0)


scores =  model.evaluate(x,Y_lables,verbose=0)
print('%s : %.2f%%'%(model.metrics_names[1],scores[1]*100))


model_yaml = model.to_yaml()
with open('model.yaml','w') as file:
    file.write(model_yaml)

model.save_weights('model.yaml.h5')

with open('model.yaml','r') as file:
    model_new_yaml = file.read()


new_model = model_from_yaml(model_new_yaml)
new_model.load_weights('model.yaml.h5')


new_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

scores =  new_model.evaluate(x,Y_lables,verbose=0)
print('%s : %.2f%%' %(model.metrics_names[1],scores[1] * 100 ))

'''
两个过程的模型结果都是相同的  : 
   acc  : 97.33%
   acc  : 97.33%

但是对于 同级的目录YAML文件格式的描述就有所不同

'''



















