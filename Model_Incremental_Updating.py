#!/usr/bin/env python
# encoding: utf-8
'''
@author: Mr_DONG
@software: PyCharm
@file: Model_Incremental_Updating.py
@time: 2018/10/9 14:25

模型增量更新
    为了保证模型的时效性, 需要定期的进行更新, 通常时间间隔为 3- 6 个月
    数据量非常大的时候, 若每次采用全部数据去重新训练模型, 则时间开销非常大
    因此 . 采用增量更新模型的方式,对模型进行训练

对于时间序的预测,增量更新相当于 默认给最新的数据增加了权重, 模型的准确度相对会比较好
实际过程中,采用增量更新模型, 需要做的与 全面更新的对比实验 , 以确保 增量恒信的可行性

'''


from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.utils import to_categorical

datasets  =datasets.load_iris()

x = datasets.data
Y = datasets.target


seed = 7
np.random.seed(seed)

x_train,x_increment,Y_train,Y_increment = train_test_split(x,Y,test_size=0.2,random_state=seed)


Y_train_labels = to_categorical(Y_train,num_classes=3)


def create_model(optimizer='rmsprop',init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))


    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model


model = create_model()
model.fit(x_train,Y_train_labels,epochs=200,batch_size=5,verbose=2)

scores = model.evaluate(x_train,Y_train_labels,verbose=0)
print('Base %s : %.2f%%' % (model.metrics_names[1],scores[1] * 100))



model_json = model.to_json()
with open('model.increment.json','w') as file:
    file.write(model_json)


model.save_weights('model.increment.json.h5')

with open('model.increment.json','r') as f:
    model_json = f.read()


new_model = model_from_json(model_json)
new_model.load_weights('model.increment.json.h5')


#  编译模型
new_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


# 增量训练模型  , 将标签 转成分类编码
Y_increment_labels = to_categorical(Y_increment,num_classes=3)
new_model.fit(x_increment,Y_increment_labels,epochs=200,batch_size=5,verbose=2)

scores = new_model.evaluate(x_increment,Y_increment_labels,verbose=0)
print('Increment %s : %.2f%% ' % (model.metrics_names[1],scores[1] * 100))


'''
结果 :  
    模型的准确度为             80.00% 
    增量模型训练完后, 准确度为  86.67%  

'''



















