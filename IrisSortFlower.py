# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold, cross_val_score
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

__time__ = '2018/10/8 19:43'
__author__ = 'Mr.DONG'
__File__ = 'IrisSortFlower.py'
__Software__ = 'PyCharm'
from sklearn import datasets
import numpy as np
'''
创建一个简单的全连接网络
    包括一个输入层(4个神经元) 两个隐藏层 一个输出层(3个神经元,多分类问题的输出层通常具有与分类类别相同的神经元个数,这里包括3个神经元)
    第一个隐藏层,4个神经元,与输入层的神经元一致,使用ReLU激活函数
    第二个隐藏层,6个神经元,同样使用 ReLU激活函数 
    
'''
datasets = datasets.load_iris()
x = datasets.data
Y = datasets.target

seed = 7
np.random.seed(seed)

# 构建模型函数
def create_model(optimizer='adam',init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4,input_dim=4,activation='relu',kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))

    # 编译模型
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,epochs=200,batch_size=5,verbose=0)
kfold = KFold(n_splits=10,shuffle=True,random_state=seed)
results= cross_val_score(model,x,Y,cv=kfold)
print('Accuracy : %.2f%% (%.2f)'% (results.mean()*100,results.std()))








