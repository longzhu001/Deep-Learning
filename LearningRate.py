from math import floor

from tensorflow.python.keras.callbacks import LearningRateScheduler

1  #!/usr/bin/env python
2  # -*- coding: utf-8 -*-
3  # @File  : LearningRate.py
4  # @Author: Mr_DONG
5  # @Date  : 2018/10/10
6  # @Desc  :
'''
在深度 学习中 使用学习率衰减 ,通常会考虑的情况 :
    1.提高初始化学习率 
        更大的学习率,在开始学习时,会快速更新权重值, 而且随着学习率的衰减可以自动调整学习率,这可以提高梯度下降的性能
    2.使用大动量
        使用较大的动量值将有助于优化算法在学习率缩小到小值时, 继续向正确的方向更新权重值

学习率衰减 : 
     选择合适的学习率 , 能够提高 随机梯度算法的性能 ,并减少训练时间
     学习率决定参数移动到最优值的速度,如果学习率过大,可能会越过最优值,反之,优化的效率可能过低,长时间算法无法收敛
     
     两种学习率衰减方法 : 线性衰退(根据epoch逐步降低学习率)
                        指数衰退(在特渡部分的epoch使用分数快速降低学习率)
                        
基于时间的线性学习率衰减实通过 SGD类中的随机梯度下降优化算法实现的
    该类具有一个decay 衰减参数
    线性学习率衰减方程如下 :
        LearningRate = LearningRate X  1/(1 + decay X epoch )

    当 decay 衰减率为0(默认值)时 , 对学习率没有影响,使用非零学习率衰减时,学习率呈线性衰减

指数衰减是通过 在固定的epoch 周期将学习率降低50%来实现的
    例如 : 初始学习率为 0.1 , 每10个epochs降低50% ,前10个使用0.1 , 接下来的10个才有0.05的学习率
    使用 LearningRateScheduler回调 ,来实现学习率的指数衰减
    epoch数作为一个参数,并将学习率返回到随机梯度下降算法中使用
    指数学习率衰减方程如下 : 
        LearningRate = InitialLearningRate X DropRate^(floor( (1+Epoch / (EpochDrop) ) ) )
'''
####################################################         线性学习率
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import datasets
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import  Dense
from tensorflow.python.keras.optimizers import SGD

dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

seed = 7
np.random.seed(seed)


def create_model(init='glorot_unifrom'):
    model = Sequential
    model.add(Dense(units=4,activation='relu',kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))
    #模型优化
    learningrate=0.1
    momentum = 0.9
    decay_rate = 0.05
    sgd = SGD(lr=learningrate,momentum=momentum,decay=decay_rate,nesterov=False)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,epochs=200,batch_size=5,verbose=1)
model.fit(x,Y)


####################################################         指数学习率

# 计算学习率
def step_decay(epoch):
    init_lrate = 0.1
    drop = 0.5
    epochs_drop = 10
    lrate = init_lrate * pow(drop,floor(1+epoch) / epochs_drop)
    return lrate

def create_model(init='glorot_unifrom'):
    model = Sequential
    model.add(Dense(units=4,activation='relu',kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))

    #模型优化
    learningrate=0.1
    momentum = 0.9
    decay_rate = 0.05

    sgd = SGD(lr=learningrate,momentum=momentum,decay=decay_rate,nesterov=False)

    # 编译模型
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    return model

# 学习率指数衰减回调
lrate = LearningRateScheduler(step_decay)
epochs = 200
model = KerasClassifier(build_fn=create_model,epochs=epochs,batch_size=5,verbose=1,callbacks=[lrate])
model.fit(x,Y)
