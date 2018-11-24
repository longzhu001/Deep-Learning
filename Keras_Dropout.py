

'''
输入层
Dropout
    是神经网络提出的一个 正则化方法 ， 其中正则化强度代表着函数的光滑程序，光滑代表连续， 联系代表可导
    原理 ： 在训练的过程中，随机的忽略部分的神经元 ， 通过 rate = 0.2  ， 每个周期更新随机忽略20% 的训练数据
    效果 ： 减弱了神经元节点间的联合适应性， 增强了泛化能力
    model.add(Dropout(rate=0.2,input_shape=(4,)))
    sgd = SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
'''
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import datasets
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.python.keras.optimizers import SGD

dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

seed = 7
np.random.seed(seed)


def create_model(init='glorot_unifrom'):
    model = Sequential
    model.add(Dropout(rate=0.2,input_shape=(4,)))
    model.add(Dense(units=4,activation='relu',kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))

    sgd = SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,epochs=200,batch_size=5,verbose=0)
kfold = KFold(n_splits=10,shuffle=True,random_state=seed)
results = cross_val_score(model,x,Y,cv=kfold)
print('Accuracy : %.2f%% (%.2f)'%(results.mean()*100,results.std()))



'''
隐藏层
Dropout
    位置 : 将在两个隐藏层之间,以及最后一个隐藏层和输出层之间使用Dropout
kernel_constraint=maxnorm(3) 权重约束,最大不超过3
     将此程序替换上面的函数 即可完成 隐藏层的 "Dropout 程序设计
'''

def create_model1(init='glorot_unifrom'):
    model = Sequential
    model.add(Dense(units=4,activation='relu',kernel_initializer=init,kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init,kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init,kernel_constraint=maxnorm(3)))

    sgd = SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model









