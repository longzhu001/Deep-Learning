# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

__time__ = '2018/10/8 19:17'
__author__ = 'Mr.DONG'
__File__ = 'pima-indians-diabetes2.py'
__Software__ = 'PyCharm'

def create_model(optimizer='adam',init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dense(12,input_dim=8,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

seed = 7
np.random.seed(seed)

dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# 分割输入变量和输出变量
x = dataset[:, 0:8]
Y = dataset[:, 8]

model = KerasClassifier(build_fn=create_model,verbose=0)

param_grid = {}
param_grid['optimizer'] = ['rmsprop','adam']
param_grid['init'] = ['glorot_uniform','normal','uniform']
param_grid['epochs'] = [50,100,150,200]
param_grid['batch_size'] = [5,10,20]
# 调参
grid = GridSearchCV(estimator=model,param_grid=param_grid)
results = grid.fit(x,Y)

# 输出结果
print('Best: %f using %s' % (results.best_score_,results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']

for mean ,std ,param in zip(means,stds,params):
    print('%f (%f) with: %r'% (mean ,std ,param ))
