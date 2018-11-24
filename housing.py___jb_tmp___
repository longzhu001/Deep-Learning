# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

__time__ = '2018/10/8 21:25'
__author__ = 'Mr.DONG'
__File__ = 'housing.py'
__Software__ = 'PyCharm'


'''
输入维度相同数量的神经元的单层完全全连接的隐藏层,13个神经元,隐藏层才有 ReLU 激活函数
由于是回归问题,不要对预测结果进行分类,所以输出层不用设置激活函数

'''
# 导入数据

from sklearn import datasets
import numpy as np
datasets = datasets.load_boston()
x = datasets.data
Y = datasets.target

seed = 7
np.random.seed(seed)

def creat_model(units_list=[13],optimizer='adam',init='normal'):
    model = Sequential()
    units = units_list[0]
    # 输入层
    model.add(Dense(units=units,activation='relu',input_dim=13,kernel_initializer=init))
    # 输出层
    model.add(Dense(units=1,kernel_initializer=init,))
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    return model

model = KerasRegressor(build_fn=creat_model,epochs=200,batch_size=5,verbose=0)
'''
steps= []
steps.append(('standardize',StandardScaler()))
steps.append(('mpl',model))
pipeline = Pipeline(steps)
kfold = KFold(n_splits=10,shuffle=True,random_state=seed)
results= cross_val_score(pipeline,x,Y,cv=kfold)
print('Baseline : %.2f (%.2f) MSE' %(results.mean(),results.std()))
'''
# 调参
param_grid = {}
param_grid['units_list'] = [[20],[13,6]]
param_grid['optimizer'] = ['rmsprop','adam']
param_grid['init'] = ['glorot_uniform','normal']
param_grid['epochs'] = [100,200]
param_grid['batch_size']=[5,20]
scaler = StandardScaler()
scaler_x=scaler.fit_transform(x)
grid = GridSearchCV(estimator=model,param_grid=param_grid)
results = grid.fit(scaler_x,Y)

# 输出结果
print('Best : %f using %s '% (results.best_score_,results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']
for mean , std , param in zip(means,stds,params):
    print('%f (%f) with : %r' % (mean,std,param))
