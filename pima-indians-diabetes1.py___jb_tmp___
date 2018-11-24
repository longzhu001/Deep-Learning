# -*- coding: utf-8 -*-
__time__ = '2018/10/8 19:00'
__author__ = 'Mr.DONG'
__File__ = 'pima-indians-diabetes1.py'
__Software__ = 'PyCharm'
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
# 设定随机种子
seed = 7
np.random.seed(seed)
# 导入数据
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# 分割输入变量和输出变量
x = dataset[:, 0:8]
Y = dataset[:, 8]
# 分割数据
kfold = StratifiedKFold(n_splits=10,random_state=seed,shuffle=True)
cvscores = []
for train,validation in kfold.split(x,Y):
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, input_dim=8, activation='sigmoid'))
    # 编译模型

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 训练模型

    model.fit(x[train], y=Y[train], epochs=150, batch_size=10,verbose=0)

    # 评估模型 ,自动和手动的评估方法
    scores = model.evaluate(x[validation], Y[validation],verbose=0)
    # 输出结果
    print('%s : %.2f%%' %(model.metrics_names[1],scores[1]*100))
    cvscores.append(scores[1] * 100)
print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores),np.std(cvscores)))
