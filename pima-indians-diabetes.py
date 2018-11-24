# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

__time__ = '2018/10/8 17:29'
__author__ = 'Mr.DONG'
__File__ = 'pima-indians-diabetes.py'
__Software__ = 'PyCharm'
'''
 多层感知器 
    印第安人糖尿病诊断
    数据集 : pima-indians-diabetes.csv
创建神经网络模型的步骤 :
    1.导入数据
    2.定义模型
    3.编译模型
    4.训练模型
    5.评估模型
    6.汇总代码
'''
import numpy as np

# 设定随机种子
seed  = 7
np.random.seed(seed)
# 导入数据
dataset =np.loadtxt('pima-indians-diabetes.csv',delimiter=',')
# 分割输入变量和输出变量
x = dataset[:,0:8]
Y = dataset[:,8]
#分割数据
x_train,x_validation,Y_train,Y_validation = train_test_split(x,Y,test_size=0.2,random_state=seed)
# 创建模型
'''
    第一层隐藏层 12 个神经元, 使用 8 个输入变量
    第二层隐藏层有8个神经元
    最后输出层有1个神经元来预测数据结果 ( 是否患有糖尿病)
    
'''
model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,input_dim=8,activation='sigmoid'))
# 编译模型
'''
 使用有效的梯度下降算法 Adam 作为优化器
 对于二进制分类问题的对数损失函数被被定义为 二进制交叉熵
'''
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# 训练模型
'''
    采用epochs 参数 对数据集进行 固定次数的迭代 
    执行射进网络中的权重更新的每个批次中所实例的个数 batch_size
    
    # 训练模型并自动评估模型
    model.fit(x=x,y=Y,epochs=150,batch_size=10,validation_split=0.2)
'''
model.fit(x=x,y=Y,epochs=150,batch_size=10)
# 评估模型 ,自动和手动的评估方法
scores = model.evaluate(x=x,y=Y)
print('\n%s : %.2f%%' %(model.metrics_names[1],scores[1]*100))

'''
结果 :  执行 150 次的部分结果
 10/768 [..............................] - ETA: 0s - loss: 8.0590 - acc: 0.5000
360/768 [=============>................] - ETA: 0s - loss: 5.1041 - acc: 0.6833
610/768 [======================>.......] - ETA: 0s - loss: 5.2318 - acc: 0.6754
768/768 [==============================] - 0s 195us/step - loss: 5.6245 - acc: 0.6510
Epoch 150/150

 10/768 [..............................] - ETA: 0s - loss: 3.2236 - acc: 0.8000
230/768 [=======>......................] - ETA: 0s - loss: 5.5362 - acc: 0.6565
500/768 [==================>...........] - ETA: 0s - loss: 5.8348 - acc: 0.6380
768/768 [==============================] - 0s 207us/step - loss: 5.6245 - acc: 0.6510

 32/768 [>.............................] - ETA: 3s
768/768 [==============================] - 0s 234us/step

acc : 65.10%

'''