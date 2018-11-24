# -*- coding: utf-8 -*-
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.layers import Embedding, Flatten, Dense
from tensorflow.python.keras.preprocessing import sequence

__time__ = '2018/10/14 21:57'
__author__ = 'Mr.DONG'
__File__ = 'IMDB_Analysis1.py'
__Software__ = 'PyCharm'

import numpy as np
'''
#词嵌入 - 将单词的正整数表示转换为词嵌入,需要指定词汇大小预期的最大数量,以及输出的每个词向量的维度
    多层感知器模型
    
'''

seed = 7
top_words = 5000
max_words = 500
out_dimension = 32
batch_size = 128
epochs  = 2

def create_model():
    model = Sequential()
    # 构建嵌入层  第一个参数 数据集的个数 , 每个数据的维度, 每个数据集的最大的长度
    model.add(Embedding(top_words,out_dimension,input_length=max_words))
    model.add(Flatten())
    model.add(Dense(units=250,activation='relu'))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    np.random.seed(seed)
    # path = 'imdb.npz'
    # f = np.load(path)
    # x_train, y_train = f['x_train'], f['y_train']
    # x_test, y_test = f['x_test'], f['y_test']
    # x_train = x_train[:5000]
    # x_test = x_test[:5000]
    # y_train = y_train[:5000]
    # y_test = y_test[:5000]
    # 限制数据集的长度
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path="F:\PycharmProjects\深度学习\imdb.npz",num_words=top_words)
    x_train =  sequence.pad_sequences(x_train,maxlen=max_words)
    x_test =  sequence.pad_sequences(x_test,maxlen=max_words)
    print(x_test.shape)
    print(x_train.shape)
    # 生成模型
    model = create_model()
    model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=batch_size,epochs=epochs,verbose=2)

'''
结果 :
(5000, 500)
(5000, 500)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 500, 32)           160000    
_________________________________________________________________
flatten (Flatten)            (None, 16000)             0         
_________________________________________________________________
dense (Dense)                (None, 250)               4000250   
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 251       
=================================================================
Total params: 4,160,501
Trainable params: 4,160,501
Non-trainable params: 0
_________________________________________________________________
Train on 25000 samples, validate on 25000 samples
Epoch 1/2
 - 64s - loss: 8.0216 - acc: 0.5000 - val_loss: 8.0590 - val_acc: 0.5000
Epoch 2/2
 - 62s - loss: 8.0590 - acc: 0.5000 - val_loss: 8.0590 - val_acc: 0.5000
'''