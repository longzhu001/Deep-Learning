# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D
from tensorflow.python.keras.preprocessing import sequence

__time__ = '2018/10/14 22:51'
__author__ = 'Mr.DONG'
__File__ = 'IMDB_Analysis2.py'
__Software__ = 'PyCharm'
'''
卷积神经网络 :
    
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
    model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
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