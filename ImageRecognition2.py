# -*- coding: utf-8 -*-
__time__ = '2018/10/11 22:25'
__author__ = 'Mr.DONG'
__File__ = 'ImageRecognition2.py'
__Software__ = 'PyCharm'
'''
改进模型 -  大型卷积神经网络
网络拓扑图 :
    1.卷积层,具有192个特征图,感受野大小5x5
    2.卷积层,具有160个特征图,感受野大小1x1
    3.卷积层,具有96个特征图,感受野大小1x1
    4.采样因子.pool_size 为 3x3 , 步长为2x2的池化层
    5.Dropout 层 , 概率为 20%
    6.卷积层,具有192个特征图,感受野大小5x5
    7.卷积层,具有192个特征图,感受野大小1x1
    8.卷积层,具有192个特征图,感受野大小1x1
    9.采样因子.pool_size 为 3x3 , 步长为2x2的池化层
    10.Dropout 层 , 概率为 50%
    11.卷积层,具有192个特征图,感受野大小5x5
    12.卷积层,具有192个特征图,感受野大小1x1
    13.卷积层,具有10个特征图,感受野大小1x1
    14.使用GLobalAveragePooling 作为最后一个池化层
    15.激活层,使用激活函数softmax
    
'''
import numpy as np
import pickle
from tensorflow.keras import optimizers
from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D,Activation,GlobalAveragePooling2D
from tensorflow.keras.constraints import maxnorm
import keras
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import LearningRateScheduler,TensorBoard


def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='latin1')
    return datadict


batch_size = 128
epochs = 200
interations = 391
num_classes = 10
dropout = 0.5
log_filepath = './nin'




def normalize_preprocessing(x_train,x_validation):
  x_train =x_train.reshape(x_train.shape[0],32,32,3).astype('float32')
  x_validation =x_validation.reshapee(x_validation.shape[0],32,32,3).astype('float32')
  mean = [125.307,122.95,113.865]
  std = [62.9932,62.0887,66.7048]
  for i in range(3):
    x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) /  std[i]
    x_validation[:,:,:,i]  = (x_validation[:,:,:,i] - mean[i]) / std[i]
  return x_train,x_validation



y_train = np_utils.to_categoriacl(y_train)
y_validation = np_utils.to_categoriacl(y_validation)
num_classes = y_validation.shape[1]


def scheduler(epoch):
  if epoch <= 60:
    return 0.05
  if epoch <=120:
    return 0.01
  if epoch<=160:
    return 0.0004

def build_model():
  model = Sequential()
  model.add(Conv2D(192,(5,5),input_shape=x_train.shape[1:],padding='same',activation='relu',kernel_regularizer=keras.regularizers.l2(0.0001),kernel_initializer=RandomNormal(stddev=0.01)))
  model.add(Dense(160,(1,1),padding='same',kernel_regularizer=keras.regularizers.l2(0.0001),kernel_initializer=RandomNormal(stddev=0.05),activation='relu'))
  model.add(Dense(96,(1,1),activation='relu',kernel_initializer=RandomNormal(stddev=0.05),kernel_regularizer=keras.regularizers.l2(0.0001),padding='same'))
  model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same'))
  model.add(Dropout(dropout))
  model.add(Conv2D(192,(5,5),activation='relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.0001),kernel_initializer=RandomNormal(stddev=0.05)))
  model.add(Conv2D(192,(1,1),padding='same',activation='relu',kernel_regularizer=keras.regularizers.l2(0.0001),kernel_initializer=RandomNormal(stddev=0.05)))
  model.add(Dropout(dropout))
  model.add(Conv2D(192,(1,1),padding='same',kernel_initializer=RandomNormal(stddev=0.05),kernel_regularizer=keras.regularizers.l2(0.0001),activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same'))
  model.add(Dropout(dropout))
  model.add(Conv2D(192,(3,3),padding='same',kernel_regularizer=keras.regularizers.l2(0.0001),kernel_initializer=RandomNormal(stddev=0.05)))
  model.add(Conv2D(192,(1,1),padding='same',kernel_regularizer=keras.regularizers.l2(0.0001),kernel_initializer=RandomNormal(stddev=0.05)))
  model.add(Conv2D(10,(1,1),padding='same',kernel_regularizer=keras.regularizers.l2(0.0001),kernel_initializer=RandomNormal(stddev=0.05)))
  model.add(GlobalAveragePooling2D())
  model.add(Activation('softmax'))


  sgd = optimizer.SGD(lr=0.1,momentun=0.9,nesterov=True)
  model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
  return model

if __name__ == '__main__':
  seed = 7
  np.random.seed(seed=seed)
  #导入数据
  fileTrain = 'data_batch_1'
  fileTest = 'test_batch'
  DataTrain = load_CIFAR_batch(fileTrain)
  DateTest = load_CIFAR_batch(fileTest)
  x_train = DataTrain['data']
  y_train = DataTrain['labels']
  x_validation = DateTest['data']
  y_validation = DateTest['labels']
  print(x_train, x_validation)
  y_train = keras.utils.to_categorical(y_train,num_classes)
  y_validation = keras.utils.to_categorical(y_validation,num_classes)

  x_train,x_validation = normalize_preprocessing(x_train,x_validation)

  # 构建神经网络
  model = build_model()
  print(model.summary())

  # 设置回调函数,实现学习率衰减
  tb_cb = TensoBoard(log_dir=log_filepath,histogram_freq=0)
  change_lr = LearningRateScheduler(schedule)
  cbks = [change_lr,tb_cb]


  model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,callbacks=cbks,validation_data=(x_validation,y_validation),verbose=2)
  model.save('nin.h5')

'''
结果 : 
  准确度 : 87.96 % 
  
'''




















