#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Improving_Iamge.py
# @Author: Mr_DONG
# @Date  : 2018/10/11
# @Desc  :
'''
Keras 中的图片增强API
    通过ImageDataGenerator类来实现增强处理功能 ：
        1.特征标准化
        2.ZCA白化
        3.随机旋转，移动，剪切和反转图像
        4.维度排序
        5.保存增强后的图像
数据集 ： 采用 Keras 中 mnist
'''
import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
# 增强图像前的图像 ， 预查看 9张图

path = 'mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']

# 显示9张图
for i in range(0,9):
    plt.subplot(331+i)
    plt.imshow(x_train[i],camp=plt.get_cmap('gray'))
plt.show()

'''
特征标准化
    对图像数据集标准化 和  对其他数据标准化是一样的 ,都可以提高神经网络算法的性能
    通过设置 : ImageDataGenerator 类中的 featurewise_center  和  featurewise_std_normalization 参数  设置为 True
    通过设置 :flow()配置batch_size 来准备数据生成器生成图像
'''
x_train= x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test = x_train.reshape(x_test.shape[0],28,28,1).astype('float32')

# 图像特征标准化
imGen =ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
imGen.fit(x_train)

for x_batch,y_batch in imGen.flow(x_train,y_train,batch_size=9):
    for i in range(0,9):
        plt.subplot(331+i)
        plt.imshow(x_train[i].reshape(28,28),camp=plt.get_cmap('gray'))
    plt.show()
    break

'''
图片ZCA 白化处理
    处理的是线性代数操作,能够减少图像像素矩阵的冗余和相关性,可以更好的将图像中的结构和特征突出显示给学习算法
    比较常用的白化处理包括 :  PCA 主成分分析 ZCA白化处理(更好的适应性)
    通过设置 zca_whitening 为True 来执行 ZCA

'''

# 图像 白化处理
imGen =ImageDataGenerator(zca_whitening=True)
imGen.fit(x_train)
# 显示图像
for x_batch,y_batch in imGen.flow(x_train,y_train,batch_size=9):
    for i in range(0,9):
        plt.subplot(331+i)
        plt.imshow(x_train[i].reshape(28,28),camp=plt.get_cmap('gray'))
    plt.show()
    break


'''
随机旋转 , 移动 ,剪切 和 反转图像
    有时候为了更好的训练时, 需要对图像进行反转,移动等操作 
    旋转 : rotation_range= 角度数   0 ~ 180
    移动 : width_shift_range=0.2,height_shift_range=0.2   0 ~ 1
    剪切 : shear_range   0 ~ 1
    反转 : horizontal_flip=True,vertical_flip=True
'''
# 图像 旋转
imGen =ImageDataGenerator(rotation_range=90)
imGen.fit(x_train)
# 显示图像
for x_batch,y_batch in imGen.flow(x_train,y_train,batch_size=9):
    for i in range(0,9):
        plt.subplot(331+i)
        plt.imshow(x_train[i].reshape(28,28),camp=plt.get_cmap('gray'))
    plt.show()
    break

# 图像 移动
imGen =ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2)
imGen.fit(x_train)
# 显示图像
for x_batch,y_batch in imGen.flow(x_train,y_train,batch_size=9):
    for i in range(0,9):
        plt.subplot(331+i)
        plt.imshow(x_train[i].reshape(28,28),camp=plt.get_cmap('gray'))
    plt.show()
    break

# 图像 剪切
imGen =ImageDataGenerator(shear_range=0.1)
imGen.fit(x_train)
# 显示图像
for x_batch,y_batch in imGen.flow(x_train,y_train,batch_size=9):
    for i in range(0,9):
        plt.subplot(331+i)
        plt.imshow(x_train[i].reshape(28,28),camp=plt.get_cmap('gray'))
    plt.show()
    break

# 图像 反转
imGen =ImageDataGenerator(horizontal_flip=True,vertical_flip=True)
imGen.fit(x_train)
# 显示图像
for x_batch,y_batch in imGen.flow(x_train,y_train,batch_size=9):
    for i in range(0,9):
        plt.subplot(331+i)
        plt.imshow(x_train[i].reshape(28,28),camp=plt.get_cmap('gray'))
    plt.show()
    break


# 创建目录并保存
try:
    os.mkdir('image')
except:
    print('The fold is exits!')
# save_to_dir='image' 目录 ,save_prefix='oct'前缀 ,save_format='png' 后缀
for x_batch,y_batch in imGen.flow(x_train,y_train,batch_size=9,save_to_dir='image',save_prefix='oct',save_format='png'):
    for i in range(0,9):
        plt.subplot(331+i)
        plt.imshow(x_train[i].reshape(28,28),camp=plt.get_cmap('gray'))
    plt.show()
    break


'''

详解 :
    ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization = False,
            samplewise_std_normalization = False,
            zca_whitening = False,
            rotation_range = 0.,
            width_shift_range = 0.,
            height_shift_range = 0.,
            shear_range = 0.,
            zoom_range = 0.,
            channel_shift_range = 0.,
            fill_mode = 'nearest',
            cval = 0.0,
            horizontal_flip = False,
            vertical_flip = False,
            rescale = None,
            preprocessing_function = None,
            data_format = K.image_data_format(),
)
 1.featurewise_center：布尔值，使输入数据集去中心化（均值为0）, 按feature执行。
 2.samplewise_center：布尔值，使输入数据的每个样本均值为0。
 3.featurewise_std_normalization：布尔值，将输入除以数据集的标准差以完成标准化, 按feature执行。
 4.samplewise_std_normalization：布尔值，将输入的每个样本除以其自身的标准差。
 5.zca_whitening：布尔值，对输入数据施加ZCA白化。
 6.rotation_range：整数，数据提升时图片随机转动的角度。随机选择图片的角度，是一个0~180的度数，取值为0~180。
 7.width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度。
 8.height_shift_range：浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度。 
 9.height_shift_range和width_shift_range是用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比例。
 10.shear_range：浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度。
 11.zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]。用来进行随机的放大。
 12.fill_mode：‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理cval：浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值。
 13.horizontal_flip：布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候。
 14.vertical_flip：布尔值，进行随机竖直翻转。
 15.rescale: 值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。16.preprocessing_function: 将被应用于每个输入的函数。该函数将在任何其他修改之前运行。该函数接受一个参数，为一张图片（秩为3的numpy array），并且输出一个具有相同shape的numpy array
 17.data_format：字符串，“channel_first”或“channel_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channel_last”对应原本的“tf”，“channel_first”对应原本的“th”。以128x128的RGB图像为例，“channel_first”应将数据组织为（3,128,128），而“channel_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channel_last”。
 18.channel_shift_range：浮点数，随机通道偏移的幅度。



'''

