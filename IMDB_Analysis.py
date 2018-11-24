# -*- coding: utf-8 -*-
__time__ = '2018/10/14 21:24'
__author__ = 'Mr.DONG'
__File__ = 'IMDB_Analysis.py'
__Software__ = 'PyCharm'
'''
情感分析实例 : 
     IMDB影评情感分析
     数据集 :
        包含25000部电影的评价信息
        Keras 提供的数据集将单词转化成整数 , 代表单词在整个数据集中的流行程度
'''

import numpy as np
from matplotlib import pyplot as plt
path = 'imdb.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']

#合并训练数据集和评估数据集  , 数组拼接
x = np.concatenate((x_train,x_test),axis=0)
y = np.concatenate((y_train,y_test),axis=0)
x1 = x[:5000]
y1 = x[:5000]
print('x shape is %s , y shape is %s'%(x.shape,y.shape))
# 保留数据集中y变量的 不同的值 , 答案的 [0  1]
print('Classes : %s'%(np.unique(y)))

print('Total words: %s'%(len(np.unique(np.hstack(x)))))

result = [len(word) for word in x]
print('Mean : %.2f words (STD: %.2f)'%(np.mean(result),np.std(result)))

# 图标展示
plt.subplot(121)
plt.boxplot(result)
plt.subplot(122)
plt.hist(result)
plt.show()

'''
结果 : 
    x shape is (50000,) , y shape is (50000,)
    Classes : [0 1]
    Total words: 88584
    Mean : 233.76 words (STD: 172.91)

'''









