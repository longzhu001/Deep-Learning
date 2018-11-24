
'''

数据集 : bank.csv   葡萄牙银行机构电话营销活动的记录
  数据集中包含 : 16个输入项目, 1个输出项目
"age";  年龄  数字
"job";   工作类型  分类 : 管理员,未知,失业者,经理,女佣,企业家,学生,蓝领,个体户,退休人员,技术人员,服务人员
"marital";  婚姻状况   二分类 是 否
"education";  教育   分类 : 未知 ,中学, 小学, 高中
"default";   默认值 ,是否具有信用    二分类 是 否
"balance";   年均余额   数字
"housing";  是否有住房贷款    二分类 是 否
"loan";     贷款   二分类 是 否
"contact";    联系方式   分类 : 未知 , 固话 , 手机
"day";   最后一次联系日   数字
"month";   最后 一次联系的月份    Jan  Feb  ....
"duration";  持续时间,上次联系的时间       数字
"campaign";  在此广告系列和此客户的联系次数    数字
"pdays";     与客户上一次联系的间隔天数   数字   -1代表没联系过
"previous";  此广告系列之前和此客户的联系次数
"poutcome"; 以前的营销活动的结果    分类   : 未知   其他   失败   成功
"y"    是否订阅了定期存款    二分类 是 否

输出  :  买 :  yes   不买 : no

    通常与同一个客户会进行多次通话沟通,客户明确购买或不买的情况下会被记录到这个数据集中
    基于现有的数据集统计,分析客户是否会购买新的产品


'''

import numpy as np

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from pandas import read_csv

# 导入数据
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

filename = 'bank.csv'
dataset = read_csv(filename,delimiter=';')
dataset['job'] = dataset['job'].replace(to_replace=['admin.','unknown','unemployed','management','housemaid','entrepreneur','student','blue-collar','self-employed','retired','technician','services'],value=[0,1,2,3,4,5,6,7,8,9,10,11])
dataset['marital']  = dataset['marital'].replace(to_replace=['married','single','divorced'],value=[0,1,2])
dataset['education'] = dataset['education'].replace(to_replace=['unknown','secondary','primary','teriary'],value=[0,2,1,3])
dataset['default'] = dataset['default'].replace(to_replace=['no','yes'],value=[0,1])
dataset['housing'] = dataset['housing'].replace(to_replace=['no','yes'],value=[0,1])
dataset['loan'] = dataset['loan'].replace(to_replace=['no','yes'],value=[0,1])
dataset['contact'] = dataset['contact'].replace(to_replace=['cellular','unknown','telephone'],value=[0,1,2])
dataset['poutcome'] = dataset['poutcome'].replace(to_replace=['unknown','other','success','failure'],value=[0,1,2,3])
dataset['month'] = dataset['month'].replace(to_replace=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],value=[1,2,3,4,5,6,7,8,9,10,11,12])
dataset['y'] = dataset['y'].replace(to_replace=['no','yes'],value=[0,1])


# 分离输入与输出
array = dataset.values
x = array[:,0:16]
Y = array[:,16]
# 设置随机种子
seed = 7
np.random.seed(seed)
#构建神经网络模型函数
def create_model(units_list=[16],optimizer='adam',init='normal'):
    # 构建模型
    model = Sequential()

    # 构建第一个隐藏和输入层
    units = units_list[0]
    model.add(Dense(units=units,activation='relu',input_dim=16,kernel_initializer=init))
    # 构建更多隐藏层
    model.add(Dense(units=1,activation='sigmoid',kernel_initializer=init))

    # 编译模型 ,损失函数 交叉熵
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model
# 创建分类模型
model = KerasClassifier(build_fn=create_model,epochs=200,batch_size=5,verbose=0)
kfold = KFold(n_splits=10,random_state=seed,shuffle=True)
results = cross_val_score(model,x,Y,cv=kfold)
print('Accuracy : %.2f%% (%.2f)' % (results.mean()*100,results.std()))



# 数据格式化   -  标准化数据 ,特别适用于在规模和分布上具有一致性的输入值
new_x = StandardScaler().fit_transform(x)
kfold = KFold(n_splits=10,random_state=seed,shuffle=True)
results = cross_val_score(model,new_x,Y,cv=kfold)
print('Accuracy : %.2f%% (%.2f)' % (results.mean()*100,results.std()))


# 输出标准化处理好,会提高数据的拟合


# 调参网格拓扑结构图
param_grid = {}
param_grid['units_list'] = [[16],[30],[16,8],[30,8]]
# 调参
grid = GridSearchCV(estimator=model,param_grid=param_grid)
results=grid.fit(new_x,Y)


# 输出结果
print('Best : %f using %s ' % (results.best_score_,results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']
for  mean,std,param in zip(means,stds,params):
    print('%f (%f) with : %r '(mean,std,param))

# 只有一个隐藏层且神经元个数为16时,神经网络的性能最好




