'''

多层感知器 进阶

JSON 序列化模型构建
  对模型进行序列化时, 会将模型结果和模型权重保存在不同的文件中
  模型权重一般保存在HDF5中, 模型的机构可以保存在 JSON 文件   或者  YAML 文件中

'''


from sklearn import datasets
import numpy as  np



# 导入数据
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.engine.saving import model_from_json
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils import to_categorical

datasets = datasets.load_iris()
x = datasets.data
Y = datasets.target

# 将标签转换成分类编码

Y_labels = to_categorical(Y,num_classes=3)

# 设定随机种子
seed = 7
np.random.seed(seed)

# 构建神经网络模型
def create_model(optimizer = 'rmsprop' , init ='glorot_uniform'  ):
    # 构建模型
    model = Sequential()
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))

    # 编译模型
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model

# 构建分类模型
model = create_model()
model.fit(x,Y_labels,epochs=200,batch_size=5,verbose=0)
scores = model.evaluate(x,Y_labels,verbose=0)
print('%s: %.2f%%'% (model.metrics_names[1],scores[1]*100))



# 将模型保存成 JSON文件
model_json = model.to_json()
with open('model.json','w') as f :
    f.write(model_json)

# 保存模型权重
model.save_weights('model.json.h5')

# 从json 文件中加载模型
with open('model.json','r') as file :
    model_json = file.read()

# 加载模型
new_model = model_from_json(model_json)
new_model.load_weights('model.json.h5')

# 编译模型
new_model.compile(loss='ctaegorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# 评估从JSON文件中加载的模型
scores = new_model.evaluate(x,Y_labels,verbose=0)
print('%s : %2.f%%' %(model.metrics_names[1],scores[1]*100))


'''
两次结果 是一样的   同时会在同级目录下生成了JSON文件和HDF5文件 : 
    acc  : 97.33%
    acc  : 97.33%

'''




