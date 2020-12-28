import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_table('Dataset/iris.data.txt',sep=',')
# Label
labels = data.iloc[:,4]

# 取2，3列的特征
x = data.iloc[:,[2,3]]
#计算训练集的平均值和标准差，拟合数据，将它转化成标准形式
x = StandardScaler().fit_transform(x)
data_x = np.array(x)

label_y = np.array(labels)
le = preprocessing.LabelEncoder()
le.fit(['Iris-setosa','Iris-versicolor', 'Iris-virginica'])
data_y = le.transform(label_y)
# print(label_y)

#将原始数据集按比例分成训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0)
x_train, x_test, y_train, y_test =np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)
# print(x_train.shape)    # (104,2)
# y.shape    # (104,)
# print(x_test.shape)     # (45,2)

lr = LogisticRegression(solver='newton-cg',multi_class='multinomial')
lr.fit(x_train,y_train)
print("Logistic Regression 模型训练集的准确率: %.3f" %lr.score(x_train,y_train))
# Logistic Regression 模型训练集的准确率: 0.990
y_hat = lr.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_hat)
print("Logistic Regression 模型的正确率: %.3f" %accuracy)
# Logistic Regression 模型的正确率: 0.889
target_list = ['setosa','versicolor','virginica']
print(metrics.classification_report(y_test,y_hat,target_names=target_list))


# 可视化分类结果，画出决策边界
# 确定坐标范围，x,y轴各表示一个特征
# 从最大值到最小值构建一系列的数据，使得它能覆盖整个的特征数据范围，然后预测这些值所属的分类，并给它们所在的区域
N, M = 500, 500 # 横纵各采样多少值
x1_min, x1_max = data_x[:,0].min(), data_x[:,0].max()
x2_min, x2_max = data_x[:,1].min(), data_x[:,1].max()
t1 = np.linspace(x1_min, x1_max,N)
t2 = np.linspace(x2_min, x2_max,M)
x1, y1 = np.meshgrid(t1, t2)    #生成网格采样点
# res = np.meshgrid(t1,t2)
# res = np.array(res)
# print(res.shape)    #(2, 500, 500)
# x1 = np.array(x1)
# y1 = np.array(y1)
# print(x1.shape)     #(500, 500)

x_test = np.stack((x1.flat,y1.flat),axis=1)     # 测试点(250000,2)
y_hat = lr.predict(x_test)  # 预测的label(250000,)
y_hat = y_hat.reshape(x1.shape) # 使之与输入形状相同 reshape --> (500,500)

marker_list = ['*', '+', 'o']
label_list = ['setosa','versicolor','virginica']
cm_light = ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
cm_dark = ['r','b','g']

plt.pcolormesh(x1,y1,y_hat,cmap=cm_light)   #根据y_hat的值分类

# 将数据集中的点显示在分类好的平面上
plt.scatter(data_x[data_y ==0, 0],data_x[data_y ==0, 1],alpha=0.8, s=50,edgecolors='red',
             c=cm_dark[0], marker=marker_list[0], label=label_list[0])
plt.scatter(data_x[data_y ==1, 0],data_x[data_y ==1, 1],alpha=0.8, s=50,edgecolors='k',
             c=cm_dark[1], marker=marker_list[1], label=label_list[1])
plt.scatter(data_x[data_y ==2, 0],data_x[data_y ==2, 1],alpha=0.8, s=50,edgecolors='k',
             c=cm_dark[2], marker=marker_list[2], label=label_list[2])

#plt.scatter(x[:, 0], x[:, 1], c=label_y.ravel(), edgecolors='k', s=50, cmap=cm_dark)

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.legend(loc=2)
# plt.xticks(())
# plt.yticks(())
plt.title("Logistic Regression: Result of Iris classification")
plt.grid()
plt.show()

