import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_table('Dataset/diabetes.tab.txt')
#print(data)

'''
# 提取BMI列特征
data_x = data.iloc[:,2]
# 提取标记
data_target = data.iloc[:,10]
#print(data_x)
#print(data_target)

# 训练样本前432个，测试样本最后10个
# 处理成二维数组
diabetes_x_train1 = data_x.iloc[:-20]    # 训练样本
diabetes_x_train2=np.array(diabetes_x_train1)
diabetes_x_train3=np.expand_dims(diabetes_x_train2,axis = 1)
#print(diabetes_x_train3)
diabetes_x_test1 = data_x.iloc[-20:]
diabetes_x_test2=np.array(diabetes_x_test1)
diabetes_x_test3=np.expand_dims(diabetes_x_test2,axis = 1)
#print(diabetes_x_test)  # 测试样本
diabetes_y_train1 = data_target[:-20]    # 训练标记
diabetes_y_train2=np.array(diabetes_y_train1)
diabetes_y_train3=np.expand_dims(diabetes_y_train2,axis = 1)
#print(diabetes_y_train3)
diabetes_y_test1 = data_target[-20:]     # 测试标记
diabetes_y_test2=np.array(diabetes_y_test1)
diabetes_y_test3=np.expand_dims(diabetes_y_test2,axis = 1)
#print(diabetes_y_test3)

# 实现线性回归
linearMdl = linear_model.LinearRegression()
#print(linearMdl)
# 训练
# X:array, 稀疏矩阵 [n_samples,n_features]
# y:array [n_samples, n_targets]
linearMdl.fit(diabetes_x_train3,diabetes_y_train3)

predict = linearMdl.predict(diabetes_x_test3)
#print("预测结果")
#print(predict)
#print("真实结果")
#print(diabetes_y_test3)

#评价结果
cost = np.mean(diabetes_y_test3 - predict)**2
print("平方和")
print(cost)
print("系数")
print(linearMdl.coef_)
print("截距")
print(linearMdl.intercept_)
print("方差")
print(linearMdl.score(diabetes_x_test3,diabetes_y_test3))

#绘图
plt.title("diabetes")
plt.xlabel("x")
plt.ylabel("y")
# 黑色，散点
plt.plot(diabetes_x_test3,diabetes_y_test3,'k.')
# 绿色，直线
plt.plot(diabetes_x_test3,predict,'g-')

for idx, n in enumerate(diabetes_x_test3):
    plt.plot([n,n],[diabetes_y_test3[idx],predict[idx]],'r-')

plt.show()
'''


#可以对数据进行标准化处理
predata_x = data.iloc[:,2]
# 提取标记
data_target = data.iloc[:,10]

data_x_mean = np.mean(predata_x)
data_x_std = np.std(predata_x)
# Normalization
data_x = (predata_x-data_x_mean) / data_x_std
# print(data_x)
#print(data_target)


# 训练样本前432个，测试样本最后20个
# 处理成二维数组
diabetes_x_train1 = data_x.iloc[:-20]    # 训练样本
diabetes_x_train2=np.array(diabetes_x_train1)
diabetes_x_train3=np.expand_dims(diabetes_x_train2,axis = 1)
#print(diabetes_x_train3)
diabetes_x_test1 = data_x.iloc[-20:]
diabetes_x_test2=np.array(diabetes_x_test1)
diabetes_x_test3=np.expand_dims(diabetes_x_test2,axis = 1)
#print(diabetes_x_test)  # 测试样本
diabetes_y_train1 = data_target[:-20]    # 训练标记
diabetes_y_train2=np.array(diabetes_y_train1)
diabetes_y_train3=np.expand_dims(diabetes_y_train2,axis = 1)
#print(diabetes_y_train3)
diabetes_y_test1 = data_target[-20:]     # 测试标记
diabetes_y_test2=np.array(diabetes_y_test1)
diabetes_y_test3=np.expand_dims(diabetes_y_test2,axis = 1)
#print(diabetes_y_test3)

# 实现线性回归
linearMdl = linear_model.LinearRegression()
#print(linearMdl)
# 训练
# X:array, 稀疏矩阵 [n_samples,n_features]
# y:array [n_samples, n_targets]
linearMdl.fit(diabetes_x_train3,diabetes_y_train3)

predict = linearMdl.predict(diabetes_x_test3)
print("预测结果")
print(predict)
print("真实结果")
print(diabetes_y_test3)

#评价结果
cost = np.mean(diabetes_y_test3 - predict)**2
print("平方和")
print(cost)
print("系数")
print(linearMdl.coef_)
print("截距")
print(linearMdl.intercept_)
#通过决定系数来来判断回归方程的拟合程度(分数越高说明拟合的程度越好)？
print("决定系数")
print(linearMdl.score(diabetes_x_test3,diabetes_y_test3))

#绘图
plt.title("diabetes")
plt.xlabel("x")
plt.ylabel("y")
# 黑色，散点
plt.plot(diabetes_x_test3,diabetes_y_test3,'k.')
# 绿色，直线
plt.plot(diabetes_x_test3,predict,'g-')

for idx, n in enumerate(diabetes_x_test3):
    plt.plot([n,n],[diabetes_y_test3[idx],predict[idx]],'r-')

plt.show()

#可视化可参考这个：https://zhuanlan.zhihu.com/p/141326006