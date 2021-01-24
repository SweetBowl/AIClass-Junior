import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt

data_raw = pd.read_csv("quake.dat",names=['Focal_depth', 'Latitude', 'Longitude','Richter'], sep=",")
data = data_raw[8:]
data = np.array(data)

labels = data[:,3]
data_x = data[:,:3]
train_x = data_x[:-178]
test_x = data_x[-178:]
# print(train_x.shape)
# print(test_x)

train_y = labels[:-178]
test_y = labels[-178:]
# print(train_y.shape)
# print(test_y)

#########################################################
# 实现线性回归
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_x, train_y)

print("Linear Regression 系数: {}".format(lr.coef_))
print("Linear Regression 截距: {}".format(lr.intercept_))
# 预测
predict = lr.predict(test_x)
# print("预测结果")
# print(predict)
# print("真实结果")
# print(test_y)

# 评价结果
MSE = metrics.mean_squared_error(predict, test_y)
RMSE = np.sqrt(metrics.mean_squared_error(predict, test_y))
print("Linear Regression 模型MSE： %.5f" %MSE)
print("Linear Regression 模型RMSE： %.5f\n" %RMSE)

plt.figure(figsize=(15,5))
# 可视化(折线图)
plt.subplot(121)
plt.plot(range(len(test_y)), test_y, 'r', label='LRTrue Data')
plt.plot(range(len(predict)), predict, 'b', label='LRPredict Data')
plt.legend()

# 可视化(散点图)
plt.subplot(122)
plt.scatter(test_y, predict)
plt.plot([test_y.min(),test_y.max()], [test_y.min(),test_y.max()], 'k--')
plt.xlabel('LRTrue')
plt.ylabel('LRPredict')
plt.show()


#########################################################
# 实现SVM回归
from sklearn.svm import SVR
svrMdl = SVR(C=0.05)
svrMdl.fit(train_x,train_y)
# 预测
predictSVR = svrMdl.predict(test_x)
# print("预测结果")
# print(predictSVR)
# print("真实结果")
# print(test_y)

# 评价结果
MSE = metrics.mean_squared_error(predictSVR, test_y)
RMSE = np.sqrt(metrics.mean_squared_error(predictSVR, test_y))
print("SVR 模型MSE： %.5f" %MSE)
print("SVR 模型RMSE： %.5f\n" %RMSE)

plt.figure(figsize=(15,5))
# 可视化(折线图)
plt.subplot(121)
plt.plot(range(len(test_y)), test_y, 'r', label='SVRTrue Data')
plt.plot(range(len(predictSVR)), predictSVR, 'b', label='SVRPredict Data')
plt.legend()
# 可视化(散点图)
plt.subplot(122)
plt.scatter(test_y, predictSVR)
plt.plot([test_y.min(),test_y.max()], [test_y.min(),test_y.max()], 'k--')
plt.xlabel('SVRTrue')
plt.ylabel('SVRPredict')
plt.show()

#########################################################
# 实现KNN回归
from sklearn.neighbors import KNeighborsRegressor
knnMdl = KNeighborsRegressor()
knnMdl.fit(train_x, train_y)
# 预测
predictknn = knnMdl.predict(test_x)
# print("预测结果")
# print(predictknn)
# print("真实结果")
# print(test_y)
# 评价结果
MSE = metrics.mean_squared_error(predictknn, test_y)
RMSE = np.sqrt(metrics.mean_squared_error(predictknn, test_y))
print("KNeighborsRegressor 模型MSE： %.5f" %MSE)
print("KNeighborsRegressor 模型RMSE： %.5f\n" %RMSE)

plt.figure(figsize=(15,5))
# 可视化(折线图)
plt.subplot(121)
plt.plot(range(len(test_y)), test_y, 'r', label='KNNTrue Data')
plt.plot(range(len(predictknn)), predictknn, 'b', label='KNNPredict Data')
plt.legend()

# 可视化(散点图)
plt.subplot(122)
plt.scatter(test_y, predictknn)
plt.plot([test_y.min(),test_y.max()], [test_y.min(),test_y.max()], 'k--')
plt.xlabel('KNNTrue')
plt.ylabel('KNNPredict')
plt.show()

#########################################################
# 实现决策树回归
from sklearn.tree import DecisionTreeRegressor
decisionTree = DecisionTreeRegressor()
decisionTree.fit(train_x, train_y)
# 预测
predictDT = decisionTree.predict(test_x)
# print("预测结果")
# print(predictDT)
# print("真实结果")
# print(test_y)

# 评价结果
MSE = metrics.mean_squared_error(predictDT, test_y)
RMSE = np.sqrt(metrics.mean_squared_error(predictDT, test_y))
print("DecisionTreeRegressor 模型MSE： %.5f" %MSE)
print("DecisionTreeRegressor 模型RMSE： %.5f\n" %RMSE)

plt.figure(figsize=(15,5))
# 可视化(折线图)
plt.subplot(121)
plt.plot(range(len(test_y)), test_y, 'r', label='DTTrue Data')
plt.plot(range(len(predictDT)), predictDT, 'b', label='DTPredict Data')
plt.legend()

# 可视化(散点图)
plt.subplot(122)
plt.scatter(test_y, predictDT)
plt.plot([test_y.min(),test_y.max()], [test_y.min(),test_y.max()], 'k--')
plt.xlabel('DTTrue')
plt.ylabel('DTPredict')
plt.show()

#########################################################
# 实现随机森林回归
from sklearn.ensemble import RandomForestRegressor
randomForest = RandomForestRegressor()
randomForest.fit(train_x,train_y)
# 预测
predictRF = randomForest.predict(test_x)
# print("预测结果")
# print(predictRF)
# print("真实结果")
# print(test_y)

# 评价结果
MSE = metrics.mean_squared_error(predictRF, test_y)
RMSE = np.sqrt(metrics.mean_squared_error(predictRF, test_y))
print("RandomForestRegressor 模型MSE： %.5f" %MSE)
print("RandomForestRegressor 模型RMSE： %.5f\n" %RMSE)

plt.figure(figsize=(15,5))
# 可视化(折线图)
plt.subplot(121)
plt.plot(range(len(test_y)), test_y, 'r', label='RFTrue Data')
plt.plot(range(len(predictRF)), predictRF, 'b', label='RFPredict Data')
plt.legend()

# 可视化(散点图)
plt.subplot(122)
plt.scatter(test_y, predictRF)
plt.plot([test_y.min(),test_y.max()], [test_y.min(),test_y.max()], 'k--')
plt.xlabel('RFTrue')
plt.ylabel('RFPredict')
plt.show()

