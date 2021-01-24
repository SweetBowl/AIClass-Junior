import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

data = pd.read_table('data_banknote_authentication.txt',
                     names=['Variance of Wavelet Transformed image', 'Skewness of Wavelet Transformed image', 'Kurtosis of Wavelet Transformed image','Entropy of image','Labels'],
                     sep=',')
data = np.array(data)

labels = data[:,-1]
data_x = data[:,:4]
train_x = data_x[:1200]
test_x = data_x[-172:]

train_y = labels[:1200]
test_y = labels[-172:]
ss = StandardScaler()

#########################################################
# 实现逻辑斯蒂回归
from sklearn.linear_model import LogisticRegression
train_x1 = ss.fit_transform(train_x)
test_x1 = ss.fit_transform(test_x)
lr = LogisticRegression()
lr.fit(train_x1,train_y)
# 预测
predict = lr.predict(test_x1)
# print("预测结果")
# print(predict)
# print("真实结果")
# print(test_y)

# 评价结果
print("Accruacy of LogisticRegression: ",lr.score(test_x1,test_y))
print(classification_report(test_y,predict,target_names=['real','fake']))
print("\n")

plt.scatter(range(len(predict)),predict,s=8,marker='.',c='red')
plt.title("The Distribution of LogisticRegression Prediction")
plt.show()
#########################################################
# 实现SVM分类
from sklearn.svm import SVC
train_x2 = ss.fit_transform(train_x)
test_x2 = ss.fit_transform(test_x)
svcMdl = SVC()
svcMdl.fit(train_x2,train_y)
# 预测
predict = svcMdl.predict(test_x2)
# print("预测结果")
# print(predict)
# print("真实结果")
# print(test_y)
# 评价结果
print("Accruacy of SVC: ",svcMdl.score(test_x2,test_y))
print(classification_report(test_y,predict,target_names=['real','fake']))
print("\n")

plt.scatter(range(len(predict)),predict,s=8,marker='.',c='red')
plt.title("The Distribution of SVC Prediction")
plt.show()
#########################################################
# 实现朴素贝叶斯分类
from sklearn.naive_bayes import GaussianNB
# 直接用 MultinomialNB 报错：ValueError: Input X must be non-negative

train_x5 = ss.fit_transform(train_x)
test_x5 = ss.fit_transform(test_x)
GaussianMdl = GaussianNB()
GaussianMdl.fit(train_x5,train_y)
# 预测
predict = GaussianMdl.predict(test_x5)
# print("预测结果")
# print(predict)
# print("真实结果")
# print(test_y)
# 评价结果
print("Accruacy of GaussianNB: ",GaussianMdl.score(test_x5,test_y))
print(classification_report(test_y,predict,target_names=['real','fake']))
print("\n")

plt.scatter(range(len(predict)),predict,s=8,marker='.',c='red')
plt.title("The Distribution of GaussianNB Prediction")
plt.show()
#########################################################
# 实现决策树分类
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(train_x,train_y)
# 预测
predict = dtc.predict(test_x)
# print("预测结果")
# print(predict)
# print("真实结果")
# print(test_y)

# 评价结果
print("Accruacy of DecisionTreeClassifier: ",dtc.score(test_x,test_y))
print(classification_report(test_y,predict,target_names=['real','fake']))
print("\n")

plt.scatter(range(len(predict)),predict,s=8,marker='.',c='red')
plt.title("The Distribution of DecisionTreeClassifier Prediction")
plt.show()
#########################################################
# 实现随机森林分类
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(train_x,train_y)
# 预测
predict = rfc.predict(test_x)
# print("预测结果")
# print(predict)
# print("真实结果")
# print(test_y)
# 评价结果
print("Accruacy of RandomForestClassifier: ",rfc.score(test_x,test_y))
print(classification_report(test_y,predict,target_names=['real','fake']))
print("\n")

plt.scatter(range(len(predict)),predict,s=8,marker='.',c='red')
plt.title("The Distribution of RandomForestClassifier Prediction")
plt.show()
