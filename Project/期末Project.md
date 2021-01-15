## 人工智能—期末Project

### 任务一：回归任务—地震等级预测

（1） 数据集：quake.dat

- 三个特征值：Focal_depth（震源深度），Latitude（纬度），Longitude（经度）；

- 真实值：Richter（里氏震级）。

（2） 数据集划分（共2178项）：前2000项为训练集，后178项为测试集。

（3） 衡量指标：均方误差（MSE）和均方根误差（RMSE）

 

### 任务二：分类任务—真假钞票鉴别（二分类）

（1） 数据集：data_banknote_authentication.txt

- 四个特征值（连续型数值）：Variance of Wavelet Transformed image；Skewness of Wavelet Transformed image；Kurtosis of Wavelet Transformed image；Entropy of image；

- 标签：0为真钞，1为假钞

（2） 数据集划分（共1372项）：前1200项为训练集，后172项为测试集。

（3） 衡量指标：查准率（Precision）和查全率（Recall）。



**要求：每个任务分别用不少于三种算法实现。例如任务一中：线性回归、SVM回归，LSTM等），并分析各种算法的结果，以及其相互比较，并分析原因。**