# AIClass-Junior
大三人工智能课程实验

## 实验一

编辑器: Jupyter Notebook or Pycharm

#### 1.线性回归: linear regression

数据集: diabete。 根据feature-bmi对病情作出prediction。

实验过程:加载数据集，划分数据集-train. _test,搭建model用于训练，得出预测结果。

#### 2.逻辑斯蒂回归: Logistic regression

数据集: iris (安德森鸢尾花卉数据集)。

实验过程：利用单个feature或者多个feature实现二元或者多元逻辑斯蒂回归。

## 实验二

#### 一、线性支持向量机（硬间隔支持向量机——Hard Margin SVM， 软间隔支持向量机——Soft Margin SVM）

1、生成数据集。使用 sklearn.datasets.make_blobs 函数为聚类任务生成数据集，输出数据集和相应的标签。
特征数设置为 2 (n_features)，标签数设置为 2 (centers);
划分数据集，80%为训练集、其余为测试集。
2、数据可视化
以第一个样本特征为 x 轴，第二个样本特征为 y 轴，绘制散点图。(根据标签着色)
3、搭建模型。svm.LinearSVC。
4、train and test。训练集注入模型，随后将训练好的模型用于测试集预测。
测试结果衡量指标(metrics)：accuacy。
5、分析讨论。调整模型参数 C，对不同结果进行分析。

#### 二、基于核方法的 SVM ( rbf kernel, etc.) 当数据线性不可分时，采用各种kernel tricks
1、数据集：iris。
特征采用,数据集前两个特征值;
划分数据集，前 130 为训练集，其余为测试集。
2、搭建模型。svm.SVC ( kernel = ‘ rbf ’ )，选择核函数。
3、train and test。训练集注入模型，随后将训练好的模型用于测试集预测
测试结果衡量指标(metrics)：accuacy，precision and recall。
4、可视化。绘制支持向量机分类边界。

## 实验三

#### 无监督聚类算法：dbscan 与 k-means

1、加载数据集，Data_for_Cluster.npz ；
X 为特征，labels_true 为标签。
2、搭建模型，k-means 与 dbscan。
3、训练模型，调参，得出分类结果。
4、结果分析及可视化。
    绘制散点图。(根据分类结果进行着色) ；
    从算法原理的角度分析两个算法优缺点，及适应的数据集特征。
5、评估标准，轮廓系数法（Silhouette Cofficient），用来评估聚类算法的效果。

## 实验四

#### 手写数字识别——CNN 的应用
1、加载数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
2、搭建 CNN 模型
优化器（optimizer）：adam
评估指标（metrics）：accuacy
3、训练好模型后，将其用于测试
