## 一、实验原理

### 1. 线性回归：Linear Regression

线性回归是利用数理统计中回归分析，来确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法，运用十分广泛。其表达形式为y = w'x+e，e为误差服从均值为0的正态分布。

在统计学中，线性回归（Linear Regression）是利用称为线性回归方程的最小平方函数对一个或多个自变量和因变量之间关系进行建模的一种回归分析。这种函数是一个或多个称为回归系数的模型参数的线性组合。只有一个自变量的情况称为简单回归，大于一个自变量情况的叫做多元回归。（这反过来又应当由多个相关的因变量预测的多元线性回归区别，而不是一个单一的标量变量。）

在线性回归中，数据使用线性预测函数来建模，并且未知的模型参数也是通过数据来估计。这些模型被叫做线性模型。最常用的线性回归建模是给定X值的y的条件均值是X的仿射函数。不太一般的情况，线性回归模型可以是一个中位数或一些其他的给定X的条件下y的条件分布的分位数作为X的线性函数表示。像所有形式的回归分析一样，线性回归也把焦点放在给定X值的y的条件概率分布，而不是X和y的联合概率分布（多元分析领域）。

线性回归是回归分析中第一种经过严格研究并在实际应用中广泛使用的类型。这是因为线性依赖于其未知参数的模型比非线性依赖于其未知参数的模型更容易拟合，而且产生的估计的统计特性也更容易确定。

线性回归模型经常用最小二乘逼近来拟合，但他们也可能用别的方法来拟合，比如用最小化“拟合缺陷”在一些其他规范里（比如最小绝对误差回归），或者在桥回归中最小化最小二乘损失函数的惩罚.相反,最小二乘逼近可以用来拟合那些非线性的模型.因此，尽管“最小二乘法”和“线性模型”是紧密相连的，但他们是不能划等号的。

#### 线性回归的步骤

1. 假设目标值（因变量）与特征值（自变量）之间线性相关（即满足一个多元一次方程，如：f(x)=w1x1+…+wnxn+b.）。
2. 然后构建损失函数。
3. 最后通过令损失函数最小来确定参数。（最关键的一步）

#### 线性回归拟合方程

以一元线性回归为例：有n组数据，自变量x(x1,x2,…,xn)，因变量y(y1,y2,…,yn)，然后我们假设它们之间的关系是：f(x)=ax+b。那么线性回归的目标就是如何让f(x)和y之间的差异最小，换句话说就是a，b取什么值的时候f(x)和y最接近。在回归问题中，均方误差是回归任务中最常用的性能度量。记J(a,b)为f(x)和y之间的差异，即

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031225640628.png#pic_center)

i代表n组数据中的第i组。
这里称J(a,b)为损失函数，明显可以看出它是个二次函数，即凸函数所以有最小值。当J(a,b)取最小值的时候，f(x)和y的差异最小，然后我们可以通过J(a,b)取最小值来确定a和b的值。

下面介绍确定a，b值的两种方法：

**方法一：最小二乘法**

损失函数J(a,b)是凸函数，那么分别关于a和b对J(a,b)求偏导，并令其为零解出a和b。![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031230145362.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020103123020160.png#pic_center)

解得：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031231002725.png#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031230931899.png#pic_center)



**方法二：梯度下降法**

梯度是一个向量，表示某一函数（该函数一般是二元及以上的）在该点处的方向导数沿着该方向取的最大值，即函数在该点沿着该方向（此梯度的方向）变化最快，变化率最大（为该梯度的模）。

在梯度下降法中，需要我们先给参数a赋一个预设值，然后再一点一点的修改a，直到J(a)取最小值时，确定a的值，即下图公式，公式中的α为学习率

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201101100653809.png#pic_center)



**方法的选用**

- 最小二乘法一般比较少用，虽然它的思想比较简单，在计算过程中需要对损失函数求导并令其为0，从而解出系数θ。但是对于计算机来说很难实现，所以一般不使用最小二乘法。
- 梯度下降法是通用的，包括更为复杂的逻辑回归算法中也可以使用，但是对于较小的数据量来说它的速度并没有优势。



#### 回归方程误差

**离差平方和**

![img](https://bkimg.cdn.bcebos.com/formula/cf21fec9f7853f59b94d3ad7ef728e24.svg)

 

![img](https://bkimg.cdn.bcebos.com/formula/2b6d70a8bd1476e7bf9d7dbeba145f0c.svg)

 

![img](https://bkimg.cdn.bcebos.com/formula/69c7b5001f156cd0953e15be0b68a6dd.svg)

其中

![img](https://bkimg.cdn.bcebos.com/formula/65e01e7642774d77cdf67412fce49a4e.svg)

 代表y的平方和；r 是相关系数，代表变异被回归直线解释的比例；

![img](https://bkimg.cdn.bcebos.com/formula/1080b39405c35883c23598980339f1b7.svg)

 就是不能被回归直线解释的变异，即SSE。

**不确定度**

#### 斜率b

法1：用

![img](https://bkimg.cdn.bcebos.com/formula/29ecac7d98ef7b8b7d4399a961ff7b42.svg)

![img](https://bkimg.cdn.bcebos.com/formula/41024347aba46f4d5b6bcbef9c7e5e37.svg)

法2：把斜率b带入

![img](https://bkimg.cdn.bcebos.com/formula/1ca230f30aa6c4bd2bb21aefe16e0aba.svg)

#### 截距a

![img](https://bkimg.cdn.bcebos.com/formula/be259598d7ef8dc7ec0a7708ab66ce2a.svg)





###2.逻辑斯蒂回归（Logistic Regression） 

Logistic Regression，虽然这个算法从名字上来看，是回归算法，但其实际上是一个分类算法。

LR回归是在线性回归模型的基础上，使用sigmoid函数，将线性模型 wTx 的结果压缩到[0,1] 之间，使其拥有概率意义。 其本质仍然是一个线性模型，实现相对简单。在广告计算和推荐系统中使用频率极高，是CTR预估模型的基本算法。同时，LR模型也是深度学习的基本组成单元。

LR回归属于概率性判别式模型，之所谓是概率性模型，是因为LR模型是有概率意义的；之所以是判别式模型，是因为LR回归并没有对数据的分布进行建模，也就是说，LR模型并不知道数据的具体分布，而是直接将判别函数，或者说是分类超平面求解了出来。

#### 逻辑斯蒂分布（Logistic Distribution）

逻辑斯蒂分布的密度函数和分布函数如下：

![截屏2020-12-27下午8.31.21](/Users/zhaoxu/Library/Application Support/typora-user-images/截屏2020-12-27下午8.31.21.png)

其中μ是位置参数，而s 是形状参数。

#### 从逻辑斯蒂分布到逻辑斯蒂回归模型

分类算法都是求解p(Ck|x)，而逻辑斯蒂回归模型，就是用当μ=0,s=1 时的逻辑斯蒂分布的概率分布函数：sigmoid函数，对p(Ck|x)进行建模，所得到的模型。

对于二分类的的逻辑斯蒂回归模型可得：![截屏2020-12-27下午8.42.53](/Users/zhaoxu/Library/Application Support/typora-user-images/截屏2020-12-27下午8.42.53.png)

a 就是我们在分类算法中的决策面。

参数向量w的求解有多种方法，梯度下降法实现相对简单，但是其收敛速度往往不尽人意，可以考虑使用随机梯度下降法来解决收敛速度的问题。但上面两种在最小值附近，都存在以一种曲折的慢速逼近方式来逼近最小点的问题。所以在LR回归的实际算法中，用到的是牛顿法，拟牛顿法（DFP、BFGS、L-BFGS）。

#### 带惩罚项的LR回归

L1正则化和L2正则化主要是用来避免模型的参数w过大，而导致的模型过拟合的问题。其实现方法就是在前面的对数似然函数后面再加上一个惩罚项。

L2惩罚的LR回归：

![截屏2020-12-27下午8.47.12](/Users/zhaoxu/Library/Application Support/typora-user-images/截屏2020-12-27下午8.47.12.png)

L1惩罚的LR回归 ：

![截屏2020-12-27下午8.47.35](/Users/zhaoxu/Library/Application Support/typora-user-images/截屏2020-12-27下午8.47.35.png)

上式中，C是用于调节目标函数和惩罚项之间关系的，C越小，惩罚力度越大，所得到的w的最优解越趋近于0，或者说参数向量越稀疏；C越大，惩罚力度越小，越能体现模型本身的特征。



### 3. Scikit-learn

Scikit-learn（以前称为scikits.learn，也称为sklearn）是针对Python 编程语言的免费软件机器学习库。它具有各种分类，回归和聚类算法，包括支持向量机，随机森林，梯度提升，*k*均值和DBSCAN等。



### 4. sklearn-linearRegression（）

(1).fit_intercept：boolean,optional,default True。是否计算截距，默认为计算。如果使用中心化的数据，可以考虑设置为False,
(2).normalize：boolean,optional,default False。标准化开关，默认关闭；该参数在fit_intercept设置为False时自动忽略。如果为True,回归会标准化输入参数：(X-X均值)/||X||；若为False，在训练模型前，可使用sklearn.preprocessing.StandardScaler进行标准化处理。
(3).copy_X：boolean,optional,default True。默认为True, 否则X会被改写。
(4).n_jobs：int,optional,default 1int。默认为1.当-1时默认使用全部CPUs ??(这个参数有待尝试)。
**属性：**
coef：array,shape(n_features, ) or (n_targets, n_features)。回归系数(斜率)。
intercept: 截距
**方法：**
(1).fit(X,y,sample_weight=None)
X:array, 稀疏矩阵 [n_samples,n_features] 
y:array [n_samples, n_targets] 
sample_weight:array [n_samples]，每条测试数据的权重，同样以矩阵方式传入。 
(2).predict(x):预测方法，将返回值y_pred
(3).get_params(deep=True)： 返回对regressor 的设置值
(4).score(X,y,sample_weight=None)：评分函数



### 5.相关系数与决定系数

**协方差与相关系数**

协方差是计算两个随机变量X X*X*和Y Y*Y* 之间的相关性的指标，定义如下：

Cov(*X*,*Y*)=E[(*X*−E*X*)(*Y*−E*Y*)]

协方差的值会随着变量量纲的变化而变化（covariance is not scale invariant），所以，提出相关系数的概念：

![截屏2020-12-28下午1.02.09](/Users/zhaoxu/Library/Application Support/typora-user-images/截屏2020-12-28下午1.02.09.png)

相关系数是用于描述两个变量*线性*相关程度的，如果r &gt; 0 r \gt 0*r*>0，呈正相关；如果r = 0 r = 0*r*=0，不相关；如果r &lt; 0 r \lt 0*r*<0，呈负相关。

**决定系数（R方）**

R方一般用在回归模型用用于评估预测值和实际值的符合程度，R方的定义如下：

![截屏2020-12-28下午1.03.13](/Users/zhaoxu/Library/Application Support/typora-user-images/截屏2020-12-28下午1.03.13.png)

上式中y是实际值，f是预测值，y ^是实际值的平均值。FVU被称为fraction of variance unexplained，RSS叫做Residual sum of squares，TSS叫做Total sum of squares。根据R^2的定义，可以看到R^2是有可能小于0的，所以R^2不是r的平方。一般地，R^2越接近1，表示回归分析中自变量对因变量的解释越好。

R^2一般用在线性模型中，但R^2不能完全反映模型预测能力的高低



### 6. sklearn-LogisticRegression（）

**参数**

(1)penalty：惩罚项，str类型，可选参数为l1和l2，默认为l2。用于指定惩罚项中使用的规范。newton-cg、sag和lbfgs求解算法只支持L2规范。L1规范假设的是模型的参数满足拉普拉斯分布，L2假设的模型参数满足高斯分布。

(2)dual：对偶或原始方法，bool类型，默认为False。对偶方法只用在求解线性多核(liblinear)的L2惩罚项上。当样本数量>样本特征的时候，dual通常设置为False。

(3)tol：停止求解的标准，float类型，默认为1e-4。就是求解到多少的时候，停止，认为已经求出最优解。

(4)c：正则化系数λ的倒数，float类型，默认为1.0。必须是正浮点型数。像SVM一样，越小的数值表示越强的正则化。

(5)fit_intercept：是否存在截距或偏差，bool类型，默认为True。

(6)intercept_scaling：仅在正则化项为”liblinear”，且fit_intercept设置为True时有用。float类型，默认为1。

(7)class_weight：用于标示分类模型中各种类型的权重，可以是一个字典或者’balanced’字符串，默认为不输入，也就是不考虑权重，即为None。

(8)random_state：随机数种子，int类型，可选参数，默认为无，仅在正则化优化算法为sag,liblinear时有用。

(9)solver：优化算法选择参数，只有五个可选参数，即newton-cg,lbfgs,liblinear,sag,saga。默认为liblinear。solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择，分别是： 

- liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
- lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
- newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
- sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。

(10)multi_class: 决定了分类方式的选择，有 ovr和multinomial两个值可以选择，默认是 ovr。ovr即one-vs-rest(OvR)，而multinomial即many-vs-many(MvM)。如果是二元逻辑回归，ovr和multinomial并没有任何区别，区别主要在多元逻辑回归上。



## 二、实验目的

1. 掌握使用sklearn实现linear regression与logistics regression

2. 掌握linear regression与Logistics regression基本原理
3. 熟悉Python编程
4. 掌握使用Python数据处理与数据可视化方法

## 三、实验内容

#### 1.线性回归: linear regression

**使用sklearn进行线性回归：线性模型:y = βX+b**
X:数据 y:目标变量 β:回归系数 b:观测噪声(bias，偏差)

**数据集:**diabete。 根据feature-bmi对病情作出prediction。

数据集中的特征值总共10项，如下:

\#年龄 #性别 #体质指数 #血压 #s1,s2,s3,s4,s4,s6(六种血清的化验数据)
y:糖尿病预测值

**实验过程:**加载数据集，划分数据集-train -test,搭建model用于训练，得出预测结果。



#### 2.逻辑斯蒂回归: Logistic regression

**使用sklearn进行逻辑斯蒂回归**

**数据集:** iris (安德森鸢尾花卉数据集)。

sepal length in cm (花萼长度) #sepal width in cm (花萼宽度)

\#petal length in cm (花瓣长度) #petal width in cm (花瓣宽度)
\#class: -- Iris Setosa (山鸢尾)
-- Iris Versicolour (变色鸢尾)
-- Iris Virginica (维吉尼亚鸢尾)

**实验过程**：利用单个feature或者多个feature实现二元或者多元逻辑斯蒂回归。

## 四、实验步骤

1. **线性回归：**

(1). 加载数据集，对数据集进行标准化处理

(2). 划分数据集-train -test，此处选用前432个为训练样本，测试样本为最后20个

(3). 使用Sklearn中的Linear Regression搭建model，用于训练。

(4). 输入数据，得出预测结果

(5). 使用平方和、系数、截距和决定系数等参数评价结果，判断回归方程的拟合程度

(6). 数据可视化

2. **逻辑斯蒂回归：**

(1). 加载数据集，选用多个feature。

(2). 划分数据集，按比例分成训练集和测试集，此处两者比例为7:3

(3). 使用Sklearn中的Logistic Regression搭建model，用于训练

(4). 输入数据，得出预测结果

(5). 使用metrics中的accuracy_scor得出模型的正确率，评价结果

(6).可视化分类结果，画出决策边界

##五、实验结果与分析（含重要数据结果分析或核心代码流程分析）

### 1. 线性回归

(1)加载数据集，对数据进行标准化处理

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_table('Dataset/diabetes.tab.txt')

predata_x = data.iloc[:,2]
# 提取标记
data_target = data.iloc[:,10]

data_x_mean = np.mean(predata_x)
data_x_std = np.std(predata_x)
# Normalization
data_x = (predata_x-data_x_mean) / data_x_std
```

(2)划分数据集-train -test，此处选用前432个为训练样本，测试样本为最后20个

```Python
# 训练样本前432个，测试样本最后20个
# 处理成二维数组
diabetes_x_train1 = data_x.iloc[:-20]    # 训练样本
diabetes_x_train2=np.array(diabetes_x_train1)
diabetes_x_train3=np.expand_dims(diabetes_x_train2,axis = 1)

diabetes_x_test1 = data_x.iloc[-20:]		 # 测试样本
diabetes_x_test2=np.array(diabetes_x_test1)
diabetes_x_test3=np.expand_dims(diabetes_x_test2,axis = 1)

diabetes_y_train1 = data_target[:-20]    # 训练标记
diabetes_y_train2=np.array(diabetes_y_train1)
diabetes_y_train3=np.expand_dims(diabetes_y_train2,axis = 1)

diabetes_y_test1 = data_target[-20:]     # 测试标记
diabetes_y_test2=np.array(diabetes_y_test1)
diabetes_y_test3=np.expand_dims(diabetes_y_test2,axis = 1)
```

(3)使用Sklearn中的Linear Regression搭建model，用于训练

```python
# 实现线性回归
linearMdl = linear_model.LinearRegression()
#print(linearMdl)

# 训练
# X:array, 稀疏矩阵 [n_samples,n_features]
# y:array [n_samples, n_targets]
linearMdl.fit(diabetes_x_train3,diabetes_y_train3)
```

(4)输入数据，得出预测结果

```python
predict = linearMdl.predict(diabetes_x_test3)
print("预测结果")
print(predict)
print("真实结果")
print(diabetes_y_test3)
```

(5)使用平方和、系数、截距和方差等参数评价结果。使用决定系数来判断回归方程的拟合程度

```python
#评价结果
cost = np.mean(diabetes_y_test3 - predict)**2
print("平方和")
print(cost)
print("系数")
print(linearMdl.coef_)
print("截距")
print(linearMdl.intercept_)

#通过决定系数来来判断回归方程的拟合程度(可认为分数越高说明拟合的程度越好)
print("决定系数")
print(linearMdl.score(diabetes_x_test3,diabetes_y_test3))
```

(6)数据可视化

```python
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
```

#### 实验结果

```
预测结果
[[225.9732401 ]
 [115.74763374]
 [163.27610621]
 [114.73638965]
 [120.80385422]
 [158.21988574]
 [236.08568105]
 [121.81509832]
 [ 99.56772822]
 [123.83758651]
 [204.73711411]
 [ 96.53399594]
 [154.17490936]
 [130.91629517]
 [ 83.3878227 ]
 [171.36605897]
 [137.99500384]
 [137.99500384]
 [189.56845268]
 [ 84.3990668 ]]
真实结果
[[233]
 [ 91]
 [111]
 [152]
 [120]
 [ 67]
 [310]
 [ 94]
 [183]
 [ 66]
 [173]
 [ 72]
 [ 49]
 [ 64]
 [ 48]
 [178]
 [104]
 [132]
 [220]
 [ 57]]

平方和
301.2601155006569
系数
[[44.62742406]]
截距
[152.91886183]
决定系数
0.4725754479822714
```

数据可视化结果

![image-20201228114318811](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201228114318811.png)

### 2. 逻辑斯蒂回归

(1). 加载数据集，对数据进行标准化处理，选用多个feature。

```python
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
```

(2). 划分数据集，按比例分成训练集和测试集，此处两者比例为7:3

```python
#将原始数据集按比例分成训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0)
x_train, x_test, y_train, y_test =np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)
```

(3). 使用Sklearn中的Logistic Regression搭建model，用于训练

参数选择：solver = 'newton-cg', multi-class = 'multinomial'

```python
lr = LogisticRegression(solver='newton-cg',multi_class='multinomial')
lr.fit(x_train,y_train)
print("Logistic Regression 模型训练集的准确率: %.3f" %lr.score(x_train,y_train))
```

(4). 输入数据，得出预测结果

```python
y_hat = lr.predict(x_test)
```

(5). 使用metrics中的accuracy_scor得出模型的正确率，评价结果

```python
accuracy = metrics.accuracy_score(y_test, y_hat)
print("Logistic Regression 模型的正确率: %.3f" %accuracy)
target_list = ['setosa','versicolor','virginica']
print(metrics.classification_report(y_test,y_hat,target_names=target_list))
```

(6).可视化分类结果，画出决策边界

```python
# 可视化分类结果，画出决策边界
# 确定坐标范围，x,y轴各表示一个特征
# 从最大值到最小值构建一系列的数据，使得它能覆盖整个的特征数据范围，然后预测这些值所属的分类，并给它们所在的区域
N, M = 500, 500 # 横纵各采样多少值
x1_min, x1_max = data_x[:,0].min(), data_x[:,0].max()
x2_min, x2_max = data_x[:,1].min(), data_x[:,1].max()
t1 = np.linspace(x1_min, x1_max,N)
t2 = np.linspace(x2_min, x2_max,M)
x1, y1 = np.meshgrid(t1, t2)    #生成网格采样点
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
```



#### 实验结果

```
Logistic Regression 模型训练集的准确率: 0.990
Logistic Regression 模型的正确率: 0.889
             precision    recall  f1-score   support
     setosa       1.00      1.00      1.00        16
 versicolor       0.79      0.94      0.86        16
  virginica       0.90      0.69      0.78        13
avg / total       0.90      0.89      0.89        45
```

数据可视化结果

![image-20201228134014126](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201228134014126.png)



## 六、总结及心得体会

本实验主要是使用linear regression进行线性回归，对病情作出预测，使用logistic regression实现二元逻辑斯蒂回归，分类鸢尾花数据集。本次实验将课堂上的知识用于实践，很好地提高了我的python编程能力，让我进一步熟悉Numpy和Matplotlib等库，同时也加深了我对线性回归和逻辑斯蒂回归的理解。

通过本次实验，我思考了线性回归的优缺点

- 优点：建模速度快，不需要很复杂的计算，在数据量大的情况下依然运行速度很快。可以根据系数给出每个变量的理解和解释
- 缺点：不能很好地拟合非线性数据
- 应用场景：线性回归简单、易于使用，但是现实生活中数据的特征和目标之间并不是简单的线性组合，所以并不能很好的解决具体问题。所以线性回归常用于数据特征稀疏，并且数据过大的问题中，可以通过线性回归进行特征筛选。

同时我也思考了为什么在深度学习如此火热的今天还使用线性回归：

一方面，线性回归所能够模拟的关系其实远不止线性关系。线性回归中的“线性”指的是系数的线性，而通过对特征的非线性变换，以及广义线性模型的推广，输出和特征之间的函数关系可以是高度非线性的。另一方面，也是更为重要的一点，线性模型的易解释性使得它在物理学、经济学、商学等领域中占据了难以取代的地位。

我也进一步比较了线性回归与逻辑斯蒂回归：

虽然逻辑回归能够用于分类，不过其本质还是线性回归。它仅在线性回归的基础上，在特征到结果的映射中加入了一层sigmoid函数（非线性）映射，即先把特征线性求和，然后使用sigmoid函数来预测。

这主要是由于线性回归在整个实数域内敏感度一致，而分类范围，只需要在[0,1]之内。而逻辑回归就是一种减小预测范围，将预测值限定为[0,1]间的一种回归模型。逻辑曲线在z=0时，十分敏感，在z>>0或z<<0处，都不敏感，将预测值限定为[0,1]。

从梯度更新视角来看，为什么线性回归在整个实数域内敏感度一致不好？

线性回归和LR的梯度更新公式是一样的,如下：

![\theta_j :={1\over m} \alpha(h_\theta(x^{(i)})-y^{(i)}) x^{(i)}_j](https://private.codecogs.com/gif.latex?%5Ctheta_j%20%3A%3D%7B1%5Cover%20m%7D%20%5Calpha%28h_%5Ctheta%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29%20x%5E%7B%28i%29%7D_j)

唯一不同的是假设函数![h_\theta(x^{(i)})](https://private.codecogs.com/gif.latex?h_%5Ctheta%28x%5E%7B%28i%29%7D%29)，![y\in \{0,1\}](https://private.codecogs.com/gif.latex?y%5Cin%20%5C%7B0%2C1%5C%7D)，对于LR而言，![h_\theta(x^{(i)})\in [0,1]](https://private.codecogs.com/gif.latex?h_%5Ctheta%28x%5E%7B%28i%29%7D%29%5Cin%20%5B0%2C1%5D)，那么梯度更新的幅度就不会太大。而线性回归![h_\theta(x^{(i)})](https://private.codecogs.com/gif.latex?h_%5Ctheta%28x%5E%7B%28i%29%7D%29)在整个实数域上，即使已经分类正确的点，在梯度更新的过程中也会大大影响到其它数据的分类，就比如有一个正样本，其输出为10000，此时梯度更新的时候，这个样本就会大大影响负样本的分类。而对于LR而言，这种非常肯定的正样本并不会影响负样本的分类情况。



## 七、对本实验过程及方法、手段的改进建议

本次实验让我学到了很多新知识，在做实验时，我花了很多时间用于数据处理和数据可视化。由于模型的构建可直接调用Sklearn，相对简单。在逻辑斯蒂回归实验中，我也尝试过选择不同的参数，使用不同的优化方法。可能是由于数据样本较少，选用的特征较少，实验结果并不是很理想。

以后的实验可引导同学们通过调节参数，扩大数据集等方法提高模型准确率，引导同学们进一步理解线性回归和逻辑斯蒂回归的原理，比较两者，学会选择适合应用场景的模型。