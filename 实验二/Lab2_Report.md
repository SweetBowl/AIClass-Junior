## 一、实验原理

![【机器学习】支持向量机 SVM（非常详细）](https://pic1.zhimg.com/v2-e833772fe2044ad9c353fb0173bd0b79_1440w.jpg?source=172ae18b)

### 1.SVM基本思想

支持向量机，因为英文名为 support vector machine，故一般简称为SVM。他是一种常用的判别方法，在机器学习领域是一个有监督的学习模式，通常用来进行模型识别，分类，回归分析以及异常值检测。

通俗的讲：支持向量机是一种两类分类模型，其基本模型定义为特征空间上的间隔最大的线性分类器，其学习策略（分割原则）便是间隔最大化，最终可转换为一个凸二次规划问题的求解。

支持向量机是许多大佬在多年研究统计学习理论基础上对线性分类器提出的另一种设计最佳准则。其原理也从线性可分说起，然后扩展到线性不可分的情况。甚至扩展到使用非线性函数中去。

SVM的主要思想可以概括为两点：

一是它针对线性可分情况进行分析，对于线性不可分情况，通过使用非线性映射算法将低维输入空间线性不可分的样本转化为高维特征空间使其可分，从而使得高维特征空间采用线性算法对样本的非线性特征进行线性分析成为可能。

二是它基于结果风险最小化理论之上在特征空间中构建最优超平面，使得学习器得到全局最优化，并且在整个样本空间的期望以某个概率满足一定上界。

**支持向量机的线性分类**：是给定一组训练实例，每个训练实例被标记为属于两个类别中的一个或另一个，SVM训练算法创建一个将新的实例分配给两个类别之一的模型，使其成为非概率二元线性分类器。SVM模型是将实例表示为空间中的点，这样映射就使得单独类别的实例被尽可能宽的明显的间隔分开。然后，将新的实例映射到同一空间，并基于他们落在间隔的哪一侧来预测所属类别。

**支持向量机的非线性分类**：除了进行线性分类之外，SVM还可以使用所谓的核技巧有效的进行非线性分类，将其输入隐式映射到高维特征空间中。当数据未被标记时，不能进行监督式学习，需要用非监督式学习，它会尝试找出数据到簇的自然聚类，并将新数据映射到这些已形成的簇。将支持向量机改进的聚类算法被称为支持向量聚类，当数据未被标记或者仅一些数据被标记时，支持向量聚类经常在工业应用中用作分类步骤的预处理。

### 2.支持向量

**线性可分**

在二维空间上，两类点被一条直线完全分开叫做线性可分。

严格的数学定义是：

![[公式]](https://www.zhihu.com/equation?tex=D_0) 和 ![[公式]](https://www.zhihu.com/equation?tex=D_1) 是 n 维欧氏空间中的两个点集。如果存在 n 维向量 w 和实数 b，使得所有属于 ![[公式]](https://www.zhihu.com/equation?tex=D_0)的点 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 都有 ![[公式]](https://www.zhihu.com/equation?tex=wx_i+%2B+b+%3E+0) ，而对于所有属于 ![[公式]](https://www.zhihu.com/equation?tex=D_1) 的点 ![[公式]](https://www.zhihu.com/equation?tex=x_j) 则有 ![[公式]](https://www.zhihu.com/equation?tex=wx_j+%2B+b+%3C+0) ，则我们称 ![[公式]](https://www.zhihu.com/equation?tex=D_0) 和 ![[公式]](https://www.zhihu.com/equation?tex=D_1) 线性可分。

**最大间隔超平面**

从二维扩展到多维空间中时，将 ![[公式]](https://www.zhihu.com/equation?tex=D_0) 和 ![[公式]](https://www.zhihu.com/equation?tex=D_1) 完全正确地划分开的 ![[公式]](https://www.zhihu.com/equation?tex=wx%2Bb%3D0) 就成了一个超平面。

为了使这个超平面更具鲁棒性，我们会去找最佳超平面，以最大间隔把两类样本分开的超平面，也称之为最大间隔超平面。

- 两类样本分别分割在该超平面的两侧；
- 两侧距离超平面最近的样本点到超平面的距离被最大化了。

**支持向量**

![img](https://pic4.zhimg.com/80/v2-0f1ccaf844905148b7e75cab0d0ee2e3_1440w.jpg)

样本中距离超平面最近的一些点，这些点叫做支持向量。

**SVM最优化问题**

SVM 想要的就是找到各类样本点到超平面的距离最远，也就是找到最大间隔超平面。任意超平面可以用下面这个线性方程来描述：

![[公式]](https://www.zhihu.com/equation?tex=w%5ETx%2Bb%3D0+%5C%5C)

通过一系列转换，最后可得到最优化问题是：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmin+%5Cfrac%7B1%7D%7B2%7D+%7C%7Cw%7C%7C%5E2+%5C+s.t.+%5Cquad+y_i%EF%BC%88w%5ETx_i%2Bb%EF%BC%89%5Cgeq+1+%5C%5C)

### 3.核函数

**线性不可分**

![img](https://img2020.cnblogs.com/blog/1226410/202005/1226410-20200521171806971-909827206.png)

这种情况的解决方法就是：将二维线性不可分样本映射到高维空间中，让样本点在高维空间线性可分，比如：

![img](https://pic1.zhimg.com/80/v2-9758d49e634c15a3e684ab84bad913ec_1440w.jpg)

对于在有限维度向量空间中线性不可分的样本，我们将其映射到更高维度的向量空间里，再通过间隔最大化的方式，学习得到支持向量机，就是非线性 SVM。 

我们用 x 表示原来的样本点，用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi%28x%29) 表示 x 映射到特征新的特征空间后到新向量。那么分割超平面可以表示为： ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29%3Dw+%5Cphi%28x%29%2Bb) 。

对于非线性 SVM 的对偶问题就变成了：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmin%5Climits_%7B%5Clambda%7D+%5B%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Clambda_i+%5Clambda_j+y_i+y_j+%28%5Cphi%28x_i%29+%5Ccdot+%5Cphi%28x_j%29%29-%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Clambda_i%5D+%5C%5C+s.t.++%5Cquad+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Clambda_iy_i+%3D+0%2C+%5Cquad+%5Clambda_i+%5Cgeq+0%2C+%5Cquad+C-%5Clambda_i-%5Cmu_i%3D0+%5C%5C)

**核函数作用**

因为低维空间映射到高维空间后维度可能会很大，如果将全部样本的点乘全部计算好，这样的计算量太大。如果存在一核函数 ![[公式]](https://www.zhihu.com/equation?tex=k%28x%2Cy%29+%3D+%28%5Cphi%28x%29%2C%5Cphi%28y%29%29) ， ![[公式]](https://www.zhihu.com/equation?tex=x_i) 与 ![[公式]](https://www.zhihu.com/equation?tex=x_j) 在特征空间的内积等于它们在原始样本空间中通过函数 ![[公式]](https://www.zhihu.com/equation?tex=k%28+x%2C+y%29) 计算的结果，我们就不需要计算高维甚至无穷维空间的内积。

核函数的引入一方面减少了计算量，另一方面也减少了存储数据的内存使用量。

**常见的核函数**

我们常用核函数有：

**线性核函数**

![[公式]](https://www.zhihu.com/equation?tex=k%28x_i%2Cx_j%29+%3D+x_i%5ETx_j+%5C%5C)

**多项式核函数**

![[公式]](https://www.zhihu.com/equation?tex=+k%28x_i%2Cx_j%29+%3D+%28x_i%5ETx_j%29%5Ed%5C%5C)

**高斯核函数**

![[公式]](https://www.zhihu.com/equation?tex=k%28x_i%2Cx_j%29+%3D+exp%28-%5Cfrac%7B%7C%7Cx_i-x_j%7C%7C%7D%7B2%5Cdelta%5E2%7D%29+%5C%5C)

这三个常用的核函数中只有高斯核函数是需要调参的。

### 4.Sklearn LinearSVC

**函数原型**

class sklearn.svm.SVC(self, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
       coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)

**参数**

- kernel：核函数，核函数是用来将非线性问题转化为线性问题的一种方法，默认是“rbf”核函数
- C：惩罚系数，用来控制损失函数的惩罚系数，类似于LR中的正则化系数。默认为1，一般需要通过交叉验证来选择一个合适的C，一般来说，噪点比较多的时候，C需要小一些。
- degree：当核函数是多项式核函数（“poly”）的时候，用来控制函数的最高次数。（多项式核函数是将低维的输入空间映射到高维的特征空间），这个参数只对多项式核函数有用，是指多项式核函数的阶数 n。如果给的核函数参数是其他核函数，则会自动忽略该参数。
- gamma：核函数系数，默认是“auto”，即特征维度的倒数。核函数系数，只对rbf poly sigmoid 有效。
- coef0：核函数常数值（ y = kx + b 的b值），只有“poly”和“sigmoid” 函数有，默认值是0.
- max_iter：最大迭代次数，默认值是 -1 ，即没有限制。
- probability：是否使用概率估计，默认是False。
- decision_function_shape：与“multi_class”参数含义类似，可以选择“ovo” 或者“ovr”（0.18版本默认是“ovo”，0.19版本为“ovr”）。OvR（one vs rest）的思想很简单，无论你是多少元分类，我们都可以看做二元分类，具体的做法是，对于第K类的分类决策，我们把所有第K类的样本作为正例，除第K类样本以外的所有样本作为负类，然后在上面做二元分类，得到第K类的分类模型。 OvO（one vs one）则是每次在所有的T类样本里面选择两类样本出来，不妨记为T1类和T2类，把所有的输出为T1 和 T2的样本放在一起，把T1作为正例，T2 作为负例，进行二元分类，得到模型参数，我们一共需要T(T-1)/2 次分类。从上面描述可以看出，OvR相对简单，但是分类效果略差（这里是指大多数样本分布情况，某些样本分布下OvR可能更好），而OvO分类相对精确，但是分类速度没有OvR快，一般建议使用OvO以达到较好的分类效果
- chache_size：缓冲大小，用来限制计算量大小，默认是200M，如果机器内存大，推荐使用500MB甚至1000MB

- tol：残差收敛条件，默认是0.0001，与LR中的一致。
- class_weight：与其他模型中参数含义一样，也是用来处理不平衡样本数据的，可以直接以字典的形式指定不同类别的权重，也可以使用balanced参数值。如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高，当然，如果你的样本类别分布没有明显的偏倚，则可以不管这个系数，选择默认的None
- verbose：是否冗余，默认为False
- random_state：随机种子的大小

　　错误项的惩罚系数。C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，也就是对测试数据的分类准确率降低。相反，减少C的话，容许训练样本中有一些误分类错误样本，泛化能力强。对于训练样本带有噪音的情况，一般采用后者，把训练样本集中错误分类的样本作为噪音。

**方法**

- decision_function(x)：获取数据集X到分离超平面的距离
- fit(x , y)：在数据集（X，y）上使用SVM模型
- get_params（[deep]）：获取模型的参数
- predict(X)：预测数值型X的标签
- score（X，y）：返回给定测试集合对应标签的平均准确率

**对象**

- support_：以数组的形式返回支持向量的索引
- support_vectors_：返回支持向量
- n_support_：每个类别支持向量的个数
- dual_coef：支持向量系数
- coef_：每个特征系数（重要性），只有核函数是LinearSVC的是可用，叫权重参数，即w
- intercept_：截距值（常数值），称为偏置参数，即b

## 二、实验目的

1. 掌握使用Sklearn实现SVM与kernel SVM
2. 掌握SVM与kernel SVM基本原理
3. 熟悉Python编程
4. 掌握使用Python数据处理和数据可视化方法

## 三、实验内容

#### 1.线性支持向量机（硬间隔支持向量机——Hard Margin SVM， 软间隔支持向量机——Soft Margin SVM）

**使用Sklearn构建线性支持向量机**

**数据集：**使用 sklearn.datasets.make_blobs 函数为聚类任务生成数据集

**实验过程**：生成数据集，划分数据集，搭建model用于训练，调节参数，分析结果，数据可视化

#### 2.基于核方法的 SVM ( rbf kernel, etc.) 当数据线性不可分时，采用各种kernel tricks

**使用Sklearn构建基于核方法的SVM**

**数据集**：Iris，特征选用前两个特征值

**实验过程**：加载数据集，划分数据集，搭建model用于训练，调节参数，分析结果，数据可视化

## 四、实验步骤

#### 1.线性支持向量机（硬间隔支持向量机——Hard Margin SVM， 软间隔支持向量机——Soft Margin SVM）

(1). 生成数据集。使用 sklearn.datasets.make_blobs 函数为聚类任务生成数据集，输出数据集和相应的标签。
特征数设置为 2 (n_features)，标签数设置为 2 (centers);
划分数据集，80%为训练集、其余为测试集。
(2). 数据可视化
以第一个样本特征为 x 轴，第二个样本特征为 y 轴，绘制散点图。(根据标签着色)
(3). 搭建模型。svm.LinearSVC。
(4). train and test。训练集注入模型，随后将训练好的模型用于测试集预测。
测试结果衡量指标(metrics)：accuacy。
(5). 分析讨论。调整模型参数 C，对不同结果进行分析。

#### 2.基于核方法的 SVM ( rbf kernel, etc.) 当数据线性不可分时，采用各种kernel tricks

(1). 数据集：iris。
特征采用,数据集前两个特征值;
划分数据集，前 130 为训练集，其余为测试集。
(2). 搭建模型。svm.SVC ( kernel = ‘ rbf ’ )，选择核函数。
(3). train and test。训练集注入模型，随后将训练好的模型用于测试集预测
测试结果衡量指标(metrics)：accuacy，precision and recall。
(4). 可视化。绘制支持向量机分类边界。

## 五、实验结果与分析（含重要数据结果分析或核心代码流程分析）

### 1.线性支持向量机

(1)生成数据集。使用 sklearn.datasets.make_blobs 函数为聚类任务生成数据集，输出数据集和相应的标签。特征数设置为 2 (n_features)，标签数设置为 2 (centers);划分数据集，80%为训练集、其余为测试集。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC  # support vector classifier

# 生成数据集
X,y = make_blobs(n_samples=100,n_features=2, centers=2,random_state=0,cluster_std=0.8)
# 划分数据集，80%为训练集
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
```

(2)数据可视化
以第一个样本特征为 x 轴，第二个样本特征为 y 轴，绘制散点图。(根据标签着色)

```python
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
```



![Lab2Data1](/Users/zhaoxu/Desktop/Lab2Data1.png)

(3)搭建模型。svm.LinearSVC。

```python
model = SVC(kernel='linear',C=0.2)
```

模型

```
SVC(C=0.2, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
```

(4)train and test。训练集注入模型，随后将训练好的模型用于测试集预测。
测试结果衡量指标(metrics)：accuacy。

```python
model.fit(x_train,y_train)
print("SVM 模型训练集准确率: %.3f" %metrics.accuracy_score(y_train,model.predict(x_train)))
y_hat = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test,y_hat)
print("SVM 模型测试集准确率: %.3f" %accuracy)
print(metrics.classification_report(y_test,y_hat))
```

**输出结果**

```
SVM 模型训练集准确率: 0.999
SVM 模型测试集准确率: 0.995
             precision    recall  f1-score   support

          0       1.00      0.99      0.99       100
          1       0.99      1.00      1.00       100

avg / total       1.00      0.99      0.99       200
```

(5)分析讨论。调整模型参数 C，对不同结果进行分析。

C=20时，

```python
model = SVC(kernel='linear',C=20)
model.fit(x_train,y_train)
```

模型

```
SVC(C=20, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
```

模型评估

```python
print("SVM 模型训练集准确率: %.3f" %metrics.accuracy_score(y_train,model.predict(x_train)))
y_hat = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test,y_hat)
print("SVM 模型测试集准确率: %.3f" %accuracy)
print(metrics.classification_report(y_test,y_hat))
```

```
SVM 模型训练集准确率: 0.999
SVM 模型测试集准确率: 1.000
             precision    recall  f1-score   support

          0       1.00      1.00      1.00       100
          1       1.00      1.00      1.00       100

avg / total       1.00      1.00      1.00       200
```

(6)数据可视化

```python
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
def plot_svc_decision(model,ax=None,plot_support=True):
    # Plot the decision function for a 2D SVC
    if ax is None:
        #得到子图
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    Y,X =np.meshgrid(y,x)
    xy = np.vstack([X.ravel(),Y.ravel()]).T
    # 绘制等高线
    P = model.decision_function(xy).reshape(X.shape)
    ax.contour(X,Y,P,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=300,linewidth=1,facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plot_svc_decision(model)
plt.show()
```

![Lab2P2](/Users/zhaoxu/Desktop/Lab2P2.png)

```python
X, y = make_blobs(n_samples=100, centers=2,random_state=0, cluster_std=0.8)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, C in zip(ax, [20, 0.2]): 
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision(model, axi)
    axi.scatter(model.support_vectors_[:, 0],model.support_vectors_[:, 1],s=300, lw=1, facecolors='none');
    axi.set_title('C = {0:.1f}'.format(C), size=14)
```

![Lab2P3](/Users/zhaoxu/Desktop/Lab2P3.png)



#### 2.基于核方法的 SVM ( rbf kernel, etc.) 

(1)加载数据集：iris；选用数据集前两个特征值;划分数据集，前 130 为训练集，其余为测试集。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC  # support vector classifier

data = pd.read_table('Dataset/iris.data.txt',sep=',')
# Label
labels = data.iloc[:,4]
# 取2，3列的特征
x = data.iloc[:,:2]
#计算训练集的平均值和标准差，拟合数据，将它转化成标准形式
x = StandardScaler().fit_transform(x)
data_x = np.array(x)
print(data_x.shape)

label_y = np.array(labels)
le = preprocessing.LabelEncoder()
le.fit(['Iris-setosa','Iris-versicolor', 'Iris-virginica'])
data_y = le.transform(label_y)

# 划分数据集
x_train = data_x[:130]
y_train = data_y[:130]
x_test = data_x[-19:]
y_test = data_y[-19:]
```

(2)搭建模型。svm.SVC ( kernel = ‘ rbf ’ )，选择核函数。

```python
model = SVC(kernel='rbf',C=10000000)
```

模型

```
SVC(C=10000000, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
```

(3)train and test。训练集注入模型，随后将训练好的模型用于测试集预测
测试结果衡量指标(metrics)：accuacy，precision and recall。

```python
model.fit(x_train,y_train)
print("kernel SVM 模型训练集准确率: %.3f" %metrics.accuracy_score(y_train,model.predict(x_train)))
y_hat = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test,y_hat)
print("kernel SVM 模型测试集准确率: %.3f" %accuracy)
print(metrics.classification_report(y_test,y_hat))
```

输出结果

```
kernel SVM 模型训练集准确率: 0.946
kernel SVM 模型测试集准确率: 0.526
             precision    recall  f1-score   support

          1       0.00      0.00      0.00         0
          2       1.00      0.53      0.69        19

avg / total       1.00      0.53      0.69        19
```

(4)可视化。绘制支持向量机分类边界。

```python
N, M = 500, 500 # 横纵各采样多少值
x1_min, x1_max = data_x[:,0].min(), data_x[:,0].max()
x2_min, x2_max = data_x[:,1].min(), data_x[:,1].max()
t1 = np.linspace(x1_min, x1_max,N)
t2 = np.linspace(x2_min, x2_max,M)
x1, y1 = np.meshgrid(t1, t2)    #生成网格采样点

x_test = np.stack((x1.flat,y1.flat),axis=1)     # 测试点(250000,2)
y_hat = model.predict(x_test)
y_hat = y_hat.reshape(x1.shape)

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
plt.title("SVM classification: Result of Iris classification")
plt.grid(b=True,ls=':')
plt.show()
```

可视化结果

![Lab2kernel](/Users/zhaoxu/Desktop/Lab2kernel.png)



## 六、总结及心得体会

### 1.SVM调参

SVM模型有两个非常重要的参数C与gamma，其中C是惩罚系数，即对误差的宽容忍，C越高，说明越不能容忍出现误差，容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差。

gamma 是选择 RBF 函数作为kernel后，该函数自带的一个参数。隐含的决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越小，gamma值越小，支持向量越多。

**(1)线性可分LinearSVC**

**调节参数C：**

![Lab2P3](/Users/zhaoxu/Desktop/Lab2P3.png)

可以看到左边这幅图C值比较大，要求比较严格，不能分错东西，隔离带中没有进入任何一个点，但是隔离带的距离比较小，泛化能力比较差。右边这幅图C值比较小，要求相对来说比较松一点，隔离带较大，但是隔离带中进入了很多的黄点和红点。那么C大一些好还是小一些好呢？这需要考虑实际问题，可以进行K折交叉验证来得到最合适的C值。

**(2)线性不可分kernel SVC**

**调节参数C**

**固定gamma=1**

C=1000

```
kernel SVM 模型训练集准确率: 0.885
kernel SVM 模型测试集准确率: 0.316
             precision    recall  f1-score   support

          1       0.00      0.00      0.00         0
          2       1.00      0.32      0.48        19

avg / total       1.00      0.32      0.48        19
```

![Lab2C1000](/Users/zhaoxu/Desktop/Lab2C1000.png)



C=100000

```
kernel SVM 模型训练集准确率: 0.946
kernel SVM 模型测试集准确率: 0.526
             precision    recall  f1-score   support

          1       0.00      0.00      0.00         0
          2       1.00      0.53      0.69        19

avg / total       1.00      0.53      0.69        19
```

![Lab2C100000](/Users/zhaoxu/Desktop/Lab2C100000.png)

C=10000000

```
kernel SVM 模型训练集准确率: 0.962
kernel SVM 模型测试集准确率: 0.526
             precision    recall  f1-score   support

          1       0.00      0.00      0.00         0
          2       1.00      0.53      0.69        19

avg / total       1.00      0.53      0.69        19
```

![Lab2C1000000](/Users/zhaoxu/Desktop/Lab2C1000000.png)

**调节参数gamma**

下面再看看另一个参数gamma值，这个参数值只是在高斯核函数里面才有，这个参数控制着模型的复杂程度，这个值越大，模型越复杂，值越小，模型就越精简。

**固定C=10000000**

Gamma=0.1

```
kernel SVM 模型训练集准确率: 0.862
kernel SVM 模型测试集准确率: 0.316
             precision    recall  f1-score   support

          1       0.00      0.00      0.00         0
          2       1.00      0.32      0.48        19

avg / total       1.00      0.32      0.48        19
```

![Lab2G0.1](/Users/zhaoxu/Desktop/Lab2G0.1.png)

Gamma='auto'(本实验中即为0.5)

```
kernel SVM 模型训练集准确率: 0.946
kernel SVM 模型测试集准确率: 0.526
             precision    recall  f1-score   support

          1       0.00      0.00      0.00         0
          2       1.00      0.53      0.69        19

avg / total       1.00      0.53      0.69        19
```

![Lab2kernel](/Users/zhaoxu/Desktop/Lab2kernel.png)

Gamma=1

```
kernel SVM 模型训练集准确率: 0.962
kernel SVM 模型测试集准确率: 0.526
             precision    recall  f1-score   support

          1       0.00      0.00      0.00         0
          2       1.00      0.53      0.69        19

avg / total       1.00      0.53      0.69        19
```

![Lab2G1](/Users/zhaoxu/Desktop/Lab2G1.png)

Gamma=10

```
kernel SVM 模型训练集准确率: 0.962
kernel SVM 模型测试集准确率: 0.421
             precision    recall  f1-score   support

          1       0.00      0.00      0.00         0
          2       1.00      0.42      0.59        19

avg / total       1.00      0.42      0.59        19
```

![Lab2G10](/Users/zhaoxu/Desktop/Lab2G10.png)

可以看出，当这个参数较大时，可以看出模型分类效果很好，但是泛化能力不太好。当这个参数较小时，可以看出模型里面有些分类是错误的，但是这个泛化能力更好，一般也应有的更多。

**小结**

核 SVM 的重要参数是正则化参数 C、核的选择以及与核相关的参数。虽然我们主要讲的是rbf核，但 scikit-learn 中还有其他选择。RBF 核只有一个参数 gamma，它是高斯核宽度的倒数。gamma 和 C 控制的都是模型复杂度，较大的值都对应更为复杂的模型。因此，这两个参数的设定通常是强烈相关的，应该同时调节。

下面再对其他调参要点做一个小结：

- 1，一般推荐在做训练之前对数据进行归一化，当然测试集的数据也要做归一化
- 2，在特征数非常多的情况下，或者样本数远小于特征数的时候，使用线性核，效果就很好了，并且只需要选择惩罚系数C即可
- 3，在选择核函数的时候，如果线性拟合效果不好，一般推荐使用默认的高斯核（rbf），这时候我们主要对惩罚系数C和核函数参数 gamma 进行调参，经过多轮的交叉验证选择合适的惩罚系数C和核函数参数gamma。
- 4，理论上高斯核不会比线性核差，但是这个理论就建立在要花费更多的时间上调参上，所以实际上能用线性核解决的问题我们尽量使用线性核函数

### 2.支持向量机优缺点

支持向量机是一种分类器。之所以称为“机”是因为它会产生一个二值决策结果，即它是一种决策“机”。支持向量机的泛化错误率较低，也就是说它具有良好的学习能力，且学到的结果具有很好的推广性。这些优点使得支持向量机十分流行，有些人认为它是监督学习中最好的定式算法。

**优点**

- 有严格的数学理论支持，可解释性强，不依靠统计方法，从而简化了通常的分类和回归问题；
- 能找出对任务至关重要的关键样本（即：支持向量）；
- 采用核技巧之后，可以处理非线性分类/回归任务；
- 最终决策函数只由少数的支持向量所确定，计算的复杂性取决于支持向量的数目，而不是样本空间的维数，这在某种意义上避免了“维数灾难”。

**缺点**

- 训练时间长。当采用 SMO 算法时，由于每次都需要挑选一对参数，因此时间复杂度为 ![[公式]](https://www.zhihu.com/equation?tex=O%28N%5E2%29)，其中 N 为训练样本的数量；
- 当采用核技巧时，如果需要存储核矩阵，则空间复杂度为 ![[公式]](https://www.zhihu.com/equation?tex=O%28N%5E2%29) ；
- 模型预测时，预测时间与支持向量的个数成正比。当支持向量的数量较大时，预测计算复杂度较高。
- 预处理数据和调参都需要非常小心。这也是为什么如今很多应用 中用的都是基于树的模型，比如随机森林或梯度提升。此外，SVM 模型很难检查。

因此支持向量机目前只适合小批量样本的任务，无法适应百万甚至上亿样本的任务。

## 七、对本实验过程及方法、手段的改进建议

本实验让我对SVM原理有了更深的理解，学会了使用Sklearn实现SVM。在做本次实验时，我用了很多时间调节模型的参数，并通过评估结果与可视化结果比较差别，选择更优的参数。这让我对模型的参数有了更深的理解。可能是由于数据样本较少，kernel SVM在测试集上的准确率不太理想。

以后的实验可引导同学们调节参数，深入探究不同参数对实验结果的影响，引导同学们理解SVM原理，学会在不同场景选用合适的模型，并尝试用不同的方法提高模型的准确率。