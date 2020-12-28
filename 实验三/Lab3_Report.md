## 一、实验原理

### 1. K-means算法

K-Means 算法是无监督的聚类算法，它实现起来比较简单，聚类效果也不错，因此应用很广泛。

K-means算法思想，即对于给定的样本集，按照样本之间的距离大小，将样本集划分为K个簇。让簇内的点尽量紧密的连在一起，而让簇间的距离尽量的大。

如果用数据表达式表示，假设簇划分为（C1, C2, ....Ck），则我们的目标是最小化平方误差E：

![img](https://img2020.cnblogs.com/blog/1226410/202004/1226410-20200422152141311-1360758607.png)　　其中μi 是簇Ci的均值向量，有时也称为质心，表达式为：

![img](https://img2020.cnblogs.com/blog/1226410/202004/1226410-20200422152421218-1607644510.png)

如果想直接求上式的最小值并不容易，这是一个 NP 难的问题，因此只能采用启发式的迭代方法。

K-means采用的启发式方法可用下面这组图描述：

![img](https://img2020.cnblogs.com/blog/1226410/202004/1226410-20200422160247818-1088392093.png)

上图a表达了初始的数据集，假设k=2，在图b中，我们随机选择了两个 k 类所对应的类别质心，即图中的红色质心和蓝色质心，然后分别求样本中所有点到这两个质心的距离，并标记每个样本的类别为和该样本距离最小的质心的类别，如图C所示，经过计算样本和红色质心和蓝色质心的距离，我们得到了所有样本点的第一轮迭代后的类别。此时我们对我们当前标记为红色和蓝色的点分别求其新的质心，如图d所示，新的红色质心和蓝色质心的位置已经发生了变动。图e和图f 重复了我们在图c和图d的过程，即将所有点的类别标记为距离最近的质心的类别并求新的质心，最终我们得到了两个类别如图f。

#### K-means算法要点

(1).对于K-Means算法，首先要注意的是 k 值的选择，一般来说，我们会根据对数据的先验经验选择一个合适的 k 值，如果没有什么先验知识，则可以通过交叉验证选择一个合适的 k 值。

(2).在确定了 k 的个数后，我们需要选择 k 个初始化的质心，就像上图 b 中的随机质心。由于我们是启发式方法，k个初始化的质心的位置选择对最后的聚类结果和运行时间都有很大的影响，因此需要选择合适的 k个质心，最后这些质心不能太近。

#### K-means算法流程

**输入**：样本集D = {x1, x2, ... xm} 聚类的簇树k，最大迭代次数 N

　　**输出**：簇划分 C = {C1, C2, ....Ck}

　　1） 从数据集D中随机选择 k 个样本作为初始的 k 个质心向量：{ μ1, μ2, ...μk }

　　2）对于 n=1, 2, ,,,, N

　　　　a） 将簇划分C初始化为Ct = Ø t=1,2,....k

　　　　b）对于i=1, 2, ....m，计算样本 xi 和各个质心向量 μj(j = 1, 2, ... k）的距离：dij = || xi - μj ||22 ，将 xi 标记最小的为 dij 所对应的类别 λi 。此时更新 Cλi = Cλi υ {xi}

　　　　c）对于 j=1,2,...k，对Cj中所有的样本点重新计算新的质心 :

![img](https://img2020.cnblogs.com/blog/1226410/202004/1226410-20200422164843998-539551122.png)

 　　　d）如果所有的k个质心向量都没有发生变换，则转到步骤3。

　　3）输出簇划分 C = {C1, C2, ... Ck}

#### 

### 2.Sklearn K-means

在scikit-learn中，包括两个K-Means的算法，一个是传统的K-Means算法，对应的类是K-Means。另一个是基于采样的 Mini Batch K-Means算法，对应的类是 MiniBatchKMeans。一般来说，使用K-Means的算法调参是比较简单的。用K-Means类的话，一般要注意的仅仅就是 k 值的选择，即参数 n_clusters：如果是用MiniBatch K-Means 的话，也仅仅多了需要注意调参的参数 batch_size，即Mini Batch 的大小。下面介绍传统K-means。

**函数原型**

*class sklearn.cluster.Kmeans* (n_clusters=8, *, *init='k-means++'*, *n_init=10*, *max_iter=300*, *tol=0.0001*, *precompute_distances='deprecated'*, *verbose=0*, *random_state=None*, *copy_x=True*, *n_jobs='deprecated'*, *algorithm='auto'*)

**参数**

**(1)n_clusters**: 即k值，一般需要多试一些值以获得较好的聚类效果。

**(2)max_iter**： 最大的迭代次数，一般如果是凸数据集的话可以不管这个值，如果数据集不是凸的，可能很难收敛，此时可以指定最大的迭代次数让算法可以及时退出循环。

**(3)n_init：**用不同的初始化质心运行算法的次数。由于K-Means是结果受初始值影响的局部最优的迭代算法，因此需要多跑几次以选择一个较好的聚类效果，默认是10，一般不需要改动。

**(4)init：** 即初始值选择的方式，可以为完全随机选择'random',优化过的'k-means++'或者自己指定初始化的k个质心。一般建议使用默认的'k-means++'。

**(5)algorithm**：有“auto”, “full” or “elkan”三种选择。"full"就是我们传统的K-Means算法， “elkan”是我们原理篇讲的elkan K-Means算法。默认的"auto"则会根据数据值是否是稀疏的，来决定如何选择"full"和“elkan”。一般数据是稠密的，那么就是 “elkan”，否则就是"full"。一般来说建议直接用默认的"auto"



### 3.DBSCAN密度聚类算法

基于密度的聚类寻找被低密度区域分离的高密度区域。DBSCAN是一种基于密度的聚类算法，这类密度聚类算法一般假定类别可以通过样本分布的紧密程度决定。同一类别的样本，他们之间是紧密相连的，也就是说，在该类别任意样本周围不远处一定有同类别的样本存在。

通过将紧密相连的样本划为一类，这样就得到了一个聚类类别。通过将所有各组紧密相连的样本划为各个不同的类别，则得到了最终的所有聚类类别结果。

如图所示：A为核心对象 ； BC为边界点 ； N为离群点； 圆圈代表 ε-邻域

![img](https://img2020.cnblogs.com/blog/1226410/202004/1226410-20200425163607381-344225475.png)

**DBSCAN算法的本质是一个发现类簇并不断扩展类簇的过程**。对于任意一点q，若他是核心点，则在该点为中心，r为半径可以形成一个类簇 c。而扩展的方法就是，遍历类簇c内所有点，判断每个点是否是核心点，若是，则将该点的 ε-邻域也划入类簇c，递归执行，直到不能再扩展类簇C。

#### **DBSCAN密度聚类思想**

DBSCAN聚类定义：由密度可达关系导出的最大密度相连的样本集合，即为最终聚类的一个类别，或者说一个簇。

DBSCAN的簇里面可以有一个或者多个核心对象。如果只有一个核心对象，则簇里其他的非核心对象样本都在这个核心对象的 ε- 邻域里；如果有多个核心对象，则簇里的任意一个核心对象的  ε- 邻域中一定有一个其他的核心对象。否则这两个核心对象无法密度可达。这些核心对象的  ε- 邻域里所有的样本的集合组成的一个 DBSCAN聚类簇。

DBSCAN任意选择一个没有类别的核心对象作为种子，然后找到所有这个核心对象能够密度可达的样本集合，即为一个聚类簇。接着继续选择另一个没有类别的核心对象去寻找密度可达的样本集合，这样就得到了另一个聚类簇。一直运行到所有核心对象都有类别为止。

### DBSCAN密度聚类步骤

(1)将所有点标记为核心点，边界点或噪声点

(2)删除噪声点

(3)为距离在Eps内的所有核心点之间赋予一条边

(4)每组连通的核心点形成一个簇

(5)将每个边界点指派到一个与之无关的核心点的簇中



### 4.Sklearn DBSCAN

**函数原型**

c*lass sklearn.cluster.DBSCAN*(*eps=0.5*, *, *min_samples=5*, *metric='euclidean'*, *metric_params=None*, *algorithm='auto'*, *leaf_size=30*, *p=None*, *n_jobs=None*)

**参数**

**(1)eps：**DBSCAN算法参数，即我们的 ε- 邻域的距离阈值，和样本距离超过ε- 的样本点不在ε- 邻域内，默认值是0.5。一般需要通过在多组值里面选择一个合适的阈值，eps过大，则更多的点会落在核心对象的ε- 邻域，此时的类别数可能会减少，本来不应该是一类的样本也会被划分为一类。反之则类别数可能会增大，本来是一类的样本却被划分开。

**(2)min_samples：**DBSCAN算法参数，即样本点要成为核心对象所需要的ε- 邻域的样本数阈值，默认是5。一般需要通过在多组值里面选择一个合适的阈值。通常和eps一起调参。在eps一定的情况下，min_smaples过大，则核心对象会过少，此时簇内部分本来是一类的样本可能会被标为噪音点，类别数也会变多。反之 min_smaples过小的话，则会产生大量的核心对象，可能会导致类别数过少。

**(3)metric：**最近邻距离度量参数，可以使用的距离度量较多，一般来说DBSCAN使用默认的欧式距离（即 p=2 的闵可夫斯基距离）就可以满足需求。可以使用的距离度量参数有：欧式距离，曼哈顿距离，切比雪夫距离，闵可夫斯基距离，马氏距离等等。

**(4)algorithm**：最近邻搜索算法参数，算法一共有三种，第一种是蛮力实现，第二种是KD树实现，第三种是球树实现，对于这个参数，一共有4种可选输入，‘brute’对应第一种蛮力实现，‘kd_tree’对应第二种KD树实现，‘ball_tree’对应第三种的球树实现， ‘auto’则会在上面三种算法中做权衡，选择一个拟合最好的最优算法。需要注意的是，如果输入样本特征是稀疏的时候，无论我们选择哪种算法，最后scikit-learn都会去用蛮力实现‘brute’。一般情况使用默认的 ‘auto’就够了。 如果数据量很大或者特征也很多，用"auto"建树时间可能会很长，效率不高，建议选择KD树实现‘kd_tree’，此时如果发现‘kd_tree’速度比较慢或者已经知道样本分布不是很均匀时，可以尝试用‘ball_tree’。而如果输入样本是稀疏的，无论选择哪个算法最后实际运行的都是‘brute’。

**(5)leaf_size：** 最近邻搜索算法参数，为使用KD树或者球树时，停止建子树的叶子节点数量的阈值。这个值越小，则生成的KD树或者球树就越大，层数越深，建树时间越长，反之，则生成的KD树或者球树会小，层数较浅，建树时间较短。默认是30，因为这个值一般只影响算法的运行速度和使用内存大小，因此一般情况可以不管它。

**(6)p：**最近邻距离度量参数。只用于闵可夫斯基距离和带权值闵可夫斯基距离中 p 值的选择， p=1为曼哈顿距离，p=2为欧式距离，如果使用默认的欧式距离就不需要管这个参数。



### 5. 聚类算法评估方法

不像监督学习的分类问题和回归问题，无监督聚类没有样本输出，也就没有比较直接的聚类评估方法。但可以从簇内的稠密程度和簇间的离散程度来评估聚类的效果。常见的方法有轮廓稀疏Silhouette Coefficient和 Calinski Harabasz Index。

#### (1)Calinski-Harabasz方法

Calinski-Harabasz 分数值 s 的数学计算公式是：

![img](https://img2020.cnblogs.com/blog/1226410/202004/1226410-20200424102450180-1260317280.png)

其中 m 为训练集样本数，k为类别数。Bk为类别之间的协方差矩阵，Wk为类别内部数据的协方差矩阵。tr为矩阵的迹。也就是说，类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的 Calinski-Harabasz 分数越高，即Calinski-Harabasz分数值 s 越大则聚类效果越好。在scikit-learn 中， Calinski-Harabasz Index对应的方法就是 metrics.calinski_harabaz_score。

#### (2)轮廓系数Silhouette Coefficient

Silhouette 系数是对聚类结果有效性的解释。可以理解为描述聚类后各个类别的轮廓清晰度的指标。其包含有两种因素——内聚度和分离度。

![img](https://img2020.cnblogs.com/blog/1226410/202004/1226410-20200427112207241-618317456.png)

计算样本 i 到同簇其他样本的平均距离ai，ai越小，说明样本 i 越应该被聚类到该簇，将 ai 称为样本 i 的簇内不相似度。计算样本 i 到其他某簇 Cj 的所有样本的平均距离 bij，称为样本 i 与簇 Cj 的不相似度。定义为样本 i 的簇间不相似度：bi = min{bi1, bi2, ...bik}。

**判断**：si 接近1，则说明样本 i 聚类合理，si 接近 -1 ，则说明样本 i 更应该分类到另外的簇，若 si 近似为0，则说明样本 i 在两个簇的边界上。

**不适用的情况**：对于簇结构为凸的数据轮廓系数较高，对于簇结构非凸的轮廓系数较低。因此，轮廓系数不能在不同的算法之间比较优劣，如统一数据下，可能KMeans的结果就比DBSCAN要好。

## 二、实验目的

1. 掌握使用sklearn实现无监督聚类算法：dbscan与k-means
2. 掌握两种无监督聚类算法：dbscan与k-means的基本原理
3. 熟悉Python编程
4. 掌握使用轮廓系数法评估聚类算法效果
5. 掌握Python数据处理和数据可视化方法

## 三、实验内容

#### 无监督聚类算法：dbscan 与 k-means

**使用sklearn构建k-means与dbscan模型**

**数据集**：Data_for_Cluster.npz，X为特征，label_true为标签

**实验过程**：加载数据集，搭建k-means与dbscan模型，训练模型，分析结果，数据可视化，使用轮廓系数评估聚类算法效果。

## 四、实验步骤

(1)加载数据集，X 为特征，labels_true 为标签。

(2)搭建模型，k-means与dbscan

(3)训练模型，调节参数，得出分类结果

(4)结果分析与可视化

(5)使用轮廓系数法评估聚类算法的效果

## 五、实验结果与分析（含重要数据结果分析或核心代码流程分析）

(1)加载数据集，x为特征，labels_true为标签

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

cluser_data = np.load('Dataset/Data_for_Cluster.npz')
print(cluser_data.files)
X = np.array(cluser_data['X'])
labels = np.array(cluser_data['labels_true'])
# 可视化数据集
plt.scatter(X[:,0],X[:,1],marker='o')
plt.show()
```

![Lab3P1](/Users/zhaoxu/Desktop/Lab3P1.png)



### **k-means**

(2)搭建model

```python
clf = KMeans(n_clusters=3, random_state=9)
```

(3)训练模型，调参，得出分类结果

```python
y_pred = clf.fit_predict(X)
```

(4)数据分析和可视化

```python
plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='o')
plt.show()
```

![Lab3P2](/Users/zhaoxu/Desktop/Lab3P2.png)

(5)评估结果，轮廓系数法（Silhouette Cofficient），用来评估聚类算法的效果

```python
# 将原始的数据X和聚类结果y_pred
# 传入对应的函数计算出该结果下的轮廓系数
score1_1 = silhouette_score(X, y_pred)
print("使用轮廓系数评估K-means3聚类效果：{}".format(score1_1))
score1_2 = calinski_harabaz_score(X,y_pred)
print("使用Calinski-Harabasz评估K-means3聚类效果：{}".format(score1_2))
```

结果

```
使用轮廓系数评估K-means3聚类效果：0.7119124295445075
使用Calinski-Harabasz评估K-means3聚类效果：5281.366586365746
```

**调节参数，令k=2**

```python
clf = KMeans(n_clusters=2, random_state=9)
y_pred = clf.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='o')
plt.show()
score2_1 = silhouette_score(X, y_pred)
print("使用轮廓系数评估K-means2聚类效果：{}".format(score2_1))
score2_2 = calinski_harabaz_score(X,y_pred)
print("使用Calinski-Harabasz评估K-means2聚类效果：{}".format(score2_2))
```

![Lab3P3](/Users/zhaoxu/Desktop/Lab3P3.png)

```
使用轮廓系数评估K-means2聚类效果：0.648374545822685
使用Calinski-Harabasz评估K-means2聚类效果：2531.442477334627
```



### **dbscan**

(2)搭建model，训练模型，调参，得出分类结果

```python
# 调参eps（默认0.5），min_samples（默认5）
y_pred = DBSCAN(eps=0.4,min_samples=10).fit_predict(X)
```

(3)数据分析和可视化

```python
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
```

![Lab3s10](/Users/zhaoxu/Desktop/Lab3s10.png)

(4)评估结果，轮廓系数法（Silhouette Cofficient），用来评估聚类算法的效果

```python
score3_1 = silhouette_score(X, y_pred)
print("使用轮廓系数评估DBSCAN聚类效果：{}".format(score3_1))
score3_2 = calinski_harabaz_score(X,y_pred)
print("使用Calinski-Harabasz评估DBSCAN聚类效果：{}".format(score3_2))
```

结果

```
使用轮廓系数评估DBSCAN聚类效果：0.6544272847686536
使用Calinski-Harabasz评估DBSCAN聚类效果：3536.9664472523896
```



## 六、总结及心得体会

### 1. 参数调节

**(1)K-means调参**

**K=2时**

轮廓系数

```
使用轮廓系数评估K-means2聚类效果：0.648374545822685
使用Calinski-Harabasz评估K-means2聚类效果：2531.442477334627
```

可视化分类结果

![Lab3P3](/Users/zhaoxu/Desktop/Lab3P3.png)

**K=3时**

轮廓系数

```
使用轮廓系数评估K-means3聚类效果：0.7119124295445075
使用Calinski-Harabasz评估K-means3聚类效果：5281.366586365746
```

可视化分类结果

![Lab3P2](/Users/zhaoxu/Desktop/Lab3P2.png)



**(2)DBSCAN调参**

**eps=0.4,min_samples=10时**

轮廓系数

```
使用轮廓系数评估DBSCAN聚类效果：0.6544272847686536
使用Calinski-Harabasz评估DBSCAN聚类效果：3536.9664472523896
```

可视化分类结果

![Lab3s10](/Users/zhaoxu/Desktop/Lab3s10.png)



**eps=0.4,min_samples=5时**

轮廓系数

```
使用轮廓系数评估DBSCAN聚类效果：0.6392886792743393
使用Calinski-Harabasz评估DBSCAN聚类效果：3526.443277268119
```

可视化分类结果

![Lab3s5](/Users/zhaoxu/Desktop/Lab3s5.png)

**eps=0.4,min_samples=15时**

轮廓系数

```
使用轮廓系数评估DBSCAN聚类效果：0.6623933046127879
使用Calinski-Harabasz评估DBSCAN聚类效果：3496.9564629325823
```

可视化分类结果

![Lab3s15](/Users/zhaoxu/Desktop/Lab3s15.png)



**eps=0.5,min_samples=10时**

轮廓系数

```
使用轮廓系数评估DBSCAN聚类效果：0.5505239194094992
使用Calinski-Harabasz评估DBSCAN聚类效果：1269.4636949688759
```

可视化分类结果

![Lab3eps05](/Users/zhaoxu/Desktop/Lab3eps05.png)



### 2.K-means小结

**K-Means的主要优点**：

(1)原理比较简单，实现容易，收敛速度快

(2)聚类效果较优（依赖K的选择）

(3)算法的可解释度比较强

(4)主要需要调参的参数仅仅是簇数 k

**K-Means的主要缺点**：

(1)K值的选取不好把握

(2)对于不是凸的数据集比较难收敛

(3)如果各隐含类别的数据不平衡，比如各隐含类别的数据量严重失衡，或者各隐含类别的方差不同，则聚类效果不佳

(4)采用迭代方法，得到的结果只能保证局部最优，不一定是全局最优（与K的个数及初值选取有关）

(5)对噪音和异常点比较的敏感（中心点易偏移）

### 3.DBSCAN小结

**DBSCAN优点**：

(1) 可以对任意形状的稠密数据集进行聚类，相对的，K-Means之类的算法一般只适用于凸数据集。

(2) 可以在聚类的同时发现异常点，对数据集中的异常点不敏感。

(3)聚类结果没有偏倚，相对的K，K-Means之类的聚类算法那初始值对聚类结果有很大的影响。

**DBSCAN缺点**：

(1)如果样本集的密度不均匀，聚类间距离相差很大时，聚类质量较差，这时用DBSCAN聚类一般不适合。

(2)如果样本集较大时，聚类收敛时间较长，此时可以对搜索最近邻时建立的KD树或者球树进行规模限制来改进。

(3)调参相对于传统的K-Means之类的聚类算法稍复杂，主要需要对距离阈值 ε ，邻域样本数阈值MinPts联合调参，不同的参数组合对最后的聚类效果有较大影响。

### 4.K-means与DBSCAN的对比

(1)k-means需要指定聚类簇数k，并且且初始聚类中心对聚类影响很大。k-means把任何点都归到了某一个类，对异常点比较敏感。DBSCAN能剔除噪声，需要指定邻域距离阈值eps和样本个数阈值MinPts，可以自动确定簇个数。

(2)K均值和DBSCAN都是将每个对象指派到单个簇的划分聚类算法，但是K均值一般聚类所有对象，而DBSCAN丢弃被它识别为噪声的对象。

(3)K均值很难处理非球形的簇和不同大小的簇。DBSCAN可以处理不同大小或形状的簇，并且不太受噪声和离群点的影响。当簇具有很不相同的密度时，两种算法的性能都很差。

(4)K均值只能用于具有明确定义的质心（比如均值或中位数）的数据。DBSCAN要求密度定义（基于传统的欧几里得密度概念）对于数据是有意义的。

(5)K均值算法的时间复杂度是O(m)，而DBSCAN的时间复杂度是O(m^2)。

(6)DBSCAN多次运行产生相同的结果，而K均值通常使用随机初始化质心，不会产生相同的结果。

(7)K均值DBSCAN和都寻找使用所有属性的簇，即它们都不寻找可能只涉及某个属性子集的簇。

(8)K均值可以发现不是明显分离的簇，即便簇有重叠也可以发现，但是DBSCAN会合并有重叠的簇。

(9)K均值可以用于稀疏的高维数据，如文档数据。DBSCAN通常在这类数据上的性能很差，因为对于高维数据，传统的欧几里得密度定义不能很好处理它们。

一般来说，如果数据集是稠密的，并且数据集不是凸的，那么用DBSCAN会比K-Means聚类效果好很多。如果数据集不是稠密的，则不推荐使用DBSCAN来聚类。

## 七、对本实验过程及方法、手段的改进建议

本次实验让我对无监督学习有了初步认识，熟悉了两种无监督学习方法：K-means与DBSCAN，学会了使用sklearn实现SVM。在做本次实验时，我用了很多时间调节模型参数，并通过轮廓系数与可视化结果比较差别，选择更优的参数，进一步地，比较两种算法的差别。

以后的实验可引导同学们调节参数，深入探究不同参数对实验结果的影响，引导同学们了解更多无监督学习方法，学会在不同场景选用合适的模型，引导同学们深入了解无监督学习的评估标准，尝试用不同方法提高模型准确率。