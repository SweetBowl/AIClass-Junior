## 一、实验原理

### 1. 卷积神经网络概念

卷积神经网络（Convolutional Neural Network，CNN）又叫卷积网络（Convolutional Network），是一种专门用来处理具有类似网格结构的数据的神经网络。卷积神经网络一词中的卷积是一种特殊的线性运算。卷积网络是指那些至少在网络的一层中使用卷积运算来代替一般的矩阵乘法的神经网络。

卷积神经网络的出现，极大的缓解了全连接神经网络中数据的波形被忽视问题。全连接神经网络在进行数据输入的时候，需要将一个二维或者三维的数据展平为一维的数据。而它在计算机中图形是一个三维的数据，因为需要存储一些类似 RGB 各个通道之间关联性的空间信息，所以三维形状中可能隐藏有值得提取的本质模式。而全连接展平会导致形状的忽视。因此需要利用卷积神经网络来保持形状的不变。

![img](https://pic3.zhimg.com/80/v2-ff46cd1067d97a86f5c2617e58c95442_1440w.jpg)

图1. 全连接神经网络与卷积神经网络的对比

### 2. 卷积神经网络的组成

卷积神经网络主要由这几类层构成：输入层、卷积层，ReLU层、池化（Pooling）层和全连接层。通过将这些层叠加起来，就可以构建一个完整的卷积神经网络。在实际应用中往往将卷积层与ReLU层共同称之为卷积层，所以卷积层经过卷积操作也是要经过激活函数的。具体说来，卷积层和全连接层（CONV/FC）对输入执行变换操作的时候，不仅会用到激活函数，还会用到很多参数，即神经元的权值w和偏差b；而ReLU层和池化层则是进行一个固定不变的函数操作。卷积层和全连接层中的参数会随着梯度下降被训练，这样卷积神经网络计算出的分类评分就能和训练集中的每个图像的标签吻合了。

#### 2.1卷积层

卷积神经网路中每层卷积层由若干卷积单元组成，每个卷积单元的参数都是通过反向传播算法优化得到的。卷积运算的目的是提取输入的不同特征，第一层卷积层可能只能提取一些低级的特征如边缘、线条和角等层级，更多层的网络能从低级特征中迭代提取更复杂的特征。

**2.1.1卷积层作用**

1. 滤波器的作用或者说是卷积的作用。卷积层的参数是有一些可学习的滤波器集合构成的。每个滤波器在空间上（宽度和高度）都比较小，但是深度和输入数据一致、。直观地来说，网络会让滤波器学习到当它看到某些类型的视觉特征时就激活，具体的视觉特征可能是某些方位上的边界，或者在第一层上某些颜色的斑点，甚至可以是网络更高层上的蜂巢状或者车轮状图案。

2. 可以被看做是神经元的一个输出。神经元只观察输入数据中的一小部分，并且和空间上左右两边的所有神经元共享参数。

3. 降低参数的数量。这个由于卷积具有“权值共享”这样的特性，可以降低参数数量，达到降低计算开销，防止由于参数过多而造成过拟合。

**2.1.2感受野**

在处理图像这样的高维度输入时，让每个神经元都与前一层中的所有神经元进行全连接是不现实的。相反，我们让每个神经元只与输入数据的一个局部区域连接。该连接的空间大小叫做神经元的感受野（receptive field），它的尺寸是一个超参数（滤波器的空间尺寸）。在深度方向上，这个连接的大小总是和输入量的深度相等。我们对待空间维度（宽和高）与深度维度是不同的：连接在空间（宽高）上是局部的，但是在深度上总是和输入数据的深度一致。

**2.1.3权值共享**

在卷积层中权值共享是用来控制参数的数量。假如在一个卷积核中，每一个感受野采用的都是不同的权重值（卷积核的值不同），那么这样的网络中参数数量将是十分巨大的。

在反向传播的时候，都要计算每个神经元对它的权重的梯度，但是需要把同一个深度切片上的所有神经元对权重的梯度累加，这样就得到了对共享权重的梯度。这样，每个切片只更新一个权重集。

有时候参数共享假设可能没有意义，特别是当卷积神经网络的输入图像是一些明确的中心结构时候。这时候我们就应该期望在图片的不同位置学习到完全不同的特征，而一个卷积核滑动地与图像做卷积都是在学习相同的特征。

#### 2.2 池化层

通常在连续的卷积层之间会周期性地插入一个池化层。它的作用是逐渐降低数据体的空间尺寸，这样的话就能减少网络中参数的数量，使得计算资源耗费变少，也能有效控制过拟合。池化层使用 MAX 操作，对输入数据体的每一个深度切片独立进行操作，改变它的空间尺寸。最常见的形式是池化层使用尺寸2x2的滤波器，以步长为2来对每个深度切片进行降采样，将其中75%的激活信息都丢掉。每个MAX操作是从4个数字中取最大值（也就是在深度切片中某个2x2的区域），深度保持不变。

池化层进行的运算一般有以下几种： 
\* 最大池化（Max Pooling）。取4个点的最大值。这是最常用的池化方法。 
\* 均值池化（Mean Pooling）。取4个点的均值。 
\* 高斯池化。借鉴高斯模糊的方法。不常用。 
\* 可训练池化。训练函数 ff ，接受4个点为输入，出入1个点。不常用。

#### 2.3归一化层

在卷积神经网络的结构中，提出了很多不同类型的归一化层，有时候是为了实现在生物大脑中观测到的抑制机制。但是这些层渐渐都不再流行，因为实践证明它们的效果即使存在，也是极其有限的。

#### 2.4全连接层

全连接层和卷积层可以相互转换： 
对于任意一个卷积层，要把它变成全连接层只需要把权重变成一个巨大的矩阵，其中大部分都是0 除了一些特定区块（因为局部感知），而且好多区块的权值还相同（由于权重共享）。 
相反地，对于任何一个全连接层也可以变为卷积层。比如，一个K＝4096 的全连接层，输入层大小为 7∗7∗512，它可以等效为一个 F=7, P=0, S=1, K=4096 的卷积层。

### 3. 卷积神经网络的结构

卷积神经网络通常是由三种层构成：卷积层，汇聚层（除非特别说明，一般就是最大值汇聚）和全连接层（简称FC）。ReLU激活函数也应该算是是一层，它逐元素地进行激活函数操作，常常将它与卷积层看作是同一层。

卷积神经网络最常见的形式就是将一些卷积层和ReLU层放在一起，其后紧跟汇聚层，然后重复如此直到图像在空间上被缩小到一个足够小的尺寸，在某个地方过渡成成全连接层也较为常见。最后的全连接层得到输出，比如分类评分等。换句话说，最常见的卷积神经网络结构如下：

INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC

其中*指的是重复次数，POOL?指的是一个可选的汇聚层。其中N >=0,通常N<=3,M>=0,K>=0,通常K<3。例如，下面是一些常见的网络结构规律：

1.INPUT -> FC ，实现一个线性分类器，此处N = M = K = 0。

2.INPUT -> CONV -> RELU -> FC，单层的卷积神经网络

3.INPUT -> [CONV -> RELU -> POOL]\*2 -> FC -> RELU -> FC，此处在每个汇聚层之间有一个卷积层，这种网络就是简单的多层的卷积神经网络。

4.INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]\*3 -> [FC -> RELU]\*2 -> FC ，此处每个汇聚层前有两个卷积层，这个思路适用于更大更深的网络（比如说这个思路就和VGG比较像），因为在执行具有破坏性的汇聚操作前，多重的卷积层可以从输入数据中学习到更多的复杂特征。

### 4. 过拟合与欠拟合

#### **欠拟合**

根本原因是特征维度过少，模型过于简单，导致拟合的函数无法满足训练集，误差较大
**解决方法**：增加特征维度，增加训练数据； 

#### **过拟合**

根本原因是特征维度过多，模型假设过于复杂，参数过多，训练数据过少，噪声过多，导致拟合的函数完美的预测训练集，但对新数据的测试集预测结果差。 过度的拟合了训练数据，而没有考虑到泛化能力。

**解决方法**：

(1).增加数据量：从数据源头获取更多数据；数据增强（Data Augmentation）。

(2).使用合适的模型：减少网络的层数、神经元个数等均可以限制网络的拟合能力。

(3).Dropout： 在训练的时候, 随机忽略掉一些神经元和神经联结 , 使这个神经网络变得”不完整”. 用一个不完整的神经网络训练一次。到第二次再随机忽略另一些, 变成另一个不完整的神经网络。有了这些随机 drop 掉的规则,可以想象其实每次训练的时候, 我们都让每一次预测结果都不会依赖于其中某部分特定的神经元。

(4).正则化：如L1和L2正则。都是针对模型中参数过大的问题引入惩罚项，依据是奥克姆剃刀原理。在深度学习中，L1会趋向于产生少量的特征，而其他的特征都是0增加网络稀疏性；而L2会选择更多的特征，这些特征都会接近于0，防止过拟合。神经网络需要每一层的神经元尽可能的提取出有意义的特征，而这些特征不能是无源之水，因此L2正则用的多一些。

(5).限制训练时间；通过评估测试。
(6).增加噪声 Noise： 输入时+权重上（高斯初始化）。

(7).数据清洗(data ckeaning/Pruning)：将错误的label 纠正或者删除错误的数据。

(8).结合多种模型： Bagging用不同的模型拟合不同部分的训练集；Boosting只使用简单的神经网络。

### 5. Adam优化器

#### Adam更新规则

计算t时间步的梯度：

![img](https://pic2.zhimg.com/80/v2-38e12384dc0d006b2f98fc9264aaca6d_1440w.jpg)

首先，计算梯度的指数移动平均数，m0 初始化为0。

类似于Momentum算法，综合考虑之前时间步的梯度动量。

β1 系数为指数衰减率，控制权重分配（动量与当前梯度），通常取接近于1的值。

默认为0.9

![img](https://pic3.zhimg.com/80/v2-7b6f1dd247e3b0af66dbbf456f4b3102_1440w.jpg)

其次，**计算梯度平方的指数移动平均数**，v0初始化为0。

β2 系数为指数衰减率，控制之前的梯度平方的影响情况。

类似于RMSProp算法，对梯度平方进行加权均值。

默认为0.999

![img](https://pic2.zhimg.com/80/v2-c867e02204cfdabc1ad73d65d166667d_1440w.jpg)

第三，由于m0初始化为0，会导致mt偏向于0，尤其在训练初期阶段。

所以，此处需要对梯度均值mt进行偏差纠正，降低偏差对训练初期的影响。

![img](https://pic4.zhimg.com/80/v2-b37903eae41c16ebf30f1ed28b61ddf3_1440w.jpg)

第四，与m0 类似，因为v0初始化为0导致训练初始阶段vt 偏向0，对其进行纠正。

![img](https://pic4.zhimg.com/80/v2-adfbb3c7f79e050cf21121a9e9da50a7_1440w.jpg)

第五，更新参数，初始的学习率α乘以梯度均值 与梯度方差 的平方根之比。

其中默认学习率α=0.001

ε=10^-8，避免除数变为0。

由表达式可以看出，对更新的步长计算，能够从梯度均值及梯度平方两个角度进行自适应地调节，而不是直接由当前梯度决定。

![img](https://pic1.zhimg.com/80/v2-4a471293949219ae58fdf20f6e6ea64c_1440w.jpg)



### 6. Tensorflow

TensorFlow是一个基于数据流编程（dataflow programming）的符号数学系统，被广泛应用于各类机器学习（machine learning）算法的编程实现，其前身是谷歌的神经网络算法库DistBelief 。Tensorflow拥有多层级结构，可部署于各类服务器、PC终端和网页并支持GPU和TPU高性能数值计算，被广泛应用于谷歌内部的产品开发和各领域的科学研

Keras是一个由Python编写的开源人工神经网络库，可以作为Tensorflow、Microsoft-CNTK和Theano的高阶应用程序接口，进行深度学习模型的设计、调试、评估、应用和可视化 。

Keras在代码结构上由面向对象方法编写，完全模块化并具有可扩展性，其运行机制和说明文档有将用户体验和使用难度纳入考虑，并试图简化复杂算法的实现难度 。Keras支持现代人工智能领域的主流算法，包括前馈结构和递归结构的神经网络，也可以通过封装参与构建统计学习模型  。在硬件和开发环境方面，Keras支持多操作系统下的多GPU并行计算，可以根据后台设置转化为Tensorflow、Microsoft-CNTK等系统下的组件

## 二、实验目的

1. 掌握使用tensorflow.keras实现简单的CNN应用
2. 掌握CNN基本原理及相关概念
3. 熟悉Python编程
4. 掌握使用Python数据处理和数据可视化方法

## 三、实验内容

#### 手写数字识别——CNN 的应用

**使用tensorflow.keras构建CNN**

**数据集**：手写数字数据集MNIST

**实验过程**：加载数据集，搭建CNN模型，训练模型，调节参数，测试模型，评估模型

## 四、实验步骤

#### 手写数字识别——CNN 的应用

(1)加载数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
(2)搭建 CNN 模型
优化器（optimizer）：adam
评估指标（metrics）：accuacy
(3)训练模型，调节参数，测试模型

## 五、实验结果与分析（含重要数据结果分析或核心代码流程分析）

1.加载数据集

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
plt.imshow(x_train[0],cmap='gray')
plt.show()
print(y_train[0])

# 对数据进行归一化处理，像素值被限定在[0,1]
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
plt.imshow(x_train[0],cmap='gray')
plt.show()

```

结果

```
5
```

![image-20201229111743232](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229111743232.png)

![image-20201229111747788](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229111747788.png)



2.搭建CNN模型

```python
model = tf.keras.models.Sequential()
# 展平图像矩阵
model.add(tf.keras.layers.Flatten())
# 添加全连接层，激活函数选用reLU
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
# 再添加一个相同的层
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
# 添加输出层，输出10个结点，代表10种不同的数字。使用softMax函数作为激活函数。
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
```

3.训练模型，测试模型，可视化

```python
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=5)

print(history.history.keys())
# 可视化
plt.plot(history.history['acc'])
plt.title('model accuracy on train')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper left')
plt.show()

train_loss, train_acc = model.evaluate(x_train,y_train)
print("训练集loss: %.4f" % train_loss)
print("训练集准确率: %.4f" % train_acc)

# 测试模型
val_loss, val_acc = model.evaluate(x_test,y_test)
print("测试集loss: %.4f" % val_loss)
print("测试集准确率: %.4f" % val_acc)

# 识别训练集
predictions = model.predict(x_test)
print("预测结果: {}".format(np.argmax(predictions[0])))

plt.imshow(x_test[0],cmap='gray')
plt.show()
```

结果

![Keras_5V](/Users/zhaoxu/Desktop/Keras_5V.png)

```
训练集loss: 0.0308
训练集准确率: 0.9899

测试集loss: 0.0877
测试集准确率: 0.9749
预测结果: 7
```

![image-20201229112230588](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229112230588.png)

![image-20201229112236650](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229112236650.png)

![image-20201229112240984](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229112240984.png)



## 六、总结及心得体会

本次实验使用的CNN模型选用adam优化器，评估指标选用accuracy。

### 1.调节参数

**epochs=5**

训练集loss: 0.0308
训练集准确率: 0.9899

测试集loss: 0.0877
测试集准确率: 0.9749

预测结果: 7

![image-20201228211310432](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201228211310432.png)



![image-20201228211320165](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201228211320165.png)



**epochs=2**

训练集loss: 0.0684
训练集准确率: 0.9792

测试集loss: 0.1058
测试集准确率: 0.9682

预测结果: 7

![image-20201229112655629](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229112655629.png)

![image-20201229112700421](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229112700421.png)



**epochs=3**

训练集loss: 0.0498
训练集准确率: 0.9844

测试集loss: 0.0932
测试集准确率: 0.9712

预测结果: 7

![image-20201229112427307](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229112427307.png)

![image-20201229112431926](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229112431926.png)



**epochs=10**

训练集loss: 0.0098
训练集准确率: 0.9968

测试集loss: 0.1084
测试集准确率: 0.9747
预测结果: 7

![image-20201229113107978](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229113107978.png)

![image-20201229113112041](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229113112041.png)



**epochs=15**

训练集loss: 0.0088
训练集准确率: 0.9970

测试集loss: 0.1344
测试集准确率: 0.9737
预测结果: 7

![image-20201229113506509](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229113506509.png)

![image-20201229113510979](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229113510979.png)



**epochs=25**

训练集loss: 0.0033
训练集准确率: 0.9989

测试集loss: 0.1486
测试集准确率: 0.9772
预测结果: 7

![image-20201229114220717](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229114220717.png)

![image-20201229114224933](/Users/zhaoxu/Library/Application Support/typora-user-images/image-20201229114224933.png)

可以发现，在epochs>5之后，继续增大epochs，模型在train上的accuracy继续提升，但提升不大，但loss在test上在增加，可以认为出现了过拟合现象。

### 2. 针对Adam优化器的改进

**1、解耦权重衰减**

![img](https://pic1.zhimg.com/80/v2-59c1cf76ea5b4c5aa4e173c5ac4f3fa8_1440w.jpg)

在每次更新梯度时，同时对其进行衰减（衰减系数w略小于1），避免产生过大的参数。

在Adam优化过程中，增加参数权重衰减项。解耦学习率和权重衰减两个超参数，能单独调试优化两个参数。

![img](https://pic1.zhimg.com/80/v2-c9c703d1f121d1041bb1973207fc4484_1440w.jpg)



2、修正指数移动均值

最近的几篇论文显示较低的β_2（如0.99或0.9）能够获得比默认值0.999更佳的结果，暗示出指数移动均值本身可能也包含了缺陷。例如在训练过程中，某个mini-batch出现比较大信息量的梯度信息，但由于这类mini-batch出现频次很少，而指数移动均值会减弱他们的作用（因为当前梯度权重 及当前梯度的平方的权重 ，权重都比较小），导致在这种场景下收敛比较差。

![img](https://pic2.zhimg.com/80/v2-bcbd44fa919f83db15e93822ee0d18e1_1440w.jpg)

AMSGrad 使用最大的 来更新梯度，而不像Adam算法中采用历史 的指数移动均值来实现。

### 3. CNN小结

虽然在实验的测量中，CNN获得了巨大的成功，但是，仍然还有很多工作值得进一步研究。首先，鉴于最近的CNN变得越来越深，它们也需要大规模的数据库和巨大的计算能力，来展开训练。人为搜集标签数据库要求大量的人力劳动。所以，大家都渴望能开发出无监督式的CNN学习方式。　　

同时，为了加速训练进程，虽然已经有一些异步的SGD算法，证明了使用CPU和GPU集群可以在这方面获得成功，但是，开放高效可扩展的训练算法依然是有价值的。在训练的时间中，这些深度模型都是对内存有高的要求，并且消耗时间的，这使得它们无法在手机平台上部署。如何在不减少准确度的情况下，降低复杂性并获得快速执行的模型，这是重要的研究方向。 　　

其次，CNN运用于新任务的一个主要障碍是：如何选择合适的超参数？比如学习率、卷积过滤的核大小、层数等等，这需要大量的技术和经验。这些超参数存在内部依赖，这会让调整变得很复杂。最近的研究显示，在学习式CNN架构的选择技巧上，存在巨大的提升空间。 　

最后，关于CNN依然缺乏统一的理论。目前的CNN模型运作模式依然是黑箱。我甚至都不知道它是如何工作的，工作原理是什么。当下，值得把更多的精力投入到研究CNN的基本规则上去。同时，正如早期的CNN发展是受到了生物视觉感知机制的启发，深度CNN和计算机神经科学二者需要进一步的深入研究。也有一些开放性的问题，比如，生物学上大脑中的学习方式如何帮助人们设计更加高效的深度模型？带权重分享的回归计算方式是否可以计算人类的视觉皮质等等。

## 七、对本实验过程及方法、手段的改进建议

本实验让我对CNN原理有了基本认识，了解了相关概念，并尝试使用keras搭建CNN，应用在手写数字识别上。由于优化器固定，所以本次实验我主要调节epoch参数，观察模型的过拟合和欠拟合现象，这也加深了我对CNN的理解。

以后的实验可引导同学们选用不同的优化器，调节参数，尝试比较不同优化器的优缺点，进一步提高模型的准确率。