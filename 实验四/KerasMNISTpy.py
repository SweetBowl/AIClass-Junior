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

#%%
model = tf.keras.models.Sequential()
# 展平图像矩阵
model.add(tf.keras.layers.Flatten())
# 添加全连接层，激活函数选用reLU
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
# 再添加一个相同的层
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
# 添加输出层，输出10个结点，代表10种不同的数字。使用softMax函数作为激活函数。
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=5)

#%%
print(history.history.keys())
# 可视化
plt.plot(history.history['acc'])
# xtick = range(1,6,1)
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.xticks(x,('1','2','3','4','5'))
plt.legend(['train'],loc='upper left')
plt.show()

#%%
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper left')
plt.show()

#%%
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ',score[0])

#%%
# 测试模型
val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss)
print(val_acc)

#%%
# 识别训练集
predictions = model.predict(x_test)
print(predictions)
print(np.argmax(predictions[0]))

#%%
plt.imshow(x_test[0],cmap='gray')
plt.show()
