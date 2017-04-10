# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:40:35 2017

@author: licencep
"""

from keras.datasets import mnist 
import matplotlib as mpl
mpl.use('TKAgg')
# the data, shuffled and split between train and test sets

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784) 
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')
X_train /= 255 
X_test /= 255
print(X_train.shape[0], 'train samples') 
print(X_test.shape[0], 'test samples')

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) 
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D

conv2d = Conv2D(32, kernel_size=(5,5), activation='sigmoid', input_shape=(28,28,1), padding='same')
print(type(conv2d))

pool = MaxPooling2D(pool_size=(2,2))
pool2 = MaxPooling2D(pool_size=(2,2))
print(type(pool))


model_3 = Sequential();

model_3.add(Conv2D(32, kernel_size=(5,5), activation='sigmoid', input_shape=(28,28,1), padding='same'))
model_3.add(pool)
model_3.add(Conv2D(64, kernel_size=(5,5), activation='sigmoid', input_shape=(14,14,1), padding='same'))
model_3.add(pool2)
model_3.add(Flatten())
model_3.add(Dense(100, activation='sigmoid'))
model_3.add(Dense(10, activation='softmax'))


model_3.summary()


from keras.optimizers import SGD
learning_rate =  0.5
sgd = SGD(learning_rate)
model_3.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

from keras.utils import np_utils
batch_size = 300
nb_epoch = 6
# convert class vectors to binary class matrices 
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10) 
model_3.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)

scores = model_3.evaluate(X_test, Y_test, verbose=0) 
print("%s: %.2f%%" % (model_3.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model_3.metrics_names[1], scores[1]*100))

import myFunctions as mf

mf.saveModel(model_3, "nn_conv")
