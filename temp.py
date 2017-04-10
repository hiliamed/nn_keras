# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

print("Test")
from keras.datasets import mnist 
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
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

# exerice 0 : affichage des images
plt.figure(figsize = (7.195, 3.841), dpi=100) 

for i in range(200):
    plt.subplot(10,20,i+1) 
    plt.imshow(X_train[i,:].reshape([28,28]), cmap='gray')
    plt.axis('off') 
plt.show()

# espace dans lequel se trouvent les images : 28*28
# la taille est 28*28.

# Exercice 1 : 
# le nombre de paramètre du modèle : 784 * 10
# pour chaque xi nous avons une couche 
# de taille 10 , est donc paramètres. 
from keras.models import Sequential
model = Sequential();

# ajouter une couche de taille 10
from keras.layers import Dense, Activation

model.add(Dense(10, input_dim = 784, name='fc1'))
model.add(Activation('softmax'))

model.summary()

# la fonction de cout est convex, on peut donc être sur d'aller vers un
# minimum global après des pas de gradient. 

from keras.optimizers import SGD
learning_rate =  0.5
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

from keras.utils import np_utils
batch_size = 300
nb_epoch = 10
# convert class vectors to binary class matrices 
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10) 
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)

scores = model.evaluate(X_test, Y_test, verbose=0) 
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Exercice 2 : Perceptron multi-couche

