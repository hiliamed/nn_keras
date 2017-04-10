# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 12:26:28 2017

@author: licencep
"""

import myFunctions as mf
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

import keras.models

perceptron = mf.loadModel("perceptron")
perceptron.summary()
perceptron.pop()
perceptron.pop()
perceptron.summary()

nn_conv = mf.loadModel("nn_conv")
nn_conv.summary()
nn_conv.pop()
nn_conv.summary()


ximg=X_test.reshape((-1,28,28,1))
#r = perceptron.predict(X_test[:1000]) 
r = nn_conv.predict(ximg[:1000]) 


#nn_conv = mf.loadModel("nn_conv")

from sklearn.manifold import  TSNE


tsne = TSNE(n_components=2, perplexity=30.0, init = 'pca', verbose=1 )
points = tsne.fit_transform(r)
labels = y_test[:1000]

#from scipy.spatial import ConvexHull

import myFunctions as mf

convex_hulls = mf.convexHulls(points,labels)
ellipses = mf.best_ellipses(points, labels)
nh = mf.neighboring_hit(points, labels)
viz = mf.Visualization(points, labels, convex_hulls, ellipses, "Perceptron", nh)
