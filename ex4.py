# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:34:01 2017

@author: licencep
"""
from sklearn.manifold import  TSNE

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

tsne = TSNE(n_components=2, perplexity=30.0, init = 'pca', verbose=1 )
points = tsne.fit_transform(X_test[0:2000,:])
labels = y_test[:2000]

#from scipy.spatial import ConvexHull

import myFunctions as mf

convex_hulls = mf.convexHulls(points,labels)
ellipses = mf.best_ellipses(points, labels)
nh = mf.neighboring_hit(points, labels)
viz = mf.Visualization(points, labels, convex_hulls, ellipses, "NN Conv", nh)
