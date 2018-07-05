# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 10:54:38 2018

@author: FVA3
"""

import matplotlib.pyplot as plt
import numpy as np
from relative_neighborhood_filter import *
import json
from sklearn.datasets import make_classification
from collections import Counter

def create_dataset(n_samples=1000, weights=(0.1, 0.9), n_classes=2,
                   class_sep=0.5, n_clusters=1):
    return make_classification(n_samples=n_samples, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters,
                               weights=list(weights),
                               class_sep=class_sep, random_state=0)
    
X, y = create_dataset(n_samples=1000, n_classes = 4, weights=(0.2, 0.25, 0.25,0.3), n_clusters = 1)

fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(X[:,0],X[:,1],c = [0 if np.unique(y)[0] == i else 1 for i in list(y)])
plt.show()


#print(np.unique(y))
#where = np.where( '/ics-traceparts-classification/mechanical-systems-and-components-of-general-use/linear-motion/linear-motion-systems/' == y)
#where = np.append(where,np.where('/ics-traceparts-classification/mechanical-systems-and-components-of-general-use/linear-motion/linear-actuators/' == y))
#X = X[where]
#y = y[where]
cur_iter = 0
nb_outliers = 100
while cur_iter < 1 and nb_outliers > 0:
    rng = RelativeNeighborhoodGraph()
    rng.fit(X,y,150, y)
    #rng.cut_edges_and_cluster()
    
#    for cluster in rng.clusters:
#        if len(cluster) != len(Counter(cluster)):
#            print(cluster)
#            print(Counter(cluster))
#            
#    
#    newX = []
#    newY = []
#    
#    for c in range(len(rng.clusters)):
#        for obj in rng.clusters[c]:
#            newX.append(X[obj])
#            newY.append(c)
#    
#    newX = np.array(newX)
#    newY = np.array(newY)
    
    m = rng.mislabelled_objects(risk = 0.1)
    cur_iter +=1
    nb_outliers = m.sum()/m.shape[0]
    X = X[~m]
    y = y[~m]
    print('% of outliers: ', nb_outliers)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(X[:,0],X[:,1],c = [0 if np.unique(y)[0] == i else 1 for i in list(y)])
    plt.show()

    
        


#
#fig, ax = plt.subplots(figsize=(15,10))
#ax.scatter(newX[:,0],newX[:,1],c = list(newY))
#plt.show()


#fig, ax = plt.subplots(figsize=(15,10))
#ax.scatter(X[:,0],X[:,1],c = [0 if np.unique(y)[0] in i else 1 for i in y])
#plt.show()


fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(X[:,0],X[:,1],c = [0 if np.unique(y)[0] == i else 1 for i in list(y)])
plt.show()



do_save = False
if do_save:
    path_out = 'e:/Datasets/After cleaning/trainList_416_ClusterCentroid_400max_vote_hard_labels_2'
    y_l = [unique[y[i]] for i in range(len(y))]
    np.save(path_out + '_signature.npy',X[~m])
    np.save(path_out + '_labels.npy',np.array(y_l)[~m])