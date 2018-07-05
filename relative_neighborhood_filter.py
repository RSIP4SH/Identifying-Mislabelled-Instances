# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 16:57:41 2018

@author: FVA3
"""
import numpy as np
from tqdm import tqdm
from math import sqrt
from collections import Counter
import scipy.stats as st

class Edge:
    def __init__(self,node1,node2,weight):
        node1.add_edge(self,weight)
        node2.add_edge(self,weight)
        self.node1 = node1
        self.node2 = node2
        self.weight = weight

class Node:
    def __init__(self, label,features = None, cgr = None):
        self.features = features
        self.label = label
        self.cgr = cgr
        self.edges = []
        self.weights = []
    def add_edge(self,node,weight):
        self.edges.append(node)
        self.weights.append(weight)


class RelativeNeighborhoodGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.features = None
        self.labels = None
        self.dist_matrix = None
        self.adjacency_matrix = None
        pass
    
    def fit(self,features,labels,neighbors = 150, cgr = None):
        self.features = features
        self.labels = labels
        self.cgr = cgr
        self.counter =  Counter(labels)
        self.dist_matrix = self.pairwise_squared_dist_matrix(features)        
        self.adjacency_matrix = np.zeros(self.dist_matrix.shape)
        self.class_matrix = self.same_class_matrix()
        self.create_graph(neighbors)
        self.weight_matrix = self.pairwise_weight_matrix()    
        
        
    def create_graph(self, neighbors = 150):
        assert(np.any(self.dist_matrix))
        assert(np.any(self.features))
        for i in tqdm(range(self.dist_matrix.shape[0])):
            row = self.dist_matrix[i,:]
            nearests = np.argsort(row)[:neighbors]
            for j in nearests:
                dist = self.dist_matrix[i,j]
                has_edge = True
                for k in nearests:
                    if k != i and k != j:
                        dist_1 = self.dist_matrix[i,k]
                        dist_2 = self.dist_matrix[j,k]
                        if max(dist_1,dist_2) < dist:
                            has_edge = False
                if has_edge:
                    self.adjacency_matrix[i,j] = 1
                    self.adjacency_matrix[j,i] = 1
                    #self.edges.append(Edge(self.nodes[i],self.nodes[j],1/(1 + sqrt(self.dist_matrix[i,j]))))
        pass
    
    def cut_edges_and_cluster(self):
        dist_matrix = self.dist_matrix*self.adjacency_matrix
        #we cut the connections between objects of different classes
        self.cut_dist_matrix = np.multiply(dist_matrix,self.class_matrix)
        self.find_clusters()
    
    def find_clusters(self):
        self.clusters = []
        self.clusters_labels = []
        self.already_clustered = []
        for i in range(self.cut_dist_matrix.shape[0]):
            if i not in self.already_clustered:
                self.cur_cluster = [] # list(connections)
                self.search_depth(i)
                self.clusters.append(self.cur_cluster)
        for i in range(len(self.clusters)):
            self.clusters_labels.append(self.labels[self.clusters[i][0]])
        self.clusters = np.array(self.clusters)
        self.clusters_labels = np.array(self.clusters_labels)
                
    def search_depth(self, i):
        if i not in self.already_clustered:
            self.already_clustered.append(i)
            self.cur_cluster.append(i)
        row = np.where(self.cut_dist_matrix[i])[0]
        for obj in row:
            if obj not in self.already_clustered:
                self.already_clustered.append(obj)
                self.cur_cluster.append(obj)
                self.search_depth(obj)    
    
    def pairwise_squared_dist_matrix(self,features):
        dot_product = np.matmul(features,np.transpose(features))
        squared_norm = np.diag(dot_product)
        distances = np.array([squared_norm,]*dot_product.shape[0]) - 2*np.diag(squared_norm) + np.array([squared_norm,]*dot_product.shape[0]).transpose() - 2*dot_product
        distances[np.where(distances < 0)] = 0
        return distances
    
    def pairwise_weight_matrix(self):
        weight_matrix = self.dist_matrix*self.adjacency_matrix
        for i in range(weight_matrix.shape[0]):
            for j in range(weight_matrix.shape[1]):
                weight_matrix[i,j] = 1/(1 + sqrt(weight_matrix[i,j])) if weight_matrix[i,j] > 0 else 0
        return weight_matrix
        
    def calculate_weight_of_cut_edges_for_each_i(self):
        '''
        Ji = sum of weight[i,j]*Ii in j, where Ii = 1 if j isn't from the same class
        of i
        In other words, Ji is the sum of weights of cut edges
        '''
        not_class_matrix = (self.class_matrix == False)
        self.cut_weights_vector = np.sum(not_class_matrix*self.weight_matrix,axis = 1)
        self.cut_weights_expectation = np.zeros(self.cut_weights_vector.shape)
        self.cut_weights_variance = np.zeros(self.cut_weights_vector.shape)
        self.normalized_cut_weights_vector = self.cut_weights_vector.copy()
        self.p_value = np.ones(self.cut_weights_vector.shape)*1000
        for i in range(self.cut_weights_expectation.shape[0]):
            proportion_of_class = self.counter[self.labels[i]]/self.labels.shape[0]
            self.cut_weights_expectation[i] = (1 - proportion_of_class)*np.sum(self.weight_matrix[i])
            self.cut_weights_variance[i] = proportion_of_class*(1 - proportion_of_class)*np.sum(self.weight_matrix[i]**2)
            if self.cut_weights_variance[i] == 0:
                self.normalized_cut_weights_vector[i] = -100
            else:
                self.normalized_cut_weights_vector[i] = (self.normalized_cut_weights_vector[i] - self.cut_weights_expectation[i])/sqrt(self.cut_weights_variance[i])
                self.p_value[i] = st.norm.cdf(self.normalized_cut_weights_vector[i])
        
    def _get_mislabelled_objects(self,risk = 0.05):
        z_score = st.norm.ppf(risk)
        return self.normalized_cut_weights_vector > z_score
    
    def mislabelled_objects(self,risk = 0.05):
        self.calculate_weight_of_cut_edges_for_each_i()
        return self._get_mislabelled_objects(risk)
        
    def same_class_matrix(self):
        matrix = np.array([self.labels,]*self.labels.shape[0])
        return matrix==matrix.transpose()
    
    def test_dist_matrix(self):
        t = np.array([[1,0,0],[0,1,0],[3,1,1]])
        d = self.pairwise_dist_matrix(t)
        result = np.sum(d < 0) == 0
        if result:
            result = d
        return result
    
    
    
    def report_separability(self):
        count = Counter(self.labels)
        for u in np.unique(self.clusters_labels):
            temp = np.where(self.cluster_labels == u)[0]
            print('Class: ', u)
            print('Number of clusters: ', temp.shape[0], 'Size of data :', count[u])
            print('Ratio: ', 1- temp.shape[0]/count[u])
    
        