"""
Graph Mining - ALTEGRAD - Dec 2020
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans

G = nx.read_edgelist( "./datasets/CA-HepTh.txt", comments="#", delimiter='\t')

############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):

    ##################
    # your code here #
    ##################
    A = nx.adjacency_matrix( G )
    inv_degree_sequence = [ 1 / G.degree(node) for node in G.nodes()]
    D = diags( inv_degree_sequence)
    n = G.number_of_nodes()
    Lrw = eye(n) - D @ A
    eig_values, eig_vectors = eigs( Lrw, k = k, which = 'SR')
    eig_vectors = eig_vectors.real
    kmean = KMeans(n_clusters = k)
    kmean.fit( eig_vectors)
    clusterings = {}
    for i, node in enumerate( G.nodes()):
        clusterings[node] = kmean.labels_[i]
    
    return clusterings



############## Task 7

##################
# your code here #
##################
largest_cc = max( nx.connected_components( G ), key=len)
GCC = G.subgraph( largest_cc)
clustering  = spectral_clustering(GCC, 50)


############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    m = G.number_of_edges()
    clusters = set(clustering.values())
    modularity = 0
    for cluster in clusters:
        nodes_in_cluster = [node  for node in G.nodes() if clustering[ node ] == cluster]
        subG = G.subgraph( nodes_in_cluster)
        lg = subG.number_of_edges()
        dc = 0
        for node in nodes_in_cluster:
            dc += G.degree(node)
        modularity += (((lg * 1.0) / m) - ((dc / (2*m)) ** 2))
    return modularity # between -1 and 1



############## Task 9

##################
# your code here #
##################
measure = modularity( GCC, clustering)
ramdom_clustering={ node: randint(1,50) for node in GCC.nodes()}
random_measure = modularity(GCC, ramdom_clustering)