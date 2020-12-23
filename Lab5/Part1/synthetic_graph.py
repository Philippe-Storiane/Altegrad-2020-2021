# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:34:13 2020

@author: philippe
"""
import networkx as nx
import numpy as np
from deepwalk import generate_walks
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from gensim.models  import Word2Vec
# Loads the web graph
G = nx.read_weighted_edgelist('../data/synthetic_graph.edgelist', delimiter=' ', create_using=nx.Graph())

max_walk_length = 10
number_walk = 20
max_window = 3
n_dim=2
measure_window=[]
measure_walk_length=[]
measure_similarities=[]
for walk_length in range(1,max_walk_length + 1):
    for window in range(1, max_window + 1):
        walks = generate_walks( G, number_walk, walk_length )
        model = Word2Vec(size=n_dim, window=window, min_count=0, sg=1, workers=8)
        model.build_vocab(walks)
        model.train(walks, total_examples=model.corpus_count, epochs=5)
        measure_window.append( window)
        measure_walk_length.append( walk_length)
        measure_similarities.append(model.wv.similarity('1','11'))
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter( measure_walk_length, measure_window, measure_similarities, color='r', marker='o')
ax.set_xlabel('walk_length')
ax.set_ylabel('window')
ax.set_zlabel('similarity')

plt.title('dim=%i, walks=%i' % (n_dim, number_walk))
plt.show()