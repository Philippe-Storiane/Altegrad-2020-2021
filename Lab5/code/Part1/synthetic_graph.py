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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from gensim.models  import Word2Vec
# Loads the web graph
G = nx.read_weighted_edgelist('../data/synthetic_graph.edgelist', delimiter=' ', create_using=nx.Graph())

max_walk_length = 20
number_walk = 10
max_window = 10
n_dim=2


walk_length = np.arange(1,number_walk + 1,1)
windows = np.arange(1,max_window + 1, 1)
walk_mesh, window_mesh = np.meshgrid( walk_length, windows)
similarity_mesh = np.zeros( window_mesh.shape)
for walk in range(walk_mesh.shape[0]):
    for window in range(walk_mesh.shape[1]):
        walks = generate_walks( G, number_walk, walk_mesh[walk, window] )
        model = Word2Vec(size=n_dim, window=window_mesh[walk,window], min_count=0, sg=1, workers=8)
        model.build_vocab(walks)
        model.train(walks, total_examples=model.corpus_count, epochs=5)
        similarity_mesh[walk,window] = model.wv.similarity('1','11')
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe( walk_mesh, window_mesh, similarity_mesh)
ax.set_xlabel('walk_length')
ax.set_ylabel('window')
ax.set_zlabel('similarity')


plt.title('dim=%i, walks=%i' % (n_dim, number_walk))
plt.show()