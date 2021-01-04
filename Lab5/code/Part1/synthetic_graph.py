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
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
from gensim.models  import Word2Vec


# Loads the web graph
G = nx.read_weighted_edgelist('../data/synthetic_graph.edgelist', delimiter=' ', create_using=nx.Graph())

max_walk_length = 20
number_walk = 10
max_window = 10
n_dim=2

fig = plt.figure()
for window in range(1, max_window + 1):
    similarities = []
    walk_lengths = []
    for walk_length in range(3, max_walk_length +1, 2):
        walk_lengths.append( walk_length )
        walks = generate_walks( G, number_walk, walk_length )
        model = Word2Vec(size=n_dim, window=window, min_count=0, sg=1, workers=8)
        model.build_vocab(walks)
        model.train(walks, total_examples=model.corpus_count, epochs=5)
        similarities.append(model.wv.similarity('1','11'))
    label="window %i" % (window)
    plt.plot( walk_lengths, similarities, label=label)

plt.xlabel('walk_length')
plt.ylabel('similarities')
plt.legend()


plt.title('dim=%i, walks=%i' % (n_dim, number_walk))
plt.show()


def unique_walk(g):
    return len(set(g))

walks_real_length = np.array( [ i for i in map(unique_walk, walks)])


def similarity_matrix( G, n_dim, window, walk_length):
    walks = generate_walks( G, number_walk, walk_length)
    contexts_size = []
    contexts_miss = 0
    neighbors = set(['1','2','3','4','5','6'])
    for walk in walks:
        indexes_1 = [ i for i, elem in enumerate(walk) if elem == '1']
        for index in indexes_1:
            min_index = max(0, index - window)
            max_index = min( len(walk) - 1, index + window)
            context = set(walk[min_index: max_index])
            contexts_size.append( len(context))
            if len( context - neighbors) > 0:
                contexts_miss += 1
    contexts_size_np = np.array( contexts_size)
    model = Word2Vec( size=n_dim, window= window, min_count = 0, sg=1, workers=8)
    model.build_vocab( walks)
    model.train( walks, total_examples = model.corpus_count, epochs=5)
    fig = plt.figure()
    similarities = np.zeros((G.number_of_nodes(),G.number_of_nodes()))
    for i in range(1,12):
        for j in range(1,12):
            similarities[i-1,j-1]=model.wv.similarity(str(i),str(j))
    cmap = plt.pcolormesh( similarities)
    fig.colorbar( cmap )
    title = "n_dim = %i, window = %i, walk_length = %i" % ( n_dim, window, walk_length)
    plt.title( title )  
    similarity_legend = "similarity %f" % ( model.wv.similarity('1','11'))
    similarity_patch = mpatches.Patch(color='red', label=similarity_legend)
    contexts_stats_legend = "avg (%i) max (%.2f) std(%.2f)" % ( np.mean(contexts_size_np), np.max( contexts_size_np), np.std( contexts_size_np))
    contexts_stats_patch = mpatches.Patch(color='red', label=contexts_stats_legend)
    contexts_miss_legend = "contexts  (%.2f) miss (%.2f)" % ( contexts_size_np.shape[0], contexts_miss / contexts_size_np.shape[0] * 100)
    contexts_miss_patch = mpatches.Patch(color='red', label=contexts_miss_legend )
    plt.legend( handles = [ similarity_patch, contexts_stats_patch, contexts_miss_patch ], loc='upper center' )
    plt.show()