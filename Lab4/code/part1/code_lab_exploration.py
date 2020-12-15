"""
Graph Mining - ALTEGRAD - Dec 2020
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
# your code here #
##################
G = nx.read_edgelist( "./datasets/CA-HepTh.txt", comments="#", delimiter='\t')
nodes = G.number_of_nodes()
edges = G.number_of_edges()
print("# nodes: " + str( nodes ))
print("# edges: " + str( edges ))


############## Task 2

##################
# your code here #
##################
components = nx.connected_components( G )

max_nodes = 0
giant_connected_component = max( components, key = len)
giant_connected_component = G.subgraph ( giant_connected_component)
gcc_nodes = giant_connected_component.number_of_nodes()
gcc_edges = giant_connected_component.number_of_edges()
print("number / radio nodes: %i / %f " % ( gcc_nodes, (gcc_nodes / nodes) * 100 ) )
print("number / radio edges: %i / %f" % ( gcc_edges, ( gcc_edges / edges) * 100 ) )




############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
##################
degree_sequence = np.array( degree_sequence )
print("minimum degree: " + str( np.min( degree_sequence )))
print("maximum degree: " + str( np.max( degree_sequence )))
print("median degree: " + str( np.median( degree_sequence )))
print("mean degree: " + str( np.mean( degree_sequence )))


############## Task 4

##################
# your code here #
##################
degree_histogram =  nx.degree_histogram( G )
fig = plt.figure()
plt.plot( degree_histogram[1:] , label="nodes vs degreee")
plt.xlabel('degree')
plt.ylabel(' number of nodes')
plt.legend(loc = 'best')
plt.show()

m=1
fig = plt.figure()
plt.loglog( np.arange(m, len(degree_histogram)), degree_histogram[m:] , label="log - log")
plt.xlabel('degree')
plt.ylabel('number nodes')
plt.legend(loc = 'best')
plt.show()



############## Task 5

##################
# your code here #
##################
transitivity = nx.transitivity( G )
print("cluster coefficient %f" % transitivity)