"""
Deep Learning on Graphs - ALTEGRAD - Dec 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 10 and 13
        
        ##################
        # your code here #
        ##################
        x_w0 = self.fc1( x_in)
        a_x_w0 = torch.matmul(adj, x_w0)
        z0 = self.relu(a_x_w0)
        
        z0 = self.dropout( z0 )

        
        z0_w1 = self.fc2(z0)
        a_z0_w1 = torch.matmul(adj, z0_w1)
        z1 = self.relu( a_z0_w1)
        
        z1_w2 = self.fc3( z1)

# very good accuracy

        return F.log_softmax(z1_w2, dim=1), a_z0_w1