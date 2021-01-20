"""
Deep Learning on Graphs - ALTEGRAD - Jan 2021
"""

import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn

def normalize_adjacency(A):
    ############## Task 1
    
    ##################
    # your code here #
    ##################
    n = A.shape[0]
    I = sp.identity(n)
    Ap = A + I
    degs = Ap.dot(np.ones(n))
    inv_degs = np.power(degs, -1)
    D_inv = sp.diags( inv_degs)
    A_normalized = D_inv.dot(Ap)
    return A_normalized


def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def loss_function(z, adj, device):
    mse_loss = nn.MSELoss()

    ############## Task 3
    
    ##################
    # your code here #
    ##################
    indices = adj.indices()
    
    y_pred = list()
    y = list()
    y_pred.append(torch.sum(torch.mul( z[indices[0,:], :],z[indices[1,:],:]), dim=1))
    y.append( torch.ones( indices.size(1)).to(device))
    
    random_indices = torch.randint( z.size(0), indices.size())
    y_pred.append( torch.sum(torch.mul(z[random_indices[0,:],:], z[random_indices[1,:],:]), dim=1))
    y.append( torch.zeros(indices.size(1)).to(device))

    y_pred = torch.cat( y_pred, dim = 0)
    y = torch.cat( y, dim = 0)

    loss = mse_loss(y_pred, y)
    return loss