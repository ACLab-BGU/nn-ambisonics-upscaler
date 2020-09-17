import numpy as np
import torch
from torch import nn
import src.utils.complex_torch as ctorch

def MSE(y_pred,y):
    '''MSE loss, normalized per sample
    y_pred and y are assumed to have shape (N,*_), where N is the batch size'''
    return nn.MSELoss()(y_pred, y) * np.prod(y.shape[1:])

def MSE_cross_scm_wiener(y_pred,y):
    Rzx_pred,x_nb_transformed = y_pred
    Rzx_true = y
    T = x_nb_transformed.shape[0]
    t_vec = torch.arange(1, T + 1)
    # x_nb_transformed is of shape: (T,N,2,Qx)
    # Rzx_pred is of shape: (T,N,2,Qz,Qx)
    # Rzx_true is of shape: (N,2,Qz,Qx)

    D = Rzx_pred - Rzx_true[None,:]
    # D is of shape: (T,N,2,Qz,Qx)

    x_nb_transformed = x_nb_transformed.permute(1,2,3,0)
    # x_nb_transformed is now of shape: (N,2,Qx,T)

    normalization_factor = ctorch.matmul(Rzx_true,x_nb_transformed,complex_dim=1) # shape (N,2,Qz,T)
    normalization_factor = torch.sum(normalization_factor**2,dim=[1,2,3]) # shape (N,)

    x_nb_transformed = x_nb_transformed[None, :].expand(T, -1, -1, -1, -1)
    # x_nb_transformed is now of shape: (T,N,2,Qx,T)

    err = ctorch.matmul(D, x_nb_transformed, complex_dim=1)  # shape (T,N,2,Qz,T)
    err = torch.sum(err ** 2, dim=[2, 3, 4])  # shape (T,N)

    err_normalized = err / normalization_factor[None,:] # shape (T,N)
    # TODO: think about the best way ro return err_normalized

    err_normalized_weighted =  ((t_vec[:,None]/t_vec.sum()) * err_normalized).sum(dim=0) # shape (N,)

    return err_normalized_weighted