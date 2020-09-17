import numpy as np
from torch import nn

def MSE(y_pred,y):
    '''MSE loss, normalized per sample
    y_pred and y are assumed to have shape (N,*_), where N is the batch size'''
    return nn.MSELoss()(y_pred, y) * np.prod(y.shape[1:])

def MSE_cross_scm_wiener(y_pred,y):
    Rzx_pred,x_transformed = y_pred
    Rzx_true = y
    T = x_transformed.shape[0]
    # x_transformed is of shape: (T, N, 2, Qx, F)
    # Rzx_pred is of shape: (T, N, Qz, Qx)
    # Rzd_true is of shape: (N, Qz, Qx)

    D = Rzx_pred - Rzx_true[None,:]
    # D is of shape: (T, N, Qz, Qx)
