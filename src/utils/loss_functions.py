import numpy as np
from torch import nn

def MSE(x,y):
    '''MSE loss, normalized per sample
    x and y are assumed to have shape (N,*_), where N is the batch size'''
    return nn.MSELoss()(x, y) * np.prod(x.shape[1:])

def MSE_cross_scm(Rzx_est,Rzx_true,x):
    a = 3
