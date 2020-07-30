import numpy as np
import torch


def l2_sq_complex(x) -> torch.Tensor:
    # x: ((N, M, L), (N, M, L))
    return torch.sum(x[0] ** 2 + x[1] ** 2) / x[0].shape[0]


def get_real_imag_parts(x):
    # x: (_, 2, _, _)

    x_real = x[:, 0]
    x_imag = x[:, 1]

    return x_real, x_imag

def cat_real_imag_parts(x_real, x_imag):
    # x_real, x_imag - (N, M, L)
    # out - (N, 2, M, L)

    shape_vec = (x_real.shape[0],1,*x_real.shape[1:])
    return torch.cat((x_real.view(shape_vec),x_imag.view(shape_vec)),dim=1)

def complex_mm(x, y):
    # x - ((N, M, L), (N, M, L))
    # y - ((N, L, K), (N, L, K))
    # out - ((N, M, K), (N, M, K))

    x_real, x_imag = x
    y_real, y_imag = y

    out_real = torch.matmul(x_real, y_real) - torch.matmul(x_imag, y_imag)
    out_imag = torch.matmul(x_real, y_imag) + torch.matmul(x_imag, y_real)

    return out_real, out_imag


def complex_outer_product(x, y=None):
    # x - ((N, Q, L), (N, Q, L))
    # y - ((N, Q, L), (N, Q, L))
    # out - ((N, Q, Q), (N, Q, Q))

    if y is None:
        y = x

    x_real, x_imag = x
    y_real, y_imag = y

    return complex_mm((x_real, x_imag), (y_real.transpose(1, 2), -y_imag.transpose(1, 2)))


def complextorch2numpy(x, dim=0):
    # convert decoupled complex torch-tensor/numpy-array, to a full complex numpy array
    # x - the decoupled tensor/array
    # dim - the dimension of the real/imag parts, must be of size 2

    assert x.shape[dim] == 2, "size of the imaginary dimensions must be 2!"
    if type(x) == torch.Tensor:
        x = x.numpy()

    x = np.moveaxis(x, dim, 0)
    x = x[0] + 1j * x[1]

    return x