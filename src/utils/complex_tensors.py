import numpy as np
import torch
from torch import nn


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

    shape_vec = (x_real.shape[0], 1, *x_real.shape[1:])
    return torch.cat((x_real.view(shape_vec), x_imag.view(shape_vec)), dim=1)


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
        x = x.detach().numpy()

    x = np.moveaxis(x, dim, 0)
    x = x[0] + 1j * x[1]

    return x


def numpy2complextorch(x, dim=0):
    # convert a complex numpy array to a complex torch-tensor/numpy-array
    # x - the numpy array
    # dim - the dimension to put the real/imag parts
    assert type(x) == np.ndarray, "input must be a numpy array"
    return torch.stack([torch.from_numpy(np.real(x)), torch.from_numpy(np.imag(x))], dim=dim)


def calc_scm(x, smoothing_dim, channels_dim, real_imag_dim, batch_dim):
    T = x.shape[smoothing_dim]
    Q = x.shape[channels_dim]
    N = x.shape[batch_dim]

    x = x.permute(real_imag_dim, batch_dim, channels_dim, smoothing_dim)
    R = torch.zeros((N, 2, Q, Q), device=x.device)  # make sure R is on the same device
    R[:, 0, :, :], R[:, 1, :, :] = complex_outer_product((x[0], x[1]))
    return R / T


def complex_tensordot(x_re, x_im, y_re, y_im, dims=2, stack_dim=1):
    z_re = torch.tensordot(x_re, y_re, dims) - torch.tensordot(x_im, y_im, dims)
    z_im = torch.tensordot(x_re, y_im, dims) + torch.tensordot(x_im, y_re, dims)
    return torch.stack((z_re, z_im), dim=stack_dim)


class ComplexConv1d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # Model components
        self.conv_re = nn.Conv1d(**kwargs)
        self.conv_im = nn.Conv1d(**kwargs)

    def forward(self, x):  # shape of x : [batch,2,channel,time]
        real = self.conv_re(x[:, 0]) - self.conv_im(x[:, 1])
        imaginary = self.conv_re(x[:, 1]) + self.conv_im(x[:, 0])
        output = torch.stack((real, imaginary), dim=1)
        return output

    @property
    def weight(self):
        return torch.stack((self.conv_re.weight, self.conv_im.weight), dim=0)

    @property
    def bias(self):
        if self.conv_re.bias is None:
            return None
        return torch.stack((self.conv_re.bias, self.conv_im.bias), dim=0)


class ComplexConv2d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # Model components
        self.conv_re = nn.Conv2d(**kwargs)
        self.conv_im = nn.Conv2d(**kwargs)

    def forward(self, x):  # shape of x : [batch, 2, channel, freq, time]
        real = self.conv_re(x[:, 0]) - self.conv_im(x[:, 1])
        imaginary = self.conv_re(x[:, 1]) + self.conv_im(x[:, 0])
        output = torch.stack((real, imaginary), dim=1)
        return output

    @property
    def weight(self):
        return torch.stack((self.conv_re.weight, self.conv_im.weight), dim=0)

    @property
    def bias(self):
        if self.conv_re.bias is None:
            return None
        return torch.stack((self.conv_re.bias, self.conv_im.bias), dim=0)
