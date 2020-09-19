import numpy as np
import torch
from torch import nn


def get_real_imag_parts(x, complex_dim):
    if type(x) == tuple:
        return x
    
    assert(x.shape[complex_dim] == 2)
    x_real = torch.narrow(x, complex_dim, 0, 1)
    x_imag = torch.narrow(x, complex_dim, 1, 1)
    return x_real, x_imag


def stack_real_imag_parts(x_real, x_imag, complex_dim, return_type):
    if return_type == "tensor":
        return torch.stack([x_real, x_imag], complex_dim)
    elif return_type == "tuple":
        return x_real, x_imag
    else:
        raise ValueError


def matmul(x, y, complex_dim, return_type="tensor"):
    x_real, x_imag = get_real_imag_parts(x, complex_dim)
    y_real, y_imag = get_real_imag_parts(y, complex_dim)

    out_real = torch.matmul(x_real, y_real) - torch.matmul(x_imag, y_imag)
    out_imag = torch.matmul(x_real, y_imag) + torch.matmul(x_imag, y_real)

    return stack_real_imag_parts(out_real, out_imag, complex_dim, return_type)


def hermite(x, dim0, dim1, complex_dim, return_type="tensor"):
    x = x.transpose(dim0, dim1)
    x_real, x_imag = get_real_imag_parts(x, complex_dim)
    x_imag = -x_imag

    return stack_real_imag_parts(x_real, x_imag, complex_dim, return_type)


def outer_product(x, y, complex_dim, return_type="tensor"):
    # simply calculates x@hermite(y)
    y = hermite(y, -2, -1, complex_dim, return_type="tuple")
    
    return matmul(x, y, complex_dim, return_type)


def solve(b, a, complex_dim, return_type="tensor"):
    a_real, a_imag = get_real_imag_parts(a, complex_dim)
    b_real, b_imag = get_real_imag_parts(b, complex_dim)

    a = torch.cat([torch.cat([a_real, -a_imag], dim=-1),
                   torch.cat([a_real,  a_imag], dim=-1)], dim=-2)
    b = torch.cat([b_real, b_imag], dim=-2)
    x, _ = torch.solve(b, a)

    x = torch.chunk(x, 2, dim=-2)
    return stack_real_imag_parts(x[0], x[1], complex_dim, return_type)


def to_numpy(x, complex_dim):
    # convert decoupled complex torch-tensor/numpy-array, to a full complex numpy array
    # x - torch tensor
    # dim - the dimension of the real/imag parts.

    if type(x) == torch.Tensor:
        x = x.detach().numpy()
    x_real, x_imag = get_real_imag_parts(x, complex_dim)
    return x_real + 1j * x_imag


def from_numpy(x, complex_dim, return_type="tensor"):
    # convert a complex numpy array to a complex torch-tensor/numpy-array
    # x - the numpy array
    # dim - the dimension to put the real/imag parts
    assert type(x) == np.ndarray, "input must be a numpy array"
    return stack_real_imag_parts(torch.from_numpy(np.real(x)),
                                 torch.from_numpy(np.imag(x)),
                                 complex_dim, return_type)


def calc_scm(x, complex_dim, smoothing_dim, channels_dim, return_type="tensor"):
    if complex_dim > smoothing_dim or complex_dim > channels_dim:
        raise NotImplementedError
    n = x.shape[smoothing_dim]
    x = _moveaxis(x, smoothing_dim, -1)
    if smoothing_dim < channels_dim:
        channels_dim -= 1
    x = _moveaxis(x, channels_dim, -2)
    
    R = outer_product(x, x, complex_dim, return_type)
    if type(R) == tuple:
        return R[0]/n, R[1]/n
    else:
        return R / n


def tensordot(x, y, complex_dim, tensordot_dims, return_type="tensor", conj_y=False):       
    x_real, x_imag = get_real_imag_parts(x, complex_dim)
    y_real, y_imag = get_real_imag_parts(y, complex_dim)
    if conj_y:
        y_imag *= -1
    out_real = torch.tensordot(x_real, y_real, tensordot_dims) - torch.tensordot(x_imag, y_imag, tensordot_dims)
    out_imag = torch.tensordot(x_real, y_imag, tensordot_dims) + torch.tensordot(x_imag, y_real, tensordot_dims)

    return stack_real_imag_parts(out_real, out_imag, complex_dim, return_type)


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


def _moveaxis(tensor: torch.Tensor, source: int, destination: int) -> torch.Tensor:
    # helper function to imitate numpy's moveaxis function.
    # copied from https://github.com/pytorch/pytorch/issues/36048
    
    dim = tensor.dim()
    perm = list(range(dim))
    if destination < 0:
        destination += dim
    perm.pop(source)
    perm.insert(destination, source)
    return tensor.permute(*perm)