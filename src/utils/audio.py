import torch


def torch_stft_nd(x, time_dim, **stft_args):
    '''
    :param x: torch tensor, with an arbitrary number of dimensions D
    :param time_dim: the dimensions of the time domain
    :param stft_args: arguments for the stft
    :return: an stft nd-array, with dimensions [*x_stripped.shape,F,T,2]
    where x_stripped has D-1 dimensions (same dimensions as x, without time_dim)
    and [F,T,2] are the freq,time,real/imag dimensions of the STFT
    '''

    # preliminaries
    dims = x.ndim

    if time_dim == dims - 1:
        permute_required = False
    else:
        permute_required = True

    if dims > 2:
        flatten_required = True
    else:
        flatten_required = False

    # perform permutation and reshaping
    if permute_required:
        x = x.transpose(dims - 1, time_dim)
    shape = x.shape
    if flatten_required:
        x = x.view(-1, shape[-1])

    # perform STFT
    stft = torch.stft(x, **stft_args)
    stft_shape = stft.shape[1:]
    stft = stft.view(*shape[:-1], *stft_shape)

    return stft
