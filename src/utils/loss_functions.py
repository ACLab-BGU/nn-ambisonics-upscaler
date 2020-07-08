import torch


def l2_outer_product(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # x: (N, 2, Q, L)
    # target: (N, 2, Q, Q)

    N = x.shape[0]
    x_tuple = get_real_imag_parts(x)
    target_tuple = get_real_imag_parts(target)

    outer_prod = complex_outer_product(x_tuple)
    d_real = target_tuple[0] - outer_prod[0]
    d_imag = target_tuple[1] - outer_prod[1]
    return l2_sq_complex((d_real, d_imag))


def l2_sq_complex(x) -> torch.Tensor:
    # x: ((N, M, L), (N, M, L))
    return torch.sum(x[0] ** 2 + x[1] ** 2) / x[0].shape[0]


def get_real_imag_parts(x):
    # x: (_, 2, _, _)

    x_real = x[:, 0]
    x_imag = x[:, 1]

    return x_real, x_imag


def complex_mm(x, y):
    # x - ((N, M, L), (N, M, L))
    # y - ((N, L, K), (N, L, K))
    # out - ((N, M, K), (N, M, K))

    x_real, x_imag = x
    y_real, y_imag = y

    out_real = torch.matmul(x_real, y_real) - torch.matmul(x_imag, y_imag)
    out_imag = torch.matmul(x_real, y_imag) + torch.matmul(x_imag, y_real)

    return out_real, out_imag


def complex_outer_product(x):
    # x - ((N, Q, L), (N, Q, L))
    # out - ((N, Q, Q), (N, Q, Q))
    x_real, x_imag = x

    return complex_mm((x_real, x_imag), (x_real.transpose(1, 2), -x_imag.transpose(1, 2)))

# def cat_real_imag_parts(x_real, x_imag):
#     # x_real, x_imag - (N, M, L)
#     # out - (N, 2, M, L)
#
#     return torch.cat(x_real, x_imag)
