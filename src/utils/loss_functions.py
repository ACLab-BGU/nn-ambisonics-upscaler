import torch

from src.utils.complex_tensors import l2_sq_complex, get_real_imag_parts, complex_outer_product


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
