import torch


def l2_outer_product(x, target):
    # x: (N, 2, Q, L)
    # target: (N, 2, Q, Q)
    d = target - torch.matmul(x, x.transpose(1, 2))
    return torch.sum(d ** 2)
