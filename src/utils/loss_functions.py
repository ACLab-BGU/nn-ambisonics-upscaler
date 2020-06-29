import torch


def l2_outer_product(x, target):
    # x: (N, Q, L)
    # target: (N, Q, Q)
    d = target - torch.matmul(x, x.transpose(1, 2))
    return torch.sum(d ** 2)
