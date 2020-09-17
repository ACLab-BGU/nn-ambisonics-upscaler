from unittest import TestCase
import torch
import src.utils.complex_torch as c
class Test(TestCase):
    def test_matmul(self):
        x = (torch.tensor([[1., 0.], [0., 1.]]),
             torch.zeros((2, 2)))
        z = c.matmul(x, x, 0, "tensor")
        print(z)
