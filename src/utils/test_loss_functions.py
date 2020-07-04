from unittest import TestCase

import torch

from src.utils.loss_functions import l2_outer_product


class Test(TestCase):
    def test_l2_outer_product(self):
        x = torch.ones(5, 4, 1)
        target = torch.ones(5, 4, 4)
        self.assertEqual(l2_outer_product(x, target), 0)
