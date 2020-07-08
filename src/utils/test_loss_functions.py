from unittest import TestCase

import torch

from src.utils.loss_functions import l2_outer_product, get_real_imag_parts, complex_mm


class Test(TestCase):
    def test_l2_outer_product(self):
        x = torch.zeros((3, 2, 2, 1))
        x[0, 0] = torch.tensor([[1], [0]])
        x[0, 1] = torch.tensor([[0], [1]])
        x[1, 0] = torch.tensor([[0], [3]])

        target = torch.zeros((3, 2, 2, 2))
        target[0, 0] = torch.tensor([[1, 0], [0, 1]])
        target[0, 1] = torch.tensor([[0, -1], [1, 0]])
        target[1, 0] = torch.tensor([[0, 0], [0, 9]])

        loss = l2_outer_product(x, target)
        self.assertEqual(loss, 0)

        # make the last sample wrong
        target[2] = torch.ones(2, 2, 2)
        los_expected = 2 * 4 / 3
        loss = l2_outer_product(x, target)
        self.assertAlmostEqual(loss, los_expected)

    def test_get_real_imag_parts(self):
        x = torch.arange(3 * 2 * 4 * 5).reshape((3, 2, 4, 5))
        x_real, x_imag = get_real_imag_parts(x)
        self.assertTrue(x_real.shape == torch.Size((3, 4, 5)))
        self.assertTrue(x_imag.shape == torch.Size((3, 4, 5)))

    def test_complex_mm(self):
        x_real = torch.tensor([[[1, 0],
                                [0, 1]],
                               [[2, 0],
                                [0, 2]]])
        x_imag = torch.tensor([[[0, 0],
                                [0, 0]],
                               [[0, 1],
                                [1, 0]]])
        y_real = torch.tensor([[[1],
                                [0]],
                               [[0],
                                [0]]])
        y_imag = torch.tensor([[[0],
                                [1]],
                               [[1],
                                [0]]])
        out_real = torch.tensor([[[1],
                                  [0]],
                                 [[0],
                                  [-1]]])
        out_imag = torch.tensor([[[0],
                                  [1]],
                                 [[2],
                                  [0]]])
        out_real_test, out_imag_test = complex_mm((x_real, x_imag), (y_real, y_imag))
        self.assertTrue(torch.equal(out_real_test, out_real))
        self.assertTrue(torch.equal(out_imag_test, out_imag))
