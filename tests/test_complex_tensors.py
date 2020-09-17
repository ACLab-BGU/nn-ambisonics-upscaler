from unittest import TestCase

import torch

from src.utils.complex_tensors_old import get_real_imag_parts_old, complex_mm, cat_real_imag_parts


class Test(TestCase):

    def test_get_real_imag_parts(self):
        x = torch.arange(3 * 2 * 4 * 5).reshape((3, 2, 4, 5))
        x_real, x_imag = get_real_imag_parts_old(x)
        self.assertTrue(x_real.shape == torch.Size((3, 4, 5)))
        self.assertTrue(x_imag.shape == torch.Size((3, 4, 5)))

    def test_cat_real_imag_parts(self):
        x = torch.arange(3 * 2 * 4 * 5).reshape((3, 2, 4, 5))
        x_real, x_imag = get_real_imag_parts_old(x)
        x_rec = cat_real_imag_parts(x_real, x_imag)
        self.assertTrue(x.shape == x_rec.shape)
        self.assertTrue(torch.equal(x,x_rec))

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
