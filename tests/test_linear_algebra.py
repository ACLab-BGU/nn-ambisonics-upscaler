from unittest import TestCase
import numpy as np

from src.utils.linear_algebra import find_nearest_PSD_mat


class Test(TestCase):
    def test_find_nearest_psd_mat(self):
        mat = np.diag([1,2,3])
        out = find_nearest_PSD_mat(mat)
        self.assertTrue(np.linalg.norm(out-mat) < 1e-9 * np.linalg.norm(mat))

        mat = np.diag([1, -2, 3])
        out = find_nearest_PSD_mat(mat)
        self.assertTrue(np.linalg.norm(out-np.diag([1, 0, 3])) < 1e-9 * np.linalg.norm(mat))
