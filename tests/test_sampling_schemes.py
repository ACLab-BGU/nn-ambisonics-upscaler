from unittest import TestCase

from src.utils.sphere.sampling_schemes import grid


class Test(TestCase):
    def test_grid(self):
        omega = grid(200, output_type='all_combs_1d')
        self.assertTrue(omega.shape == (2, 200))

        omega = grid(200, output_type='all_combs_2d')
        self.assertTrue(omega.shape == (2, 10, 20))

        th, ph = grid(200)
        self.assertTrue(th.shape == (10,))
        self.assertTrue(ph.shape == (20,))

        th, ph = grid(200, output_type='tuple_2d')
        self.assertTrue(th.shape == (10, 20))
        self.assertTrue(ph.shape == (10, 20))
