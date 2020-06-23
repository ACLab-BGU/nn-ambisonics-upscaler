from unittest import TestCase


class Test(TestCase):
    def test_load_free_field_raw_file(self):
        from src.data.make_dataset import load_free_field_raw_file
        import numpy as np
        R = load_free_field_raw_file(1)
        self.assertEqual(R.shape, (513, 49, 49))
        self.assertAlmostEqual(R[0, 0, 0], 0.020269437564138)
        self.assertAlmostEqual(np.imag(R[0, 1, 0]), -0.011099281007683)
        self.assertAlmostEqual(np.imag(R[26, 2, 4]), -0.019799927271249)
