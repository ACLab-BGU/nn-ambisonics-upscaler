from unittest import TestCase

import torch


class TestModel(TestCase):
    pass


class TestBaseModel(TestCase):
    def test_forward(self):
        x = torch.zeros(40,50,2)
        import src.models.fc_model as fc
        m = fc.BaseModel(100, 10, 2, [50, 15])
        out = m.forward(x)
        self.assertEqual(out.shape, (40, 10))
