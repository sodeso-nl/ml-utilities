from unittest import TestCase
import numpy as np

import so_ml_tools as _soml


class Test(TestCase):

    def test_is_one_hot_encoded(self):
        y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(_soml.util.onehot.is_one_hot_encoded(y))

        y = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        self.assertFalse(_soml.util.onehot.is_one_hot_encoded(y))

        y = np.array([1, 0, 0])
        self.assertFalse(_soml.util.onehot.is_one_hot_encoded(y))

        y = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertFalse(_soml.util.onehot.is_one_hot_encoded(y))