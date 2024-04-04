from unittest import TestCase
import numpy as np
import tensorflow as tf

import so_ml_tools as _soml


class Test(TestCase):

    def test_dataset_from_array_univariate_horizon_1(self):
        array = np.arange(0, 70, 1, dtype=np.int64)
        df = _soml.timeseries.data.dataset_from_array_univariate(
            array=array,
            window_size=3,
            horizon_size=1,
            batch_size=2)

        features, labels = next(iter(df))

        self.assertTrue(tf.reduce_all(tf.equal(features.numpy(), [[0, 1, 2], [1, 2, 3]])))
        self.assertTrue(tf.reduce_all(tf.equal(labels.numpy(), [[3, 4]])))

    def test_dataset_from_array_univariate_horizon_2(self):
        array = np.arange(0, 70, 1, dtype=np.int64)
        df = _soml.timeseries.data.dataset_from_array_univariate(
            array=array,
            window_size=3,
            horizon_size=2,
            batch_size=2)

        features, labels = next(iter(df))

        self.assertTrue(tf.reduce_all(tf.equal(features.numpy(), [[0, 1, 2], [1, 2, 3]])))
        self.assertTrue(tf.reduce_all(tf.equal(labels.numpy(), [[3, 4], [4, 5]])))
