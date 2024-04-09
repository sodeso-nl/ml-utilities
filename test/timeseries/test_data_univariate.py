from unittest import TestCase
import numpy as _np

import so_ml_tools as _soml


class Test(TestCase):

    def test_dataset_from_array_univariate_window_3_horizon_1(self):
        array = _np.arange(0, 70, 1, dtype=_np.int64)
        df = _soml.timeseries.data.dataset_from_array_univariate(
            data=array,
            window_size=3,
            horizon_size=1,
            batch_size=2)

        features, labels = _soml.tf.dataset.get_features_and_labels(dataset=df, max_samples=2)
        _np.testing.assert_array_equal(x=features, y=_np.array([[0, 1, 2], [1, 2, 3]]))
        _np.testing.assert_array_equal(x=labels, y=_np.array([[3], [4]]))

    def test_dataset_from_array_univariate_window_3_horizon_2(self):
        array = _np.arange(0, 70, 1, dtype=_np.int64)
        df = _soml.timeseries.data.dataset_from_array_univariate(
            data=array,
            window_size=3,
            horizon_size=2)

        features, labels = _soml.tf.dataset.get_features_and_labels(dataset=df, max_samples=2)
        _np.testing.assert_array_equal(x=features, y=_np.array([[0, 1, 2], [1, 2, 3]]))
        _np.testing.assert_array_equal(x=labels, y=_np.array([[3, 4], [4, 5]]))

    def test_dataset_from_array_univariate_window_5_horizon_1_centered(self):
        array = _np.arange(0, 70, 1, dtype=_np.int64)
        df = _soml.timeseries.data.dataset_from_array_univariate(
            data=array,
            window_size=5,
            horizon_size=1,
            centered=True,
            batch_size=2)

        features, labels = _soml.tf.dataset.get_features_and_labels(dataset=df, max_samples=2)
        _np.testing.assert_array_equal(x=features, y=_np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]))
        _np.testing.assert_array_equal(x=labels, y=_np.array([[2], [3]]))

    def test_dataset_from_array_univariate_window_5_horizon_2_centered(self):
        array = _np.arange(0, 70, 1, dtype=_np.int64)
        df = _soml.timeseries.data.dataset_from_array_univariate(
            data=array,
            window_size=5,
            horizon_size=2,
            centered=True,
            batch_size=2)

        features, labels = _soml.tf.dataset.get_features_and_labels(dataset=df, max_samples=2)
        _np.testing.assert_array_equal(x=features, y=_np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]))
        _np.testing.assert_array_equal(x=labels, y=_np.array([[2, 3], [3, 4]]))
