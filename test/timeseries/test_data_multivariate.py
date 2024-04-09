from unittest import TestCase
import numpy as _np

import so_ml_tools as _soml


class Test(TestCase):

    def test_dataset_from_array_univariate_window_3_horizon_1(self):
        array = _np.reshape(_np.arange(0, 70, 1, dtype=_np.int64), newshape=(10, 7))
        df = _soml.timeseries.data.dataset_from_array_multivariate(
            data=array,
            label_column_idx=0,
            window_size=3,
            horizon_size=1,
            batch_size=2)

        expected_features = _np.array([
            [
                [0, 1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12, 13],
                [14, 15, 16, 17, 18, 19, 20],
            ], [
                [7, 8, 9, 10, 11, 12, 13],
                [14, 15, 16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25, 26, 27],
            ]
        ])

        expected_labels = _np.array([
            [21],
            [28]
        ])

        features, labels = _soml.tf.dataset.get_features_and_labels(dataset=df, max_samples=2)
        _np.testing.assert_array_equal(x=features, y=expected_features)
        _np.testing.assert_array_equal(x=labels, y=expected_labels)

    def test_dataset_from_array_univariate_window_3_horizon_2(self):
        array = _np.reshape(_np.arange(0, 70, 1, dtype=_np.int64), newshape=(10, 7))
        df = _soml.timeseries.data.dataset_from_array_multivariate(
            data=array,
            label_column_idx=0,
            window_size=3,
            horizon_size=2,
            batch_size=2)

        expected_features = _np.array([
            [
                [0, 1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12, 13],
                [14, 15, 16, 17, 18, 19, 20],
            ], [
                [7, 8, 9, 10, 11, 12, 13],
                [14, 15, 16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25, 26, 27],
            ]
        ])

        expected_labels = _np.array([
            [21, 28],
            [28, 35]
        ])

        features, labels = _soml.tf.dataset.get_features_and_labels(dataset=df, max_samples=2)
        _np.testing.assert_array_equal(x=features, y=expected_features)
        _np.testing.assert_array_equal(x=labels, y=expected_labels)

    def test_dataset_from_array_univariate_window_4_horizon_2_centered(self):
        array = _np.reshape(_np.arange(0, 70, 1, dtype=_np.int64), newshape=(10, 7))
        df = _soml.timeseries.data.dataset_from_array_multivariate(
            data=array,
            label_column_idx=0,
            window_size=4,
            horizon_size=2,
            centered=True,
            batch_size=2)

        expected_features = _np.array([
            [
                [0, 1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12, 13],
                [14, 15, 16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25, 26, 27]
            ], [
                [7, 8, 9, 10, 11, 12, 13],
                [14, 15, 16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25, 26, 27],
                [28, 29, 30, 31, 32, 33, 34]
            ]
        ])

        expected_labels = _np.array([
            [14, 21],
            [21, 28]
        ])

        features, labels = _soml.tf.dataset.get_features_and_labels(dataset=df, max_samples=2)
        _np.testing.assert_array_equal(x=features, y=expected_features)
        _np.testing.assert_array_equal(x=labels, y=expected_labels)