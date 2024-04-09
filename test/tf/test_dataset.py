from unittest import TestCase
import tensorflow as _tf
import numpy as _np

import so_ml_tools as _soml


class Test(TestCase):

    def test_split_1D(self):
        x = _np.arange(start=10, stop=20, step=1)
        y = _np.arange(start=20, stop=30, step=1)
        ds = _tf.data.Dataset.from_tensor_slices(tensors=(x, y))

        train_ds, validate_ds, test_ds = _soml.tf.dataset.split(dataset=ds, split_percentages=[0.4, 0.4, 0.2])

        train_x, train_y = _soml.tf.dataset.get_features_and_labels(dataset=train_ds)
        validate_x, validate_y = _soml.tf.dataset.get_features_and_labels(dataset=validate_ds)
        test_x, test_y = _soml.tf.dataset.get_features_and_labels(dataset=test_ds)

        _np.testing.assert_array_equal(train_x, _np.array([[10], [11], [12], [13]]))
        _np.testing.assert_array_equal(train_y, _np.array([[20], [21], [22], [23]]))

        _np.testing.assert_array_equal(validate_x, _np.array([[14], [15], [16], [17]]))
        _np.testing.assert_array_equal(validate_y, _np.array([[24], [25], [26], [27]]))

        _np.testing.assert_array_equal(test_x, _np.array([[18], [19]]))
        _np.testing.assert_array_equal(test_y, _np.array([[28], [29]]))

    def test_split_2D(self):
        x = _np.reshape(_np.arange(start=10, stop=20, step=1), newshape=(5, 2))
        y = _np.reshape(_np.arange(start=20, stop=30, step=1), newshape=(5, 2))
        ds = _tf.data.Dataset.from_tensor_slices(tensors=(x, y))

        train_ds, validate_ds, test_ds = _soml.tf.dataset.split(dataset=ds, split_percentages=[0.4, 0.4, 0.2])

        train_x, train_y = _soml.tf.dataset.get_features_and_labels(dataset=train_ds)
        validate_x, validate_y = _soml.tf.dataset.get_features_and_labels(dataset=validate_ds)
        test_x, test_y = _soml.tf.dataset.get_features_and_labels(dataset=test_ds)

        _np.testing.assert_array_equal(train_x, _np.array([[10, 11], [12, 13]]))
        _np.testing.assert_array_equal(train_y, _np.array([[20, 21], [22, 23]]))

        _np.testing.assert_array_equal(validate_x, _np.array([[14, 15], [16, 17]]))
        _np.testing.assert_array_equal(validate_y, _np.array([[24, 25], [26, 27]]))

        _np.testing.assert_array_equal(test_x, _np.array([[18, 19]]))
        _np.testing.assert_array_equal(test_y, _np.array([[28, 29]]))

    def test_get_features(self):
        x = _np.arange(start=10, stop=20, step=1)
        y = _np.arange(start=20, stop=30, step=1)
        ds = _tf.data.Dataset.from_tensor_slices(tensors=(x, y)).batch(2)

        features = _soml.tf.dataset.get_features(dataset=ds)
        _np.testing.assert_array_equal(features,
                                       _np.array([[10], [11], [12], [13], [14], [15], [16], [17], [18], [19]]))

    def test_get_features_with_max_samples(self):
        x = _np.arange(start=10, stop=20, step=1)
        y = _np.arange(start=20, stop=30, step=1)
        ds = _tf.data.Dataset.from_tensor_slices(tensors=(x, y)).batch(2)

        features = _soml.tf.dataset.get_features(dataset=ds, max_samples=4)
        _np.testing.assert_array_equal(features,
                                       _np.array([[10], [11], [12], [13]]))

    def test_get_lables(self):
        x = _np.arange(start=10, stop=20, step=1)
        y = _np.arange(start=20, stop=30, step=1)
        ds = _tf.data.Dataset.from_tensor_slices(tensors=(x, y)).batch(2)

        labels = _soml.tf.dataset.get_labels(dataset=ds)
        _np.testing.assert_array_equal(labels,
                                       _np.array([[20], [21], [22], [23], [24], [25], [26], [27], [28], [29]]))

    def test_get_lables_max_samples(self):
        x = _np.arange(start=10, stop=20, step=1)
        y = _np.arange(start=20, stop=30, step=1)
        ds = _tf.data.Dataset.from_tensor_slices(tensors=(x, y)).batch(2)

        labels = _soml.tf.dataset.get_labels(dataset=ds, max_samples=4)
        _np.testing.assert_array_equal(labels,
                                       _np.array([[20], [21], [22], [23]]))

    def test_get_features_and_labels(self):
        x = _np.arange(start=10, stop=20, step=1)
        y = _np.arange(start=20, stop=30, step=1)
        ds = _tf.data.Dataset.from_tensor_slices(tensors=(x, y)).batch(2)

        features, labels = _soml.tf.dataset.get_features_and_labels(dataset=ds)
        _np.testing.assert_array_equal(features,
                                       _np.array([[10], [11], [12], [13], [14], [15], [16], [17], [18], [19]]))
        _np.testing.assert_array_equal(labels,
                                       _np.array([[20], [21], [22], [23], [24], [25], [26], [27], [28], [29]]))

    def test_get_features_and_labels_max_samples(self):
        x = _np.arange(start=10, stop=20, step=1)
        y = _np.arange(start=20, stop=30, step=1)
        ds = _tf.data.Dataset.from_tensor_slices(tensors=(x, y)).batch(2)

        features, labels = _soml.tf.dataset.get_features_and_labels(dataset=ds, max_samples=4)
        _np.testing.assert_array_equal(features,
                                       _np.array([[10], [11], [12], [13]]))
        _np.testing.assert_array_equal(labels,
                                       _np.array([[20], [21], [22], [23]]))