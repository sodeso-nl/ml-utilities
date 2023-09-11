from unittest import TestCase
import numpy as np
import tensorflow as tf

import so_ml_tools as _soml


class Test(TestCase):
    def test_is_multiclass_classification_true(self):
        y_prob = [[0.98, 0.2], [0.4, 0.6]]
        self.assertTrue(_soml.util.label.is_multiclass_classification(y_prob=y_prob))
        self.assertTrue(_soml.util.label.is_multiclass_classification(y_prob=np.array(y_prob)))
        self.assertTrue(_soml.util.label.is_multiclass_classification(y_prob=tf.constant(y_prob)))

    def test_is_multiclass_classification_false(self):
        y_prob = [[0.98], [0.45]]
        self.assertFalse(_soml.util.label.is_multiclass_classification(y_prob=y_prob))
        self.assertFalse(_soml.util.label.is_multiclass_classification(y_prob=np.array(y_prob)))
        self.assertFalse(_soml.util.label.is_multiclass_classification(y_prob=tf.constant(y_prob)))

    def test_is_binary_classification_true(self):
        y_prob = [[0.98], [0.45]]
        self.assertTrue(_soml.util.label.is_binary_classification(y_prob=y_prob))
        self.assertTrue(_soml.util.label.is_binary_classification(y_prob=np.array(y_prob)))
        self.assertTrue(_soml.util.label.is_binary_classification(y_prob=tf.constant(y_prob)))

    def test_is_binary_classification_false(self):
        y_prob = [[0.98, 0.02], [0.4, 0.6]]
        self.assertFalse(_soml.util.label.is_binary_classification(y_prob=y_prob))
        self.assertFalse(_soml.util.label.is_binary_classification(y_prob=np.array(y_prob)))
        self.assertFalse(_soml.util.label.is_binary_classification(y_prob=tf.constant(y_prob)))

    def test_probability_to_class(self):
        y_prob = [
            [0.75, 0.20, 0.05],
            [0.26, 0.24, 0.50]
        ]

        result = _soml.util.label.probability_to_class(y_prob=y_prob)
        self.assertTrue(np.array_equal(result, np.array([0, 2])))

        result = _soml.util.label.probability_to_class(y_prob=np.array(y_prob))
        self.assertTrue(np.array_equal(result, np.array([0, 2])))

        result = _soml.util.label.probability_to_class(y_prob=tf.constant(y_prob))
        self.assertTrue(tf.reduce_all(tf.equal(result, tf.constant([0, 2], dtype=tf.int64))))

    def test_probability_to_binary(self):
        y_prob = [
            [0.56],
            [0.32],
            [0.98]
        ]

        result = _soml.util.label.probability_to_binary(y_prob=y_prob)
        self.assertTrue(np.array_equal(result, np.array([[1.], [0.], [1.]])))

        result = _soml.util.label.probability_to_binary(y_prob=np.array(y_prob))
        self.assertTrue(np.array_equal(result, np.array([[1.], [0.], [1.]])))

        result = _soml.util.label.probability_to_binary(y_prob=tf.constant(y_prob))
        self.assertTrue(tf.reduce_all(tf.equal(result, tf.constant(value=[[1], [0], [1]], dtype=tf.int8))))

    def test_to_prediction_multi_class(self):
        y_prob = [
            [0.75, 0.20, 0.05],
            [0.26, 0.24, 0.50]
        ]

        result = _soml.util.label.to_prediction(y_prob=y_prob)
        self.assertTrue(np.array_equal(result, np.array([0, 2])))

        result = _soml.util.label.to_prediction(y_prob=np.array(y_prob))
        self.assertTrue(np.array_equal(result, np.array([0, 2])))

        result = _soml.util.label.to_prediction(y_prob=tf.constant(y_prob))
        self.assertTrue(tf.reduce_all(tf.equal(result, tf.constant([0, 2], dtype=tf.int64))))

    def test_to_prediction_binary(self):
        y_prob = [
            [0.56],
            [0.32],
            [0.98]
        ]

        result = _soml.util.label.to_prediction(y_prob=y_prob)
        self.assertTrue(np.array_equal(result, np.array([[1], [0], [1]])))

        result = _soml.util.label.to_prediction(y_prob=np.array(y_prob))
        self.assertTrue(np.array_equal(result, np.array([[1], [0], [1]])))

        result = _soml.util.label.to_prediction(y_prob=tf.constant(y_prob))
        self.assertTrue(tf.reduce_all(tf.equal(result, tf.constant([[1], [0], [1]], dtype=tf.int8))))
