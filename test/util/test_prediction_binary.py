from unittest import TestCase
import numpy as _np
import so_ml_tools.util.prediction as _prediction


class Test(TestCase):

    def test_is_binary_classification(self):
        y = _np.array([0, 1, 0, 1])
        self.assertTrue(_prediction.is_binary_classification(y))

        y = _np.array([0.2, 1.0, 0.0, 0.8])
        self.assertTrue(_prediction.is_binary_classification(y))

        y = _np.array([[0], [1], [0], [1]])
        self.assertTrue(_prediction.is_binary_classification(y))

        y = _np.array([[0.2], [1.0], [0.0], [0.8]])
        self.assertTrue(_prediction.is_binary_classification(y))

        y = _np.array([[0, 1], [1, 0], [0, 0]])
        self.assertFalse(_prediction.is_binary_classification(y))

        y = _np.array([[0], [1], [2]])
        self.assertFalse(_prediction.is_binary_classification(y))

        y = _np.array([])
        self.assertFalse(_prediction.is_binary_classification(y))

    def test_is_binary_probabilities(self):
        self.assertFalse(_prediction.is_binary_probability(y=_np.array([-0.1])))
        self.assertFalse(_prediction.is_binary_probability(y=_np.array([1.1])))
        self.assertFalse(_prediction.is_binary_probability(y=_np.array([[-0.1]])))
        self.assertFalse(_prediction.is_binary_probability(y=_np.array([[-0.1]])))

        self.assertTrue(_prediction.is_binary_probability(y=_np.array([0])))
        self.assertTrue(_prediction.is_binary_probability(y=_np.array([1])))
        self.assertTrue(_prediction.is_binary_probability(y=_np.array([0.1, 0.8])))
        self.assertTrue(_prediction.is_binary_probability(y=_np.array([1.0, 0.0])))

        self.assertTrue(_prediction.is_binary_probability(y=_np.array([0.1])))
        self.assertFalse(_prediction.is_binary_probability(y=_np.array([[1.1]])))
        self.assertFalse(_prediction.is_binary_probability(y=_np.array([[1.1]])))
        self.assertTrue(_prediction.is_binary_probability(y=[[0.1], [0.8]]))
        self.assertTrue(_prediction.is_binary_probability(y=_np.array([[1.0], [0.0]])))

    def test_binary_probability_to_prediction(self):
        with self.assertRaises(ValueError):
            y_prob = [-0.1]
            _prediction.binary_probability_to_prediction(y=y_prob)

        with self.assertRaises(ValueError):
            y_prob = [[0.1, 0.9], [0.9, 0.1]]
            _prediction.binary_probability_to_prediction(y=y_prob)

        y_prob = [[0.4]]
        y_pred = _prediction.binary_probability_to_prediction(y=y_prob)
        _np.testing.assert_array_equal(y_pred, _np.array([[0]]))
        y_pred = _prediction.binary_probability_to_prediction(y=y_prob, maintain_shape=False)
        _np.testing.assert_array_equal(y_pred, _np.array([0]))

        y_prob = [[0.4], [0.8]]
        y_pred = _prediction.binary_probability_to_prediction(y=y_prob)
        _np.testing.assert_array_equal(y_pred, _np.array([[0], [1]]))
        y_pred = _prediction.binary_probability_to_prediction(y=y_prob, maintain_shape=False)
        _np.testing.assert_array_equal(y_pred, _np.array([0, 1]))

        y_prob = [[0.4], [0.6]]
        y_pred = _prediction.binary_probability_to_prediction(y=y_prob, threshold=0.39)
        _np.testing.assert_array_equal(y_pred, _np.array([[1], [1]]))
