from unittest import TestCase
import numpy as _np
import so_ml_tools.util.prediction as _prediction


class Test(TestCase):

    def test_is_multiclass_classification(self):
        y = _np.array([[0, 1], [1, 0], [0, 0]])
        self.assertTrue(_prediction.is_multiclass_classification(y))

        y = _np.array([[0], [1], [0], [1]])
        self.assertFalse(_prediction.is_multiclass_classification(y))

        y = _np.array([0, 1, 0, 1])
        self.assertFalse(_prediction.is_multiclass_classification(y))

        y = _np.array([])
        self.assertFalse(_prediction.is_multiclass_classification(y))

    def test_is_multiclass_propabilities(self):
        y_prob = _np.array([0.2, 0.3, 0.5])
        self.assertFalse(_prediction.is_multiclass_propabilities(y_prob))

        y_prob = _np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
        self.assertTrue(_prediction.is_multiclass_propabilities(y_prob))

        y_prob = _np.array([[0.4], [0.9]])
        self.assertFalse(_prediction.is_multiclass_propabilities(y_prob))

    def test_multiclass_probability_to_prediction(self):
        with self.assertRaises(ValueError):
            y_prob = [0.1, 0.9]
            _prediction.multiclass_probability_to_prediction(y=y_prob)

        with self.assertRaises(ValueError):
            y_prob = [[0.1], [0.9]]
            _prediction.multiclass_probability_to_prediction(y=y_prob)

        y_prob = [[0.1, 0.9]]
        y_pred = _prediction.multiclass_probability_to_prediction(y=y_prob)
        _np.testing.assert_array_equal(y_pred, _np.array([1]))

        y_prob = [[0.1, 0.0, 0.9], [0.5, 0.2, 0.3]]
        y_pred = _prediction.multiclass_probability_to_prediction(y=y_prob)
        _np.testing.assert_array_equal(y_pred, _np.array([2, 0]))

        y_prob = [[0.1, 0.0, 0.9], [0.5, 0.2, 0.3]]
        y_pred = _prediction.multiclass_probability_to_prediction(y=y_prob, maintain_shape=True)
        _np.testing.assert_array_equal(y_pred, _np.array([[2], [0]]))
