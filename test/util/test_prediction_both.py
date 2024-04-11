from unittest import TestCase
import numpy as _np
import so_ml_tools.util.prediction as _prediction


class Test(TestCase):

    def test_probability_to_prediction_multiclass(self):
        # with self.assertRaises(ValueError):
        #     y_prob = [0.1, 0.9]
        #     _prediction.probability_to_prediction(y=y_prob)

        y_prob = [[0.1, 0.0, 0.9], [0.5, 0.2, 0.3]]
        y_pred = _prediction.probability_to_prediction(y=y_prob, maintain_shape=True)
        _np.testing.assert_array_equal(y_pred, _np.array([[2], [0]]))

    def test_probability_to_prediction_binary(self, **kwags):
        with self.assertRaises(ValueError):
            y_prob = [-0.1]
            _prediction.binary_probability_to_prediction(y=y_prob)

        y_prob = [[0.4]]
        y_pred = _prediction.binary_probability_to_prediction(y=y_prob)
        _np.testing.assert_array_equal(y_pred, _np.array([[0]]))
        y_pred = _prediction.probability_to_prediction(y=y_prob, maintain_shape=False)
        _np.testing.assert_array_equal(y_pred, _np.array([0]))

        y_prob = [[0.51]]
        y_pred = _prediction.binary_probability_to_prediction(y=y_prob)
        _np.testing.assert_array_equal(y_pred, _np.array([[1]]))
        y_pred = _prediction.probability_to_prediction(y=y_prob, maintain_shape=False)
        _np.testing.assert_array_equal(y_pred, _np.array([1]))
