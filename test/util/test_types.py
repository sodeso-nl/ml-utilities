from unittest import TestCase

import pandas as _pd
import numpy as _np
import tensorflow as _tf
import so_ml_tools as _soml


class Test(TestCase):
    def test_to_numpy_pandas_series(self):
        series = _pd.Series(data=[1, 2, 3])
        result = _soml.util.types.to_numpy(series)
        expected = _np.array([1, 2, 3])
        _np.testing.assert_array_equal(result, expected)

    def test_to_numpy_pandas_dataframe(self):
        pandas = _pd.DataFrame(data=[1, 2, 3])
        result = _soml.util.types.to_numpy(pandas)
        expected = _np.array([[1], [2], [3]])
        _np.testing.assert_array_equal(result, expected)

    def test_to_numpy_tensorflow_tensors(self):
        tensor = _tf.constant([1, 2, 3])
        result = _soml.util.types.to_numpy(tensor)
        expected = _np.array([1, 2, 3])
        _np.testing.assert_array_equal(result, expected)

    def test_to_numpy_list_1D(self):
        list = [1, 2, 3]
        result = _soml.util.types.to_numpy(list)
        expected = _np.array([1, 2, 3])
        _np.testing.assert_array_equal(result, expected)

    def test_to_numpy_list_2D(self):
        list = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        result = _soml.util.types.to_numpy(list)
        expected = _np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        _np.testing.assert_array_equal(result, expected)