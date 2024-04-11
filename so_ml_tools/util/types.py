import numpy as _np
import pandas as _pd
import tensorflow as _tf
from typing import Union as _Union


def to_numpy(value: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor]) -> _np.ndarray:
    """Converts input data to a NumPy array.

        Args:
            value (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
                Input data to be converted to a NumPy array.

        Returns:
            numpy.ndarray: Converted NumPy array.

        Raises:
            ValueError: If the input data type is unsupported.
    """
    if isinstance(value, list):
        return _np.array(value)
    elif isinstance(value, _np.ndarray):
        return value
    elif isinstance(value, (_pd.DataFrame, _pd.Series)):
        return value.to_numpy()
    elif isinstance(value, _tf.Tensor):
        return value.numpy()
    elif hasattr(value, 'numpy'):
        return value.numpy()
    elif hasattr(value, 'to_numpy'):
        return value.to_numpy()

    raise ValueError("Unsupported data type. Please provide a tensor, dataset, dataframe, or series.")