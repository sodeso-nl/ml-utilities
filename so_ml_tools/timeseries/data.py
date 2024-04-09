import tensorflow as _tf
import pandas as _pd
import keras as _ks
import numpy as _np
import math as _math
import so_ml_tools as _soml

from typing import Union as _Union


def make_naive_predictions(y_true):
    """
    Method that creates a naive prediction from the given truth labels:

    Input:
        [0, 1, 2, 3, 4, 5, 6, 7]

    Output:
        [1, 2, 3, 4, 5, 6, 7]
        [0, 1, 2, 3, 4, 5, 6]


    Args:
        y_true: the truth labels for which we want to make a naive prediction

    Returns:
        y_true and the naive prediction based on the given truth labels

    """
    naive_predictions = y_true[:-1]
    return y_true[1:], naive_predictions


def dataset_from_array_multivariate(
        data: _Union[list, _np.ndarray, _pd.DataFrame, _tf.Tensor],
        label_column_idx,
        window_size: int,
        horizon_size: int = 1,
        centered: bool = False,
        batch_size=None) -> _tf.data.Dataset:
    """Creates a TensorFlow dataset from a multivariate time series array.

        Args:
            data (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
                Input array to be checked.: Input containing the multivariate time series data.
            label_column_idx (int): Index of the column representing the target label.
            window_size (int): Size of the sliding window used for creating sequences.
            horizon_size (int, optional): Size of the prediction horizon. Defaults to 1.
            centered (bool, optional): Whether to center the sliding window around each label.
                Defaults to False.
            batch_size (int, optional): Number of samples per batch. Defaults to None.

        Returns:
            tensorflow.data.Dataset: TensorFlow dataset containing the input sequences and corresponding labels.

        Raises:
            AssertionError: If horizon_size is greater than window_size or if centered is True but window_size is too small.

        Examples:
            >>> import numpy as np
            >>> import tensorflow as tf
            >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
            >>> dataset = dataset_from_array_multivariate(array, label_column_idx=0, window_size=2, horizon_size=1)
            >>> for batch in dataset:
            ...     print(batch)
            (array([[[1, 2, 3], [4, 5, 6]]]), array([[7], [10]]))
        """
    assert horizon_size < window_size, f"horizon_size '{horizon_size}' must be smaller then window_size '{window_size}'"
    assert not centered or (
            centered and window_size > 2), f"Centered window requested but window_size is too small: '{window_size}', needs to be at least 3 or higher."

    x = _soml.util.types.to_numpy(data)

    label_offset = window_size
    if centered:
        label_offset = _math.floor(window_size / 2)

    y = x[label_offset:, label_column_idx]

    if horizon_size > 1:
        y = [y[start_idx:start_idx + horizon_size] for start_idx, _ in enumerate(y)]

        # Remove last set of lines which are not complete
        y = y[0:-(window_size - horizon_size)]

    return _ks.utils.timeseries_dataset_from_array(
        x,
        y,
        sequence_length=window_size,
        sequence_stride=1,
        batch_size=batch_size)


def dataset_from_array_univariate(
        data: _Union[list, _np.ndarray, _pd.DataFrame, _tf.Tensor],
        window_size: int,
        horizon_size: int = 1,
        centered: bool = False,
        batch_size=None) -> _tf.data.Dataset:
    """Creates a TensorFlow dataset from a univariate time series array.

        Args:
            data (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor): Data array containing the univariate time series data.
            window_size (int): Size of the sliding window used for creating sequences.
            horizon_size (int, optional): Size of the prediction horizon. Defaults to 1.
            centered (bool, optional): Whether to center the sliding window around each label.
                Defaults to False.
            batch_size (int, optional): Number of samples per batch. Defaults to None.

        Returns:
            tensorflow.data.Dataset: TensorFlow dataset containing the input sequences and corresponding labels.

        Raises:
            AssertionError: If horizon_size is greater than window_size or if centered is True but window_size is too small.

        Examples:
            >>> import numpy as np
            >>> import tensorflow as tf
            >>> array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
            >>> dataset = dataset_from_array_univariate(array, window_size=3, horizon_size=1)
            >>> for batch in dataset:
            ...     print(batch)
            (array([[1, 2, 3]]), array([4]))
        """
    assert horizon_size <= window_size, f"horizon_size '{horizon_size}' must be smaller or equal to window_size '{window_size}'"
    assert not centered or (
            centered and window_size > 2), f"Centered window requested but window_size is too small: '{window_size}', needs to be at least 3 or higher."

    x = _soml.util.types.to_numpy(data).ravel()

    label_offset = window_size
    if centered:
        label_offset = _math.floor(window_size / 2)

    y = x[label_offset:]

    if horizon_size > 1:
        y = [y[start_idx:start_idx + horizon_size] for start_idx, _ in enumerate(y)]
        # Remove last set of labels which are not complete horizons
        y = y[:-(window_size - horizon_size)]

    return _ks.utils.timeseries_dataset_from_array(
        x,
        y,
        sequence_length=window_size,
        batch_size=batch_size)
