import tensorflow as _tf
import pandas as _pd
import keras as _ks
import numpy as _np
import math as _math


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


def dataset_from_tensor_multivariate(
        tensor: _tf.Tensor,
        label_column_idx,
        window_size,
        horizon_size=1,
        centered=False,
        batch_size=None):
    return dataset_from_array_multivariate(
        array=tensor.numpy(),
        label_column_idx=label_column_idx,
        window_size=window_size,
        horizon_size=horizon_size,
        centered=centered,
        batch_size=batch_size)


def dataset_from_dataframe_multivariate(
        dataframe: _pd.DataFrame,
        label_column: str,
        window_size: int,
        horizon_size: int = 1,
        centered: bool = False,
        batch_size=None):
    label_idx = dataframe.columns.get_loc(label_column)
    return dataset_from_array_multivariate(
        array=dataframe.to_numpy(),
        label_column_idx=label_idx,
        window_size=window_size,
        horizon_size=horizon_size,
        centered=centered,
        batch_size=batch_size)


def dataset_from_array_multivariate(
        array: _np.ndarray,
        label_column_idx,
        window_size: int,
        horizon_size: int = 1,
        centered: bool = False,
        batch_size=None) -> _tf.data.Dataset:
    assert horizon_size <= window_size, f"horizon_size '{horizon_size}' must be smaller or equal to window_size '{window_size}'"
    assert not centered or (
            centered and window_size > 2), f"Centered window requested but window_size is too small: '{window_size}', needs to be at least 3 or higher."

    x = array

    label_offset = window_size
    if centered:
        label_offset = _math.floor(window_size / 2)

    y = x[label_offset:, label_column_idx]

    if horizon_size > 1:
        horizon_start_offset = label_offset
        horizon_end_offset = label_offset + horizon_size
        y = [y[start_idx + horizon_start_offset:start_idx + horizon_end_offset]
             for start_idx, _ in enumerate(y)]

        # Remove last set of lines which are not complete
        y = y[0:-window_size]

    return _ks.utils.timeseries_dataset_from_array(
        x,
        y,
        sequence_length=window_size,
        sequence_stride=1,
        batch_size=batch_size)


def dataset_from_tensor_univariate(
        tensor: _tf.Tensor,
        window_size: int,
        horizon_size: int = 1,
        centered: bool = False,
        batch_size=None) \
        -> _tf.data.Dataset:
    """
    Creates a TensorFlow Dataset from a given tensor object with the specified window size and horizon size. To
    center the labels set the `centered=True` argument.

    Args:
        tensor: the tensor object
        window_size: the size of the window
        horizon_size: the size of the horizon (smaller or equal to window_size)
        centered: flag to indicate if the labels should be centered in relation to the window
        batch_size: batch size for the dataset

    Returns:
        the dataset.
    """
    return dataset_from_array_univariate(
        array=tensor.numpy().ravel(),
        window_size=window_size,
        horizon_size=horizon_size,
        centered=centered,
        batch_size=batch_size)


def dataset_from_dataframe_univariate(
        dataframe: _pd.DataFrame,
        window_size: int,
        horizon_size: int = 1,
        centered: bool = False,
        batch_size=None) \
        -> _tf.data.Dataset:
    """
    Creates a TensorFlow Dataset from a given dataframe object with the specified window size and horizon size. To
    center the labels set the `centered=True` argument.

    Args:
        dataframe: the dataframe object
        window_size: the size of the window
        horizon_size: the size of the horizon (smaller or equal to window_size)
        centered: flag to indicate if the labels should be centered in relation to the window
        batch_size: batch size for the dataset

    Returns:
        the dataset.
    """
    return dataset_from_array_univariate(
        array=dataframe.to_numpy().ravel(),
        window_size=window_size,
        horizon_size=horizon_size,
        centered=centered,
        batch_size=batch_size)


def dataset_from_array_univariate(
        array: _np.ndarray,
        window_size: int,
        horizon_size: int = 1,
        centered: bool = False,
        batch_size=None) -> _tf.data.Dataset:
    """
    Creates a TensorFlow Dataset from a given array object with the specified window size and horizon size. To
    center the labels set the `centered=True` argument.

    Args:
        array: the array object
        window_size: the size of the window
        horizon_size: the size of the horizon (smaller or equal to window_size)
        centered: flag to indicate if the labels should be centered in relation to the window
        batch_size: batch size for the dataset

    Returns:
        the dataset.
    """
    assert horizon_size <= window_size, f"horizon_size '{horizon_size}' must be smaller or equal to window_size '{window_size}'"
    assert not centered or (
                centered and window_size > 2), f"Centered window requested but window_size is too small: '{window_size}', needs to be at least 3 or higher."

    x = array.ravel()

    label_offset = window_size
    if centered:
        label_offset = _math.floor(window_size / 2)

    y = x[label_offset:]

    if horizon_size > 1:
        horizon_start_offset = label_offset
        horizon_end_offset = label_offset + horizon_size
        y = [x[start_idx + horizon_start_offset:start_idx + horizon_end_offset]
             for start_idx, _ in enumerate(y)]

        # Remove last set of labels which are not complete horizons
        y = y[:-window_size + 1]

    return _ks.utils.timeseries_dataset_from_array(
        x,
        y,
        sequence_length=window_size,
        batch_size=batch_size)
