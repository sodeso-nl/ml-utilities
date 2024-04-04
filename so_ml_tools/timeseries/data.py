import tensorflow as _tf
import pandas as _pd
import keras as _ks
import numpy as _np


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
        batch_size=None):
    return dataset_from_array_multivariate(
        array=tensor.numpy(),
        label_column_idx=label_column_idx,
        window_size=window_size,
        horizon_size=horizon_size,
        batch_size=batch_size)


def dataset_from_dataframe_multivariate(
        dataframe: _pd.DataFrame,
        label_column: str,
        window_size: int,
        horizon_size: int = 1,
        batch_size=None):
    label_idx = dataframe.columns.get_loc(label_column)
    return dataset_from_array_multivariate(
        array=dataframe.to_numpy(),
        label_column_idx=label_idx,
        window_size=window_size,
        horizon_size=horizon_size,
        batch_size=batch_size)


def dataset_from_array_multivariate(
        array: _np.ndarray,
        label_column_idx,
        window_size: int,
        horizon_size: int = 1,
        batch_size=None) -> _tf.data.Dataset:
    x = array
    y = x[:, label_column_idx]

    return dataset_from_arrays_multivariate(
        x=x,
        y=y,
        window_size=window_size,
        horizon_size=horizon_size,
        batch_size=batch_size)


def dataset_from_arrays_multivariate(
        x: _np.ndarray,
        y: _np.ndarray,
        window_size: int,
        horizon_size: int = 1,
        batch_size=None) -> _tf.data.Dataset:
    if horizon_size > 1:
        horizon_start_offset = window_size
        horizon_end_offset = window_size + horizon_size
        y = [y[start_idx + horizon_start_offset:start_idx + horizon_end_offset] for start_idx, _ in
             enumerate(y[window_size:])]

        # Remove last set of lines which are not complete
        y = y[0:-horizon_size + 1]

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
        batch_size=None) \
        -> _tf.data.Dataset:
    return dataset_from_array_univariate(
        array=tensor.numpy().ravel(),
        window_size=window_size,
        horizon_size=horizon_size,
        batch_size=batch_size)


def dataset_from_dataframe_univariate(
        dataframe: _pd.DataFrame,
        window_size: int,
        horizon_size: int = 1,
        batch_size=None) \
        -> _tf.data.Dataset:
    return dataset_from_array_univariate(
        array=dataframe.to_numpy().ravel(),
        window_size=window_size,
        horizon_size=horizon_size,
        batch_size=batch_size)


def dataset_from_array_univariate(
        array: _np.ndarray,
        window_size: int,
        horizon_size: int = 1,
        batch_size=None) -> _tf.data.Dataset:
    x = array.ravel()
    y = x[window_size:]

    return dataset_from_arrays_univariate(
        x=x,
        y=y,
        window_size=window_size,
        horizon_size=horizon_size,
        batch_size=batch_size)


def dataset_from_arrays_univariate(
        x: _np.ndarray,
        y: _np.ndarray,
        window_size: int,
        horizon_size: int = 1,
        batch_size=None) -> _tf.data.Dataset:
    x = x.ravel()
    if horizon_size > 1:
        horizon_start_offset = window_size
        horizon_end_offset = window_size + horizon_size
        y = [x[start_idx + horizon_start_offset:start_idx + horizon_end_offset]
             for start_idx, _ in enumerate(y)]

        # Remove last set of lines which are not complete
        y = y[0:-horizon_size + 1]

    return _ks.utils.timeseries_dataset_from_array(
        x,
        y,
        sequence_length=window_size,
        batch_size=batch_size)


# def make_windows_univariate(x, window_size, horizon):
#     """
#     Turns 1 1D array into a 2D array of sequential labelled windows of window_size with horizon size labels.
#
#     E.g. if horizon = 1 and window_size = 3, then
#     X = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#
#     Output:
#         Features:
#         [
#             [0, 1, 2, 3, 4, 5, 6]
#             [1, 2, 3, 4, 5, 6, 7]
#         ]
#
#         Labels:
#         [
#             [7]
#             [8]
#         ]
#
#     Args:
#         x: A 1D Numpy array or Tensor or a single column Pandas dataframe
#         window_size: the number of time steps to use as features
#         horizon:  the number of time steps to use as labels
#
#     """
#     if not isinstance(x, _tf.Tensor):
#         x = _tf.convert_to_tensor(value=x)
#
#     x = x.numpy().ravel()
#
#     # 1. Create a window of specific window_size (add the horizon on the end for labelling later)
#     window_step = _np.expand_dims(_np.arange(window_size + horizon), axis=0)
#
#     # 2. Create a 2D array for multiple window step (minus 1 to account for 0 indexing)
#     window_indexes = window_step + _np.expand_dims(_np.arange(len(x) - (window_size + horizon - 1)),
#                                                   axis=0).T  # Create 2D array of windows of window size window_size.
#
#     # 3. Index on the target array (a time series) with 2D array of multiple window steps
#     windowed_array = x[window_indexes]
#
#     # 4. get the labelled windows
#     windows, labels = _get_labelled_windows(windowed_array, horizon=horizon)
#     return windows, labels


# def _get_labelled_windows(x, horizon):
#     """
#     Creates labels for windowed dataset.
#
#     E.g. if horizon = 1
#     Input: [0, 1, 2, 3, 4, 5, 6, 7] -> Output: ([0, 1, 2, 3, 4, 5, 6], [7])
#     """
#     return x[:, :-horizon], x[:, -horizon:]