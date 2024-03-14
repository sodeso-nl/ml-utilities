import tensorflow as _tf
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


def make_windows(x, window_size, horizon):
    """
    Turns 1 1D array into a 2D array of sequential labelled windows of window_size with horizon size labels.

    E.g. if horizon = 1 and window_size = 3, then
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    Output:
        Features:
        [
            [0, 1, 2, 3, 4, 5, 6]
            [1, 2, 3, 4, 5, 6, 7]
        ]

        Labels:
        [
            [7]
            [8]
        ]

    Args:
        x: A 1D Numpy array or Tensor or a single column Pandas dataframe
        window_size: the number of time steps to use as features
        horizon:  the number of time steps to use as labels

    """
    if not isinstance(x, _tf.Tensor):
        x = _tf.convert_to_tensor(value=x)

    # Check that we only have a single feature
    assert x.ndim == 1, (f"Shape of x is not correct {x.shape}, should be 1D array (ie. [1, 2, 3, 4]), please check "
                         f"the following:"
                         f"* Does your data contain multiple columns, if so, select only a single column, if it is a "
                         f"Pandas dataframe then use df['col'], if it is a tensor use slicing, for example t[:,2] to "
                         f"select the third column, If your data is column based then convert it to row based "
                         f"using tf.squeeze()")

    x = x.numpy()

    # 1. Create a window of specific window_size (add the horizon on the end for labelling later)
    window_step = _np.expand_dims(_np.arange(window_size + horizon), axis=0)

    # 2. Create a 2D array for multiple window step (minus 1 to account for 0 indexing)
    window_indexes = window_step + _np.expand_dims(_np.arange(len(x) - (window_size + horizon - 1)),
                                                  axis=0).T  # Create 2D array of windows of window size window_size.

    # 3. Index on the target array (a time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]

    # 4. get the labelled windows
    windows, labels = _get_labelled_windows(windowed_array, horizon=horizon)
    return windows, labels


def _get_labelled_windows(x, horizon):
    """
    Creates labels for windowed dataset.

    E.g. if horizon = 1
    Input: [0, 1, 2, 3, 4, 5, 6, 7] -> Output: ([0, 1, 2, 3, 4, 5, 6], [7])
    """
    return x[:, :-horizon], x[:, -horizon:]