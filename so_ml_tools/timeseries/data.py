import tensorflow as _tf
import numpy as _np


def get_labelled_windows(x, horizon):
    """
    Creates labels for windowed dataset.

    E.g. if horizon = 1
    Input: [0, 1, 2, 3, 4, 5, 6, 7] -> Output: ([0, 1, 2, 3, 4, 5, 6], [7])
    """
    return x[:, :-horizon], x[:, -horizon:]


def make_windows(x, window_size, horizon):
    """
    Turns 1 1D array into a 2D array of sequential labelled windows of window_size with horizon size labels.
    """
    if not isinstance(x, _tf.Tensor):
        x = _tf.convert_to_tensor(value=x)

    # Check that we only have a single feature
    assert x.ndim == 1, f"x contains multiple features {x.ndim}, this method only handles a single feature, use x['col'] to use a single feature."

    # Convert the object to a numpy object so we can use slicing.
    x = x.numpy()

    # 1. Create a window of specific window_size (add the horizon on the end for labelling later)
    window_step = _np.expand_dims(_np.arange(window_size + horizon), axis=0)

    # 2. Create a 2D array for multiple window step (minus 1 to account for 0 indexing)
    window_indexes = window_step + _np.expand_dims(_np.arange(len(x) - (window_size + horizon - 1)),
                                                  axis=0).T  # Create 2D array of windows of window size window_size.

    # 3. Index on the target arraay (a time series) with 2D array of multiple widnow steps
    windowed_array = x[window_indexes]

    # 4. get the labelled widnows
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
    return windows, labels
