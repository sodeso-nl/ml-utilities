import numpy as _np


# Create function to view NumPy arrays as windows
def make_windows(x, window_size, horizon):
    """
    Turns 1 1D array into a 2D array of sequential labelled windows of window_size with horizon size labels.
    """
    # 1. Create a window of specific window_size (add the horizon on the end for labelling later)
    window_step = _np.expand_dims(_np.arange(window_size+horizon), axis=0)

    # 2. Create a 2D array for multiple window step (minus 1 to account for 0 indexing)
    window_indexes = window_step + _np.expand_dims(_np.arange(len(x)-(window_size + horizon - 1)), axis=0).T # Create 2D array of windows of window size window_size.

    # 3. Index on the target arraay (a time series) with 2D array of multiple widnow steps
    windowed_array = x[window_indexes]

    #4. get the labelled widnows
    windows, labels = windowed_array[:, :-horizon], windowed_array[:, -horizon:]
    return windows, labels
