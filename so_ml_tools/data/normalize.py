import numpy as _np


def normalize_vector(x: _np.array):
    """
    Normalizes an array containing vectors of x/y coordinates so that the array does not contain
    negative values.

    :param x: the vector containing values from -X to +X which need to be normalized between 0 and 1
    :return: the normalized vector.
    """
    x = x + (_np.abs(_np.min(x[:, 0])))
    x = x / _np.max(x[:, 0])
    x = x + (_np.abs(_np.min(x[:, 1])))
    return x / _np.max(x[:, 1])
