import numpy as np


def convert_to_numpy_array_if_neccesary(value):
    """
    Converts the given object to a Numpy Array (when possible)

    :param value: The object to convert
    :return: The Numpy array
    """
    if not isinstance(value, np.ndarray):
        return np.array(value)

    return value


def to_ordinal(y):
    """
    Converts sparse encoded labels to ordinal labels:

    One-Hot:
    [
        [0.33, 0.78, 0.23],
        [0.64, 0.32, 0.11]
    ]

    Ordinal:
    [1, 0]

    :param y: Sparse encoded labels
    :return: Ordinal encoded labels
    """
    return np.argmax(y, axis=1)


def to_binary(y):
    """
    Converts predicted encoded labels with a value between 0 & 1 to 0 or 1.:

    Ordinal:
    [0.33, 0.67, 0.45, 0.99]

    Ordinal:
    [0, 1, 0, 1]

    :param y: Predicted encoded labels
    :return: Ordinal encoded labels
    """
    return np.round(y)


def is_multiclass_classification(y):
    """
    Return True if the y value is a multiclass classification, for example:
    [
        [0.33, 0.78, 0.23],
        [0.64, 0.32, 0.11]
    ]

    :param y: the label to check if it is a multiclass classification
    :return: True if it is multiclass classification, False if not.
    """
    return y.ndim == 2 and len(y[0]) > 1


def is_binary_classification(y):
    """
    Return True if the y value is a binary classification, for example:
    [
        [0.56],
        [0.32],
        [0.98]
    ]

    :param y: the label to check if it is a binary classification
    :return: True if it is binary classification, False if not.
    """
    return y.ndim == 2 and len(y[0]) == 1