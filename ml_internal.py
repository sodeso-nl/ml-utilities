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


def sparse_labels(y):
    """
    DONE
    Converts multiclass dense label predictions to sparse labels:

    One-Hot:
    [
        [0.33, 0.78, 0.23],
        [0.64, 0.32, 0.11],
        [0.22, 0.36, 0.76]
    ]

    Ordinal:
    [1, 0, 2]

    :param y: Dense encoded labels
    :return: Sparse encoded labels
    """
    return np.argmax(y, axis=1)


def binarize_labels(y):
    """
    DONE
    Converts scaled classification prediction labels to binarized classification prediction labels:

    Scaled:
    [
        [0.56],
        [0.32],
        [0.98]
    ]

    Ordinal:
    [
        [1],
        [0],
        [1]
    ]

    :param y: Scaled classification prediction labels
    :return: Binarized classification prediction labels
    """
    return np.round(y)


def is_label_dense(y):
    """
    DONE
    Return True if the y value is a multiclass classification where the values represent the probability of the result
    to be that specific class, for example, when we have three possible outcomes:
    [
        [0.33, 0.78, 0.23],
        [0.64, 0.32, 0.11]
    ]

    :param y: the label to check if it is a multiclass classification prediction
    :return: True if it is a multiclass classification, False if not.
    """
    return y.ndim == 2 and len(y[0]) > 1


def is_label_scaled(y):
    """
    Return True if the y value is a scaled classification prediction where the value represents a probability between 0 and 1,
    for example, when we have three different predictions:
    [
        [0.56],
        [0.32],
        [0.98]
    ]

    :param y: the label to check if it is a scaled classification prediction
    :return: True if it is a scaled classification, False if not.
    """
    return y.ndim == 2 and len(y[0]) == 1


def convert_to_sparse_or_binary(y_true, y_pred):
    y_pred = convert_to_numpy_array_if_neccesary(y_pred)
    y_true = convert_to_numpy_array_if_neccesary(y_true)

    if is_label_dense(y_true):
        y_true = sparse_labels(y_true)

    if is_label_dense(y_pred):
        y_pred = sparse_labels(y_pred)
    elif is_label_scaled(y_pred):
        y_pred = binarize_labels(y_pred)

    return y_true, y_pred
