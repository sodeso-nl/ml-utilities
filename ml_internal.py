import numpy as np
import tensorflow as tf


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
    Converts multiclass dense label predictions to sparse labels:

    Dense:
    [
        [0.33, 0.78, 0.23],
        [0.64, 0.32, 0.11],
        [0.22, 0.36, 0.76]
    ]

    Sparse:
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


def is_image_float32_and_not_normalized(x):
    """
    Convenience method to check if an image in Tensor format is of type float32 but not normalized (values between
    0..255 instead of 0..1)
    :param x: Tensor containing the image(s)
    :return:
    """
    return x.dtype == tf.float32 and tf.math.reduce_max(x).numpy() > 1.0


def convert_to_sparse_or_binary(y):
    """
    Converts dense values to sparse or scaled values to binary.
    :param y:
    :return:
    """
    y = convert_to_numpy_array_if_neccesary(y)

    if is_label_dense(y):
        y = sparse_labels(y)
    elif is_label_scaled(y):
        y = binarize_labels(y)

    return y
