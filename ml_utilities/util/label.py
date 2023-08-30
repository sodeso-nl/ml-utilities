import numpy as np
import tensorflow as tf

from typing import Union


def is_multiclass_classification(y_prob: Union[tf.Tensor, np.array]) -> bool:
    """Return True if the y value is a multiclass classification where the values represent the probability of the result
    to be that specific class, for example, when we have three possible outcomes:
    [
        [0.33, 0.78, 0.23],
        [0.64, 0.32, 0.11]
    ]

    Args:
        y_prob: the label to check if it is a multiclass classification prediction

    Returns:
        True if it is a multiclass classification, False if not.

    Raises:
        TypeError: If `x` is neither a tf.Tensor or np.array.
    """
    if tf.is_tensor(y_prob):
        return y_prob.get_shape().ndims == 2 and y_prob.shape[1] > 1
    elif isinstance(y_prob, np.ndarray):
        return y_prob.ndim == 2 and y_prob.shape[1] > 1

    raise TypeError('y should be of type tf.Tensor or np.array.')


def is_binary_classification(y_prob: Union[tf.Tensor, np.array]) -> bool:
    """Return True if the y value is a binary classification prediction where the value represents a probability between 0 and 1,
    for example, when we have three different predictions:
    [
        [0.56],
        [0.32],
        [0.98]
    ]

    Args:
        y_prob: the label to check if it is a binary classification prediction

    Returns:
        True if it is a binary classification, False if not.

    Raises:
        TypeError: If `x` is neither a tf.Tensor or np.array.
    """
    if tf.is_tensor(y_prob):
        return y_prob.get_shape().ndims == 2 and y_prob.shape[1] == 1
    elif isinstance(y_prob, np.ndarray):
        return y_prob.ndim == 2 and y_prob.shape[1] == 1

    raise TypeError('y should be of type tf.Tensor or np.array.')


def probability_to_class(y_prob: Union[tf.Tensor, np.array]) -> Union[tf.Tensor, np.array]:
    """Converts multiclass dense label predictions to sparse labels:

    Probability:
    [
        [0.33, 0.78, 0.23],
        [0.64, 0.32, 0.11],
        [0.22, 0.36, 0.76]
    ]

    Class:
    [1, 0, 2]

    Args:
        y_prob: the probabilities matrix either a 'tf.Tensor' or a 'np.array'

    Returns:
        The sparse encoded classes

    Raises:
        TypeError: If `x` is neither a tf.Tensor or np.array.
    """
    if tf.is_tensor(y_prob):
        return tf.argmax(y_prob, axis=1)
    elif isinstance(y_prob, np.ndarray):
        return np.argmax(y_prob, axis=1)

    raise TypeError('y should be of type tf.Tensor or np.array.')


def probability_to_binary(y_prob: Union[tf.Tensor, np.array]):
    """
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

    Args:
        y_prob: the probabilities matrix either a 'tf.Tensor' or a 'np.array'
        dtype: (optional) the destination type.

    Returns:
        Binarized prediction labels

    Raises:
        TypeError: If `x` is neither a tf.Tensor or np.array.
    """
    if tf.is_tensor(y_prob):
        y1 = tf.round(y_prob)
        return tf.cast(y1, dtype=tf.int8)
    elif isinstance(y_prob, np.ndarray):
        y1 = np.round(a=y_prob, decimals=0)
        return y1.astype(int)

    raise TypeError('y should be of type tf.Tensor or np.array.')


def to_prediction(y_prob: Union[tf.Tensor, np.array]) -> Union[tf.Tensor, np.array]:
    """
    Determines if the probability is a multiclass or binary classification and then will
    return the prediction in either class or binary form.

    Args:
        y_prob: the probabilities matrix either a 'tf.Tensor' or a 'np.array'
        dtype:(optional, only applicable for binary classification) parameter to change the dtype

    Return:
        A tensor or numpy array with the prediction.
    """
    if is_multiclass_classification(y_prob=y_prob):
        y_prob = probability_to_class(y_prob=y_prob)
    elif is_binary_classification(y_prob=y_prob):
        y_prob = probability_to_binary(y_prob=y_prob)

    return y_prob
