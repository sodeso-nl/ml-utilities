import numpy as _np
import tensorflow as _tf

from typing import Union as _Union


def is_multiclass_classification(y_prob: any) -> bool:
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
        TypeError: If `y_prob` is neither a 'list', 'tf.Tensor' or 'np.array'.
    """
    assert y_prob is not None, "y_prob is null"

    if isinstance(y_prob, list):
        return len(y_prob) > 0 and isinstance(y_prob[0], (list | _np.ndarray)) and len(y_prob[0]) > 1  # [[0.60, 0.10, 0.30], [....]]
    elif _tf.is_tensor(y_prob):
        return y_prob.get_shape().ndims == 2 and y_prob.shape[1] > 1  # [[0.60, 0.10, 0.30], [....]]
    elif isinstance(y_prob, _np.ndarray):
        return y_prob.ndim == 2 and y_prob.shape[1] > 1  # [[0.60, 0.10, 0.30], [....]]

    raise TypeError('y should be of type tf.Tensor or np.array.')


def is_binary_classification(y_prob: any) -> bool:
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
        TypeError: If `y_prob` is neither a  'list', 'tf.Tensor' or 'np.array'.
    """
    assert y_prob is not None, "y_prob is null"

    if isinstance(y_prob, list):
        if len(y_prob) > 0 and isinstance(y_prob[0], (list | _np.ndarray)):
            if len(y_prob[0]) == 1: # [[1], [0], [1]]
                if 0 <= _np.max(y_prob) <= 2:
                    return True
        elif len(y_prob) > 0: # [0, 1, 1, 0, 1]
            if 0 <= max(y_prob) <= 2:
                return True

        return False
    if _tf.is_tensor(y_prob):
        if y_prob.get_shape().ndims == 2 and y_prob.shape[1] == 1 and 0 <= _tf.math.reduce_max(y_prob) <= 2:  # [[1], [0], [1]]
            return True
        elif y_prob.get_shape().ndims == 1 and 0 <= _tf.math.reduce_max(y_prob) <= 2:  # [0, 1, 1, 0, 1]
            return True

        return False
    elif isinstance(y_prob, _np.ndarray):
        if y_prob.ndim == 2 and y_prob.shape[1] == 1 and 0 <= _np.max(y_prob) <= 2:  # [[1], [0], [1]]
            return True
        elif y_prob.ndim == 1 and 0 <= _np.max(y_prob) <= 2:  # [0, 1, 1, 0, 1]
            return True

        return False

    raise TypeError('y_prob should be of type list, tf.Tensor or np.array.')


def probability_to_class(y_prob: any) -> any:
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
        y_prob: the probabilities matrix either a 'list', 'tf.Tensor' or a 'np.array'

    Returns:
        The sparse encoded classes

    Raises:
        TypeError: If `y_prob` is neither a 'list', 'tf.Tensor' or 'np.array'.
    """
    assert y_prob is not None, "y_prob is null"

    if isinstance(y_prob, list):
        return list(_np.argmax(y_prob, axis=1))
    elif _tf.is_tensor(y_prob):
        return _tf.argmax(y_prob, axis=1)
    elif isinstance(y_prob, _np.ndarray):
        return _np.argmax(y_prob, axis=1)

    raise TypeError('y_prob should be of type list, tf.Tensor or np.array.')


def probability_to_binary(y_prob: any) -> any:
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
        y_prob: the probabilities matrix either a 'list', 'tf.Tensor' or a 'np.array'
        dtype: (optional) the destination type.

    Returns:
        Binarized prediction labels

    Raises:
        TypeError: If `y_prob` is neither a 'list', 'tf.Tensor' or 'np.array'.
    """
    assert y_prob is not None, "y_prob is null"

    if isinstance(y_prob, list):
        return list(_np.round(y_prob))
    if _tf.is_tensor(y_prob):
        y1 = _tf.round(y_prob)
        return _tf.cast(y1, dtype=_tf.int8)
    elif isinstance(y_prob, _np.ndarray):
        y1 = _np.round(a=y_prob, decimals=0)
        return y1.astype(int)

    raise TypeError('y_prob should be of type list, tf.Tensor or np.array.')


def to_prediction(y_prob: any) -> any:
    """
    Determines if the probability is a multiclass or binary classification and then will
    return the prediction in either class or binary form.

    Args:
        y_prob: the probabilities matrix either a 'list', 'tf.Tensor' or a 'np.array'
        dtype:(optional, only applicable for binary classification) parameter to change the dtype

    Return:
        A tensor or numpy array with the prediction.
    """
    assert y_prob is not None, "y_prob is null"

    if is_multiclass_classification(y_prob=y_prob):
        y_prob = probability_to_class(y_prob=y_prob)
    elif is_binary_classification(y_prob=y_prob):
        y_prob = probability_to_binary(y_prob=y_prob)

    return y_prob


def is_one_hot(x):
    """
    Check if the given x is a one-hot encoded value.

    Args:
        x: the value to check
    """
    return (x.sum(axis=1)-_np.ones(x.shape[0])).sum() == 0
