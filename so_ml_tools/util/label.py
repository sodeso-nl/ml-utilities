import numpy as _np
import tensorflow as _tf
import tensorflow.keras as _ks
import so_ml_tools as _soml


def to_categorical(y, num_classes=None, dtype=None):
    """
    Converts a class vector (integers) to binary class matrix.

    E.g. for use with `categorical_crossentropy`.

    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
          as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.
    """
    return _ks.utils.to_categorical(y, num_classes, dtype=dtype)


def diff_indexes(y_true, y_pred=None, y_prob=None):
    if isinstance(y_true, _tf.data.Dataset):
        y_true = _soml.tf.dataset.get_labels(dataset=y_true)

    y_true = _soml.util.prediction.probability_to_prediction(y_probs=y_true)
    if y_pred is None and y_prob is not None:
        y_pred = _soml.util.prediction.probability_to_prediction(y_probs=y_prob)
    elif y_pred is None and y_prob is None:
        raise "Must specify 'y_pred' or 'y_prob'"

    compare_to = y_pred

    # Check if the shapes match, if not, try to fix it.
    if y_pred.shape != y_true.shape:
        compare_to = _np.reshape(y_pred, newshape=y_true.shape)

    compared_result = y_true == compare_to
    return _np.where(compared_result == False)[0]
