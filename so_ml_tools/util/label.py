import numpy as _np
import tensorflow as _tf
import tensorflow.keras as _ks
import so_ml_tools as _soml


def to_categorical(y, num_classes=None, dtype=None):
    """
    Converts a class vector (integers) to binary class matrix (one-hot encoded).

    Args:
        y (array-like): Class vector to be converted into a matrix (integers from 0 to num_classes - 1).
        num_classes (int, optional): Total number of classes. If not provided, the number of classes is inferred from
            the input data. Default is None.
        dtype (str or numpy.dtype, optional): Data type of the output matrix. If not provided, the default data type
            is used. Default is None.

    Returns:
        numpy.ndarray: Binary matrix representation of the input class vector. Each row corresponds to one sample, and
        each column corresponds to one class. The value at (i, j) represents whether sample i belongs to class j,
        where 1 indicates membership and 0 indicates non-membership.

    Raises:
        ValueError: If y contains values outside the range [0, num_classes).
        ValueError: If num_classes is not provided and cannot be inferred from the input data.

    Examples:
        >>> y = [0, 1, 2, 1]
        >>> to_categorical(y)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.],
               [0., 1., 0.]], dtype=float32)

    References:
        - Original implementation: https://github.com/keras-team/keras/blob/main/keras/utils/np_utils.py#L11

    """
    return _ks.utils.to_categorical(y, num_classes, dtype=dtype)


def diff_indexes(y_true, y_pred=None, y_prob=None):
    if isinstance(y_true, _tf.data.Dataset):
        y_true = _soml.tf.dataset.get_labels(dataset=y_true)

    y_true = _soml.util.prediction.probability_to_prediction(y=y_true)
    if y_pred is None and y_prob is not None:
        y_pred = _soml.util.prediction.probability_to_prediction(y=y_prob)
    elif y_pred is None and y_prob is None:
        raise "Must specify 'y_pred' or 'y_prob'"

    compare_to = y_pred

    # Check if the shapes match, if not, try to fix it.
    if y_pred.shape != y_true.shape:
        compare_to = _np.reshape(y_pred, newshape=y_true.shape)

    compared_result = y_true == compare_to
    return _np.where(compared_result == False)[0]
