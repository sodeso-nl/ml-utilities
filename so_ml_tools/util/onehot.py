import numpy as _np
import so_ml_tools as _soml


def is_one_hot_encoded(value) -> bool:
    """Check if the input array represents a one-hot encoded array.

    Args:
        value (numpy.ndarray or any): Input array to be checked.

    Returns:
        bool: True if the input array represents a one-hot encoded array, False otherwise.

    Raises:
        None

    Examples:
        >>> import numpy as np
        >>> is_one_hot_encoded(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        True
        >>> is_one_hot_encoded(np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]))
        False
    """
    if not isinstance(value, _np.ndarray):
        value = _soml.util.types.to_numpy(value=value)

    if len(value.shape) != 2:
        return False

    if not _np.all((value == 0) | (value == 1)):
        return False

    if not _np.all(_np.sum(value, axis=1) == 1):
        return False

    return True


def one_hot_to_indices(value) -> _np.ndarray:
    """Converts a one-hot encoded array to a list of indices.

    Args:
        value (numpy.ndarray or any): Input array to be converted.

    Returns:
        numpy.ndarray: Array of indices corresponding to the one-hot encoding.

    Raises:
        ValueError: If the input is not a one-hot encoded array or not a one-dimensional NumPy array.

    Examples:
        >>> import numpy as np
        >>> one_hot_to_indices(np.array([1, 0, 0]))
        [0]
        >>> one_hot_to_indices(np.array([0, 1, 0]))
        [1]
        >>> one_hot_to_indices(np.array([0, 0, 1]))
        [2]
    """
    if not isinstance(value, _np.ndarray):
        value = _soml.util.types.to_numpy(value=value)

    if is_one_hot_encoded(value):
        return value

    if not isinstance(value, _np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    if len(value.shape) != 1:
        raise ValueError("Input must be a one-dimensional array.")

    indices = _np.where(value == 1)[0]
    return indices.tolist()