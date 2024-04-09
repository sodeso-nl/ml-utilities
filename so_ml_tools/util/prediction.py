import numpy as _np
import tensorflow as _tf
import pandas as _pd
import so_ml_tools as _soml

from typing import Union as _Union


def is_binary_classification(y: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor]) -> bool:
    """Check if the input array represents a binary classification.

        Args:
            y (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
                Input array to be checked.

        Returns:
            bool: True if the input array represents a binary classification, False otherwise.

        Raises:
            None
        """
    if not isinstance(y, _np.ndarray):
        y = _soml.util.types.to_numpy(value=y)

    if len(y.shape) == 1 and len(y) > 0:
        return True

    if len(y.shape) == 2:
        return y.shape[1] == 1

    return False


def is_binary_probability(y_probs: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor]) -> bool:
    """Check if the input array represents binary probabilities.

        Args:
            y_probs (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
                Input array to be checked.

        Returns:
            bool: True if the input array represents binary probabilities, False otherwise.

        Raises:
            None
        """
    if not isinstance(y_probs, _np.ndarray):
        y_probs = _soml.util.types.to_numpy(value=y_probs)

    if not is_binary_classification(y=y_probs):
        return False

    if _np.all((y_probs >= 0) & (y_probs <= 1)):
        return True

    elif len(y_probs.shape) == 2:
        # Check if the second dimension only contains a 1-dimensional array
        if all(isinstance(y_prob, _np.ndarray) and len(y_prob.shape) == 1 for y_prob in y_probs):
            return all(_np.all((y_prob >= 0) & (y_prob <= 1)) for y_prob in y_probs)

    return False


def binary_probability_to_prediction(y_probs: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor],
                                     maintain_shape: bool = True, threshold: float = 0.5) -> _np.ndarray:
    """Converts binary probabilities to binary predictions based on a threshold.

        Args:
            y_probs (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
                Input array of binary probabilities to be converted.
            maintain_shape (bool, optional): Whether to maintain the shape of the input array. Defaults to True.
            threshold (float, optional): Threshold value for binary classification. Defaults to 0.5.

        Returns:
            numpy.ndarray: Array of binary predictions.

        Raises:
            ValueError: If the input array does not represent binary probabilities.
        """
    if not isinstance(y_probs, _np.ndarray):
        y_probs = _soml.util.types.to_numpy(value=y_probs)

    if not is_binary_probability(y_probs):
        raise ValueError('Input must be a binary probability array.')

    y_preds = _np.where(y_probs > threshold, 1, 0).astype(_np.int64)
    if maintain_shape:
        return y_preds

    return y_preds.ravel()


def is_multiclass_classification(y: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor]) -> bool:
    """Check if the input array represents multiclass classification.

        Args:
            y (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
                Input array to be checked.

        Returns:
            bool: True if the input array represents multiclass classification, False otherwise.

        Raises:
            None
        """
    if not isinstance(y, _np.ndarray):
        y = _soml.util.types.to_numpy(value=y)

    return len(y.shape) == 2 and len(y[0]) > 1


def is_multiclass_propabilities(y_probs: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor]) -> bool:
    """Check if the input array represents multiclass probabilities.

        Args:
            y_probs (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
                Input array to be checked.

        Returns:
            bool: True if the input array represents multiclass probabilities, False otherwise.

        Raises:
            None
        """
    if not isinstance(y_probs, _np.ndarray):
        y_probs = _soml.util.types.to_numpy(value=y_probs)

    if not is_multiclass_classification(y=y_probs):
        return False

    return _np.all(_np.isclose(_np.sum(y_probs, axis=1), 1.0))


def multiclass_probability_to_prediction(y_probs: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor],
                                         maintain_shape=False) -> _np.ndarray:
    """Converts multiclass probabilities to multiclass predictions.

    Args:
        y_probs (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
            Input array of multiclass probabilities to be converted.
        maintain_shape (bool, optional): Whether to maintain the shape of the input array. Defaults to False.

    Returns:
        numpy.ndarray: Array of multiclass predictions.

    Raises:
        ValueError: If the input array does not represent multiclass probabilities.
    """
    if not isinstance(y_probs, _np.ndarray):
        y_probs = _soml.util.types.to_numpy(value=y_probs)

    if not is_multiclass_propabilities(y_probs):
        raise ValueError('Input must be a multi-class probability array.')

    if maintain_shape:
        return _np.reshape(_np.argmax(y_probs, axis=1), newshape=(y_probs.shape[0], 1))

    return _np.argmax(y_probs, axis=1)


def probability_to_prediction(y_probs: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor],
                              maintain_shape=None) -> _np.ndarray:
    """Converts probabilities to binary or multiclass predictions based on the input array shape.

        Args:
            y_probs (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
                Input array of probabilities to be converted.
            maintain_shape (bool, optional): Whether to maintain the shape of the input array.
                If None, the function automatically determines the shape based on the input type.
                Defaults to None.

        Returns:
            numpy.ndarray: Array of binary or multiclass predictions.

        Raises:
            ValueError: If the input array does not represent binary or multiclass probabilities.
    """
    if not isinstance(y_probs, _np.ndarray):
        y_probs = _soml.util.types.to_numpy(value=y_probs)

    if is_multiclass_propabilities(y_probs=y_probs):
        if maintain_shape is None:
            maintain_shape = False
        return multiclass_probability_to_prediction(y_probs, maintain_shape=maintain_shape)
    elif is_binary_probability(y_probs=y_probs):
        if maintain_shape is None:
            maintain_shape = True
        return binary_probability_to_prediction(y_probs, maintain_shape=maintain_shape)

    raise ValueError('Input must be either a binary or multi-class probability array.')
