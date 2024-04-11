import numpy as _np
import tensorflow as _tf
import pandas as _pd
import so_ml_tools as _soml

from typing import Union as _Union


def is_binary_classification(y: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor]) -> bool:
    """
        Checks if the input array represents a multiclass classification problem.

        Parameters:
        y : Union[list, numpy.ndarray, pandas.Series, pandas.DataFrame, tensorflow.Tensor]
            The input array representing the target variable.

        Returns:
        bool
            True if the input array represents a multiclass classification problem, False otherwise.
        """
    if not isinstance(y, _np.ndarray):
        y = _soml.util.types.to_numpy(value=y)

    if y.size == 0:
        return False

    # When there is one dimension we can have two situations, one is they are probabilities so all
    # values have a real number between 0 and (including) 1, or it can be 0 or 1 when we talk about
    # predictions.
    if len(y.shape) == 1:
        return _np.all(_np.logical_and(y >= 0, y <= 1))

    # # When there are two dimensions where the second dimension contains multiple values
    # # we expect a probability type of data, so values should be between 0 and (including) 1
    # if len(y.shape) == 2 and len(y[0]) > 1:
    #     return _np.all(_np.logical_and(y >= 0, y <= 1))

    # When there are two dimensions where the second dimension only contains a single
    # value we expect this to be whole numbers only.
    if len(y.shape) == 2 and len(y[0]) == 1:
        return _np.all(_np.logical_and(y >= 0, y <= 1))

    return False


def is_binary_probability(y: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor]) -> bool:
    """Check if the input array represents binary probabilities.

        Args:
            y (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
                Input array to be checked.

        Returns:
            bool: True if the input array represents binary probabilities, False otherwise.

        Raises:
            None
        """
    if not isinstance(y, _np.ndarray):
        y = _soml.util.types.to_numpy(value=y)

    if not is_binary_classification(y=y):
        return False

    if _np.all((y >= 0) & (y <= 1)):
        return True

    elif len(y.shape) == 2:
        # Check if the second dimension only contains a 1-dimensional array
        if all(isinstance(y_prob, _np.ndarray) and len(y_prob.shape) == 1 for y_prob in y):
            return all(_np.all((y_prob >= 0) & (y_prob <= 1)) for y_prob in y)

    return False


def binary_probability_to_prediction(y: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor],
                                     maintain_shape: bool = True, threshold: float = 0.5) -> _np.ndarray:
    """Converts binary probabilities to binary predictions based on a threshold.

        Args:
            y (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
                Input array of binary probabilities to be converted.
            maintain_shape (bool, optional): Whether to maintain the shape of the input array. Defaults to True.
            threshold (float, optional): Threshold value for binary classification. Defaults to 0.5.

        Returns:
            numpy.ndarray: Array of binary predictions.

        Raises:
            ValueError: If the input array does not represent binary probabilities.
        """
    if not isinstance(y, _np.ndarray):
        y = _soml.util.types.to_numpy(value=y)

    if not is_binary_probability(y):
        raise ValueError('Input must be a binary probability array.')

    y_preds = _np.where(y > threshold, 1, 0).astype(_np.int64)
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


def is_multiclass_propabilities(y: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor]) -> bool:
    """Check if the input array represents multiclass probabilities.

        Args:
            y (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
                Input array to be checked.

        Returns:
            bool: True if the input array represents multiclass probabilities, False otherwise.

        Raises:
            None
        """
    if not isinstance(y, _np.ndarray):
        y = _soml.util.types.to_numpy(value=y)

    if not is_multiclass_classification(y=y):
        return False

    return _np.all(_np.isclose(_np.sum(y, axis=1), 1.0))


def multiclass_probability_to_prediction(y: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor],
                                         maintain_shape=False) -> _np.ndarray:
    """Converts multiclass probabilities to multiclass predictions.

    Args:
        y (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
            Input array of multiclass probabilities to be converted.
        maintain_shape (bool, optional): Whether to maintain the shape of the input array. Defaults to False.

    Returns:
        numpy.ndarray: Array of multiclass predictions.

    Raises:
        ValueError: If the input array does not represent multiclass probabilities.
    """
    if not isinstance(y, _np.ndarray):
        y = _soml.util.types.to_numpy(value=y)

    if not is_multiclass_propabilities(y):
        raise ValueError('Input must be a multi-class probability array.')

    if maintain_shape:
        return _np.reshape(_np.argmax(y, axis=1), newshape=(y.shape[0], 1))

    return _np.argmax(y, axis=1)


def probability_to_prediction(y: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor],
                              maintain_shape=None) -> _np.ndarray:
    """Converts probabilities to binary or multiclass predictions based on the input array shape.

        Args:
            y (list or numpy.ndarray or pandas.Series or pandas.DataFrame or tensorflow.Tensor):
                Input array of probabilities to be converted.
            maintain_shape (bool, optional): Whether to maintain the shape of the input array.
                If None, the function automatically determines the shape based on the input type.
                Defaults to None.

        Returns:
            numpy.ndarray: Array of binary or multiclass predictions.

        Raises:
            ValueError: If the input array does not represent binary or multiclass probabilities.
    """
    if not isinstance(y, _np.ndarray):
        y = _soml.util.types.to_numpy(value=y)

    if is_multiclass_classification(y=y):
        if is_multiclass_propabilities(y=y):
            if maintain_shape is None:
                maintain_shape = False
            return multiclass_probability_to_prediction(y, maintain_shape=maintain_shape)
        else:
            return y
    elif is_binary_classification(y=y):
        if is_binary_probability(y=y):
            if maintain_shape is None:
                maintain_shape = True
            return binary_probability_to_prediction(y, maintain_shape=maintain_shape)
        else:
            return y

    raise ValueError('Input must be either a binary or multi-class probability array.')
