import numpy as _np
import tensorflow as _tf
import pandas as _pd
import so_ml_tools as _soml

from typing import Union as _Union


def show_wrong_predicted_images(
        x: _Union[list[list], _np.ndarray, _tf.Tensor],
        y_true: _Union[list[list], _np.ndarray, _tf.Tensor, _pd.Series],
        y_pred: _Union[list[list], _np.ndarray, _tf.Tensor, _pd.Series] = None,
        y_prob: _Union[list[list], _np.ndarray, _tf.Tensor] = None,
        class_names: list[str] = None,
        shape=(4, 8)):
    if _soml.util.onehot.is_one_hot_encoded(value=y_true):
        y_true = _soml.util.onehot.one_hot_to_indices(value=y_true)
        y_true = _np.reshape(a=y_true, newshape=(len(y_true), 1))

    y_true = _soml.util.prediction.probability_to_prediction(y=y_true)
    if y_pred is None and y_prob is not None:
        y_pred = _soml.util.prediction.probability_to_prediction(y=y_prob, maintain_shape=True)
    elif y_pred is None and y_prob is None:
        raise "Must specify 'y_pred' or 'y_prob'"

    x = _soml.util.types.to_numpy(value=x)

    # Get the indices of the labels that are different
    indices = _soml.util.label.diff_indexes(y_true=y_true, y_pred=y_pred)
    X_test_failed = x[indices]
    y_test_failed = y_pred[indices]
    _soml.data.image.show_images_from_nparray_or_tensor(x=X_test_failed, y=y_test_failed, class_names=class_names, shape=shape)
