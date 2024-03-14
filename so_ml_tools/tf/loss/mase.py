import tensorflow as _tf
import numpy as _np
import pandas as _pd


def mean_absolute_scaled_error(y_true, y_pred, seasonality: int = None):
    """
    Calculates the mean absolute scaled error

    Args:
        y_true: the ground truth
        y_pred: the predictions
        seasonality: (int) the seasonality, 7=weekly, etc...

    Returns:
        The mean absolute scaled error
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    mae = _np.mean(_np.abs(y_true - y_pred))

    # Find MAE of naive forecast (no seasonality)
    if seasonality is not None:
        mad = _np.mean(_np.abs(y_true[seasonality:] - y_true[:-seasonality]))
    else:
        mad = _np.mean(_np.abs(y_true[1:] - y_true[:-1]))

    return mae / mad


def _to_numpy(x):
    if not isinstance(x, _np.ndarray):
        if isinstance(x, _pd.DataFrame) | isinstance(x, _pd.Series) | isinstance(x, _pd.DatetimeIndex):
            return x.to_numpy()
        elif isinstance(x, _tf.Tensor):
            return x.numpy()
        else:
            return _tf.convert_to_tensor(value=x).numpy()

    return x