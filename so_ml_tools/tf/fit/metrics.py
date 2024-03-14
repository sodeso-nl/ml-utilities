import tensorflow as _tf
import numpy as _np
import pandas as _pd
from so_ml_tools.tf.loss.mase import mean_absolute_scaled_error as _mean_absolute_scaled_error


def evaluate_preds(y_true, y_pred, seasonality: int = None) -> dict:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    y_true = y_true.astype(dtype=_np.float32).ravel()
    y_pred = y_pred.astype(dtype=_np.float32).ravel()

    mae = _tf.keras.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred).numpy()
    mse = _tf.keras.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred).numpy()
    rmse = _np.sqrt(mse)
    mape = _tf.keras.metrics.mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred).numpy()
    mase = _mean_absolute_scaled_error(y_true=y_true, y_pred=y_pred, seasonality=seasonality)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'mase': mase
    }


def _to_numpy(x):
    if not isinstance(x, _np.ndarray):
        if isinstance(x, _pd.DataFrame) | isinstance(x, _pd.Series) | isinstance(x, _pd.DatetimeIndex):
            return x.to_numpy()
        elif isinstance(x, _tf.Tensor):
            return x.numpy()
        else:
            return _tf.convert_to_tensor(value=x).numpy()

    return x