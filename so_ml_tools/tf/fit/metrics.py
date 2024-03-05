import tensorflow as _tf
import numpy as _np


def mean_absolute_scaled_error(y_true, y_pred):
    """
    Implement MASE (assuming no seasonality of data).
    """
    mae = _tf.reduce_mean(_tf.abs(y_true - y_pred))

    # Find MAE of naive forecast (no seasonality)
    mae_naive_no_season = _tf.reduce_mean(_tf.abs(y_true[1:] - y_true[:-1]))

    return mae / mae_naive_no_season


def evaluate_preds(y_true, y_pred) -> dict:
    y_true = _tf.cast(y_true, dtype=_tf.float32)
    y_pred = _tf.cast(y_pred, dtype=_tf.float32)

    mse = _tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = _np.sqrt(mse)

    if not isinstance(rmse, _np.floating):
        rmse = rmse.numpy()

    return {
        'mae': _tf.keras.metrics.mean_absolute_error(y_true, y_pred).numpy(),
        'mse': mse.numpy(),
        'rmse': rmse,
        'mape': _tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred).numpy(),
        'mase': mean_absolute_scaled_error(y_true, y_pred).numpy()
    }
