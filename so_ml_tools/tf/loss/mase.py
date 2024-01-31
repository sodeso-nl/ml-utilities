import tensorflow as _tf


def mean_absolute_scaled_error(y_true, y_pred, seasonality: int = None):
    """
    Implement MASE (assuming no seasonality of data).
    """
    mae = _tf.reduce_mean(_tf.abs(y_true - y_pred))

    #  Mean Absolute Deviation (MAD) of the historical data
    if seasonality is not None:
        mad = _tf.reduce_mean(_tf.abs(y_true[seasonality:] - y_true[:-seasonality]))
    else:
        mad = _tf.reduce_mean(_tf.abs(y_true[1:] - y_true[:-1]))

    return mae / mad
