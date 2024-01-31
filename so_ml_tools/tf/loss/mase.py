import tensorflow as tf


def mean_absolute_scaled_error(y_true, y_pred, seasonality: int = None):
    """
    Implement MASE (assuming no seasonality of data).
    """
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    #  Mean Absolute Deviation (MAD) of the historical data
    if seasonality is not None:
        mad = tf.reduce_mean(tf.abs(y_true[seasonality:] - y_true[:-seasonality]))
    else:
        mad = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))

    return mae / mad
