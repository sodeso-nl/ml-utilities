import tensorflow as tf
import so_ml_tools as soml


def add_batch_to_tensor(x: tf.Tensor) -> tf.Tensor:
    """
    Adds a batch size to the given tensor if x = (224, 224, 3) then the result will be (0, 224, 224, 3)

    Args:
        x: The tensor

    Returns:
        A tensor with a batch size of 0.
    """
    return tf.expand_dims(x, axis=0)


def probability_to_class(y_prob: tf.Tensor) -> tf.Tensor:
    """
    See ml.util.label.probability_to_class
    """
    return soml.util.label.probability_to_class(y_prob=y_prob)


def probability_to_binary(y_prob: tf.Tensor) -> tf.Tensor:
    """
    See ml.util.label.probability_to_binary
    """
    return soml.util.label.probability_to_binary(y_prob=y_prob)


def to_prediction(y_prob: tf.Tensor, dtype=None) -> tf.Tensor:
    """
    See ml.util.label.to_prediction
    """
    return soml.util.label.to_prediction(y_prob=y_prob, dtype=dtype)
