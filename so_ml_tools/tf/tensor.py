import tensorflow as _tf
import so_ml_tools as _soml


def add_batch_to_tensor(x: _tf.Tensor) -> _tf.Tensor:
    """
    Adds a batch size to the given tensor if x = (224, 224, 3) then the result will be (0, 224, 224, 3)

    Args:
        x: The tensor

    Returns:
        A tensor with a batch size of 0.
    """
    return _tf.expand_dims(x, axis=0)


def probability_to_class(y_prob: _tf.Tensor) -> _tf.Tensor:
    """
    See ml.util.label.probability_to_class
    """
    return _soml.util.label.probability_to_class(y_prob=y_prob)


def probability_to_binary(y_prob: _tf.Tensor) -> _tf.Tensor:
    """
    See ml.util.label.probability_to_binary
    """
    return _soml.util.label.probability_to_binary(y_prob=y_prob)


def to_prediction(y_prob: _tf.Tensor) -> _tf.Tensor:
    """
    See ml.util.label.to_prediction
    """
    return _soml.util.label.to_prediction(y_prob=y_prob)
