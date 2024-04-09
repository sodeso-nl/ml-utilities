import tensorflow as _tf


def add_batch_to_tensor(x: _tf.Tensor) -> _tf.Tensor:
    """
    Adds a batch size to the given tensor if x = (224, 224, 3) then the result will be (0, 224, 224, 3)

    Args:
        x: The tensor

    Returns:
        A tensor with a batch size of 0.
    """
    return _tf.expand_dims(x, axis=0)