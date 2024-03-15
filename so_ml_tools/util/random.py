import tensorflow as _tf
import numpy as _np


def set_seed(seed):
    """
    Set the seed value for TensorFlow, Keras and Numpy.
    Args:
        seed:

    Returns:

    """
    _tf.random.set_seed(seed=seed)
    _tf.keras.utils.set_random_seed(seed=seed)
    _np.random.seed(seed=seed)
