import tensorflow as tf


def add_batch_to_tensor(x):
    return tf.expand_dims(x, axis=0)
