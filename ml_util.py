import numpy as np

from sklearn.model_selection import train_test_split
from keras import mixed_precision


def set_single_precision_policy() -> None:
    """
    Disable mixed precision by using the float32 policy.
    """
    policy = mixed_precision.Policy("float32")
    mixed_precision.set_global_policy(policy)


def set_mixed_precision_policy_for_gpu() -> None:
    """
    Enable mixed precision by using the mixed_float16 policy, use this policy
    for GPU acceleration.

    For details: https://www.tensorflow.org/guide/mixed_precision
    """
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)


def set_mixed_precision_policy_for_tpu() -> None:
    """
    Enable mixed precision by using the mixed_bfloat16 policy, use this policy
    for TPU acceleration.

    For details: https://www.tensorflow.org/guide/mixed_precision
    """
    policy = mixed_precision.Policy("mixed_bfloat16")
    mixed_precision.set_global_policy(policy)


def get_mixed_precision_policy():
    """
    Returns the current configured mixed-precision policy.
    :return: the current configured mixed-precision policy.
    """
    return mixed_precision.global_policy()


def normalize_xy_data(x):
    """
    Normalizes an array containing vectors of x/y coordinates so that the array does not contain
    negative values.

    :param x: the vector containing values from -X to +X which need to be normalized between 0 and 1
    :return: the normalized vector.
    """
    x = x + (np.abs(np.min(x[:, 0])))
    x = x / np.max(x[:, 0])
    x = x + (np.abs(np.min(x[:, 1])))
    return x / np.max(x[:, 1])


def split_train_test_data(*arrays, test_size=.2, train_size=.8, random_state=42, shuffle=True):
    """
    Usage:

    X_train, X_test, y_train, y_test =
        split_train_test_data(X, y)
    """
    return train_test_split(*arrays,
                            test_size=test_size,
                            train_size=train_size,
                            random_state=random_state,
                            shuffle=shuffle)


