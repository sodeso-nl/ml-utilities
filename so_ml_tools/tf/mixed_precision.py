from keras import mixed_precision as _mixed_precision


def set_single_precision_policy() -> None:
    """
    Disable mixed precision by using the float32 policy.
    """
    policy = _mixed_precision.Policy("float32")
    _mixed_precision.set_global_policy(policy)


def set_mixed_precision_policy_for_gpu() -> None:
    """
    Enable mixed precision by using the mixed_float16 policy, use this policy
    for GPU acceleration.

    For details: https://www.tensorflow.org/guide/mixed_precision
    """
    policy = _mixed_precision.Policy("mixed_float16")
    _mixed_precision.set_global_policy(policy)


def set_mixed_precision_policy_for_tpu() -> None:
    """
    Enable mixed precision by using the mixed_bfloat16 policy, use this policy
    for TPU acceleration.

    For details: https://www.tensorflow.org/guide/mixed_precision
    """
    policy = _mixed_precision.Policy("mixed_bfloat16")
    _mixed_precision.set_global_policy(policy)


def get_mixed_precision_policy():
    """
    Returns the current configured mixed-precision policy.
    :return: the current configured mixed-precision policy.
    """
    return _mixed_precision.global_policy()
