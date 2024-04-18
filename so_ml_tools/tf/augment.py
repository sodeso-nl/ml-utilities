import tensorflow as _tf
import keras as _ks
import random as _random
import numpy as _np
import so_ml_tools as _soml

from typing import Union as _Union


def generate_samples(augmentation: _Union[_ks.Sequential, _ks.layers.Layer], x: _Union[list, _np.ndarray, _tf.Tensor],
                     y: _Union[list, _np.ndarray, _tf.Tensor], shape=(4, 8)) -> (_np.ndarray, _np.ndarray):
    """
    Generates augmented samples by applying augmentation to input data.

    Args:
        augmentation (Sequential or Layer): Augmentation model to apply.
        x (list or ndarray or Tensor): Input data.
        y (list or ndarray or Tensor): Labels corresponding to input data.
        shape (tuple of int, optional): Shape of the output data (default is (4, 8)).

    Returns:
        tuple of ndarray: Augmented images (x_augmented_images) and corresponding labels (y_augmented_images).
    """
    if not isinstance(x, _np.ndarray):
        x = _soml.util.types.to_numpy(value=x)

    if not isinstance(y, _np.ndarray):
        y = _soml.util.types.to_numpy(value=y)

    # pick a random value
    x_idx = _random.randrange(0, len(x))
    x_image = x[x_idx]
    y_label = y[x_idx]

    x_augmented_images = _np.array(_np.expand_dims(x_image, axis=0))
    y_augmented_images = _np.full(shape=(shape[0] * shape[1], len(y_label)), fill_value=y_label)
    for i in range((shape[0] * shape[1]) - 1):
        image = augmentation(_np.expand_dims(x_image, axis=0)).numpy()
        x_augmented_images = _np.vstack((x_augmented_images, image))

    return x_augmented_images, y_augmented_images


def show_samples(augmentation: _ks.Sequential, x: _Union[list, _np.ndarray, _tf.Tensor],
                 y: _Union[list, _np.ndarray, _tf.Tensor], class_names: list[str] = None, shape=(4, 8)):
    """
    Show augmented samples by displaying images and corresponding labels.

    Args:
        augmentation (Sequential): Augmentation model used to generate samples.
        x (list or ndarray or Tensor): Input data.
        y (list or ndarray or Tensor): Labels corresponding to input data.
        class_names (list of str, optional): Names of classes (default is None).
        shape (tuple of int, optional): Shape of the grid for displaying images (default is (4, 8)).
    """
    x_augmented_images, y_augmented_images = generate_samples(augmentation=augmentation, x=x, y=y, shape=shape)
    _soml.data.image.show_images_from_nparray_or_tensor(
        x=x_augmented_images,
        y=y_augmented_images,
        class_names=class_names,
        shape=shape)
