import random

import tensorflow as tf
import matplotlib as plt
import so_ml_tools as soml
import numpy as np


def image_as_tensor(image, img_shape=(224, 224), scale=True):
    """
        Reads in an image from filename, turns it into a tensor and reshapes into
        specified shape (img_shape, img_shape, channels)

        :param filename: path to target image
        :param img_shape: tuple with height/width dimension of target image size
        :param scale: scale pixel values from 0-255 to 0-1 or not
        :return: Image tensor of shape (img_shape, img_shape, 3)
        """

    # Decode image into tensor
    img = tf.io.decode_image(contents=image, channels=3)

    # # Resize the image (height / width)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(images=img, size=[img_shape[0], img_shape[1]])

    if not scale:
        return tf.cast(img * 255, tf.int32)
    else:
        return img


def is_image_float32_and_not_normalized(x):
    """
    Convenience method to check if an image in Tensor format is of type float32 but not normalized (values between
    0..255 instead of 0..1)
    :param x: Tensor containing the image(s)
    :return:
    """
    return x.dtype == tf.float32 and tf.math.reduce_max(x).numpy() > 1.0


def show_images_from_nparray_or_tensor(x, y, class_names: list[str] = None, indices=None, shape=(4, 6), cmap='gray'):
    """
    Shows images stored in a tensor / numpy array. The array should be a vector of images.

    :param X: is an array containing vectors of images.
    :param y: are the associated labels
    :param class_names: the labels of the classes
    :param indices: None to pick random, otherwise an array of indexes to display
    :param shape: is the number of images to display
    :param cmap: is the color map to use, use "gray" for gray scale images, use None for default.
    """
    y = soml.util.label.to_prediction(y_prob=y, dtype=np.float32)

    if is_image_float32_and_not_normalized(x):
        x = tf.cast(x=x, dtype=tf.uint8)

    if indices:
        assert shape[0] * shape[1] <= len(
            indices), f"Size of shape ({shape[0]}, {shape[1]}), with a total of " \
                      f"{shape[0] * shape[1]} images, is larger then number of indices supplied ({len(indices)})."
        for i in indices:
            if i > len(x):
                assert False, f"Values of indices point to an index ({i}) which is out of bounds of X (length: {len(x)})"

    fig = plt.figure(figsize=(shape[1] * 3, shape[0] * 3))
    fig.patch.set_facecolor('gray')
    for i in range(shape[0] * shape[1]):
        ax = plt.subplot(shape[0], shape[1], i + 1)
        ax.axis('off')

        if indices is None:
            rand_index = random.choice(range(len(x)))
        else:
            rand_index = indices[i]

        plt.imshow(x[rand_index], cmap=cmap)

        if soml.util.label.is_multiclass_classification(y):
            class_index = soml.util.label.probability_to_class(y)
        else:
            # Integer encoded labels
            class_index = y[rand_index]

        if class_names is None:
            plt.title(class_index, color='white')
        else:
            plt.title(f", {class_index}, {x[rand_index].shape}", color='white')
    plt.show()


def show_single_image_from_nparray_or_tensor(image, title="", figsize=(10, 8), cmap='gray'):
    """
    Shows images stored in a tensor / numpy array. The array should be a vector of images.

    :param image: the image to plot
    :param title: the title to display above the image
    :param figsize: Size of output figure (default=(10, 8)).
    :param cmap: is the collor map to use, use "gray" for gray scale images, use None for default.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('gray')
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(f"{title} {image.shape}", color='white')
    plt.show()
