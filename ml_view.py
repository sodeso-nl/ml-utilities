import os
import random
import logging

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import ml_internal as mlint

# noinspection PyUnresolvedReferences
from tensorflow.data import Dataset


def show_random_image_from_disk(target_dir, target_class, shape=(4, 6), cmap='gray') -> None:
    """
    Shows a random image from the file system.
    :param target_dir: The target directory, for example /food101/train
    :param target_class: The target class name, for example /steak
    :param shape: is the number of images in a grid to display
    :param cmap: is the color map to use, use "gray" for gray scale images, use None for default.
    """
    try:
        target_folder = os.path.join(target_dir, target_class)

        # Get a random image path
        fig = plt.figure(figsize=(shape[1] * 3, shape[0] * 3))
        fig.patch.set_facecolor('gray')
        for i in range(0, shape[0] * shape[1]):
            ax = plt.subplot(shape[0], shape[1], i + 1)
            ax.axis('off')

            image_filename = random.sample(os.listdir(target_folder), 1)
            image_fullpath = os.path.join(target_folder, image_filename[0])
            img = mpimg.imread(image_fullpath)
            plt.imshow(img, cmap=cmap)
            plt.title(f"{image_filename[0]}: {img.shape}", color='white')

        plt.show()
    except Exception as e:
        logging.error(e)


def show_images_from_dataset(dataset, shape=(4, 8), cmap='gray') -> None:
    """
    Shows images stored in a dataset.

    :param dataset: the dataset from which we want to show images
    :param shape: the shape, where the shape width * height should reflect the batch size
    :param cmap: is the color map to use, use "gray" for gray scale images, use None for default.
    """
    assert isinstance(dataset, Dataset), f"The dataset supplied is not a tensorflow.data.Dataset."

    # Retrieve first batch, depending on the initalization of the dataset the batch size is default 32
    # so when performing a take of (1) we retreive the first batch
    batches = dataset.take(1)

    # Use an iterator to get the first batch of images and labels
    batch_iter = iter(batches)
    x, y_true = next(batch_iter)

    assert shape[0] * shape[1] == len(x), f"Size of shape ({shape[0]}, {shape[1]}), with a total of " \
                  f"{shape[0] * shape[1]} images, is not equal to the batch size of the dataset ({len(x)})."

    if len(x) != shape[0] * shape[1]:
        raise TypeError('dataset is not a Dataset')

    show_images_from_nparray_or_tensor(x=x, y=y_true, class_labels=dataset.class_names, shape=shape, cmap=cmap)
    plt.show()


def show_images_from_nparray_or_tensor(x, y, class_labels=None, indices=None, shape=(4, 6), cmap='gray'):
    """
    Shows images stored in a tensor / numpy array. The array should be a vector of images.

    :param X: is an array containing vectors of images.
    :param y: are the associated labels
    :param class_labels: the labels of the classes
    :param indices: None to pick random, otherwise an array of indexes to display
    :param shape: is the number of images to display
    :param cmap: is the collor map to use, use "gray" for gray scale images, use None for default.
    """
    y = mlint.convert_to_sparse_or_binary(y=y)

    if mlint.is_image_float32_and_not_normalized(x):
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

        if mlint.is_label_dense(y):
            class_index = mlint.sparse_labels(y)
        else:
            # Integer encoded labels
            class_index = y[rand_index]

        if class_labels is None:
            plt.title(class_index, color='white')
        else:
            plt.title(f"{class_labels[class_index]}, {class_index}, {x[rand_index].shape}", color='white')
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
