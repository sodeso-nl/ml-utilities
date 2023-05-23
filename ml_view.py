import os
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import ml_internal as mli


def show_random_image_from_disk(target_dir, target_class):
    """
    Shows a random image from the file system.
    :param target_dir: The target directory, for example /food101/train
    :param target_class: The target class name, for example /steak
    """
    # Setup the target directory (we'll view images from here)
    target_folder = os.path.join(target_dir, target_class)

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)

    # Read in the image and plot it using matplotlib
    img = mpimg.imread(os.path.join(target_folder, random_image[0]))
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")

    print(f"image shape: {img.shape}") # Show the shape of the image.
    return img


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

        if mli.is_multiclass_classification(y):
            class_index = mli.to_ordinal(y)
        else:
            # Integer encoded labels
            class_index = y[rand_index]

        if class_labels is None:
            plt.title(class_index, color='white')
        else:
            plt.title("{name}: {idx}".format(name=class_labels[class_index], idx=class_index), color='white')


def plot_single_image_from_nparray_or_tensor(image, title="", figsize=(10, 8), cmap='gray'):
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
    plt.title(f"{title}", color='white')
