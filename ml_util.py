import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


def get_labels_from_dataset(dataset, index_only=True):
    """
    Returns the labels from a (batched)Dataset

    :param dataset: the dataset from which we want the labels.
    :param index_only: to create an indexed list or keep the one-hot encoded.
    :return: the labels
    """
    y_labels = []
    for images, labels in dataset.unbatch():  # Un-batch the test data and get images and labels
        if index_only:
            y_labels.append(labels.numpy().argmax())  # Append the index which has the largest value (one-hot)
        else:
            y_labels.append(labels.numpy())

    return y_labels


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


def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    specified shape (img_shape, img_shape, channels)

    Args:
      filename (str): path to target image
      image_shape (int): height/width dimension of target image size
      scale (bool): scale pixel values from 0-255 to 0-1 or not

    Returns:
      Image tensor of shape (img_shape, img_shape, 3)
    """
    # Read in the image
    img = tf.io.read_file(filename=filename)

    # Decode image into tensor
    img = tf.io.decode_image(contents=img, channels=3)

    # Resize the image (height / width)
    img = tf.image.resize(images=img, size=[img_shape, img_shape])

    if scale:
        return img / 255.
    else:
        return img
