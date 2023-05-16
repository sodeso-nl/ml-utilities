import numpy as np

from sklearn.model_selection import train_test_split


def get_labels_from_dataset(dataset, index_only=True):
    """
    Returns the labels from a (batched)Dataset

    :param dataset: the dataset from which we want the labels.
    :param index_only: to create an indexed list or keep the one-hot encoded.
    :return: the labels
    """
    y_labels = []
    for images, labels in dataset.unbatch(): # Un-batch the test data and get images and labels
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
