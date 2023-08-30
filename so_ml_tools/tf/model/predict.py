import tensorflow as tf
import numpy as np
import so_ml_tools as soml


def predict_classification_dataset_without_labels(model: tf.keras.Model, dataset: tf.data.Dataset):
    """
    Performs predictions on the given model for all entries in the dataset, will
    return y_prob. Does not work well when dataset is being shuffled.

    :param model: model to perform the predictions on
    :param dataset: dataset containing the data to use for the predictions
    :return: y_true, y_pred
    """
    y_prob = None
    for x in dataset:
        batch_pred = model.predict(x)
        if y_prob is None:
            y_prob = batch_pred
        else:
            y_prob = np.concatenate((y_prob, batch_pred), axis=0)

    return y_prob


def predict_classification_dataset_with_labels(model: tf.keras.Model, dataset: tf.data.Dataset):
    """
    Performs predictions on the given model for all entries in the dataset, will
    return y_pred and y_true as a tuple. Usefull for when the dataset has shuffling enabled
    :param model: model to perform the predictions on
    :param dataset: dataset containing the data to use for the predictions
    :return: y_true, y_pred
    """
    y_prob = None
    y_true = None
    for x, y in dataset:
        batch_pred = model.predict(x)
        if y_prob is None:
            y_prob = batch_pred
        else:
            y_prob = np.concatenate((y_prob, batch_pred), axis=0)

        if y_true is None:
            y_true = y
        else:
            y_true = np.concatenate((y_true, y), axis=0)
    return y_true, y_prob


def predict_regression(model: tf.keras.Model, x):
    """
    Performs predictions on the given model for all entries in the dataset, will
    return y_pred and y_true as a tuple. Use full for when the dataset has shuffling enabled
    :param model: model to perform the predictions on
    :param dataset: dataset containing the data to use for the predictions
    :return: y_true, y_pred
    """
    y_pred = None
    for values in x:
        batch_pred = model.predict(x=soml.tf.tensor.add_batch_to_tensor(values))
        if y_pred is None:
            y_pred = batch_pred
        else:
            y_pred = np.concatenate((y_pred, batch_pred), axis=0)

    return y_pred