import tensorflow as _tf
import matplotlib.pyplot as _plt
import so_ml_tools as _soml


def get_class_names_from_dataset_info(ds_info):
    """
    Returns the labels from the dataset info object which is created from loading a
    TensorFlow Dataset (tensorflow_datasets)

    :param ds_info: The dataset info object
    :return: The labels / class names
    """
    return ds_info.features["label"].names


def get_class_names(dataset: _tf.data.Dataset):
    """
    Returns the class names from the dataset
    :param dataset: the dataset from which we want the class names.
    :return: the class names
    """
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    return dataset.class_names


def get_labels(dataset: _tf.data.Dataset):
    """
    Returns the labels from a (batched)Dataset

    :param dataset: the dataset from which we want the labels.
    :return: the labels
    """
    if not isinstance(dataset, _tf.data.Dataset):
      raise TypeError('dataset is not a tf.data.Dataset')

    input_dataset = dataset._input_dataset
    while not hasattr(input_dataset, '_batch_size') and hasattr(input_dataset, '_input_dataset'):
      input_dataset = input_dataset._input_dataset

    if hasattr(input_dataset, '_batch_size'):
      dataset = dataset.unbatch()

    y_labels = []
    for _, labels in dataset:
        y_labels.append(labels.numpy())

    return y_labels


def show_images_from_dataset(dataset: _tf.data.Dataset, shape=(4, 8)):
    _soml.data.image.show_images_from_dataset(dataset=dataset, shape=shape)
