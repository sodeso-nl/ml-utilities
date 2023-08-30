import tensorflow as tf
import matplotlib as plt
import so_ml_utilities as somlu


def get_class_names_from_dataset_info(ds_info):
    """
    Returns the labels from the dataset info object which is created from loading a
    TensorFlow Dataset (tensorflow_datasets)

    :param ds_info: The dataset info object
    :return: The labels / class names
    """
    return ds_info.features["label"].names


def get_class_names(dataset: tf.data.Dataset):
    """
    Returns the class names from the dataset
    :param dataset: the dataset from which we want the class names.
    :return: the class names
    """
    if not isinstance(dataset, tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    return dataset.class_names


def get_labels(dataset: tf.data.Dataset):
    """
    Returns the labels from a (batched)Dataset

    :param dataset: the dataset from which we want the labels.
    :return: the labels
    """
    if not isinstance(dataset, tf.data.Dataset):
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


def show_images_from_dataset(dataset: tf.data.Dataset, shape=(4, 8)):
    assert isinstance(dataset, tf.data.Dataset), f"The dataset supplied is not a tensorflow.data.Dataset."

    # Retrieve first batch, depending on the initalization of the dataset the batch size is default 32
    # so when performing a take of (1) we retreive the first batch
    batches = dataset.take(1)

    # Use an iterator to get the first batch of images and labels
    batch_iter = iter(batches)
    x, y_true = batch_iter.next()

    assert shape[0] * shape[1] == len(x), f"Size of shape ({shape[0]}, {shape[1]}), with a total of " \
                  f"{shape[0] * shape[1]} images, is not equal to the batch size of the dataset ({len(x)})."

    if len(x) != shape[0] * shape[1]:
        raise TypeError('dataset is not a Dataset')

    somlu.data.image.show_images_from_nparray_or_tensor(x=x, y=y_true, class_names=dataset.class_names, shape=shape)
    plt.show()
