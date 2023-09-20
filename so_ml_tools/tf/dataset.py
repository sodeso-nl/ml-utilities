import tensorflow as _tf
import so_ml_tools as _soml


def describe(dataset: _tf.data.Dataset) -> None:
    """
    Describes the input / output shapes and dtype's for the given `tf.data.Dataset`

    Args:
        dataset: a `tf.data.Dataset`
    """
    if isinstance(dataset.element_spec, tuple):
        inputs = dataset.element_spec[0]
        outputs = dataset.element_spec[1]

        print(f"          | {'shape':<16} | dtype")
        for i, input in enumerate(inputs):
            print(f"Input  #{i} | {str(input.shape):<16} | {str(input.dtype.name)}")

        if isinstance(outputs, _tf.TensorSpec):
            print(f"Output    | {str(outputs.shape):<16} | {str(outputs.dtype.name)}")


def batch_and_prefetch(dataset: _tf.data.Dataset) -> _tf.data.Dataset:
    """
    Returns a dataset with batching enabled (size 32) and prefetching.

    Args:
        dataset: a `tf.data.Dataset`

    Returns:
        A `tf.data.Dataset` with batching and prefetching.
    """
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    if is_batched(dataset=dataset):
        print('Dataset is already batched, only prefetch is added.')
        return dataset.prefetch(_tf.data.AUTOTUNE)

    return dataset.batch(batch_size=32).prefetch(_tf.data.AUTOTUNE)


def get_class_names_from_dataset_info(ds_info: dict):
    """
    Returns the labels from the dataset info object which is created from loading a
    TensorFlow Dataset (tensorflow_datasets)

    :param ds_info: The dataset info object
    :return: The labels / class names
    """
    return ds_info["features"]["label"].names


def get_class_names(dataset: _tf.data.Dataset):
    """
    Returns the class names from the dataset
    :param dataset: the dataset from which we want the class names.
    :return: the class names
    """
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    if not hasattr(dataset, 'class_names'):
        raise TypeError("dataset does not have a ´class_names´ attribute defined.")

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


def is_batched(dataset: _tf.data.Dataset) -> bool:
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    if hasattr(dataset, '_batch_size'):
        return True

    input_dataset = dataset._input_dataset
    while not hasattr(input_dataset, '_batch_size') and hasattr(input_dataset, '_input_dataset'):
        input_dataset = input_dataset._input_dataset

    return hasattr(input_dataset, '_batch_size')


def show_images_from_dataset(dataset: _tf.data.Dataset, shape=(4, 8)):
    _soml.data.image.show_images_from_dataset(dataset=dataset, shape=shape)
