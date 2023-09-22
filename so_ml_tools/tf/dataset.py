import tensorflow as _tf
import so_ml_tools as _soml


def describe_pipeline(dataset: _tf.data.Dataset):
    """
    Describes the different steps in the `tf.data.Dataset` pipeline.

    :param dataset: the dataset.
    """
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    input_dataset = dataset
    print(f"{input_dataset.__class__.__name__}", end="")

    while hasattr(input_dataset, '_input_dataset'):
        print(" -> ", end="")
        input_dataset = input_dataset._input_dataset
        print(f"{input_dataset.__class__.__name__}", end="")


def describe_inputs_and_outputs(dataset: _tf.data.Dataset) -> None:
    """
    Describes the input / output shapes and dtype's for the given `tf.data.Dataset`

    Args:
        dataset: a `tf.data.Dataset`
    """
    if isinstance(dataset.element_spec, tuple):
        inputs = dataset.element_spec[0]
        outputs = dataset.element_spec[1]

        print(f"          | {'dtype':<16} | shape")
        # In case of a single input
        if isinstance(inputs, _tf.TensorSpec):
            print(f"Input     | {str(inputs.dtype.name):<16} | {str(inputs.shape)}")
        else:
            # In case of multiple concatenated datasets.
            for i, input in enumerate(inputs):
                print(f"Input  #{i} | {str(input.dtype.name):<16} | {str(input.shape)}")

        if isinstance(outputs, _tf.TensorSpec):
            print(f"Output    | {str(outputs.dtype.name):<16} | {str(outputs.shape)}")


def add_rescaling_mapping(dataset: _tf.data.Dataset) -> _tf.data.Dataset:
    """
    Adds a rescale mapping to the dataset (0-255 -> 0-1)

    Preferably do this before using `optimize_pipeline`

    Args:
        dataset: a `tf.data.Dataset`

    Returns:
        A `tf.data.Dataset` with rescaling.
    """
    return dataset.map(_rescale)


def _rescale(x, y):
    return x / 255., y


def optimize_pipeline(dataset: _tf.data.Dataset) -> _tf.data.Dataset:
    """
    Returns a dataset with (when possible), batching, caching and prefetch in that order, if any of the steps has
    already been applied then it will skip this. For example, creating a dataset using load_image_dataset_from_directory
    will already create a batching dataset, so this method will only add caching and prefetching.

    Preferably do this after using `add_rescaling_mapping`

    Args:
        dataset: a `tf.data.Dataset`

    Returns:
        A `tf.data.Dataset` with batching and prefetching.
    """
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    # Add batching when possible.
    if is_batched(dataset=dataset):
        print('Dataset is already batched.')
    else:
        print('Batching added to dataset.')
        dataset = dataset.batch(batch_size=32)

    # Add caching when possible.
    if is_cached(dataset=dataset):
        print('Dataset is already cached.')
    else:
        print('Caching added to dataset.')
        dataset = dataset.cache()

    # Add prefetching when possible.
    if is_prefetched(dataset=dataset):
        print('Dataset is already prefetched.')
    else:
        print('Prefetching added to dataset.')
        dataset = dataset.prefetch(_tf.data.AUTOTUNE)

    return dataset


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

    input_dataset = dataset
    while not hasattr(input_dataset, 'class_names') and hasattr(input_dataset, '_input_dataset'):
        input_dataset = input_dataset._input_dataset

    if not hasattr(input_dataset, 'class_names'):
        raise TypeError("dataset does not have a ´class_names´ attribute defined.")

    return input_dataset.class_names


def get_labels(dataset: _tf.data.Dataset):
    """
    Returns the labels from a (batched)Dataset

    :param dataset: the dataset from which we want the labels.
    :return: the labels
    """
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    input_dataset = dataset
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

    input_dataset = dataset
    while not input_dataset.__class__.__name__ == '_BatchDataset' and hasattr(input_dataset, '_input_dataset'):
        input_dataset = input_dataset._input_dataset

    return input_dataset.__class__.__name__ == '_BatchDataset'


def is_prefetched(dataset: _tf.data.Dataset) -> bool:
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    input_dataset = dataset
    while not input_dataset.__class__.__name__ == '_PrefetchDataset' and hasattr(input_dataset, '_input_dataset'):
        input_dataset = input_dataset._input_dataset

    return input_dataset.__class__.__name__ == '_PrefetchDataset'


def is_cached(dataset: _tf.data.Dataset) -> bool:
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    input_dataset = dataset
    while not input_dataset.__class__.__name__ == 'CacheDataset' and hasattr(input_dataset, '_input_dataset'):
        input_dataset = input_dataset._input_dataset

    return input_dataset.__class__.__name__ == 'CacheDataset'


def is_shuffled(dataset: _tf.data.Dataset) -> bool:
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    input_dataset = dataset
    while not input_dataset.__class__.__name__ == '_ShuffleDataset' and hasattr(input_dataset, '_input_dataset'):
        input_dataset = input_dataset._input_dataset

    return input_dataset.__class__.__name__ == '_ShuffleDataset'


def show_images_from_dataset(dataset: _tf.data.Dataset, shape=(4, 8)):
    _soml.data.image.show_images_from_dataset(dataset=dataset, shape=shape)
