import tensorflow as _tf
import so_ml_tools as _soml
import numpy as _np

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
        print(f"{'dtype':<16} | shape")
        for inputs in dataset.element_spec:
            # In case of a single input
            if isinstance(inputs, _tf.TensorSpec):
                print(f"{str(inputs.dtype.name):<16} | {str(inputs.shape)}")
            else:
                # In case of multiple concatenated datasets.
                for i, input in enumerate(inputs):
                    print(f"{str(input.dtype.name):<16} | {str(input.shape)}")


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


def add_batching(dataset: _tf.data.Dataset, batch_size=32) -> _tf.data.Dataset:
    if is_batched(dataset=dataset):
        print('WARN: Dataset is already batched.')
    return dataset.batch(batch_size=batch_size)


def add_caching(dataset: _tf.data.Dataset) -> _tf.data.Dataset:
    if is_cached(dataset=dataset):
        print('WARN: Dataset is already cached.')
    return dataset.cache()


def add_shuffling(dataset: _tf.data.Dataset, buffer_size=_tf.data.AUTOTUNE) -> _tf.data.Dataset:
    if is_shuffled(dataset=dataset):
        print('WARN: Dataset is already shuffled.')

    if buffer_size == _tf.data.AUTOTUNE:
        if is_batched(dataset=dataset):
            batch_size = get_batch_size(dataset)
            buffer_size = batch_size * 8
        else:
            buffer_size = 1000

        print(f'Buffer size for shuffle has been set to {buffer_size}')
    return dataset.shuffle(buffer_size=buffer_size)


def add_prefetching(dataset: _tf.data.Dataset, buffer_size=_tf.data.AUTOTUNE) -> _tf.data.Dataset:
    if is_prefetched(dataset=dataset):
        print('WARN: Dataset is already prefetched.')
    return dataset.prefetch(buffer_size=buffer_size)


def get_class_names_from_dataset_info(ds_info: dict):
    """
    Returns the labels from the dataset info object which is created from loading a
    TensorFlow Dataset (tensorflow_datasets)

    :param ds_info: The dataset info object
    :return: The labels / class names
    """
    if hasattr(ds_info, 'features'):
        return ds_info.features['label'].names

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
        # noinspection PyProtectedMember
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
    # if not isinstance(dataset, _tf.data.Dataset):
    #     raise TypeError('dataset is not a tf.data.Dataset')
    #
    # input_dataset = dataset
    # while not hasattr(input_dataset, '_batch_size') and hasattr(input_dataset, '_input_dataset'):
    #     # noinspection PyProtectedMember
    #     input_dataset = input_dataset._input_dataset
    #
    # if hasattr(input_dataset, '_batch_size'):
    #     dataset = dataset.unbatch()
    #
    # y_labels = []
    # for _, labels in dataset:
    #     y_labels.append(labels.numpy())
    #
    # return y_labels
    y_labels = _np.array([], dtype=_np.int64)
    itr = iter(dataset)
    while True:
        try:
            _, batch_labels = next(itr)
            y_labels = _np.append(y_labels, batch_labels.numpy())
        except StopIteration:
            break
    return y_labels


def get_batch_dataset(dataset: _tf.data.Dataset) -> _tf.data.Dataset:
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    input_dataset = dataset
    while not input_dataset.__class__.__name__ == '_BatchDataset' and hasattr(input_dataset, '_input_dataset'):
        # noinspection PyProtectedMember
        input_dataset = input_dataset._input_dataset

    if input_dataset.__class__.__name__ == '_BatchDataset':
        return input_dataset


def get_batch_size(dataset: _tf.data.Dataset) -> int:
    batch_dataset = get_batch_dataset(dataset=dataset)
    if not batch_dataset:
        # noinspection PyUnresolvedReferences
        # noinspection PyProtectedMember
        return batch_dataset._batch_size

    return -1


def is_batched(dataset: _tf.data.Dataset) -> bool:
    return not get_batch_dataset(dataset=dataset)


def is_prefetched(dataset: _tf.data.Dataset) -> bool:
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    input_dataset = dataset
    while not input_dataset.__class__.__name__ == '_PrefetchDataset' and hasattr(input_dataset, '_input_dataset'):
        # noinspection PyProtectedMember
        input_dataset = input_dataset._input_dataset

    return input_dataset.__class__.__name__ == '_PrefetchDataset'


def is_cached(dataset: _tf.data.Dataset) -> bool:
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    input_dataset = dataset
    while not input_dataset.__class__.__name__ == 'CacheDataset' and hasattr(input_dataset, '_input_dataset'):
        # noinspection PyProtectedMember
        input_dataset = input_dataset._input_dataset

    return input_dataset.__class__.__name__ == 'CacheDataset'


def is_shuffled(dataset: _tf.data.Dataset) -> bool:
    if not isinstance(dataset, _tf.data.Dataset):
        raise TypeError('dataset is not a tf.data.Dataset')

    input_dataset = dataset
    while not input_dataset.__class__.__name__ == '_ShuffleDataset' and hasattr(input_dataset, '_input_dataset'):
        # noinspection PyProtectedMember
        input_dataset = input_dataset._input_dataset

    return input_dataset.__class__.__name__ == '_ShuffleDataset'


def show_images_from_dataset(dataset: _tf.data.Dataset, class_names = None, shape=(4, 8)):
    _soml.data.image.show_images_from_dataset(dataset=dataset, class_names=class_names, shape=shape)
