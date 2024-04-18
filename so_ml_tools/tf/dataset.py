import numpy as np
import tensorflow as _tf
import numpy as _np
from typing import Union as _Union
from typing import Tuple as _Tuple


def split(dataset: _tf.data.Dataset, split_percentages: list[float]) -> tuple[_tf.data.Dataset, ...]:
    """Splits a TensorFlow dataset into multiple datasets based on the provided split percentages.

    Args:
        dataset (tensorflow.data.Dataset): Input TensorFlow dataset to be split.
        split_percentages (List[float]): List of float values representing the percentages of each split.

    Returns:
        Tuple[tensorflow.data.Dataset, ...]: Tuple of TensorFlow datasets after splitting.

    Raises:
        AssertionError: If the sum of split percentages is not equal to 1.0.

    Examples:
        >>> import tensorflow as tf
        >>> dataset = tf.data.Dataset.range(10)
        >>> splits = split(dataset, split_percentages=[0.7, 0.2, 0.1])
        >>> for split in splits:
        ...     print(list(split.as_numpy_iterator()))
        [0, 1, 2, 3, 4, 5, 6], [7, 8], [9]
    """
    assert sum(split_percentages) == 1.0, "The sum of split percentages should be equal to 1.0"

    # Convert percentages to actual lengths from the dataset.
    split_lengths = [int(split_pct * len(dataset)) for split_pct in split_percentages]

    split_datasets = []
    for split_length in split_lengths:
        split_datasets.append(dataset.take(split_length))
        dataset = dataset.skip(split_length)

    return tuple(split_datasets)


def describe_pipeline(dataset: _tf.data.Dataset):
    """
    Describes the different steps in the `tf.data.Dataset` pipeline.

    :param dataset: the dataset.
    """
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


def get_labels(dataset: _tf.data.Dataset, max_samples: int = None) -> _np.ndarray:
    """
    Returns the labels from a (batched)Dataset

    :param dataset: the dataset from which we want the labels.
    :param max_samples: the maximum number of samples to return
    :return: the labels
    """
    if is_shuffled(dataset=dataset):
        print("WARNING: Dataset is shuffled, retrieving features and labels separate will result in "
              "different order of features and labels")

    if is_batched(dataset=dataset):
        dataset = dataset.unbatch()

    current_sample = 0
    all_labels = None
    itr = dataset.as_numpy_iterator()
    while max_samples is None or current_sample < max_samples:
        try:
            _, labels = next(itr)
            if all_labels is None:
                all_labels = _np.expand_dims(labels, axis=0)
            else:
                all_labels = _np.vstack((all_labels, np.expand_dims(labels, axis=0)))
            current_sample += 1
        except StopIteration:
            break

    return all_labels


def get_features(dataset: _tf.data.Dataset, max_samples: int = None) -> _np.ndarray:
    """
    Returns the features from a (batched)Dataset

    :param dataset: the dataset from which we want the features.
    :param max_samples: the maximum number of samples to return
    :return: the features
    """
    if is_shuffled(dataset=dataset):
        print("WARNING: Dataset is shuffled, retrieving features and labels separate will result in "
              "different order of features and labels")

    if is_batched(dataset=dataset):
        dataset = dataset.unbatch()

    current_sample = 0
    all_features = None
    itr = dataset.as_numpy_iterator()
    while max_samples is None or current_sample < max_samples:
        try:
            features, _ = next(itr)
            if all_features is None:
                all_features = _np.expand_dims(features, axis=0)
            else:
                all_features = _np.vstack((all_features, _np.expand_dims(features, axis=0)))
            current_sample += 1
        except StopIteration:
            break

    return all_features


def get_features_and_labels(dataset: _tf.data.Dataset, max_samples: int = None) -> _Tuple[_np.array, _np.ndarray]:
    """
    Returns the features and labels from a (batched)Dataset as a tupple (features, labels)

    :param dataset: the dataset from which we want the features and labels.
    :param max_samples: the maximum number of samples to return
    :return: the features and labels
    """
    if is_shuffled(dataset=dataset):
        print("WARNING: Dataset is shuffled, retrieved features and labels will match accordingly, however performing"
              " this action twice will result in different order.")

    if is_batched(dataset=dataset):
        dataset = dataset.unbatch()

    current_sample = 0
    all_features = None
    all_labels = None
    itr = dataset.as_numpy_iterator()
    while max_samples is None or current_sample < max_samples:
        try:
            features, labels = next(itr)
            if all_features is None:
                all_features = _np.expand_dims(features, axis=0)
                all_labels = _np.expand_dims(labels, axis=0)
            else:
                all_features = _np.vstack((all_features, _np.expand_dims(features, axis=0)))
                all_labels = _np.vstack((all_labels, _np.expand_dims(labels, axis=0)))
            current_sample += 1
        except StopIteration:
            break

    return all_features, all_labels


def get_batch_dataset(dataset: _tf.data.Dataset) -> _Union[_tf.data.Dataset, None]:
    input_dataset = dataset
    while not input_dataset.__class__.__name__ == '_BatchDataset' and hasattr(input_dataset, '_input_dataset'):
        # noinspection PyProtectedMember
        input_dataset = input_dataset._input_dataset

    if input_dataset.__class__.__name__ == '_BatchDataset':
        return input_dataset

    return None


def get_batch_size(dataset: _tf.data.Dataset) -> int:
    batch_dataset = get_batch_dataset(dataset=dataset)
    if batch_dataset is not None:
        # noinspection PyUnresolvedReferences
        # noinspection PyProtectedMember
        return batch_dataset._batch_size

    return -1


def is_batched(dataset: _tf.data.Dataset) -> bool:
    return get_batch_dataset(dataset=dataset) is not None


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
