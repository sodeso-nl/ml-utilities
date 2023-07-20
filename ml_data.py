import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder

from keras.utils import image_dataset_from_directory

########################################################################################################################
# General
########################################################################################################################


def one_hot_encode_column(*columns: list[np.ndarray]) -> (OneHotEncoder, list[np.ndarray]):
    """
    One-Hot encodes all given columns using all values of all the provided columns.

    Example usage:

    encoder, train_labels_one_hot, val_labels_one_hot, test_labels_one_hot = \
    one_hot_encode_column(
        train_df['target'],
        val_df['target'],
        test_df['target']
    )

    If you get the ValueError:

    all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)

    Then that means one of the lists passed in has ha different shape then the others.

    :param columns: one or more columns
    :return: first value is the encoder, all other values are in same order as the columns but encoded
    """
    encoder = OneHotEncoder(sparse_output=False)

    # Preprocess the columns
    preprocessed_columns = []
    for column in columns:
        if isinstance(column, pd.DataFrame) or isinstance(column, pd.Series):
            column = column.to_numpy()

        # Make sure the column is two dimensional
        if column.ndim == 1:
            column = column.reshape(-1, 1)

        preprocessed_columns.append(column)


    # Concatenate all columns and fit the encoder
    encoder.fit(np.concatenate(preprocessed_columns))

    return_values = []
    # Transform each list now individually
    for preprocessed_column in preprocessed_columns:
        return_values.append(encoder.transform(preprocessed_column))

    return_values.insert(0, encoder)
    return return_values


########################################################################################################################
# General
########################################################################################################################

def dataframe_convert_column_as_type(dataframe: pd.DataFrame, column_name: str, dtype=np.float64, inplace=True) -> pd.DataFrame:
    """
    Converts the dtype of the given column to the specified dtype.

    :param dataframe: the pd.DataFrame
    :param column_name: the column name
    :param dtype: the newly to assign dtype.
    :param inplace: return a new instance of the DataFrame (False) or adjust the given DataFrame
    :return: see inplace
    """
    if inplace:
        dataframe[column_name] = dataframe[column_name].astype(dtype)

    return dataframe


def dataframe_delete_null_rows(dataframe: pd.DataFrame, column_name: str, inplace=True) -> pd.DataFrame:
    """
    Deletes all rows containing a null values inside the given column.

    :param dataframe: the pd.DatFrame
    :param column_name: the column name
    :param inplace: return a new instance of the DataFrame (False) or adjust the given DataFrame
    :return: see inplace
    """
    return dataframe.drop(dataframe[dataframe[column_name].isnull()].index, inplace = inplace)


def dataframe_delete_rows_not_number(dataframe: pd.DataFrame, column_name: str, inplace=True) -> pd.DataFrame:
    """
    Deletes all rows containing non-number values inside the given column.

    :param dataframe: the pd.DatFrame
    :param column_name: the column
    :param inplace: return a new instance of the DataFrame (False) or adjust the given DataFrame
    :return: see inplace
    """
    return dataframe.drop(dataframe[pd.to_numeric(dataframe[column_name], errors='coerce').isna()].index, inplace=inplace)


def dataframe_drop_columns(dataframe: pd.DataFrame, column_names: list[str]) -> None:
    """
    Removes (inplace) the given column names from the dataframe.

    :param dataframe: the pd.DatFrame
    :param column_names: the names of the columns to remove
    """
    if not type(column_names) == list and column_names is not None:
        column_names = [column_names]

    for c in column_names:
        dataframe.drop(c, axis=1, inplace=True)


def dataframe_describe(dataframe: pd.DataFrame, column_names: list[str] = None, round=2):
    # If the column_names argument is not a list then create a list
    if not type(column_names) == list and column_names is not None:
        column_names = [column_names]

    # If we don't have a list of column names then create a histogram for every column.
    if column_names is not None:
        columns = list(dataframe[column_names].columns)
    else:
        columns = list(dataframe.columns)

    data = []
    for c in columns:
        v = dataframe[c]

        _mean = _std = _min = _q25 = _q50 = _q75 = _max = z_min = z_max = np.NAN
        if is_numeric_dtype(v):
            _mean = v.mean()
            _std = v.std()
            _min = v.min()
            _q25 = v.quantile(.25)
            _q50 = v.quantile(.50)
            _q75 = v.quantile(.75)
            _max = v.max()

            # Calculate lower value when Z-Score = -3
            z_min = -3 * _std + _mean

            # Calculate upper value when Z-Score = 3
            z_max = 3 * _std + _mean

        data.append([
            v.name, v.dtype, v.count(), v.isna().sum(), v.nunique(), _mean, _std, z_min, z_max, _min, _q25,
            _q50, _q75, _max
        ])

    print(f"Total number of rows: {len(dataframe)}")
    return pd.DataFrame(columns=["Column", "DType", "NotNull", "Null", "Unique", "Mean", "Std", "Z-Min", "Z-Max", "Min", "25%", "50%", "75%", "Max"], data=data).round(round)


def add_batch_to_tensor(x: tf.Tensor):
    """
    Adds a batch size to the given tensor if x = (224, 224, 3) then the result will be (0, 224, 224, 3)
    :param x: The tensor
    :return: The tensor with the batch size
    """
    return tf.expand_dims(x, axis=0)


def image_as_tensor(image, img_shape=(224, 224), scale=True):
    """
        Reads in an image from filename, turns it into a tensor and reshapes into
        specified shape (img_shape, img_shape, channels)

        :param filename: path to target image
        :param img_shape: tuple with height/width dimension of target image size
        :param scale: scale pixel values from 0-255 to 0-1 or not
        :return: Image tensor of shape (img_shape, img_shape, 3)
        """

    # Decode image into tensor
    img = tf.io.decode_image(contents=image, channels=3)

    # # Resize the image (height / width)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(images=img, size=[img_shape[0], img_shape[1]])

    if not scale:
        return tf.cast(img * 255, tf.int32)
    else:
        return img


########################################################################################################################
# TensorFlow Dataset
########################################################################################################################


def load_food101_from_tdfs(img_shape=(224, 224), rescale=1/255., shuffle_buffer_size=1000):
    """
    Loads the Food-101 dataset and adds a preprocessor to
    scale the data

    https://www.tensorflow.org/datasets/catalog/food101

    This dataset contains only a 'train' and 'validation' set.
    :param img_shape: A tuple with height/width
    :param rescale: The rescale value to which the image will be multiplied, default from 0-255 -> 0-1, set to None to disable
    :param shuffle_buffer_size: Buffer size for shuffling, higher value will consume more memory
    :return: A tuple of train and test data with preprocessing and feature information
    """
    (train_data, test_data), ds_info = tfds.load("food101",
                                                 split=["train", "validation"],
                                                 shuffle_files=True,
                                                 as_supervised=True,  # Data gets returned in tuple format (data, label)
                                                 with_info=True)

    def preprocessor(image, label):
        image = tf.image.resize(image, [img_shape[0], img_shape[1]])  # Reshape target image
        if rescale:
            image = image * rescale

        return tf.cast(image, tf.float32), label  # Return (float32_image, label) tuple

    train_data = train_data.map(map_func=preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle train_data and turn it into batches and prefetch is (load it faster)
    train_data = train_data \
        .shuffle(buffer_size=shuffle_buffer_size) \
        .batch(batch_size=32) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)

    # Map preprocessing function to the test data
    test_data = test_data.map(map_func=preprocessor, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size=32) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)

    return (train_data, test_data), ds_info


def get_class_names_from_dataset_info(ds_info):
    """
    Returns the labels from the dataset info object which is created from loading a
    TensorFlow Dataset (tensorflow_datasets)

    :param ds_info: The dataset info object
    :return: The labels / class names
    """
    return ds_info.features["label"].names


########################################################################################################################
# (Batched)Dataset
########################################################################################################################

def load_image_dataset_from_directory(directory: str,
                                      label_mode='categorical',
                                      image_size=(256, 256),
                                      batch_size=32,
                                      class_names=None,
                                      color_mode='rgb',
                                      shuffle=True, validation_split=None):
    return image_dataset_from_directory(directory=directory,
                                        label_mode=label_mode,
                                        image_size=image_size,
                                        batch_size=batch_size,
                                        class_names=class_names,
                                        color_mode=color_mode,
                                        shuffle=shuffle,
                                        validation_split=validation_split
                                        )


def get_class_names_from_dataset(dataset: tf.data.Dataset):
    """
    Returns the class names from the dataset
    :param dataset: the dataset from which we want the class names.
    :return: the class names
    """
    if not isinstance(dataset, tf.data.Dataset):
      raise TypeError('dataset is not a tf.data.Dataset')

    return dataset.class_names


def get_labels_from_dataset(dataset: tf.data.Dataset):
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
