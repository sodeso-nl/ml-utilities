import tensorflow as tf
import tensorflow_datasets as tfds

import ml_internal as mlint

from keras.utils import image_dataset_from_directory

########################################################################################################################
# General
########################################################################################################################


def add_batch_to_tensor(x):
    """
    Adds a batch size to the given tensor if x = (224, 224, 3) then the result will be (0, 224, 224, 3)
    :param x: The tensor
    :return: The tensor with the batch size
    """
    return tf.expand_dims(x, axis=0)


def load_image_as_tensor(filename, img_shape=(224, 224), scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    specified shape (img_shape, img_shape, channels)

    :param filename: path to target image
    :param img_shape: tuple with height/width dimension of target image size
    :param scale: scale pixel values from 0-255 to 0-1 or not
    :return: Image tensor of shape (img_shape, img_shape, 3)
    """

    # Read in the image
    img = tf.io.read_file(filename=filename)

    # Decode image into tensor
    img = tf.io.decode_image(contents=img, channels=3)

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

def load_image_dataset_from_directory(directory,
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


def get_labels_from_dataset(dataset, index_only=True):
    """
    Returns the labels from a (batched)Dataset

    :param dataset: the dataset from which we want the labels.
    :param index_only: to create an indexed list or keep the one-hot encoded.
    :return: the labels / class names
    """
    dataset_unbatched = dataset.unbatch()
    _, y = next(iter(dataset_unbatched))
    if not mlint.is_multiclass_classification(y) and index_only:
        raise TypeError('dataset is not a multiclass classification, index_only must be False')

    y_labels = []
    for images, labels in dataset_unbatched:  # Un-batch the test data and get images and labels
        if index_only:
            y_labels.append(labels.numpy().argmax())  # Append the index which has the largest value (one-hot)
        else:
            y_labels.append(labels.numpy())

    return y_labels

