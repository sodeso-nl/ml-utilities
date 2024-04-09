import random as _random
import os as _os
import logging as _logging

import tensorflow as _tf
import matplotlib.pyplot as _plt
import so_ml_tools as _soml


def load_image_as_tensor(filename: str, img_shape=(224, 224), scale=True) -> _tf.Tensor:
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    specified shape (img_shape, img_shape, channels)

    Args:
        filename: path to target image
        img_shape: A `tuple` with the dimensions of the image.
        scale: A `bool` indicating if scaling needs to be performed (0-255 -> 0-1)

    Returns:
        A 'tf.Tensor' of shape (img_shape, 3) with of dtype 'tf.int32'.
    """

    img = _tf.io.read_file(filename=filename)
    return image_as_tensor(image=img, img_shape=img_shape, scale=scale)


def image_as_tensor(image, img_shape: tuple = (224, 224), scale: bool = True) -> _tf.Tensor:
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    specified shape (img_shape, img_shape, channels)
            
    Args:
        image: A `Tensor` of type `string`. 0-D. The encoded image bytes.
        img_shape: A `tuple` with the dimensions of the image.
        scale: A `bool` indicating if scaling needs to be performed (0-255 -> 0-1)

    Returns:
        A 'tf.Tensor' of shape (img_shape, 3) with of dtype 'tf.int32'.
    """
    # Decode image into tensor
    img = _tf.io.decode_image(contents=image, channels=3)

    # # Resize the image (height / width)
    img = _tf.image.convert_image_dtype(img, _tf.float32)
    img = _tf.image.resize(images=img, size=[img_shape[0], img_shape[1]])

    if not scale:
        return _tf.cast(x=img * 255, dtype=_tf.int32)
    else:
        return img


def is_image_float32_and_not_normalized(x) -> bool:
    """
    Convenience method to check if an image in `tf.Tensor` format is of type `float32` but not normalized (values between
    0..255 instead of 0..1)

    Args:
        x: A `tf.Tensor`

    Returns:
        A `bool` indicating if the tensor is of type `float32` but not normalized.
    """
    return x.dtype == _tf.float32 and _tf.math.reduce_max(x).numpy() > 1.0


def show_images_from_nparray_or_tensor(x, y, class_names: list[str] = None, shape: tuple = (4, 6),
                                       cmap: str = 'gray') -> None:
    """
    Shows images stored in a tensor / numpy array. The array should be a vector of images.

    Args:
        x: a `tf.Tensor` containg the images
        y: a 'tf.Tensor' containing the labels associated with `x`
        class_names: a `list` of class names
        shape: A `tuple` specifying the number of images to display (width, height)
        cmap: A 'str' with the color map to use for displaying the images.

    Returns:

    """
    # Convert to categorical first so we can use the values as indices.
    if _soml.util.onehot.is_one_hot_encoded(value=y):
        y = _soml.util.onehot.one_hot_to_indices(value=y)

    if _soml.data.image.is_image_float32_and_not_normalized(x):
        x = _tf.cast(x=x, dtype=_tf.uint8)

    fig = _plt.figure(figsize=(shape[1] * 3, shape[0] * 3))
    fig.patch.set_facecolor('gray')
    for i in range(shape[0] * shape[1]):
        ax = _plt.subplot(shape[0], shape[1], i + 1)
        ax.axis('off')

        # if indices is None:
        rand_index = _random.choice(range(len(x)))
        # else:
        #     rand_index = indices[i]

        _plt.imshow(x[rand_index], cmap=cmap)

        class_index = y[rand_index]
        if class_names is None:
            _plt.title(class_index, color='white')
        else:
            _plt.title(f"{class_names[class_index]}, {x[rand_index].shape}", color='white')
    _plt.show()


def show_single_image_from_nparray_or_tensor(image, title="", figsize=(10, 8), cmap='gray') -> None:
    """
    Shows images stored in a tensor / numpy array. The array should be a vector of images.

    Args:
        image: the image to plot
        title: the title to display above the image
        figsize: Size of output figure (default=(10, 8)).
        cmap: is the collor map to use, use "gray" for gray scale images, use None for default.

    Returns:
        None
    """
    fig = _plt.figure(figsize=figsize)
    fig.patch.set_facecolor('gray')
    _plt.imshow(image, cmap=cmap)
    _plt.axis('off')
    _plt.title(f"{title} {image.shape}", color='white')
    _plt.show()


def show_random_image_from_disk(target_dir, target_class, shape=(4, 6), cmap='gray') -> None:
    """
    Shows a random image from the file system.

    Args:
        target_dir: The target directory, for example /food101/train
        target_class: The target class name, for example /steak
        shape: is the number of images in a grid to display
        cmap: is the color map to use, use "gray" for gray scale images, use None for default.

    Returns:
        None
    """
    try:
        target_folder = _os.path.join(target_dir, target_class)

        # Get a random image path
        fig = _plt.figure(figsize=(shape[1] * 3, shape[0] * 3))
        fig.patch.set_facecolor('gray')
        for i in range(0, shape[0] * shape[1]):
            ax = _plt.subplot(shape[0], shape[1], i + 1)
            ax.axis('off')
            image_filename = _random.sample(_os.listdir(target_folder), 1)
            image_fullpath = _os.path.join(target_folder, image_filename[0])
            img = _plt.imread(image_fullpath)
            _plt.imshow(img, cmap=cmap)
            _plt.title(f"{image_filename[0]}: {img.shape}", color='white')

        _plt.show()
    except Exception as e:
        _logging.error(e)


def show_images_from_dataset(dataset: _tf.data.Dataset, class_names=None, shape=(4, 8)) -> None:
    """
    Displays a number of images based on the shape size, the shape size must be the same
    as the batch size of the dataset. Default shape size is (4, 8) so the batch size
    should be 32.

    Args:
        dataset: the dataset containing the images
        class_names: optional list of class names, if none provided then it will be extracted from dataset.
        shape: the shape size

    Returns:
        None
    """
    assert isinstance(dataset, _tf.data.Dataset), f"The dataset supplied is not a tensorflow.data.Dataset."
    x, y = _soml.tf.dataset.get_features_and_labels(dataset=dataset, max_samples=shape[0] * shape[1])

    if class_names is None:
        class_names = _soml.tf.dataset.get_class_names(dataset=dataset)

    show_images_from_nparray_or_tensor(x=x, y=y, class_names=class_names, shape=shape)
