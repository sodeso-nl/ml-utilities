import os
import tarfile
import zipfile
import urllib

import tensorflow as tf

def download_file(source, destination) -> None:
    """
    Download a file from source location and store it at the specified destination location.
    :param source: the source URL
    :param destination: the destination path and filename
    """
    path = os.path.dirname(destination)
    if path:
        os.makedirs(path, exist_ok=True)
    else:
        os.path.join('./', destination)

    # noinspection PyUnresolvedReferences
    urllib.request.urlretrieve(source, destination)


def extract_tgz(file, folder='.') -> None:
    """
    Extracts a .tgz file in the specified destination.
    :param file: the .tgz file
    :param folder: the destination location, if not specified then same path as source file.
    """
    if folder is None:
        folder = os.path.dirname(file)
    tgz_file = tarfile.open(file)
    tgz_file.extractall(path=folder)
    tgz_file.close()


def extract_zip(file, folder='.') -> None:
    """
    Extracts a .zip file in the specified destination.

    :param file: the path and filename
    :param folder: the destination location, if not specified then same path as source file.
    """
    if folder is None:
        folder = os.path.dirname(file)
    zip_file = zipfile.ZipFile(file)
    zip_file.extractall(path=folder)
    zip_file.close()


def list_dir_summary(folder) -> None:
    """
    Lists a summary of the specified folder
    :param folder: the folder to list the contents of
    """
    for dirpath, dirnames, filenames in os.walk(folder):
        print(f"There are {len(dirnames)} directories, and {len(filenames)} in {dirpath}")


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