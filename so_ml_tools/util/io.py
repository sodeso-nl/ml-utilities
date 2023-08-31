import os as _os
import tarfile as _tarfile
import zipfile as _zipfile
import urllib as _urllib
import tensorflow as _tf


def download_file(source='https://www.download.com/file.zip', filepath='./data/file.zip') -> None:
    """
    Download a file from source location and store it at the specified destination location.

    Args:
        source: the source URL
        filepath: the destination path and filename
    """
    path = _os.path.dirname(filepath)
    if path:
        _os.makedirs(path, exist_ok=True)
    else:
        _os.path.join('./', filepath)

    # noinspection PyUnresolvedReferences
    _urllib.request.urlretrieve(source, filepath)
    print(f'Download of {source} completed.')


def extract_tgz(filepath='./data/file.tar.gz', folder='./data') -> None:
    """
    Extracts a .tgz file in the specified destination.

    Args:
        filepath: the .tgz file
        folder: the destination location, if not specified then same path as source file.
    """
    if folder is None:
        folder = _os.path.dirname(filepath)
    tgz_file = _tarfile.open(filepath)
    tgz_file.extractall(path=folder)
    tgz_file.close()
    print(f"Extraction of {filepath} completed.")


def extract_zip(filepath='./data/file.zip', folder='./data') -> None:
    """
    Extracts a .zip file in the specified destination.

    Args:
        filepath: the path and filename
        folder: the destination location, if not specified then same path as source file.
    """
    if folder is None:
        folder = _os.path.dirname(filepath)
    zip_file = _zipfile.ZipFile(filepath)
    zip_file.extractall(path=folder)
    zip_file.close()


def list_dir_summary(folder='./data') -> None:
    """
    Lists a summary of the specified folder

    Args:
        folder: the folder to list the contents of
    """
    for dirpath, dirnames, filenames in _os.walk(folder):
        print(f"There are {len(dirnames)} directories, and {len(filenames)} in {dirpath}")


def load_image_as_tensor(filename: str, img_shape=(224, 224), scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    specified shape (img_shape, img_shape, channels)

    Args:
        filename: path to target image
        img_shape: tuple with height/width dimension of target image size
        scale: scale pixel values from 0-255 to 0-1 or not

    Returns:
        Image tensor of shape (img_shape, img_shape, 3)
    """

    # Read in the image
    img = _tf.io.read_file(filename=filename)

    # Decode image into tensor
    img = _tf.io.decode_image(contents=img, channels=3)

    # # Resize the image (height / width)
    img = _tf.image.convert_image_dtype(img, _tf.float32)
    img = _tf.image.resize(images=img, size=[img_shape[0], img_shape[1]])

    if not scale:
        return _tf.cast(img * 255, _tf.int32)
    else:
        return img
