import os as _os
import tarfile as _tarfile
import zipfile as _zipfile
import urllib as _urllib


def download_file(source='https://www.download.com/file.zip', filepath='./data/file.zip', overwrite=False) -> None:
    """
    Download a file from source location and store it at the specified destination location.

    Args:
        source: the source URL
        filepath: the destination path and filename
        overwrite: if the file already exists then skip the download
    """
    if not overwrite and _os.path.exists(filepath):
        print(f'Download of {source} skipped, file {filepath} exists.')
        return

    path = _os.path.dirname(filepath)
    if path:
        _os.makedirs(path, exist_ok=True)
    else:
        _os.path.join('./', filepath)

    # noinspection PyUnresolvedReferences
    print(f'Download of {source} started.')
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

    print(f"Extraction of {filepath} started.")
    with _tarfile.open(name=filepath) as tObject:
        tObject.extractall(path=folder)

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

    print(f"Extraction of {filepath} started.")
    with _zipfile.ZipFile(file=filepath) as zObject:
        zObject.extractall(path=folder)

    print(f"Extraction of {filepath} completed.")


def list_dir_summary(folder='./data') -> None:
    """
    Lists a summary of the specified folder

    Args:
        folder: the folder to list the contents of
    """
    for dirpath, dirnames, filenames in _os.walk(folder):
        print(f"There are {len(dirnames)} directories, and {len(filenames)} in {dirpath}")
