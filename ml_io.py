import os
import tarfile
import zipfile
import urllib


def download_file(source, destination):
    """
    Download a file from source location and store it at the specified destination location.
    :param source: the source URL
    :param destination: the destination path and filename
    """
    path = os.path.dirname(destination)
    if not path:
        os.makedirs(path, exist_ok=True)
    else:
        os.path.join('./', destination)

    urllib.request.urlretrieve(source, destination)


def extract_tgz(file, folder='.'):
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


def extract_zip(file, folder='.'):
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


def list_dir_summary(folder):
    """
    Lists a summary of the specified folder
    :param folder: the folder to list the contents of
    """
    for dirpath, dirnames, filenames in os.walk(folder):
        print(f"There are {len(dirnames)} directories, and {len(filenames)} in {dirpath}")
