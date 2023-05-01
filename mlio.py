import os
import tarfile
import urllib


def download_file(source, destination):
    """
    Download a file from source location and store it at the specified destination location.
    :param source: the source URL
    :param destination: the destination path and filename
    """
    path = os.path.dirname(destination)
    os.makedirs(path, exist_ok=True)
    urllib.request.urlretrieve(source, destination)


def extract_tgz(file):
    """
    Extracts a .tgz file in the same path as where the .tgz file exists.
    :param file: the path and filename
    """
    path = os.path.dirname(file)
    housing_tgz = tarfile.open(file)
    housing_tgz.extractall(path=path)
    housing_tgz.close()