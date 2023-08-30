from setuptools import setup, find_packages
from so_ml_tools import __version__

DESCRIPTION = 'Machine Learning Convenience Functions'

with open("requirements.txt", "r") as fh:
    install_requires = fh.read()

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="so-ml-tools",
    version=__version__,
    author="Ronald Mathies",
    description="Utilities library for TensorFlow",
    long_description_content_type="text/plain",
    long_description="A various set of utility functions that can be used in conjunction with Tensorflow.",
    packages=find_packages(),
    install_requires=install_requires,
    keywords=['python', 'machine', 'deep', 'learning', 'tensorflow'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux"
    ]
)