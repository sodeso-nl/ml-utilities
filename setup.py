from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Machine Learning Convenience Functions'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="ml-utilities",
    version=VERSION,
    author="Ronald Mathies",
    description="Utilities library for TensorFlow",
    long_description_content_type="text/plain",
    long_description="A various set of utility functions that can be used in conjunction with Tensorflow.",
    packages=find_packages(),
    install_requires=[
        # 'scikit-learn>=1.2.2',
        # 'matplotlib>=3.7.1',
        # 'pandas>=2.0.0',
        # 'tensorflow>=2.8.0'
    ],
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