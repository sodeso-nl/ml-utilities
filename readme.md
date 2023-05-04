Initial setup environment

To initially setup your environment for Apple Silicon please execute the following instructions:

Install miniforge3:

>> brew install miniforge

Create a new environment in conda:

>> conda env create -f environment.yml

Activate the newly created environment

>> conda activate ml-utilities

To delete the created environment

>> conda remove -n ml-utilities --all

To de-activate the environment

>> conda deactivate