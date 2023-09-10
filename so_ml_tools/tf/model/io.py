import os as _os
import pickle as _pickle

import tensorflow as _tf
import keras as _keras

import so_ml_tools as _soml


def save_model(model: _tf.keras.Model, filepath="./models/model.h5", save_format="h5") -> None:
    """
    This method will call the default model.save method.

    :param model: the model to save
    :param filepath: String or PathLike, path to the file to save the model to.
    :param save_format: The format to store the data (tf/h5)
    """
    model.save(filepath=filepath, save_format=save_format)


def load_model(filepath="./models/model.h5", custom_objects=None) -> _keras.Model:
    """
    This method will call the default load_model method.

    Important: When loading a model containing TensorFlow Hub layers you will need to provide this
    as a custom object:

    custom_objects={"KerasLayer": hub.KerasLayer}

    However, this is not needed when the model was saved in the SaveModel format since these
    details are present in the saved state.

    :param filepath: String or PathLike, path to the file to save the model to.
    :param custom_objects: custom objects for example KerasLayer from TensorFlow Hub
    :return: the loaded model
    """
    return _tf.keras.models.load_model(filepath=filepath, custom_objects=custom_objects)


def save_weights(model: _tf.keras.Model, filepath: str, save_format="h5") -> None:
    """
    Saves only the weights of the model (see model.save_weights )

    :param model:
    :param filepath:
    :param save_format:
    """
    path = _os.path.dirname(filepath)
    if path:
        _os.makedirs(path, exist_ok=True)
    model.save_weights(filepath, save_format=save_format)


def load_weights(model: _tf.keras.Model, filepath: str) -> None:
    """
    Loads the weights into the given model.

    Disables trainable on the model and on all the layers, then loads the weights and
    restores the trainable configuration on the model and weights.

    :param model: the model to load the weights into
    :param filepath: String or PathLike, path to the file to save the weights
                to. When saving in TensorFlow format, this is the prefix used
                for checkpoint files (multiple files are generated). Note that
                the '.h5' suffix causes weights to be saved in HDF5 format.
    """

    # Collect all layers that are trainable
    trainable_layer_names = _soml.tf.model.layer.collect_layer_names(model, recursive=True, include_trainable=True,
                                                                     include_non_trainable=False)

    # Disable training completely
    _soml.tf.model.layer.set_trainable_on_layers(model, trainable=False)

    # Load the weights
    model.load_weights(filepath)

    _soml.tf.model.layer.set_trainable_on_layers(model, layer_names=trainable_layer_names, trainable=True)


def save_model_alt(model: _tf.keras.Model, name: str, directory="./models", format='h5') -> None:
    """
    Alternative solution to saving a model since the default implementation has issues with augmentation layers.

    Use in conjunction with load_model_alt, this method will create two files, one .pkl file
    with the configuration of the model, and a (for example .h5) file containing the weights of the model.

    :param model: The model to save
    :param directory: The target directory to save the model to
    :param name: The name of the model, used for filenames, use without extension
    :param format: The format to store the data (tf/h5)
    """
    config = model.get_config()

    # Check if target directory exists, if not, create it
    _os.makedirs(directory, exist_ok=True)

    config_file = _os.path.join(directory, name + '.pkl')
    with open(config_file, 'wb') as fp:
        _pickle.dump(config, fp)

    model_file = _os.path.join(directory, name + '.' + format)
    model.save_weights(model_file, save_format=format)


def load_model_alt(name: str, directory="./models", format='h5'):
    """
    Alternative solution to saving a model since the default implementation has issues with augemntation layers.

    This method will create a new model based on the configuration stored in a .pkl file and the weights stored in
    a (for example .h5) file.

    Use in conjunction with save_model_alt

    :param directory: The target directory to save the model to
    :param name: The name of the model, used for filenames, use without extension
    :param format: The format to store the data (tf/h5)
    :return: the loaded model
    """
    config_file = _os.path.join(directory, name + '.pkl')
    with open(config_file, 'rb') as fp:
        model = _keras.Model.from_config(_pickle.load(fp))
        model_file = _os.path.join(directory, name + '.' + format)
        model.load_weights(model_file)
        return model
