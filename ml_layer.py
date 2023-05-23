import os
import pickle

from keras import Model


def collect_layers(model, recursive=True, include_trainable=True, include_non_trainable=True):
    """
    Collects layers from the given model, if the model is functional then there might be sub-branches
    inside the model, for example:

    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomHeight(.2),
        RandomWidth(.2),
        RandomRotation(.2),
        RandomZoom(.2),
    ], name="data_augemtation")

    inputs = Input(shape=INPUT_SHAPE, name="input_layer")
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D(name='global_average_pooling_2d')(x)
    outputs = Dense(10, activation="softmax", name="output_layer")(x)
    model_2 = Model(inputs, outputs)

    Where in this case the data_augmentation is a sub-branch inside the functional model, use recursive to also
    traverse these layers.

    :param model: the model
    :param recursive: follow recursive
    :param include_trainable: include layers that are set as trainable (default: True)
    :param include_non_trainable: include layers that are set as non-trainable (default: True)
    :return: the layers
    """
    layers = []

    for layer in model.layers:
        if hasattr(layer, 'layers') and recursive:
            layers = [*layers, *collect_layers(layer, recursive=recursive, include_trainable=include_trainable, include_non_trainable=include_non_trainable)]
        else:
            if (include_trainable and layer.trainable) or (include_non_trainable and layer.trainable is False):
                layers.append(layer)

    return layers


def collect_layer_names(model, recursive=True, include_trainable=True, include_non_trainable=True):
    layers = collect_layers(model, recursive=recursive, include_trainable=include_trainable, include_non_trainable=include_non_trainable)
    return list(map(lambda layer: layer.name, layers))


def set_trainable_on_layers(model, layer_names=None, trainable=True):
    for layer in collect_layers(model, recursive=True, include_trainable=True, include_non_trainable=True):
        if layer_names is None or layer.name in layer_names:
            layer.trainable = trainable


def set_trainable_on_first_n_layers(model, n, trainable=True):
    layers = collect_layers(model, recursive=True, include_trainable=True, include_non_trainable=True)
    for layer in layers[:n]:
        layer.trainable = trainable


def set_trainable_on_last_n_layers(model, n, trainable=True):
    layers = collect_layers(model, recursive=True, include_trainable=True, include_non_trainable=True)
    for layer in layers[-n:]:
        layer.trainable = trainable


def list_model(model, recursive=True, include_trainable=True, include_non_trainable=True):
    layers = collect_layers(model, recursive=recursive, include_trainable=include_trainable, include_non_trainable=include_non_trainable)
    list_layers(layers=layers, include_trainable=include_trainable, include_non_trainable=include_non_trainable)


def list_layers(layers, include_trainable=True, include_non_trainable=True):
    layer_name_col_width = len(max(list(map(lambda l: l.name, layers)), key=len))
    layer_type_col_width = len(max(list(map(lambda l: type(l).__name__, layers)), key=len))
    layer_shape_col_width = len(max(list(map(lambda l: str(l.output_shape), layers)), key=len))

    print(f"{'Row':<5} | {'Name (Type)':<{layer_name_col_width + layer_type_col_width + 3}} | Trainable | Output Shape")
    for layer_number, layer in enumerate(layers):
        if (include_trainable and layer.trainable) or (include_non_trainable and layer.trainable is False):
            print(f"{layer_number:<5} | {layer.name:<{layer_name_col_width}} ({type(layer).__name__:<{layer_type_col_width}}) | {str(layer.trainable):<9} | "
                  f"{str(layer.output_shape):<{layer_shape_col_width}}")


########################################################################################################################
# Saving / loading models
########################################################################################################################


def save_weights(model, filepath, save_format="h5"):
    """
    Saves
    :param model:
    :param filepath:
    :param save_format:
    :return:
    """
    path = os.path.dirname(filepath)
    if path:
        os.makedirs(path, exist_ok=True)
    model.save(filepath, save_format=save_format)


def load_weights(model, filepath):
    """
    Disables trainable on the model and on all the layers, then loads the weights and
    restores the trainable configuration on the model and weights.

    :param model: the model to load the weights into
    :param filepath: String or PathLike, path to the file to save the weights
                to. When saving in TensorFlow format, this is the prefix used
                for checkpoint files (multiple files are generated). Note that
                the '.h5' suffix causes weights to be saved in HDF5 format.
    """

    # Collect all layers that are trainable
    trainable_layer_names = collect_layer_names(model, recursive=True, include_trainable=True, include_non_trainable=False)

    # Disable training completely
    set_trainable_on_layers(model, trainable=False)

    # Load the weights
    model.load_weights(filepath,)

    set_trainable_on_layers(model, layer_names=trainable_layer_names, trainable=True)


def save_model_alt(model, directory, name, format='h5'):
    """
    Alternative solution to saving a model since the default implementation has issues with augemntation layers.

    Use in conjunction with load_model_alt

    :param model: The model to save
    :param directory: The target directory to save the model to
    :param name: The name of the model, used for filenames, use without extension
    :param format: The format to store the data (tf/h5)
    """
    config = model.get_config()

    # Check if target directory exists, if not, create it
    os.makedirs(directory, exist_ok=True)

    config_file = os.path.join(directory, name + '.pkl')
    with open(config_file, 'wb') as fp:
        pickle.dump(config, fp)

    model_file = os.path.join(directory, name + '.' + format)
    model.save(model_file, save_format="tf")


def load_model_alt(directory, name, format='h5'):
    """
    Alternative solution to saving a model since the default implementation has issues with augemntation layers.

    Use in conjunction with save_model_alt

    :param directory: The target directory to save the model to
    :param name: The name of the model, used for filenames, use without extension
    :param format: The format to store the data (tf/h5)
    :return: the loaded model
    """
    config_file = os.path.join(directory, name + '.pkl')
    with open(config_file, 'rb') as fp:
        model = Model.from_config(pickle.load(fp))
        model_file = os.path.join(directory, name + '.' + format)
        model.load_weights(model_file)
        return model