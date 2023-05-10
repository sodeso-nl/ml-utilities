def collect_trainable_layers(model):
    """
    Returns the names of all layers that are trainable.
    :param model: the model
    :return: a list of names of layers that have layer.trainable = True
    """
    trainable_layers = []

    for layer in model.layers:
        if hasattr(layer, 'layers'):
            trainable_layers = [*trainable_layers, *collect_trainable_layers(layer)]
        else:
            if layer.trainable:
                trainable_layers.append(layer.name)

    return trainable_layers


def enable_trainable_layers(model, layer_names):
    """
    Set layer.trainable = True on all layers with a name in the layer_names list within the specified model.
    :param model: the model on which the layers should be enabled for training
    :param layer_names: the names of the layers that should be enabled.
    """
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            enable_trainable_layers(layer, layer_names)
        else:
            if layer.name in layer_names:
                layer.trainable = True


def disable_trainable_layers(model, layer_names):
    """
    Set layer.trainable = False on all layers with a name in the layer_names list within the specified model.
    :param model: the model on which the layers should be disabled for training
    :param layer_names: the names of the layers that should be disabled.
    """
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            disable_trainable_layers(layer, layer_names)
        else:
            if layer.name in layer_names:
                layer.trainable = False


def disable_trainable_layers_all(model):
    """
    Disables trainable on the model and on all the layers.
    :param model: the model to disable training on.
    """
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            disable_trainable_layers_all(layer)
        else:
            layer.trainable = False


def load_weights(model, filepath):
    """
    Disables trainable on the model and on all the layers.
    :param model: the model to load the weights into
    :param filepath: String or PathLike, path to the file to save the weights
                to. When saving in TensorFlow format, this is the prefix used
                for checkpoint files (multiple files are generated). Note that
                the '.h5' suffix causes weights to be saved in HDF5 format.
    """

    # Collect all layers that are trainable
    trainable_layers = collect_trainable_layers(model)

    # Disable training completely
    model.trainable = False

    # Load the weights
    model.load_weights(filepath)

    # Set model to trainable (which in turn will also set all layers to trainable.
    model.trainable = True

    # Now disable all layers to be trainable so that we can then enable only the layers that were trainable before.
    disable_trainable_layers_all(model)
    enable_trainable_layers(model, trainable_layers)
