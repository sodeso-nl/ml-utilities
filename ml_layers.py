def collect_layers(model, recursive=True, include_trainable=True, include_non_trainable=True):
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
    for layer in model.layers[:-n]:
        layer.trainable = trainable


def set_trainable_on_last_n_layers(model, n, trainable=True):
    for layer in model.layers[-n:]:
        layer.trainable = trainable


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
    trainable_layer_names = collect_layer_names(model, recursive=True, include_trainable=True, include_non_trainable=False)

    # Disable training completely
    set_trainable_on_layers(model, trainable=False)

    # Load the weights
    model.load_weights(filepath)

    set_trainable_on_layers(model, layer_names=trainable_layer_names, trainable=True)


def list_layers(model, recursive=True, include_trainable=True, include_non_trainable=True):
    layers = collect_layers(model, recursive=recursive, include_trainable=include_trainable, include_non_trainable=include_non_trainable)
    layer_name_col_width = len(max(list(map(lambda layer: layer.name, layers)), key=len))
    layer_type_col_width = len(max(list(map(lambda layer: type(layer).__name__, layers)), key=len))
    layer_shape_col_width = len(max(list(map(lambda layer: str(layer.output_shape), layers)), key=len))

    print(f"{'Row':<5} | {'Name (Type)':<{layer_name_col_width + layer_type_col_width + 3}} | Trainable | Output Shape")
    for layer_number, layer in enumerate(layers):
        if (include_trainable and layer.trainable) or (include_non_trainable and layer.trainable is False):
            print(f"{layer_number:<5} | {layer.name:<{layer_name_col_width}} ({type(layer).__name__:<{layer_type_col_width}}) | {str(layer.trainable):<9} | "
                  f"{str(layer.output_shape):<{layer_shape_col_width}}")