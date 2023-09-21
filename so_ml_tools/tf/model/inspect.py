import tensorflow as _tf


def collect_layers(model: _tf.keras.Model, recursive=True, include_trainable=True, include_non_trainable=True) -> list:
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
            layers = [*layers, *collect_layers(layer, recursive=recursive, include_trainable=include_trainable,
                                               include_non_trainable=include_non_trainable)]
        else:
            if (include_trainable and layer.trainable) or (include_non_trainable and layer.trainable is False):
                layers.append(layer)

    return layers


def collect_layer_names(model: _tf.keras.Model, recursive=True, include_trainable=True, include_non_trainable=True) -> \
        list[str]:
    layers = collect_layers(model, recursive=recursive, include_trainable=include_trainable,
                            include_non_trainable=include_non_trainable)
    return list(map(lambda layer: layer.name, layers))


def set_trainable_on_layers(model: _tf.keras.Model, layer_names=None, trainable=True) -> None:
    for layer in collect_layers(model, recursive=True, include_trainable=True, include_non_trainable=True):
        if layer_names is None or layer.name in layer_names:
            layer.trainable = trainable


def set_trainable_on_first_n_layers(model: _tf.keras.Model, n, trainable=True) -> None:
    layers = collect_layers(model, recursive=True, include_trainable=True, include_non_trainable=True)
    for layer in layers[:n]:
        layer.trainable = trainable


def set_trainable_on_last_n_layers(model: _tf.keras.Model, n, trainable=True) -> None:
    layers = collect_layers(model, recursive=True, include_trainable=True, include_non_trainable=True)
    for layer in layers[-n:]:
        layer.trainable = trainable


def list_model(model, recursive=True, include_trainable=True, include_non_trainable=True) -> None:
    layers = collect_layers(model, recursive=recursive, include_trainable=include_trainable,
                            include_non_trainable=include_non_trainable)
    list_layers(layers=layers, include_trainable=include_trainable, include_non_trainable=include_non_trainable)

    total_params = sum([_tf.size(var).numpy() for var in model.variables])
    trainable_params = sum([_tf.size(var).numpy() for var in model.trainable_variables])
    non_trainable_params = total_params - trainable_params

    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")


def list_layers(layers: list[_tf.keras.layers.Layer], include_trainable=True, include_non_trainable=True) -> None:
    layer_name_col_width = len(max(list(map(lambda l: l.name, layers)), key=len))
    layer_type_col_width = len(max(list(map(lambda l: type(l).__name__, layers)), key=len))
    layer_dtype_col_width = len(max(list(map(lambda l: str(l.dtype), layers)), key=len))
    layer_dtype_policy_col_width = len(max(list(map(lambda l: str(l.dtype_policy.name), layers)), key=len))

    print(
        f"{'row':<5} | {'name (type)':<{layer_name_col_width + layer_type_col_width + 3}} | {'dtype':<{layer_dtype_col_width}} | {'policy':<{layer_dtype_policy_col_width}} | trainable | output shape")
    for layer_number, layer in enumerate(layers):
        if (include_trainable and layer.trainable) or (include_non_trainable and layer.trainable is False):
            params = sum([_tf.size(var).numpy() for var in layer.trainable_variables])
            print(
                f"{layer_number:<5} | {layer.name:<{layer_name_col_width}} ({type(layer).__name__:<{layer_type_col_width}}) | {str(layer.dtype):<{layer_dtype_col_width}} | {str(layer.dtype_policy.name):<{layer_dtype_policy_col_width}} | {str(layer.trainable):<9} | "
                f"{str(layer.output_shape)}{params:,}")
