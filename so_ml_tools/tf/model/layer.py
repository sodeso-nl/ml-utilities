import tensorflow as _tf

import so_ml_tools as _soml


def set_trainable_on_layers(model: _tf.keras.Model, layer_names=None, trainable=True) -> None:
    for layer in _soml.tf.model.inspect.collect_layers(model, recursive=True, include_trainable=True, include_non_trainable=True):
        if layer_names is None or layer.name in layer_names:
            layer.trainable = trainable


def set_trainable_on_first_n_layers(model: _tf.keras.Model, n, trainable=True) -> None:
    layers = _soml.tf.model.inspect.collect_layers(model, recursive=True, include_trainable=True, include_non_trainable=True)
    for layer in layers[:n]:
        layer.trainable = trainable


def set_trainable_on_last_n_layers(model: _tf.keras.Model, n, trainable=True) -> None:
    layers = _soml.tf.model.inspect.collect_layers(model, recursive=True, include_trainable=True, include_non_trainable=True)
    for layer in layers[-n:]:
        layer.trainable = trainable
