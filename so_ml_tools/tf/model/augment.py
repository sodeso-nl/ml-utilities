import keras as _keras
import so_ml_tools as _soml


def create_augmentation_layer(random_flip_h=True,
                              random_flip_v=True,
                              random_rotate=True,
                              random_height=True,
                              random_width=True,
                              random_translation=True,
                              random_zoom=True,
                              verbose=1) -> _keras.Sequential:
    steps = []
    if random_flip_h or random_flip_v:
        if random_flip_h and random_flip_v:
            steps.append(_keras.layers.RandomFlip('horizontal_and_vertical'))
        elif random_flip_h:
            steps.append(_keras.layers.RandomFlip('horizontal'))
        elif random_flip_v:
            steps.append(_keras.layers.RandomFlip('vertical'))

    if random_rotate:
        steps.append(_keras.layers.RandomRotation(.1))

    if random_height:
        steps.append(_keras.layers.RandomHeight(.1))

    if random_width:
        steps.append(_keras.layers.RandomWidth(.1))

    if random_zoom:
        steps.append(_keras.layers.RandomZoom(.1))

    if random_translation:
        steps.append(_keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1))

    augmentation_layers = _keras.Sequential(steps, name='data_augmentation')
    if verbose:
        _soml.tf.model.inspect.list_model(model=augmentation_layers, recursive=False)

    return augmentation_layers
