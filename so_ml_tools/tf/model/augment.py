import keras as _keras


def create_augmentation_layer(random_flip_h=True,
                              random_flip_v=True,
                              random_rotate=True,
                              random_height=False,
                              random_width=False,
                              random_translation=True,
                              random_zoom=True) -> _keras.Sequential:
    steps = []
    if random_flip_h or random_flip_v:
        if random_flip_h and random_flip_v:
            steps.append(_keras.layers.RandomFlip('horizontal_and_vertical'))
        elif random_flip_h:
            steps.append(_keras.layers.RandomFlip('horizontal'))
        elif random_flip_v:
            steps.append(_keras.layers.RandomFlip('vertical'))

    if random_rotate:
        steps.append(_keras.layers.RandomRotation(.2))

    if random_height:
        steps.append(_keras.layers.RandomHeight(.2))

    if random_width:
        steps.append(_keras.layers.RandomWidth(.2))

    if random_zoom:
        steps.append(_keras.layers.RandomZoom(.2))

    if random_translation:
        steps.append(_keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1))

    return _keras.Sequential(steps, name='data_augmentation')
