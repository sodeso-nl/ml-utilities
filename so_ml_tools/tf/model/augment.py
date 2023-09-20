import keras as _keras


def create_augmentation_layer(random_flip_h=True,
                              random_flip_v=True,
                              random_rotate=True,
                              random_height=True,
                              random_width=True,
                              random_zoom=True,
                              rescaling=True) -> _keras.Sequential:
    steps = []
    if random_flip_h or random_flip_v:
        if random_flip_h and random_flip_v:
            steps.append(_keras.layers.RandomFlip(mode='horizontal_and_vertical'))
        elif random_flip_h:
            steps.append(_keras.layers.RandomFlip(mode='horizontal'))
        elif random_flip_v:
            steps.append(_keras.layers.RandomFlip(mode='vertical'))

    if random_rotate:
        steps.append(_keras.layers.RandomRotation(factor=.2))

    if random_height:
        steps.append(_keras.layers.RandomHeight(factor=.2))

    if random_width:
        steps.append(_keras.layers.RandomWidth(factor=.2))

    if random_zoom:
        steps.append(_keras.layers.RandomZoom(factor=.2))

    if rescaling:
        steps.append(_keras.layers.Rescaling(factor=1 / 255.))

    return _keras.Sequential(steps, name='data_augmentation')
