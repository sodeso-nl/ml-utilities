import os as _os
import random as _random
import logging as _logging

import matplotlib.pyplot as _plt
import tensorflow as _tf


def load_image_dataset_from_directory(directory: str,
                                      label_mode='categorical',
                                      image_size=(256, 256),
                                      batch_size=32,
                                      class_names=None,
                                      color_mode='rgb',
                                      shuffle=True, validation_split=None):
    return _tf.keras.utils.image_dataset_from_directory(directory=directory,
                                                        label_mode=label_mode,
                                                        image_size=image_size,
                                                        batch_size=batch_size,
                                                        class_names=class_names,
                                                        color_mode=color_mode,
                                                        shuffle=shuffle,
                                                        validation_split=validation_split
                                                        )
