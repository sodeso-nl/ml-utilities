import os
import random
import logging

import matplotlib.pyplot as plt
import tensorflow as tf


def load_image_dataset_from_directory(directory: str,
                                      label_mode='categorical',
                                      image_size=(256, 256),
                                      batch_size=32,
                                      class_names=None,
                                      color_mode='rgb',
                                      shuffle=True, validation_split=None):
    return tf.data.Dataset.image_dataset_from_directory(directory=directory,
                                                        label_mode=label_mode,
                                                        image_size=image_size,
                                                        batch_size=batch_size,
                                                        class_names=class_names,
                                                        color_mode=color_mode,
                                                        shuffle=shuffle,
                                                        validation_split=validation_split
                                                        )


def show_random_image_from_disk(target_dir, target_class, shape=(4, 6), cmap='gray') -> None:
    """
    Shows a random image from the file system.
    :param target_dir: The target directory, for example /food101/train
    :param target_class: The target class name, for example /steak
    :param shape: is the number of images in a grid to display
    :param cmap: is the color map to use, use "gray" for gray scale images, use None for default.
    """
    # Setup the target directory (we'll view images from here)
    try:
        target_folder = os.path.join(target_dir, target_class)

        # Get a random image path
        fig = plt.figure(figsize=(shape[1] * 3, shape[0] * 3))
        fig.patch.set_facecolor('gray')
        for i in range(0, shape[0] * shape[1]):
            ax = plt.subplot(shape[0], shape[1], i + 1)
            ax.axis('off')

            image_filename = random.sample(os.listdir(target_folder), 1)
            image_fullpath = os.path.join(target_folder, image_filename[0])
            img = plt.image.mpimg.imread(image_fullpath)
            plt.imshow(img, cmap=cmap)
            plt.title(f"{image_filename[0]}: {img.shape}", color='white')

        plt.show()
    except Exception as e:
        logging.error(e)