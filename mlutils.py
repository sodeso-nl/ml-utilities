import matplotlib.pyplot as plt
import numpy as np
import random
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def plot_model_history(history, figsize=(10, 6)):
    """
    Plots (when available), the validation loss and accuracy, training loss and accuracy and learning rate.

    :param history: the history object returned from fitting a model.
    :param figsize: figure size (default: (10, 6))
    :return: two graphs, one for loss and one for accuracy
    """
    plt.figure(figsize=figsize)

    # Plot the traiing loss and accuracy
    plt.plot(history.history['loss'], label='Training loss', color='#0000FF', linewidth=1.5)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation loss', color='#00FF00', linewidth=1.5)

    # Plot the learning rate
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning rate', color='#000000', linewidth=1.5, linestyle='--')

    plt.title('Loss', size=20)
    plt.xticks(history.epoch)
    plt.xlabel('Epoch', size=14)
    plt.legend()

    # Start a new figure
    plt.figure(figsize=figsize, facecolor='#FFFFFF')

    # Plot the validation loss and accuracy
    plt.plot(history.history['accuracy'], label='Training accuracy', color='#0000FF', linewidth=1.5)
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation accuracy', color='#00FF00', linewidth=1.5)

    # Plot the learning rate
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning rate', color='#000000', linewidth=1.5, linestyle='--')
    plt.title('Accuracy', size=20)
    plt.xticks(history.epoch)
    plt.xlabel('Epoch', size=14)
    plt.legend()

    plt.show()


def plot_xy_data_with_label(X, y):
    """
    Plots a graph of the values of X.

    :param X: is an array containing vectors of x/y coordinates.
    :param y: are the associated labels (0=blue, 1=red)
    """
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "r^")

    # X contains two features, x1 and x2
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20)

    # Displaying the plot.
    plt.show()


def normalize_xy_data(X):
    """
    Normalizes an array containing vectors of x/y coordinates so that the array does not contain
    negative values.

    :param X: the vector containing values from -X to +X which need to be normalized between 0 and 1
    :return: the normalized vector.
    """
    X = X + (np.abs(np.min(X[:, 0])))
    X = X / np.max(X[:, 0])
    X = X + (np.abs(np.min(X[:, 1])))
    return X / np.max(X[:, 1])


def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary created by a model predicting on X.

    Inspired by the following two websites:
    https://cs231n.github.io/neural-networks-case-study

    :param model: the sequence model.
    :param X: array containing vectors with x/y coordinates
    :param y: are the associated labels (0=blue, 1=red)
    """
    # Define the axis boundaries of the plot and create a meshgrid.
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Create X value (we're going to make predictions on these)
    x_in = np.c_[xx.ravel(), yy.ravel()]  # Stack 2D arrays together

    # Make predictions
    y_pred = model.predict(x_in)

    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification")
        # We have to reshape our predictions to get them ready for plotting.
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def find_learning_rate_division(learning_rate, epochs):
    """
    Finds the optimal division which can be used in a learning rate scheduler.

    epochs = 50
    initial_lr = 0.001

    division = find_learning_rate_division(initial_lr, epochs)
    lr_scheduler = LearningRateScheduler(lambda epoch: initial_lr * 10 ** (epoch / division))

    :param learning_rate: initial learning rate
    :param epochs: number of epochs that the model will train for
    :return: the minimum learning rate, maximum learning rate and the division which can be used in the scheduler.
    """
    min_lr = 0.
    max_lr = 0.
    division = 1000
    while max_lr < 0.1:
        min_lr = learning_rate * 10 ** (1 / division)
        max_lr = learning_rate * 10 ** (epochs / division)
        division -= 1
    return (min_lr, max_lr, division)


def split_train_test_data(*arrays,
                          test_size=.2,
                          train_size=.8,
                          random_state=42,
                          shuffle=True):
    """
    Usage:

    X_train, X_test, y_train, y_test =
        split_train_test_data(X, y)
    """
    return train_test_split(*arrays,
                            test_size=test_size,
                            train_size=train_size,
                            random_state=random_state,
                            shuffle=shuffle)


def show_images_from_nparray_or_tensor(X, y, class_labels=None, indexes=None, size=4, cmap='gray'):
    """
    Shows images stored in a tensor / numpy array. The array should be a vector of images.

    :param X: is an array containing vectors of images.
    :param y: are the associated labels
    :param indexes: None to pick random, otherwise an array of indexes to display
    :param size: is the number of images to display
    :param cmap: is the collor map to use, use "gray" for gray scale images, use None for default.
    """

    plt.figure(figsize=(size * 4, size * 2))
    for i in range(size):
        ax = plt.subplot(2, size, i + 1)
        ax.axis('off')

        if indexes is None:
            rand_index = random.choice(range(len(X)))
        else:
            rand_index = indexes[i]

        plt.imshow(X[rand_index], cmap=cmap)

        if class_labels is None:
            plt.title(y[rand_index])
        else:
            plt.title(class_labels[y[rand_index]])

        plt.axis = (False)


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
    """
      y_true    =      The actual result
      y_pred    =      The predicted result
      classes   =      The classes in case of a multi-class validation
      figsize   =      Size of the graph
      text_size =      Font-size
      Plots the decision boundary created by a model predicting on X.
      Inspired by the following two websites:
      https://cs231n.github.io/neural-networks-case-study
    """

    # For the situation where y_pred is a multi-class non one-hot-encoded value:
    # y_pred = [[0.9, 0.6], [0.2, 0.7]]
    # and y_true contains the indices of which class it is.
    # y_true = [9, 2, 1, 1, 6]
    if y_pred.ndim == 2 and isinstance(y_pred[0][0], np.floating) and y_true.ndim == 1 and isinstance(y_true[0],
                                                                                                      np.integer):
        y_pred = np.argmax(y_pred, axis=1)

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize our confusion matrix
    n_classes = cm.shape[0]
    # Let's prettify it
    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    # Set labels to be classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    # label the axes
    ax.set(title="Confusion matrix",
           xlabel="predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    # Adjust label size
    ax.yaxis.label.set_size(text_size);
    ax.xaxis.label.set_size(text_size);
    ax.title.set_size(text_size);
    # Set treshold for different colors
    threshold = (cm.max() + cm.min()) / 2.
    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}, ({cm_norm[i, j] * 100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)
