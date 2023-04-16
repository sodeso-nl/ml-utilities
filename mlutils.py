import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split


def plot_model_history(history, figsize=(10, 6)):
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
    X       = is an array containing vectors of x/y coordinates.
    y       = are the associated labels (0=blue, 1=red)
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
    :param X: the vector containing values from -X to +X which need to be normalized between 0 and 1
    :return: the normalized vector.
    """
    X = X + (np.abs(np.min(X[:, 0])))
    X = X / np.max(X[:, 0])
    X = X + (np.abs(np.min(X[:, 1])))
    return X / np.max(X[:, 1])


def plot_decision_boundary(model, X, y):
    """
    model =      The sequence model.
    X     =      Array containing vectors with x/y coordinates
    y     =      Are the associated labels (0=blue, 1=red)

    Plots the decision boundary created by a model predicting on X.
    Inspired by the following two websites:
    https://cs231n.github.io/neural-networks-case-study
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


def find_learning_rate_range(learning_rate, epochs):
    """
    Arguments:
      learning_rate =            starting learning rate
      epochs        =            number of epochs to train for

    Returns:
      Minimum learning rate
      Maximum learning rate
      Division to use in the LearningRateScheduler (lambda epoch: learning_rate * 10 ** (epoch / division))
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
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True):
    return train_test_split(arrays,
                     test_size=test_size,
                     train_size=train_size,
                     random_state=random_state,
                     shuffle=shuffle)