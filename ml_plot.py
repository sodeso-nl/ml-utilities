import ml_internal as mli

import matplotlib.pyplot as plt
import numpy as np
import random
import itertools

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.data import Dataset


########################################################################################################################
# Plotting pandas Dataframe
########################################################################################################################


def plot_histogram_from_dataframe(x, columns):
    """
    Plots a histogram for each of the specified columns in the DataFrame X.

    :param X: a dataframe
    :param columns: columns which exist within the DataFrame.
    """
    for c in columns:
        x.hist(c)


########################################################################################################################
# Plotting history methods
########################################################################################################################


def plot_consecutive_histories(histories, labels, figsize=(10, 6)):
    """
    Plots (when available), the validation loss and accuracy, training loss and accuracy and learning rate.

    :param histories: the history objects returned from fitting models.
    :param labels: the labels for each history object for seperating the epochs
    :param figsize: figure size (default: (10, 6))
    :return: two graphs, one for loss and one for accuracy
    """
    all_loss_history = []
    all_val_loss_history = []
    all_accuracy_history = []
    all_val_accuracy_history = []
    all_mae_history = []
    all_val_mae_history = []
    all_lr_history = []

    first_epoch = 10000
    last_epoch = -1
    for history in histories:
        first_epoch = min(first_epoch, min(history.epoch))
        last_epoch = max(last_epoch, max(history.epoch))

        all_loss_history = [*all_loss_history, *history.history['loss']]
        all_val_loss_history = [*all_val_loss_history, *history.history['val_loss']]

        if 'accuracy' in history.history:
            all_accuracy_history = [*all_accuracy_history, *history.history['accuracy']]
            all_val_accuracy_history = [*all_val_accuracy_history, *history.history['val_accuracy']]

        if 'mae' in history.history:
            all_mae_history = [*all_mae_history, *history.history['mae']]
            all_val_mae_history = [*all_val_mae_history, *history.history['val_mae']]

        if 'lr' in history.history:
            all_lr_history = [*all_lr_history, *history.history['lr']]

    epoch_labels = range(first_epoch + 1, last_epoch + 2)
    ticks = range(len(epoch_labels))
    plt.figure(figsize=figsize, facecolor='#FFFFFF')
    _plot_history_graph_line(all_loss_history, label='Training loss', color='#0000FF')
    _plot_history_graph_line(all_val_loss_history, label='Validation loss', color='#00FF00')
    _plot_history_graph_line(all_lr_history, label='Learning rate', color='#000000', linestyle='dashed')
    _plot_history_ends(histories, labels)
    plt.title('Loss', size=20)
    plt.xticks(ticks, epoch_labels)
    plt.xlabel('Epoch', size=14)
    plt.legend()

    if all_accuracy_history:
        # Start a new figure
        plt.figure(figsize=figsize, facecolor='#FFFFFF')
        _plot_history_graph_line(all_accuracy_history, label='Training accuracy', color='#0000FF')
        _plot_history_graph_line(all_val_accuracy_history, label='Validation accuracy', color='#00FF00')
        _plot_history_graph_line(all_lr_history, label='Learning rate', color='#000000', linestyle='dashed')
        _plot_history_ends(histories, labels)
        plt.title('Accuracy', size=20)
        plt.xticks(ticks, epoch_labels)
        plt.xlabel('Epoch', size=14)
        plt.legend()

    if all_mae_history:
        # Start a new figure
        plt.figure(figsize=figsize, facecolor='#FFFFFF')
        _plot_history_graph_line(all_mae_history, label='Training mae', color='#0000FF')
        _plot_history_graph_line(all_val_mae_history, label='Validation mae', color='#00FF00')
        _plot_history_graph_line(all_lr_history, label='Learning rate', color='#000000', linestyle='dashed')
        _plot_history_ends(histories, labels)
        plt.title('Mean Absolute Accuracy', size=20)
        plt.xticks(ticks, epoch_labels)
        plt.xlabel('Epoch', size=14)
        plt.legend()


def plot_history(history, figsize=(10, 6)):
    plot_consecutive_histories([history], ["Start history"], figsize=figsize)


########################################################################################################################
# Plotting methods for linear / logistic regression
########################################################################################################################


def plot_xy_data_with_label(x, y):
    """
    Plots a graph of the values of x whereas x contains vectors of x/y coordinates and y
    is the label (0 or 1).

    :param X: is an array containing vectors of x/y coordinates.
    :param y: are the associated labels (0=blue, 1=red)
    """
    plt.plot(x[:, 0][y == 1], x[:, 1][y == 1], "bs")
    plt.plot(x[:, 0][y == 0], x[:, 1][y == 0], "r^")

    # X contains two features, x1 and x2
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20)

    # Displaying the plot.
    plt.show()


def plot_decision_boundary(model, x, y):
    """
    Plots the decision boundary created by a model predicting on X.

    Inspired by the following two websites:
    https://cs231n.github.io/neural-networks-case-study

    :param model: the sequence model.
    :param X: array containing vectors with x/y coordinates
    :param y: are the associated labels (0=blue, 1=red)
    """
    # Define the axis boundaries of the plot and create a meshgrid.
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1

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
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(15, 15), text_size=10, norm=False, savefig=False):
    """
      Plots a confusion matrix of the given data.

      :param y_true: Array of truth labels (must be same shape as y_pred).
      :param y_pred: Array of predicted labels (must be same shape as y_true).
      :param class_names: Array of class labels (e.g. string form). If `None`, integer labels are used.
      :param figsize: Size of output figure (default=(15, 15)).
      :param text_size: Size of output figure text (default=10).
      :param norm: normalize values or not (default=False).
      :param savefig: save confusion matrix to file (default=False).

      Plots the decision boundary created by a model predicting on X.
      Inspired by the following two websites:
      https://cs231n.github.io/neural-networks-case-study
    """

    if isinstance(y_true, Dataset):
        raise TypeError('y_true is a dataset, please get the labels from the dataset using '
                        '\'y_labels = get_labels_from_dataset(dataset=dataset, index_only=True)\'')

    # If y_true or y_pred is not a Numpy array then try to convert it.
    y_true = mli.convert_to_numpy_array_if_neccesary(y_true)
    y_pred = mli.convert_to_numpy_array_if_neccesary(y_pred)

    # If the y_true labels are one-hot encoded then convert them to integer encoded labels.
    if mli.is_multiclass_classification(y_true):
        y_true = mli.to_ordinal(y_true)

    # Check if we need to convert multi-class classification one-hot encoding to index or
    # if we are dealing with binary classification, then we need to round the numer to either 0 or 1
    if mli.is_multiclass_classification(y_pred):
        y_pred = mli.to_ordinal(y_pred)
    elif mli.is_binary_classification(y_pred):
        y_pred = mli.to_binary(y_pred)

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
    if class_names:
        labels = class_names
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
    ax.title.set_size(text_size)

    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    # Set treshold for different colors
    threshold = (cm.max() + cm.min()) / 2.
    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    if savefig:
        fig.savefig("./confusion_matrix.png")


def plot_classification_report_f1_score(y_true, y_pred, class_names, figsize=(10, 8)):
    """
    Creates a horizontal bar graph with the F1-Scores of the y_true / y_pred.
    :param y_true: Array of truth labels (must be same shape as y_pred).
    :param y_pred: Array of predicted labels (must be same shape as y_true).
    :param class_names: Array of class labels (e.g. string form). If `None`, integer labels are used.
    :param figsize: Size of output figure (default=(10, 8)).
    """
    if isinstance(y_true, Dataset):
        raise TypeError(
            'y_true is a dataset, please get the labels from the dataset using '
            '\'y_labels = get_labels_from_dataset(dataset=dataset, index_only=True)\'')

    # If y_true or y_pred is not a Numpy array then try to convert it.
    y_true = mli.convert_to_numpy_array_if_neccesary(y_true)
    y_pred = mli.convert_to_numpy_array_if_neccesary(y_pred)

    # If the y_true labels are one-hot encoded then convert them to integer encoded labels.
    if mli.is_multiclass_classification(y_true):
        y_true = mli.to_ordinal(y_true)

    # Check if we need to convert multi-class classification one-hot encoding to index or
    # if we are dealing with binary classification, then we need to round the numer to either 0 or 1
    if mli.is_multiclass_classification(y_pred):
        y_pred = mli.to_ordinal(y_pred)
    elif mli.is_binary_classification(y_pred):
        y_pred = mli.to_binary(y_pred)

    # Generate classification report from SKLearn.
    report_dict = classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names, output_dict=True)

    # Collect all the F1-scores
    f1_scores = {}
    for k, v in report_dict.items():
        if k == 'accuracy':  # Cut-off when the accuracy key is visible.
            break
        else:
            f1_scores[k] = v['f1-score']

    # Sort F1-scores from best to worst
    f1_scores_sorted = dict(sorted(f1_scores.items(), key=lambda item: item[1], reverse=True))

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(f1_scores_sorted))
    scores = ax.barh(y_pos, f1_scores_sorted.values(), align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(f1_scores_sorted.keys())

    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('F1-score')
    ax.set_title('F1-Scores')
    ax.bar_label(scores, fmt='%.2f')

    # Remove anoying white space at top / bottom.
    plt.margins(y=0)

    plt.show()


########################################################################################################################
# Internal methods for plotting history
########################################################################################################################


def _plot_history_ends(histories, labels):
    """
    Internal method which will plot a vertical line showing where a histories last epoch is visible.

    :param histories: the history objects returned from fitting models.
    :param labels: the labels for each history object for seperating the epochs
    """
    for idx, history in enumerate(histories):
        plt.plot([min(history.epoch), min(history.epoch)], plt.ylim(), label=f'{labels[idx]}')


def _plot_history_graph_line(data, label, color, linestyle='solid'):
    """
    Internal method which will plot the information from the fit histroy.

    :param data: the data to plot
    :param label: the label associated with the data
    :param color: color of the line
    :param linestyle: line-style of the line (default: solid)
    """
    if data:
        plt.plot(data, label=label, color=color, linestyle=linestyle, linewidth=1.5)
