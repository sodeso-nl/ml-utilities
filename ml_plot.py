import ext.ml_internal as mlint

import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
import collections.abc
import pandas as pd
import tensorflow as tf

from pandas.api.types import is_numeric_dtype, is_object_dtype

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.interpolate import interp1d

########################################################################################################################
# Plotting pandas Dataframe
########################################################################################################################


def plot_histogram_from_dataframe(dataframe: pd.DataFrame, column_names: list[str] = None, log=False, min_nunique=3, max_nunique=50, figsize=None, cols=3, verbose=1):
    """
    Plots a histogram for each of the numeric columns in the DataFrame.

    :param dataframe: the pandas dataframe
    :param column_names: columns which exist within the DataFrame if none specified all columns will be processed
    :param log: set to True to enable logarithmic scaling
    :param min_nunique: minimum number of unique values present, if lower then this then no graph will be displayed (since it is basically boolean)
    :param max_nunique: maximum number of unique values present, only applicable to object column types since these cannot be binned
    :param figsize: size of the plot, if None specified then one is calculated
    :param cols: number of plots on the horizontal axis
    :param verbose: display messages when columns are not visualized
    """
    # assert column_names is not None, "column_names cannot be None"

    # If the column_names argument is not a list then create a list
    if not type(column_names) == list and column_names is not None:
        column_names = [column_names]

    # If we don't have a list of column names then create a histogram for every column.
    if column_names is not None:
        columns = list(dataframe[column_names].columns)
    else:
        columns = list(dataframe.columns)

    # Calculate the number of rows / columns for the subplot.
    rows = max(int(len(columns) / cols), 1)
    cols = min(cols, len(columns))
    rows += 1 if rows * cols < len(columns) else 0

    # If figsize is not specified then calculate the fig-size
    if figsize is None:
        figsize = (17, rows * 4)

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

    # If we have more then one column then flatten the axis so we can loop through them,
    # if we have only one column then create list containing the axis so we can still loop through it.
    if len(columns) > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    # Horizontal / vertical padding between the histograms.
    fig.tight_layout(h_pad=10, w_pad=5)

    n = 0
    for c in columns:
        v = dataframe[c]
        nunique = v.nunique()

        if is_numeric_dtype(v) and min_nunique < nunique:

            z_min = -3 * v.std() + v.mean()
            z_max = 3 * v.std() + v.mean()

            # Make sure we do not get more then 50 bins.
            nunique = 50 if nunique > 50 else nunique
            v.hist(ax=axs[n], bins=nunique + 1, log=log, facecolor='#2ab0ff', edgecolor='#169acf', align='left', linewidth=0.1)

            # Only show z_min when it is not before the graph starts.
            legend = False
            if z_max < v.max():
                axs[n].plot([z_max, z_max], plt.ylim(), label='Z-Max')
                legend = True

            # Only show z_min when it is not before the graph starts.
            if z_min > v.min():
                axs[n].plot([z_min, z_min], plt.ylim(), label='Z-Min')
                legend = True

            if legend:
                axs[n].legend()
        elif is_object_dtype(v) and min_nunique < nunique <= max_nunique:
            v.hist(ax=axs[n], bins=range(nunique + 1), log=log, facecolor='#2ab0ff', edgecolor='#169acf', align='left', linewidth=0.1, width = .5)
        else:
            if verbose:
                print(f"Column '{v.name}' is not visualized, the number of nunique values ({nunique}) either exceeds {max_nunique} or is lower then {min_nunique}.")
            continue

        axs[n].set(title=v.name)

        # Only rotate the x labels
        for tick in axs[n].get_xticklabels():
            tick.set_rotation(45)

        # Remove unnecessary white space on the left/right side of the graph.
        axs[n].margins(x=0)
        n += 1

    for i in range(n, rows*cols):
        fig.delaxes(axs[i])

    plt.show()


########################################################################################################################
# Plotting history methods
########################################################################################################################


def plot_consecutive_histories(histories: list[tf.keras.callbacks.History], labels: list[str], figsize=(10, 6)):
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


def plot_history(history: tf.keras.callbacks.History, figsize=(10, 6)):
    plot_consecutive_histories([history], ["Start history"], figsize=figsize)


########################################################################################################################
# Plotting methods for linear / logistic regression
########################################################################################################################

def plot_xy_data_with_label(x, y) -> None:
    """
    Plots a graph of the values of x whereas x contains vectors of x/y coordinates and y
    is the label (0 or 1).

    :param x: is an array containing vectors of x/y coordinates.
    :param y: are the associated labels (0=blue, 1=red)
    """
    plt.plot(x[:, 0][y == 1], x[:, 1][y == 1], "bs")
    plt.plot(x[:, 0][y == 0], x[:, 1][y == 0], "r^")

    # X contains two features, x1 and x2
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20)

    # Displaying the plot.
    plt.show()


def plot_decision_boundary(model: tf.keras.Model, x, y) -> None:
    """
    Plots the decision boundary created by a model predicting on X.

    Inspired by the following two websites:
    https://cs231n.github.io/neural-networks-case-study

    :param model: the sequence model.
    :param x: array containing vectors with x/y coordinates
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
    # noinspection PyUnresolvedReferences
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    # noinspection PyUnresolvedReferences
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


########################################################################################################################
# Plotting methods for classification models
########################################################################################################################

def plot_classification_confusion_matrix(y_true, y_pred, class_names: list[str] = None, figsize=(15, 15), text_size=10, norm=False, savefig=False) -> None:
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

    if isinstance(y_true, tf.data.Dataset):
        raise TypeError('y_true is a dataset, please get the labels from the dataset using '
                        '\'y_labels = get_labels_from_dataset(dataset=dataset, index_only=True)\'')

    y_true = mlint.convert_to_sparse_or_binary(y=y_true)
    y_pred = mlint.convert_to_sparse_or_binary(y=y_pred)

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize our confusion matrix
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    # noinspection PyUnresolvedReferences
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


def table_quality_metrics(y_true, y_pred):
    """
    calculates model accuracy, precision, recall and F1-Score
    :param y_true:
    :param y_pred:
    :return:
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate precision, recall and F1-score using "weighted" average, weighted will also take the amount of samples for each in mind.
    model_precission, model_recall, model_f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = \
        {
            "accuracy" : model_accuracy,
            "precision": model_precission,
            "recall": model_recall,
            "f1": model_f1_score
        }
    return dict(sorted(model_results.items()))


def plot_classification_report_f1_score(y_true, y_pred, class_names: list[str], figsize=(10, 8)) -> None:
    """
    Creates a horizontal bar graph with the F1-Scores of the y_true / y_pred.
    :param y_true: Array of truth labels (must be same shape as y_pred).
    :param y_pred: Array of predicted labels (must be same shape as y_true).
    :param class_names: Array of class labels (e.g. string form). If `None`, integer labels are used.
    :param figsize: Size of output figure (default=(10, 8)).
    """
    if isinstance(y_true, tf.data.Dataset):
        raise TypeError(
            'y_true is a dataset, please get the labels from the dataset using '
            '\'y_labels = get_labels_from_dataset(dataset=dataset, index_only=True)\'')

    y_true = mlint.convert_to_sparse_or_binary(y=y_true)
    y_pred = mlint.convert_to_sparse_or_binary(y=y_pred)

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


def plot_classification_prediction_confidence_histogram(y_true, y_pred, class_names: list[str], figsize=(8, 4)):
    bins = range(0, 110, 10)

    if mlint.is_label_dense(y_true):
        fig, axs = plt.subplots(nrows=max(int(len(class_names) / 4), 1), ncols=min(4, len(class_names)), figsize=figsize)
        axs = axs.flatten()

        # Spacing between the graphs
        fig.tight_layout(h_pad=5, w_pad=5)

        # Set the ticks for all the subplots
        plt.setp(axs, xticks=bins)

        # For each column (class) in y_true
        for n in range(y_true.shape[1]):

            # Concatenate the y_true for the n'th class and y_pred for the n'th class together.
            y = np.concatenate((y_true[:, [n]], y_pred[:, [n]]), axis=1)

            # Filter out only the rows that are applicable to the n'th class (where y_true == 1)
            y = y[np.in1d(y[:, 0], [1])]

            # Delete first column (originally y_true)
            y = np.delete(y, np.s_[0:1], axis=1)

            # Round the values
            y = (np.round(y, decimals=1) * 100).astype(dtype=int)

            # Plot a histogram with the given values.
            class_name = str(n) if class_names is None else class_names[n]
            axs[n].hist(y, log=True, bins=11, facecolor='#2ab0ff', edgecolor='#169acf', align='left', linewidth=0.5,
                     label=class_name)
            axs[n].set(title=class_name, xlabel='Confidence (%)', ylabel='Predictions')
        plt.show()
    else:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        # Spacing between the graphs
        fig.tight_layout(h_pad=5, w_pad=5)

        # Set the ticks for all the subplots
        plt.setp(axs, xticks=bins)

        # In binary classification we have two possible best outcomes, 0 and 1, both ends mean that something has
        # been positively identified, so we need to first extract the two classes based on y_true (0 is the first class
        # and 1 is the second class), then we need to translate the value ranges, for results
        # of class 1 (y_true[:] == 1) the closer the number to 1 the higher de confidence, and for results
        # of class 0 (y_true[:] == 0) the close the number to 0 the higher de confidence.
        # For this we use the interp1d function.
        positivate_range_translation = interp1d([0.,1.],[0,100])
        negative_range_translation = interp1d([0.,1.],[100,0])

        y_true_int = mlint.binarize_labels(y_true, dtype=np.int32)
        for n in range(2):
            # Filter out only the rows thar are applicatie to the n'th class
            y_pred_single_class = y_pred[np.in1d(y_true_int[:, 0], [n])]

            # Convert ranges as described above
            if n == 1:
                y_pred_single_class_translated = positivate_range_translation(y_pred_single_class)
            else:
                y_pred_single_class_translated = negative_range_translation(y_pred_single_class)

            # Sort the values
            y = (np.round(y_pred_single_class_translated / 10) * 10).astype(dtype=int)

            # Plot a graph with the given values.
            class_name = str(n) if class_names is None else class_names[n]
            axs[n].hist(y, log=True, bins=11, facecolor='#2ab0ff', edgecolor='#169acf', align='left', linewidth=0.5,
                        label=class_name)
            axs[n].set(title=class_name, xlabel='Confidence (%)', ylabel='Predictions')


def plot_classification_prediction_confidence(y_true, y_pred, class_names: list[str], figsize=(10, 8)):
    plt.figure(figsize=figsize)

    if mlint.is_label_dense(y_true):
        # For each column (class) in y_true
        for n in range(y_true.shape[1]):
            # Concatenate the y_true for the n'th class and y_pred for the n'th class together.
            y_combined = np.concatenate((y_true[:, [n]], y_pred[:, [n]]), axis=1)

            # Filter out only the rows that are applicable to the n'th class (where y_true == 1)
            y_combined_class = y_combined[np.in1d(y_combined[:, 0], [1])]

            # Delete first column (originally y_true)
            y_class = np.delete(y_combined_class, np.s_[0:1], axis=1)

            # Sort the values
            y_class_sorted = y_class[y_class[:, 0].argsort()] * 100

            # Plot a graph with the given values.
            class_name = str(n) if class_names is None else class_names[n]
            plt.plot(y_class_sorted, label=class_name, linewidth=1.5)
    else:
        # In binary classification we have two possible best outcomes, 0 and 1, both ends mean that something has
        # been positively identified, so we need to first extract the two classes based on y_true (0 is the first class
        # and 1 is the second class), then we need to translate the value ranges, for results
        # of class 1 (y_true[:] == 1) the closer the number to 1 the higher de confidence, and for results
        # of class 0 (y_true[:] == 0) the close the number to 0 the higher de confidence.
        # For this we use the interp1d function.
        positivate_range_translation = interp1d([0.,1.],[0,100])
        negative_range_translation = interp1d([0.,1.],[100,0])

        y_true_int = mlint.binarize_labels(y_true, dtype=np.int32)
        for n in range(2):
            # Filter out only the rows thar are applicatie to the n'th class
            y_pred_single_class = y_pred[np.in1d(y_true_int[:, 0], [n])]

            # Convert ranges as described above
            if n == 1:
                y_pred_single_class_translated = positivate_range_translation(y_pred_single_class)
            else:
                y_pred_single_class_translated = negative_range_translation(y_pred_single_class)

            # Sort the values
            y_pred_single_class_translated_sorted = y_pred_single_class_translated[y_pred_single_class_translated[:, 0].argsort()]

            # Plot a graph with the given values.
            class_name = str(n) if class_names is None else class_names[n]
            plt.plot(y_pred_single_class_translated_sorted, label=class_name, linewidth=1.5)


    # X contains two features, x1 and x2
    plt.xlabel("Predictions", fontsize=20)
    plt.ylim([0, 100])
    plt.yticks(np.round(np.arange(0, 105, 5), 1))
    plt.ylabel(r"Confidence (%)", fontsize=20)
    plt.legend()
    # Displaying the plot.
    plt.show()


########################################################################################################################
# Plotting methods for regression models
########################################################################################################################

def plot_regression_y_pred_vs_y_true(y_true, y_pred, figsize=(10, 8)):
    # Merge the two columns into a single new numpy array.
    m = np.append(y_true.round(2), y_pred.round(2), axis=1)

    # Add a third column with the difference
    m_a = np.append(m, y_true - y_pred, axis=1)

    # sort based on y_true
    s = m_a[m_a[:,0].argsort()]
    plt.figure(figsize=figsize, facecolor='#FFFFFF')
    plt.plot(s[:,0], label="y_true", color="#0000FF", linestyle="solid", linewidth=1.5)
    plt.plot(s[:,1], label="y_pred", color="#FF0000", linestyle="solid", linewidth=1.5)
    plt.plot(s[:,2], label="diff", color="#FF0000", linestyle="solid", linewidth=1.5)
    plt.title('y_true vs y_pred with difference', size=20)
    plt.xlabel('Predictions', size=14)
    plt.ylabel('Value', size=14)
    plt.legend()


def show_outliers_multiclass(x, y_true, y_pred, y_prob, top=10):
    """
    Displays the top X (default 10) outliers between y_true and y_pred.
    :param x: data to display
    :param y_true: actual labels [3, 3, 2, ..., 4, 4, 1]
    :param y_pred: predictions [3, 3, 2, ..., 4, 2, 1]
    :param y_prob: probabilities (multiclass) [[0.30974227, ...], [0.32494593, ...], ..]
    :param top: how many outliers to return
    :return: the outliers
    """
    data = [[y_true[i], x.numpy().item(), y_prob[i][x]] for i, x in enumerate(y_pred)]
    data_df = pd.DataFrame(data, columns=['y_true', 'y_pred', 'y_prob'])
    # Functions like zip()
    all_df = pd.concat([x, data_df], axis=1)
    outliers = all_df[all_df['y_true'] != all_df['y_pred']].sort_values('y_prob', ascending=False)[:top]
    return outliers


def show_outliers_binary(x, y_true, y_pred, top=10):
    """
    Displays the top X (default 10) outliers between y_true and y_pred.
    :param x: data to display
    :param y_true: labels
    :param y_pred: predictions
    :param top: how many rows to display
    :return: dataframe containing the top X.
    """
    x_copy = x.copy()
    x_copy["y_true"] = y_true.round(0)
    x_copy["y_pred"] = y_pred.round(0)

    diff = y_true - y_pred
    x_copy["diff"] = diff.round(0)

    mean_positive = diff[diff['price'] > 0].mean()[0]
    mean_negative = diff[diff['price'] < 0].mean()[0]

    positive_outliers = x_copy[x_copy["diff"] > mean_positive]
    negative_outliers = x_copy[x_copy["diff"] < mean_negative]

    positive_outliers_sorted = positive_outliers.sort_values(['diff'], ascending=False, inplace=False)[:top]
    negative_outliers_sorted = negative_outliers.sort_values(['diff'], ascending=True, inplace=False)[:top]

    return pd.concat([positive_outliers_sorted, negative_outliers_sorted])

########################################################################################################################
# Internal methods for plotting history
########################################################################################################################


def _plot_history_ends(histories: list[tf.keras.callbacks.History], labels: list[str]) -> None:
    """
    Internal method which will plot a vertical line showing where a histories last epoch is visible.

    :param histories: the history objects returned from fitting models.
    :param labels: the labels for each history object for seperating the epochs
    """
    for idx, history in enumerate(histories):
        plt.plot([min(history.epoch), min(history.epoch)], plt.ylim(), label=f'{labels[idx]}')


def _plot_history_graph_line(data, label, color, linestyle='solid') -> None:
    """
    Internal method which will plot the information from the fit histroy.

    :param data: the data to plot
    :param label: the label associated with the data
    :param color: color of the line
    :param linestyle: line-style of the line (default: solid)
    """
    if data:
        plt.plot(data, label=label, color=color, linestyle=linestyle, linewidth=1.5)
