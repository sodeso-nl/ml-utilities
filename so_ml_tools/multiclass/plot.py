import itertools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import so_ml_tools as soml
import sklearn as sklearn


def confusion_matrix(y_true, y_prob, class_names: list[str] = None, figsize=(15, 15), text_size=10, norm=False, savefig=False) -> None:
    """
      Plots a confusion matrix of the given data.

      :param y_true: Array of truth labels (must be same shape as y_pred).
      :param y_prob: Array of probability labels (must be same shape as y_true).
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
                        '\'y_labels = ml.tf.dataset.get_labels(dataset=dataset)\'')

    y_true = soml.util.label.to_prediction(y_prob=y_true)
    y_pred = soml.util.label.to_prediction(y_prob=y_prob)

    # Create the confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
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


def classification_prediction_confidence(y_true, y_pred, class_names: list[str], figsize=(10, 8)):
    plt.figure(figsize=figsize)

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

    # X contains two features, x1 and x2
    plt.xlabel("Predictions", fontsize=20)
    plt.ylim([0, 100])
    plt.yticks(np.round(np.arange(0, 105, 5), 1))
    plt.ylabel(r"Confidence (%)", fontsize=20)
    plt.legend()
    # Displaying the plot.
    plt.show()


def classification_prediction_confidence_histogram(y_true, y_pred, class_names: list[str], figsize=(8, 4)):
    bins = range(0, 110, 10)

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


def report_f1_score(y_true, y_pred, class_names: list[str], figsize=(10, 8)) -> None:
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

    y_true = soml.util.label.to_prediction(y_prob=y_true)
    y_pred = soml.util.label.to_prediction(y_prob=y_pred)

    # Generate classification report from SKLearn.
    report_dict = sklearn.metrics.classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names, output_dict=True)

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
