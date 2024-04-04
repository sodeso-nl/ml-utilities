from sklearn.metrics import confusion_matrix as _confusion_matrix

import itertools as _itertools

import matplotlib.pyplot as _plt
import numpy as _np
import sklearn as _sk
import tensorflow as _tf
import so_ml_tools as _soml

from scipy.interpolate import interp1d as _interp1d

from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import classification_report as _classification_report


def roc_curve(y_true, y_prob=None, figsize=(5, 5), label_color='black'):
    """
    Plots the ROC curve for the given values. Also calculates the ROC-AUC value and the
    optimal threshold. The optimal threshold is also returned as a value.

    Args:
        y_true: Array of truth labels (must be same shape as y_pred).
        y_prob: Array of probabilities (then y_pred is not necessary), must be same shape as y_true.
        figsize: Size of output figure (default=(15, 15)).
        label_color: label color of ticks, titles, x/y axis values / labels.

    Returns:
        The optimal threshold value
    """
    if isinstance(y_true, _tf.data.Dataset):
        raise TypeError('y_true is a dataset, please get the labels from the dataset using '
                        '\'y_labels = soml.tf.dataset.get_labels(dataset=dataset)\'')

    y_true = _soml.util.label.to_prediction(y_prob=y_true)

    precision_fpr, sensitivity_tpr, thresholds = _sk.metrics.roc_curve(y_true=y_true, y_score=y_prob)

    # Calculate Area Under Curve
    auc = _sk.metrics.auc(precision_fpr, sensitivity_tpr)

    # Calculate optimal threshold value:
    youden_index = sensitivity_tpr - precision_fpr
    optimal_idx = _np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = _plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)  # Transparant background

    ax.plot(precision_fpr, sensitivity_tpr)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='dashed')  # Draw diagonal line
    ax.plot(precision_fpr[optimal_idx], sensitivity_tpr[optimal_idx],
            marker="o",
            markersize=5,
            markeredgecolor="red",
            markerfacecolor="red")  # Draw dot for optimal threshold

    ax.set(title=f'ROC-Curve \nArea Under Curve (AUC): {round(auc * 100)}%\nOptimal Threshold: {optimal_threshold}',
           xlabel="False Positive Rate",
           ylabel="True Positive Rate")

    ax.xaxis.label.set_color(label_color)  # Set color of x-axis label
    ax.tick_params(axis='x', colors=label_color)  # Set color of x-axis ticks.

    ax.yaxis.label.set_color(label_color)  # Set color of y-axis label
    ax.tick_params(axis='y', colors=label_color)  # Set color of y-axis ticks.
    ax.title.set_color(label_color)  # Set color of title

    _plt.xticks(rotation=70)

    _plt.show()

    return optimal_threshold


def confusion_matrix(y_true, y_pred=None, y_prob=None, class_names: list[str] = None, figsize=(15, 15), text_size=10,
                     norm=False, savefig=False, label_color='black') -> None:
    """
   Plots a confusion matrix of the given data.

    Args:
      y_true: Array of truth labels (must be same shape as y_pred).
      y_pred: Array of predictions (then y_prob is not necessary), must be same shape as y_true.
      y_prob: Array of probabilities (then y_pred is not necessary), must be same shape as y_true.
      class_names: Array of class labels (e.g. string form). If `None`, integer labels are used.
      figsize: Size of output figure (default=(15, 15)).
      text_size: Size of output figure text (default=10).
      norm: normalize values or not (default=False).
      savefig: save confusion matrix to file (default=False).
        label_color: label color of ticks, titles, x/y axis values / labels.

    Returns:
        None
    """

    if isinstance(y_true, _tf.data.Dataset):
        raise TypeError('y_true is a dataset, please get the labels from the dataset using '
                        '\'y_labels = soml.tf.dataset.get_labels(dataset=dataset)\'')

    y_true = _soml.util.label.to_prediction(y_prob=y_true)
    if y_pred is None and y_prob is not None:
        y_pred = _soml.util.label.to_prediction(y_prob=y_prob)
    elif y_pred is None and y_prob is None:
        raise "Must specify 'y_pred' or 'y_prob'"

    # Create the confusion matrix
    cm = _confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, _np.newaxis]  # normalize our confusion matrix
    n_classes = cm.shape[0]

    fig, ax = _plt.subplots(figsize=figsize)

    # Set background to transparent.
    fig.patch.set_alpha(0.0)

    # noinspection PyUnresolvedReferences
    # Color bar on the right side.
    cax = ax.matshow(cm, cmap=_plt.cm.Blues)
    cb = fig.colorbar(cax)
    cb.set_label('', color=label_color)
    cb.ax.yaxis.set_tick_params(color=label_color)
    _plt.setp(_plt.getp(cb.ax.axes, 'yticklabels'), color=label_color)

    # Set labels to be classes
    if class_names is not None:
        labels = class_names
    else:
        labels = _np.arange(cm.shape[0])

    # label the axes
    ax.set(title="Confusion matrix",
           xlabel="predicted label",
           ylabel="Actual label",
           xticks=_np.arange(n_classes),
           yticks=_np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    ax.xaxis.label.set_color(label_color) # Set color of x-axis label
    ax.tick_params(axis='x', colors=label_color) # Set color of x-axis ticks.
    ax.yaxis.label.set_color(label_color)  # Set color of y-axis label
    ax.tick_params(axis='y', colors=label_color)  # Set color of y-axis ticks.
    ax.title.set_color(label_color) # Set color of title

    # Set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.title.set_size(text_size)

    _plt.xticks(rotation=70, fontsize=text_size)
    _plt.yticks(fontsize=text_size)

    # Set treshold for different colors
    threshold = (cm.max() + cm.min()) / 2.
    # Plot the text on each cell
    for i, j in _itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            _plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                      horizontalalignment="center",
                      color="white" if cm[i, j] > threshold else "black",
                      size=text_size)
        else:
            _plt.text(j, i, f"{cm[i, j]}",
                      horizontalalignment="center",
                      color="white" if cm[i, j] > threshold else "black",
                      size=text_size)

    if savefig:
        fig.savefig("./confusion_matrix.png")


def decision_boundary(model: _tf.keras.Model, x, y) -> None:
    """
    Plots the decision boundary created by a model predicting on X.

    Inspired by the following two websites:
    https://cs231n.github.io/neural-networks-case-study

    Args:
        model: the sequence model.
        x: array containing vectors with x/y coordinates
        y: are the associated labels (0=blue, 1=red)

    Returns:
        None
    """
    # Define the axis boundaries of the plot and create a meshgrid.
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1

    xx, yy = _np.meshgrid(_np.linspace(x_min, x_max, 100),
                          _np.linspace(y_min, y_max, 100))

    # Create X value (we're going to make predictions on these)
    x_in = _np.c_[xx.ravel(), yy.ravel()]  # Stack 2D arrays together

    # Make predictions
    y_pred = model.predict(x_in)

    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification")
        # We have to reshape our predictions to get them ready for plotting.
        y_pred = _np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classification")
        y_pred = _np.round(y_pred).reshape(xx.shape)

    # Plot the decision boundary
    # noinspection PyUnresolvedReferences
    _plt.contourf(xx, yy, y_pred, cmap=_plt.cm.RdYlBu, alpha=0.7)
    # noinspection PyUnresolvedReferences
    _plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=_plt.cm.RdYlBu)
    _plt.xlim(xx.min(), xx.max())
    _plt.ylim(yy.min(), yy.max())


import numpy as _np
import matplotlib.pyplot as _plt


def y_pred_vs_y_true(y_true, y_pred, figsize=(20, 8)) -> None:
    """
    Shows a graph with the predicted values vs the truth labels.

    Args:
        y_true: the truth labels
        y_pred: the predicted values
        figsize: size of the graph

    Returns:
        None
    """
    # Merge the two columns into a single new numpy array.
    ytrue_ypred = _np.append(y_true.round(2), y_pred.round(2), axis=1)

    # Add a third column with the difference
    ytrue_ypred_diff = _np.append(ytrue_ypred, y_true - y_pred, axis=1)

    mae = _np.round(_np.mean(_np.abs(ytrue_ypred_diff[:, 2])), 5)

    # sort based on y_true
    ytrue_ypred_diff = ytrue_ypred_diff[ytrue_ypred_diff[:, 0].argsort()]
    _plt.figure(figsize=figsize, facecolor='#FFFFFF')
    _plt.plot(ytrue_ypred_diff[:, 1], label="y_pred", color="#FF0000", linestyle="solid", linewidth=1)
    _plt.plot(ytrue_ypred_diff[:, 0], label="y_true", color="#00FF00", linestyle="solid", linewidth=1)
    _plt.plot(ytrue_ypred_diff[:, 2], label="diff", color="#0000FF", linestyle="solid", linewidth=1)
    _plt.title(f'y_true vs y_pred with difference {mae}', size=20)
    _plt.xlabel('Predictions', size=14)
    _plt.ylabel('Value', size=14)
    _plt.legend()


def report_f1_score(y_true, y_pred=None, y_prob=None, class_names: list[str] = None, figsize=(10, 8)) -> None:
    """
    Creates a horizontal bar graph with the F1-Scores of the y_true / y_pred.

    Args:
        y_true: the truth labels
        y_pred: (optional) the predictions (either y_pred or y_prob should be supplied)
        y_prob: (optional) the probabilities (either y_pred or y_prob should be supplied)
        class_names: Array of class labels (e.g. string form). If `None`, integer labels are used.
        figsize: Size of output figure (default=(10, 8)).

    Returns:
        A 'dict' containing accuracy, precision, recall, f1 score and support
    """
    if isinstance(y_true, _tf.data.Dataset):
        raise TypeError(
            'y_true is a dataset, please get the labels from the dataset using '
            '\'y_labels = get_labels_from_dataset(dataset=dataset, index_only=True)\'')

    y_true = _soml.util.label.to_prediction(y_prob=y_true)
    if y_pred is None and y_prob is not None:
        y_pred = _soml.util.label.to_prediction(y_prob=y_prob)
    elif y_pred is None and y_prob is None:
        raise "y_pred or y_prob argument should be provided."

    # Generate classification report from SKLearn.
    report_dict = _classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names,
                                         output_dict=True)

    # Collect all the F1-scores
    f1_scores = {}
    for k, v in report_dict.items():
        if k == 'accuracy':  # Cut-off when the accuracy key is visible.
            break
        else:
            f1_scores[k] = v['f1-score']

    # Sort F1-scores from best to worst
    f1_scores_sorted = dict(sorted(f1_scores.items(), key=lambda item: item[1], reverse=True))

    _plt.rcdefaults()
    fig, ax = _plt.subplots(figsize=figsize)
    y_pos = range(len(f1_scores_sorted))
    scores = ax.barh(y_pos, f1_scores_sorted.values(), align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(f1_scores_sorted.keys())

    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('F1-score')
    ax.set_title('F1-Scores')
    ax.bar_label(scores, fmt='%.2f')

    # Remove anoying white space at top / bottom.
    _plt.margins(y=0)

    _plt.show()


def prediction_confidence(y_true, y_prob, class_names: list[str], figsize=(10, 8)):
    """
    Creates a plot showing the confidence of the predictions, the plot will display the number of predictions vs
    the confidence (how close was the prediction to the actual truth label). In a perfect world the
    plot would be a straight vertical line on the right (all predictions are exactly 1 or 0 compared with their
    truth label). Since this is never the case the more optimal plot would be a vertical line around 0 on the x-axis
    with a small bend in the top left corner following a horizontal line around 100%.

    Args:
        y_true: Array of truth labels, must be same shape as y_pred.
        y_prob: Array of probabilities, must be same shape as y_true.
        class_names: Array of class labels (e.g. string form). If `None`, integer labels are used.
        figsize: Size of output figure (default=(10, 8)).

    Returns:
        None
    """
    _plt.figure(figsize=figsize)

    if _soml.util.label.is_multiclass_classification(y_prob=y_true):

        # For each column (class) in y_true
        for n in range(y_true.shape[1]):
            # Concatenate the y_true for the n'th class and y_pred for the n'th class together.
            y_combined = _np.concatenate((y_true[:, [n]], y_prob[:, [n]]), axis=1)

            # Filter out only the rows that are applicable to the n'th class (where y_true == 1)
            y_combined_class = y_combined[_np.in1d(y_combined[:, 0], [1])]

            # Delete first column (originally y_true)
            y_class = _np.delete(y_combined_class, _np.s_[0:1], axis=1)

            # Sort the values
            y_class_sorted = y_class[y_class[:, 0].argsort()] * 100

            # Plot a graph with the given values.
            class_name = str(n) if class_names is None else class_names[n]
            _plt.plot(y_class_sorted, label=class_name, linewidth=1.5)
    elif _soml.util.label.is_binary_classification(y_prob=y_true):
        # In binary classification we have two possible best outcomes, 0 and 1, both ends mean that something has
        # been positively identified, so we need to first extract the two classes based on y_true (0 is the first class
        # and 1 is the second class), then we need to translate the value ranges, for results
        # of class 1 (y_true[:] == 1) the closer the number to 1 the higher de confidence, and for results
        # of class 0 (y_true[:] == 0) the close the number to 0 the higher de confidence.
        # For this we use the interp1d function.
        positivate_range_translation = _interp1d([0., 1.], [0, 100])
        negative_range_translation = _interp1d([0., 1.], [100, 0])

        y_true_int = _soml.util.label.probability_to_binary(y_true)
        for n in range(2):
            # Filter out only the rows thar are applicatie to the n'th class
            y_pred_single_class = y_prob[_np.in1d(y_true_int[:, 0], [n])]

            # Convert ranges as described above
            if n == 1:
                y_pred_single_class_translated = positivate_range_translation(y_pred_single_class)
            else:
                y_pred_single_class_translated = negative_range_translation(y_pred_single_class)

            # Sort the values
            y_pred_single_class_translated_sorted = y_pred_single_class_translated[
                y_pred_single_class_translated[:, 0].argsort()]

            # Plot a graph with the given values.
            class_name = str(n) if class_names is None else class_names[n]
            _plt.plot(y_pred_single_class_translated_sorted, label=class_name, linewidth=1.5)
    else:
        raise "Could not determine if y_prob is a multi-class classification or a binary classification."

    # X contains two features, x1 and x2
    _plt.xlabel("Predictions", fontsize=20)
    _plt.ylim([0, 100])
    _plt.yticks(_np.round(_np.arange(0, 105, 5), 1))
    _plt.ylabel(r"Confidence (%)", fontsize=20)
    _plt.legend()
    # Displaying the plot.
    _plt.show()


def prediction_confidence_histogram(y_true, y_prob, class_names: list[str], log=True, figsize=(8, 4)):
    """
    Creates a histogram showing the confidence of the predictions. The digest of this method is the same
    as for the ´prediction_confidence´ method except that it displays the information as a histogram.

    Args:
        y_true: Array of truth labels, must be same shape as y_pred.
        y_prob: Array of probabilities, must be same shape as y_true.
        class_names: Array of class labels (e.g. string form). If `None`, integer labels are used.
        log: use a logarithmic scale for the y-axis or not.
        figsize: Size of output figure (default=(10, 8)).

    Returns:
        None
    """
    bins = range(0, 110, 10)

    if _soml.util.label.is_multiclass_classification(y_prob=y_true):
        fig, axs = _plt.subplots(nrows=max(int(len(class_names) / 4), 1), ncols=min(4, len(class_names)),
                                 figsize=figsize)
        axs = axs.flatten()

        # Spacing between the graphs
        fig.tight_layout(h_pad=5, w_pad=5)

        # Set the ticks for all the subplots
        _plt.setp(axs, xticks=bins)

        # For each column (class) in y_true
        for idx in range(y_true.shape[1]):
            # Concatenate the y_true for the n'th class and y_pred for the n'th class together.
            y = _np.concatenate((y_true[:, [idx]], y_prob[:, [idx]]), axis=1)

            # Filter out only the rows that are applicable to the n'th class (where y_true == 1)
            y = y[_np.in1d(y[:, 0], [1])]

            # Delete first column (originally y_true)
            y = _np.delete(y, _np.s_[0:1], axis=1)

            # Round the values
            y = (_np.round(y, decimals=1) * 100).astype(dtype=int)

            # Plot a histogram with the given values.
            __plot_histogram(class_names=class_names, axs=axs, y=y, idx=idx)
    elif _soml.util.label.is_binary_classification(y_prob=y_true):
        fig, axs = _plt.subplots(nrows=1, ncols=2, figsize=figsize)

        # Spacing between the graphs
        fig.tight_layout(h_pad=5, w_pad=5)

        # Set the ticks for all the subplots
        _plt.setp(axs, xticks=bins)

        # In binary classification we have two possible best outcomes, 0 and 1, both ends mean that something has
        # been positively identified, so we need to first extract the two classes based on y_true (0 is the first class
        # and 1 is the second class), then we need to translate the value ranges, for results
        # of class 1 (y_true[:] == 1) the closer the number to 1 the higher de confidence, and for results
        # of class 0 (y_true[:] == 0) the close the number to 0 the higher de confidence.
        # For this we use the interp1d function.
        positivate_range_translation = _interp1d([0., 1.], [0, 100])
        negative_range_translation = _interp1d([0., 1.], [100, 0])

        y_true_int = _soml.util.label.probability_to_binary(y_true)
        for idx in range(2):
            # Filter out only the rows thar are applicatie to the n'th class
            y_pred_single_class = y_prob[_np.in1d(y_true_int[:, 0], [idx])]

            # Convert ranges as described above
            if idx == 1:
                y_pred_single_class_translated = positivate_range_translation(y_pred_single_class)
            else:
                y_pred_single_class_translated = negative_range_translation(y_pred_single_class)

            # Sort the values
            y = (_np.round(y_pred_single_class_translated / 10) * 10).astype(dtype=int)

            # Plot a histogram with the given values.
            __plot_histogram(class_names=class_names, axs=axs, y=y, idx=idx, log=log)
    else:
        raise "Could not determine if y_prob is a multi-class classification or a binary classification."

    _plt.show()


def __plot_histogram(class_names: list[str], axs, y, idx, log=True) -> None:
    class_name = str(idx) if class_names is None else class_names[idx]
    axs[idx].hist(y, log=log, bins=11, facecolor='#2ab0ff', edgecolor='#169acf', align='left', linewidth=0.5,
                  label=class_name)
    axs[idx].set(title=class_name, xlabel='Confidence (%)', ylabel='Predictions')
