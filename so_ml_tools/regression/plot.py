import matplotlib.pyplot as _plt
import numpy as _np
import tensorflow as _tf

import so_ml_tools as _soml

from scipy.interpolate import interp1d as _interp1d


def confusion_matrix(y_true, y_pred=None, y_prob=None, class_names: list[str] = None, figsize=(15, 15), text_size=10, norm=False, savefig=False) -> None:
    """
      Plots a confusion matrix of the given data.

      :param y_true: Array of truth labels (must be same shape as y_pred).
      :param y_pred: Array of predictions (then y_prob is not necessary), must be same shape as y_true.
      :param y_prob: Array of probabilities (then y_pred is not necessary), must be same shape as y_true.
      :param class_names: Array of class labels (e.g. string form). If `None`, integer labels are used.
      :param figsize: Size of output figure (default=(15, 15)).
      :param text_size: Size of output figure text (default=10).
      :param norm: normalize values or not (default=False).
      :param savefig: save confusion matrix to file (default=False).

      Plots the decision boundary created by a model predicting on X.
      Inspired by the following two websites:
      https://cs231n.github.io/neural-networks-case-study
    """
    _soml.multiclass.plot.confusion_matrix(y_true, y_prob, class_names, figsize, text_size, norm, savefig)


def prediction_confidence(y_true, y_pred, class_names: list[str], figsize=(10, 8)):
    _plt.figure(figsize=figsize)

    # In binary classification we have two possible best outcomes, 0 and 1, both ends mean that something has
    # been positively identified, so we need to first extract the two classes based on y_true (0 is the first class
    # and 1 is the second class), then we need to translate the value ranges, for results
    # of class 1 (y_true[:] == 1) the closer the number to 1 the higher de confidence, and for results
    # of class 0 (y_true[:] == 0) the close the number to 0 the higher de confidence.
    # For this we use the interp1d function.
    positivate_range_translation = _interp1d([0.,1.],[0,100])
    negative_range_translation = _interp1d([0.,1.],[100,0])

    y_true_int = _soml.util.label.probability_to_binary(y_true)
    for n in range(2):
        # Filter out only the rows thar are applicatie to the n'th class
        y_pred_single_class = y_pred[_np.in1d(y_true_int[:, 0], [n])]

        # Convert ranges as described above
        if n == 1:
            y_pred_single_class_translated = positivate_range_translation(y_pred_single_class)
        else:
            y_pred_single_class_translated = negative_range_translation(y_pred_single_class)

        # Sort the values
        y_pred_single_class_translated_sorted = y_pred_single_class_translated[y_pred_single_class_translated[:, 0].argsort()]

        # Plot a graph with the given values.
        class_name = str(n) if class_names is None else class_names[n]
        _plt.plot(y_pred_single_class_translated_sorted, label=class_name, linewidth=1.5)

    # X contains two features, x1 and x2
    _plt.xlabel("Predictions", fontsize=20)
    _plt.ylim([0, 100])
    _plt.yticks(_np.round(_np.arange(0, 105, 5), 1))
    _plt.ylabel(r"Confidence (%)", fontsize=20)
    _plt.legend()
    # Displaying the plot.
    _plt.show()


def prediction_confidence_histogram(y_true, y_pred, class_names: list[str], figsize=(8, 4)):
    bins = range(0, 110, 10)

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
    positivate_range_translation = _interp1d([0.,1.],[0,100])
    negative_range_translation = _interp1d([0.,1.],[100,0])

    y_true_int = _soml.util.label.probability_to_binary(y_true)
    for n in range(2):
        # Filter out only the rows thar are applicatie to the n'th class
        y_pred_single_class = y_pred[_np.in1d(y_true_int[:, 0], [n])]

        # Convert ranges as described above
        if n == 1:
            y_pred_single_class_translated = positivate_range_translation(y_pred_single_class)
        else:
            y_pred_single_class_translated = negative_range_translation(y_pred_single_class)

        # Sort the values
        y = (_np.round(y_pred_single_class_translated / 10) * 10).astype(dtype=int)

        # Plot a graph with the given values.
        class_name = str(n) if class_names is None else class_names[n]
        axs[n].hist(y, log=True, bins=11, facecolor='#2ab0ff', edgecolor='#169acf', align='left', linewidth=0.5,
                    label=class_name)
        axs[n].set(title=class_name, xlabel='Confidence (%)', ylabel='Predictions')
    _plt.show()


def report_f1_score(y_true, y_pred, class_names: list[str], figsize=(10, 8)) -> None:
    _soml.multiclass.plot.report_f1_score(y_true, y_pred, class_names, figsize)


def xy_data_with_label(x, y) -> None:
    """
    Plots a graph of the values of x whereas x contains vectors of x/y coordinates and y
    is the label (0 or 1).

    :param x: is an array containing vectors of x/y coordinates.
    :param y: are the associated labels (0=blue, 1=red)
    """
    _plt.plot(x[:, 0][y == 1], x[:, 1][y == 1], "bs")
    _plt.plot(x[:, 0][y == 0], x[:, 1][y == 0], "r^")

    # X contains two features, x1 and x2
    _plt.xlabel(r"$x_1$", fontsize=20)
    _plt.ylabel(r"$x_2$", fontsize=20)

    # Displaying the plot.
    _plt.show()


def decision_boundary(model: _tf.keras.Model, x, y) -> None:
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


def y_pred_vs_y_true(y_true, y_pred, figsize=(10, 8)) -> None:
    """
    Shows a graph with the predicted values vs the truth labels.
    :param y_true: the truth labels
    :param y_pred: the predicted values
    :param figsize: size of the graph
    """
    # Merge the two columns into a single new numpy array.
    m = _np.append(y_true.round(2), y_pred.round(2), axis=1)

    # Add a third column with the difference
    m_a = _np.append(m, y_true - y_pred, axis=1)

    # sort based on y_true
    s = m_a[m_a[:,0].argsort()]
    _plt.figure(figsize=figsize, facecolor='#FFFFFF')
    _plt.plot(s[:,0], label="y_true", color="#0000FF", linestyle="solid", linewidth=1.5)
    _plt.plot(s[:,1], label="y_pred", color="#FF0000", linestyle="solid", linewidth=1.5)
    _plt.plot(s[:,2], label="diff", color="#FF0000", linestyle="solid", linewidth=1.5)
    _plt.title('y_true vs y_pred with difference', size=20)
    _plt.xlabel('Predictions', size=14)
    _plt.ylabel('Value', size=14)
    _plt.legend()