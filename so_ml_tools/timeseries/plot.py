import matplotlib.pyplot as _plt
import seaborn as _sns
import itertools as _itertools
import numpy as _np
import tensorflow as _tf


def timeseries(x: list[_np.ndarray],
               y: list[_np.ndarray],
               labels: list[str] = None,
               title: str = None,
               x_label: str = None,
               y_label: str = None,
               figsize: tuple = (8, 6),
               label_color='black') -> None:
    """
    Plots time series data, where `x` is a single array containing the date/time and `y` can
    be one or more arrays of labels and predictions.

    In case `y` has a larger horizon then one, for example to predict 7 days, then the data will be
    reduced using the mean operation.

    The use this method you can call it in the following ways:

    timeseries(
        x=[
            X_history,
            X_pred
        ],
        y=[
            y_history,
            y_pred
        ],
        labels=['Label', 'Pred'],
        start=300,
        x_label='Time', y_label
        ='Price')

    Note that filling in the values for X and y might be different in certain circumstances, most often this has
    to do with the shape, make sure they match.

    Where day is set to the day in the range of the horizon, in this case the third day.

    Args:
        x: A list containing one ore more arrays of dates/times
        y: A list containing one ore more arrays of labels and predictions
        labels: An array of labels where each label corresponds to an entry in the list of y.
        title: An optional title for the graph.
        x_label: An optional x-axis label for the graph.
        y_label: An optional y-axis label for the graph.
        figsize: An optional figure size. Default is (8, 6)
        label_color: An optional color for the labels of the axis, axis values and title.
    """
    fig, ax = _plt.subplots(figsize=figsize)

    color_palette = _itertools.cycle(_sns.color_palette(palette="tab10", n_colors=len(x)))
    for idx, _y in enumerate(y):
        _x = x[idx]

        # If the horizon of the data contains more then one day then reduce the
        if hasattr(_y, 'ndim') and _y.ndim > 1:
            _y = _tf.reduce_mean(_y, axis=1)
            if labels is not None and len(labels) > idx:
                labels[idx] = f"{labels[idx]} (mean)"
        else:
            _y = y[idx]

        _label = 'None'
        if labels is not None and len(labels) > idx:
            _label = labels[idx - 1]



        _sns.lineplot(
            x=_x,
            y=_y,
            color=next(color_palette),
            # s=7,
            ax=ax,
            label=_label)

    if x_label:
        _plt.xlabel(x_label)
    if y_label:
        _plt.ylabel(y_label)
    if title:
        _plt.title(title)

    fig.patch.set_alpha(0.0)  # Transparant background

    ax.tick_params(axis='x', labelrotation=90)

    ax.xaxis.label.set_color(label_color)  # Set color of x-axis label
    ax.tick_params(axis='x', colors=label_color)  # Set color of x-axis ticks.

    ax.yaxis.label.set_color(label_color)  # Set color of y-axis label
    ax.tick_params(axis='y', colors=label_color)  # Set color of y-axis ticks.
    ax.title.set_color(label_color)  # Set color of title

    if labels:
        _plt.legend()

    _plt.show()
