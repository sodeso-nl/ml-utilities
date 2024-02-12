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
               start: int = None,
               figsize: tuple = (8, 6),
               label_color='black') -> None:
    """
    Plots time series data, where `x` is a single array containing the date/time and `y` can
    be one or more arrays of labels and predictions.

    In case `y` has a larger horizon then one, for example to predict 7 days, then the data will be
    reduced using the mean operation.

    The use this method you can call it in the following ways:

    timeseries(
        x=[X[-len(dates):], X[-len(dates):]],
        y=[y_label, Y_pred],
        labels=['Label', 'Pred'],
        start=300,
        x_label='Time', y_label
        ='Price')

    Where both y_label and y_pred contain one ore more arrays of predictions (for example, a horizon of 1, 2, 3 days),
    in case you only want to plot one of the days do the following:

    day = 2

    timeseries(
        x=[X[-len(dates):], X[-len(dates):]],
        y=[y_label[:,day], Y_pred[:,day]],
        labels=['Label', 'Pred'],
        start=300,
        x_label='Time', y_label
        ='Price')

    Where day is set to the day in the range of the horizon, in this case the third day.

    Args:
        x: A list containing one ore more arrays of dates/times
        y: A list containing one ore more arrays of labels and predictions
        labels: An array of labels where each label corresponds to an entry in the list of y.
        title: An optional title for the graph.
        x_label: An optional x-axis label for the graph.
        y_label: An optional y-axis label for the graph.
        start: An optional starting project for plotting the graph.
        figsize: An optional figure size. Default is (8, 6)
        label_color: An optional color for the labels of the axis, axis values and title.
    """
    fig, ax = _plt.subplots(figsize=figsize)

    color_palette = _itertools.cycle(_sns.color_palette(palette="tab10", n_colors=len(x)))
    for idx, _y in enumerate(y):
        _x = x[idx]

        # If the horizon of the data contains more then one day then reduce the
        if _y.ndim > 1:
            _y = _tf.reduce_mean(_y, axis=1)
            labels[idx] = f"{labels[idx]} (mean)"

        _sns.lineplot(
            x=_x[start:],
            y=_y[start:],
            color=next(color_palette),
            # s=7,
            ax=ax,
            label=None if labels is None else labels[idx])

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
