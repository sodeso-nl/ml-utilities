import matplotlib.pyplot as _plt
import seaborn as _sns
import itertools


def timeseries(x, y, labels: list[str] = None, title=None, x_label=None, y_label=None, label_color='black'):
    fig, ax = _plt.subplots(figsize=(8, 6))

    color_palette = itertools.cycle(_sns.color_palette(palette="tab10", n_colors=len(x)))
    for idx, _x in enumerate(x):
        _sns.scatterplot(x=_x, y=y[idx], color=next(color_palette), ax=ax, label=None if labels is None else labels[idx])

    if x_label:
        _plt.xlabel(x_label)
    if y_label:
        _plt.ylabel(y_label)
    if title:
        _plt.title(title)

    fig.patch.set_alpha(0.0)  # Transparant background

    ax.xaxis.label.set_color(label_color)  # Set color of x-axis label
    ax.tick_params(axis='x', colors=label_color)  # Set color of x-axis ticks.

    ax.yaxis.label.set_color(label_color)  # Set color of y-axis label
    ax.tick_params(axis='y', colors=label_color)  # Set color of y-axis ticks.
    ax.title.set_color(label_color)  # Set color of title

    if labels:
        _plt.legend()

    _plt.show()
