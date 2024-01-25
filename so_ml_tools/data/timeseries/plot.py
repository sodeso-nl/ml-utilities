import matplotlib.pyplot as _plt
import seaborn as _sns
import itertools


def timeseries(x, y, labels: list[str] = None, title: str = None, x_label: str = None, y_label: str = None,
               start: int = None, figsize: tuple = (8, 6), label_color='black'):
    fig, ax = _plt.subplots(figsize=figsize)

    color_palette = itertools.cycle(_sns.color_palette(palette="tab10", n_colors=len(x)))
    for idx, _x in enumerate(x):
        # _sns.scatterplot(
        _sns.lineplot(
            x=_x[start:],
            y=y[idx][start:],
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

    ax.xaxis.label.set_color(label_color)  # Set color of x-axis label
    ax.tick_params(axis='x', colors=label_color)  # Set color of x-axis ticks.

    ax.yaxis.label.set_color(label_color)  # Set color of y-axis label
    ax.tick_params(axis='y', colors=label_color)  # Set color of y-axis ticks.
    ax.title.set_color(label_color)  # Set color of title

    if labels:
        _plt.legend()

    _plt.show()