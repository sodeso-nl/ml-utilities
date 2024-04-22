import matplotlib.pyplot as _plt
import seaborn as _sns
import numpy as _np
import so_ml_tools as _soml


def plot_predictions(x, y_true, y_prob, start_index=None, end_index=None, title='Predictions', figsize=(30, 8)):
    fig, ax = _plt.subplots(figsize=figsize)

    if not isinstance(x, _np.ndarray):
        x = _soml.util.types.to_numpy(x)
    if not isinstance(y_true, _np.ndarray):
        y_true = _soml.util.types.to_numpy(y_true)
    if not isinstance(y_prob, _np.ndarray):
        y_prob = _soml.util.types.to_numpy(y_prob)

    if start_index is None:
        start_index = 0

    if end_index is None:
        end_index = len(x)

    assert end_index > start_index, 'end_index must be larger then start_index'

    print(f'{start_index}, {end_index}')

    _sns.lineplot(x=x.ravel()[start_index:end_index],
                  y=y_true.ravel()[start_index:end_index],
                  ax=ax, label='Actual')

    _sns.lineplot(x=x.ravel()[start_index:end_index],
                  y=y_prob.ravel()[start_index:end_index],
                  ax=ax, label='Predicted')

    _plt.title(f'{title} (range: {start_index}-{end_index})')
    _plt.legend()
    ax.set_ylabel("Value")
    ax.set_xlabel("Time")
    _plt.show()
