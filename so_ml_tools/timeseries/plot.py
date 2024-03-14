import matplotlib.pyplot as _plt
import seaborn as _sns
import numpy as _np
import tensorflow as _tf
import pandas as _pd


def _to_numpy(x):
    if not isinstance(x, _np.ndarray):
        if isinstance(x, _pd.DataFrame) | isinstance(x, _pd.Series) | isinstance(x, _pd.DatetimeIndex):
            return x.to_numpy()
        elif isinstance(x, _tf.Tensor):
            return x.numpy()
        else:
            return _tf.convert_to_tensor(value=x).numpy()

    return x


def plot_predictions(x, y_true, y_prob, start_index=None, end_index=None, title='Predictions', figsize=(30, 8)):
    fig, ax = _plt.subplots(figsize=figsize)

    x = _to_numpy(x)
    y_true = _to_numpy(y_true)
    y_prob = _to_numpy(y_prob)

    if start_index is None:
        start_index = 0

    if end_index is None:
        end_index = len(x)

    assert end_index > start_index, 'end_index must be larger then start_index'

    print(f'{start_index}, {end_index}')

    _sns.lineplot(x=x.ravel()[start_index:end_index],
                  y=y_true.ravel()[start_index:end_index],
                  ax=ax)

    _sns.lineplot(x=x.ravel()[start_index:end_index],
                  y=y_prob.ravel()[start_index:end_index],
                  ax=ax)

    _plt.title(f'{title} (range: {start_index}-{end_index})')
    ax.set_ylabel("Value")
    ax.set_xlabel("Time")
    _plt.show()
