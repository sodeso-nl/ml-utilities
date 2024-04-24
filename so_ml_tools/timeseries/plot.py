import matplotlib.pyplot as _plt
import seaborn as _sns
import numpy as _np
import pandas as _pd
import so_ml_tools as _soml


def lineplot(dataframe: _pd.DataFrame, x_column: str = None, y_columns: list[str] = None, figsize=None):
    """
    Plots line plots for specified columns in a DataFrame.

    Args:
        dataframe (_pd.DataFrame): The DataFrame containing the data to plot.
        x_column (str, optional): The column to use as the x-axis values. Defaults to None.
        y_columns (list[str], optional): The list of column names to plot on the y-axis. Defaults to None.
        figsize (tuple, optional): The size of the figure (width, height) in inches. Defaults to None.

    Example:
        >>> import pandas as pd
        >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        >>> df = pd.DataFrame(data)
        >>> lineplot(df)
        >>> lineplot(df, x_column='index', y_columns=['A', 'B'])
    """
    if y_columns is not None:
        columns = list(dataframe[y_columns].columns)
    else:
        columns = dataframe.select_dtypes(include=_np.number).columns.values

    if figsize is None:
        figsize = (20, 3*len(columns))

    if x_column is not None:
        df_with_index = dataframe.set_index(x_column)
        df_with_index[columns].plot(subplots=True, figsize=figsize)
    else:
        dataframe[columns].plot(subplots=True, figsize=figsize)


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
