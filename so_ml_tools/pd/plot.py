import matplotlib.pyplot as _plt
import seaborn as _sns
import pandas as _pd


def count_categories(dataframe: _pd.DataFrame, label_column: str = None, column_names: list[str] = None, cols=3, figsize: tuple = None):
    if column_names is None:
        column_names = dataframe.select_dtypes(exclude='number').columns.tolist()

    rows = max(int(len(column_names) / cols), 1)
    cols = min(cols, len(column_names))
    rows += 1 if rows * cols < len(column_names) else 0

    # If figsize is not specified then calculate the fig-size
    if figsize is None:
        figsize = (17, rows * 4)

    fig, axs = _plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

    # If we have more then one column then flatten the axis so we can loop through them,
    # if we have only one column then create list containing the axis so we can still loop through it.
    if len(column_names) > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    n = 0
    for column_name in column_names:
        _sns.countplot(ax=axs[n], data=dataframe, x=column_name, hue=label_column)
        n += 1

    _plt.show()


def histogram_for_columns(dataframe: _pd.DataFrame, column_names: list[str] = None, log=False,
                                  min_nunique: int = 0, max_nunique: int = 50, figsize: tuple = None, cols=3,
                                  verbose=1):
    """
    Plots a histogram for each of the numeric columns in the DataFrame.

    :param dataframe: the pandas dataframe
    :param column_names: columns which exist within the DataFrame if none specified all columns that are numeric
        will be processed
    :param log: set to True to enable logarithmic scaling
    :param min_nunique: minimum number of unique values present, if lower then this then no graph will be displayed (since it is basically boolean)
    :param max_nunique: maximum number of unique values present, only applicable to object column types since these cannot be binned
    :param figsize: size of the plot, if None specified then one is calculated
    :param cols: number of plots on the horizontal axis
    :param verbose: display messages when columns are not visualized
    """
    # assert column_names is not None, "column_names cannot be None"

    # If the column_names argument is not a list then create a list
    if not type(column_names) == list and column_names is not None:
        column_names = [column_names]

    # If we don't have a list of column names then create a histogram for every column.
    if column_names is not None:
        columns = list(dataframe[column_names].columns)
    else:
        columns = list(dataframe.columns)

    # Calculate the number of rows / columns for the subplot.
    rows = max(int(len(columns) / cols), 1)
    cols = min(cols, len(columns))
    rows += 1 if rows * cols < len(columns) else 0

    # If figsize is not specified then calculate the fig-size
    if figsize is None:
        figsize = (17, rows * 4)

    fig, axs = _plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

    # If we have more then one column then flatten the axis so we can loop through them,
    # if we have only one column then create list containing the axis so we can still loop through it.
    if len(columns) > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    # Horizontal / vertical padding between the histograms.
    fig.tight_layout(h_pad=10, w_pad=5)

    n = 0
    for c in columns:
        v = dataframe[c]
        nunique = v.nunique()

        if _pd.api.types.is_numeric_dtype(v) and min_nunique < nunique:
            # Make sure we do not get more then 50 bins.
            nunique = 50 if nunique > 50 else nunique
            v.hist(ax=axs[n], bins=nunique + 1, log=log, facecolor='#2ab0ff', edgecolor='#169acf', align='left', linewidth=0.1)

            color = iter(["black", "darkred", "red", "orangered", 'limegreen', 'green', 'darkgreen'])
            axs[n].axvline(v.mean(), label=f'Mean', color=next(color), linestyle='dashed')

            z_score_min = [-1 * v.std() + v.mean(), -2 * v.std() + v.mean(), -3 * v.std() + v.mean()]
            for idx, value in enumerate(z_score_min):
                # Only show when it is not before the graph starts.
                c = next(color)
                if value > v.min():
                    axs[n].axvline(value, label=f'(-{idx+1}σ)', color=c, linestyle='dashed')

            z_score_max = [1 * v.std() + v.mean(), 2 * v.std() + v.mean(), 3 * v.std() + v.mean()]
            for idx, value in enumerate(z_score_max):
                # Only show when it is not after the graph starts.
                c = next(color)
                if value < v.max():
                    axs[n].axvline(value, label=f'({idx+1}σ)', color=c, linestyle='dashed')

            axs[n].legend()
        else:
            if verbose:
                print(f"Column '{v.name}' is not visualized, the number of nunique values ({nunique}) either exceeds {max_nunique} or is lower then {min_nunique}.")
            continue

        axs[n].set(title=v.name)

        # Only rotate the x labels
        for tick in axs[n].get_xticklabels():
            tick.set_rotation(45)

        # Remove unnecessary white space on the left/right side of the graph.
        axs[n].margins(x=0)
        axs[n].grid(axis='x')
        n += 1

    for i in range(n, rows*cols):
        fig.delaxes(axs[i])

    _plt.show()
