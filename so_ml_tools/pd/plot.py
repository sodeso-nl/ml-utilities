import matplotlib.pyplot as _plt
import seaborn as _sns
import pandas as _pd
import numpy as _np
import scipy as _sp


def count_categories(dataframe: _pd.DataFrame, label_column: str = None, column_names: list[str] = None, cols=3,
                     figsize: tuple = None):
    """
    Create count plots for categorical columns in a DataFrame, optionally segmented by a label column.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - label_column (str): The column used for segmentation (optional). If provided, each count plot will be segmented by the unique values in this column.
    - column_names (List[str]): List of column names to create count plots for. If not provided, all non-numeric columns in the DataFrame will be used.
    - cols (int): Number of columns in the grid layout for count plots.
    - figsize (Tuple[int, int]): Size of the entire figure (width, height). If not provided, the figure size is calculated based on the number of rows and a default height.

    Returns:
    - None

    Example:
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Sample DataFrame
    data = {'Category': ['A', 'B', 'A', 'C', 'B', 'C'], 'Label': [1, 0, 1, 1, 0, 1]}
    df = pd.DataFrame(data)

    # Create count plots for categorical columns
    count_categories(dataframe=df, label_column='Label', cols=2)
    ```
    """
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
                          min_nunique: int = 0, max_nunique: int = 50, figsize: tuple = None, cols=1,
                          verbose=1, label_color='black'):
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
    :param label_color: label color of ticks, titles, x/y axis values / labels.
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
    fig.patch.set_alpha(0.0)  # Transparant background

    n = 0
    for c in columns:
        v = dataframe[c]
        nunique = v.nunique()

        if _pd.api.types.is_numeric_dtype(v) and min_nunique < nunique:
            # Make sure we do not get more then 50 bins.
            nunique = 50 if nunique > 50 else nunique
            bins = 'auto'
            if nunique < 3:
                bins = 2
            _sns.histplot(data=v, kde=bins == 'auto', stat='count', bins=bins, ax=axs[n], kde_kws={'bw_adjust': .15})

            if nunique > 3:
                # Display the mean
                axs[n].text(v.mean(), axs[n].get_ylim()[1] + 10, 'Mean', rotation='vertical', ha='center', va='bottom')

                axs[n].axvline(v.mean(), label=f'Mean ({v.mean():.2f})', color='black', linestyle='dotted')

                # Display the Z-Score
                color = iter(["red", "red", "red", 'red', 'red', 'red'])
                z_score_min = [-3 * v.std() + v.mean(), -2 * v.std() + v.mean(), -1 * v.std() + v.mean()]
                for idx, value in enumerate(z_score_min):
                    # Only show when it is not before the graph starts.
                    c = next(color)
                    if value > v.min():
                        axs[n].text(value, axs[n].get_ylim()[1] + 10, f'-{3 - idx}σ', rotation='vertical', ha='center',
                                    va='bottom', color=label_color)
                        axs[n].axvline(value, label=f'-{3 - idx}σ ({value:.2f})', color=c, linestyle='dotted')

                z_score_max = [1 * v.std() + v.mean(), 2 * v.std() + v.mean(), 3 * v.std() + v.mean()]
                for idx, value in enumerate(z_score_max):
                    # Only show when it is not after the graph starts.
                    c = next(color)
                    if value < v.max():
                        axs[n].text(value, axs[n].get_ylim()[1] + 10, f'{idx + 1}σ', rotation='vertical', ha='center',
                                    va='bottom', color=label_color)
                        axs[n].axvline(value, label=f'{idx + 1}σ ({value:.2f})', color=c, linestyle='dotted')

                # Display the IREQ and lower / upper bounds.
                q1 = v.quantile(q=.25)
                q3 = v.quantile(q=.75)
                irq = (q3 - q1)
                irql = q1 - (1.5 * irq)
                irqu = q3 + (1.5 * irq)

                if axs[n].get_xlim()[1] > irql > axs[n].get_xlim()[0]:
                    axs[n].text(irql, axs[n].get_ylim()[1] + 10, 'IRQ-L', rotation='vertical', ha='center', va='bottom',
                                color=label_color)
                    axs[n].axvline(irql, label=f'IRQ-L ({irql:.2f})', color='green', linestyle='dotted')

                if axs[n].get_xlim()[1] > q1 > axs[n].get_xlim()[0]:
                    axs[n].text(q1, axs[n].get_ylim()[1] + 10, 'Q1', rotation='vertical', ha='center', va='bottom',
                                color=label_color)
                    axs[n].axvline(q1, label=f'Q1 ({q1:.2f})', color='green', linestyle='dotted')

                if axs[n].get_xlim()[1] > q1 > axs[n].get_xlim()[0]:
                    axs[n].text(q3, axs[n].get_ylim()[1] + 10, 'Q3', rotation='vertical', ha='center', va='bottom',
                                color=label_color)
                    axs[n].axvline(q3, label=f'Q3 ({q3:.2f})', color='green', linestyle='dotted')

                if axs[n].get_xlim()[1] > irqu > axs[n].get_xlim()[0]:
                    axs[n].text(irqu, axs[n].get_ylim()[1] + 10, 'IRQ-U', rotation='vertical', ha='center', va='bottom',
                                color=label_color)
                    axs[n].axvline(irqu, label=f'IRQ-U ({irqu:.2f})', color='green', linestyle='dotted')

                axs[n].legend()
            axs[n].grid(False)
        else:
            if verbose:
                print(
                    f"Column '{v.name}' is not visualized, the number of nunique values ({nunique}) either exceeds {max_nunique} or is lower then {min_nunique} or the values are not numeric.")
            continue

        # Only rotate the x labels
        for tick in axs[n].get_xticklabels():
            tick.set_rotation(45)

        # Remove unnecessary white space on the left/right side of the graph.
        axs[n].margins(x=0)
        axs[n].grid(axis='y')

        axs[n].xaxis.label.set_color(label_color)  # Set color of x-axis label
        axs[n].tick_params(axis='x', colors=label_color)  # Set color of x-axis ticks.

        axs[n].yaxis.label.set_color(label_color)  # Set color of y-axis label
        axs[n].tick_params(axis='y', colors=label_color)  # Set color of y-axis ticks.
        axs[n].title.set_color(label_color)  # Set color of title

        n += 1

    for i in range(n, rows * cols):
        fig.delaxes(axs[i])

    _plt.show()


def correlation(dataframe: _pd.DataFrame, numeric_only=False, method: str = 'pearson', figsize: tuple = (12, 12),
                label_color='black') -> None:
    """
    Create a correlation heatmap for the given DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - numeric_only (bool): Whether to include only numeric columns.
    - method (str): The method to use for the correlation heatmap (pearson, spearman or kendall)
    - figsize (tuple): Size of the figure.
    - label_color (str): Color of axis labels and title.

    Returns:
    - None
    """
    # Validate input parameters
    if not isinstance(dataframe, _pd.DataFrame):
        raise ValueError("Input 'dataframe' must be a Pandas DataFrame.")

    if not isinstance(figsize, tuple):
        raise ValueError("'figsize' must be a tuple.")

    if method not in ['pearson', 'kendall', 'spearman']:
        raise ValueError("Invalid correlation method. Supported methods: 'pearson', 'kendall', 'spearman'.")

    # Create the correlation dataset
    df_corr = dataframe.corr(numeric_only=numeric_only, method=method)

    # Create a mask so that only the lower left triangle remains.
    mask = _np.triu(_np.ones_like(df_corr, dtype=bool))

    # Create heatmap.
    fig, ax = _plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)  # Transparent background
    _sns.heatmap(data=df_corr.round(2), mask=mask, vmin=-1, vmax=1, annot=True, cmap='coolwarm',
                 ax=ax)

    # Customize labels and title
    ax.xaxis.label.set_color(label_color)  # Set color of x-axis label
    ax.tick_params(axis='x', colors=label_color)  # Set color of x-axis ticks.

    ax.yaxis.label.set_color(label_color)  # Set color of y-axis label
    ax.tick_params(axis='y', colors=label_color)  # Set color of y-axis ticks.

    ax.title.set_color(label_color)  # Set color of title
    ax.set_title('Correlation')
    _plt.show()


def correlation_pb(dataframe: _pd.DataFrame, continuous_columns: list, dichotomous_column: str, figsize: tuple = (12, 6),
                   label_color='black') -> None:
    """
    Create a correlation bar plot between continuous features and a dichotomous feature using point-biserial correlation coefficient.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame with continuous and dichotomous features.
    - continuous_columns (list): List of continuous variable columns.
    - dichotomous_column (str): Column name of the dichotomous variable.
    - figsize (tuple): Size of the figure.
    - label_color (str): Color of axis labels and title.

    Returns:
    - None
    """
    # Validate input parameters
    if not isinstance(dataframe, _pd.DataFrame):
        raise ValueError("Input 'dataframe' must be a Pandas DataFrame.")

    if not isinstance(figsize, tuple):
        raise ValueError("'figsize' must be a tuple.")

    if not isinstance(continuous_columns, list) or not all(isinstance(col, str) for col in continuous_columns):
        raise ValueError("'continuous_columns' must be a list of continuous variable column names.")

    if not isinstance(dichotomous_column, str):
        raise ValueError(
            "'dichotomous_column' must be a string representing the column name of the dichotomous variable.")

    if dichotomous_column not in dataframe.columns:
        raise ValueError(f"'{dichotomous_column}' not found in the DataFrame columns.")

    for col in continuous_columns:
        if col not in dataframe.columns:
            raise ValueError(f"'{col}' not found in the DataFrame columns.")

    # Drop Null columns.
    dataframe = dataframe.dropna()

    # Calculate point-biserial correlation coefficients
    corr_results = [_sp.stats.pointbiserialr(dataframe[col], dataframe[dichotomous_column]) for col in continuous_columns]

    # Extract correlation coefficients and p-values
    corr_coefficients = [result[0] for result in corr_results]
    p_values = [result[1] for result in corr_results]

    # Set up the matplotlib figure
    fig, ax = _plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)  # Transparent background

    # Create a color palette for the bars
    color_palette = _sns.color_palette("coolwarm", len(continuous_columns))

    # Create a bar plot using Seaborn
    _sns.barplot(x=continuous_columns, y=corr_coefficients, ax=ax, palette=color_palette)

    for i, (value, p_value) in enumerate(zip(corr_coefficients, p_values)):
        ax.text(i, value, f"c:{value:.2f}, p:{p_value:.4f}", ha='center', va='bottom')

    # Customize labels and title
    ax.set_xlabel("Continuous Variables", color=label_color)
    ax.xaxis.label.set_color(label_color)
    ax.tick_params(axis='x', colors=label_color)

    ax.set_ylabel("Point-Biserial Correlation Coefficient", color=label_color)
    ax.yaxis.label.set_color(label_color)
    ax.tick_params(axis='y', colors=label_color)

    ax.title.set_color(label_color)
    _plt.title(f'Correlation with {dichotomous_column}')

    _plt.show()
