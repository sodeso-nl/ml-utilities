import pandas as pd
import numpy as np


def convert_column_to_type(dataframe: pd.DataFrame, columns: list[str], dtype=np.float64, inplace=True) -> pd.DataFrame:
    """
    Converts the dtype of the given column to the specified dtype.

    :param dataframe: the pd.DataFrame
    :param columns: A `list` of column names
    :param dtype: the newly to assign dtype.
    :param inplace: return a new instance of the DataFrame or adjust the given DataFrame
    :return: see inplace
    """
    result = dataframe
    if not inplace:
        result = dataframe.copy(deep=True)

    for column in columns:
        result[column] = result[column].astype(dtype)

    return result


def delete_null_rows(dataframe: pd.DataFrame, column_name: str, inplace=True) -> pd.DataFrame:
    """
    Deletes all rows containing a null values inside the given column.

    :param dataframe: the pd.DatFrame
    :param column_name: the column name
    :param inplace: return a new instance of the DataFrame (False) or adjust the given DataFrame
    :return: see inplace
    """
    return dataframe.drop(dataframe[dataframe[column_name].isnull()].index, inplace = inplace)


def delete_rows_not_numeric(dataframe: pd.DataFrame, column_name: str, inplace=True) -> pd.DataFrame:
    """
    Deletes all rows containing non-number values inside the given column.

    :param dataframe: the pd.DatFrame
    :param column_name: the column
    :param inplace: return a new instance of the DataFrame (False) or adjust the given DataFrame
    :return: see inplace
    """
    return dataframe.drop(dataframe[pd.to_numeric(dataframe[column_name], errors='coerce').isna()].index, inplace=inplace)


def drop_columns(dataframe: pd.DataFrame, column_names: list[str]) -> None:
    """
    Removes (inplace) the given column names from the dataframe.

    :param dataframe: the pd.DatFrame
    :param column_names: the names of the columns to remove
    """
    if not type(column_names) == list and column_names is not None:
        column_names = [column_names]

    for c in column_names:
        dataframe.drop(c, axis=1, inplace=True)


def describe(dataframe: pd.DataFrame, column_names: list[str] = None, round=2):
    # If the column_names argument is not a list then create a list
    if not type(column_names) == list and column_names is not None:
        column_names = [column_names]

    # If we don't have a list of column names then create a histogram for every column.
    if column_names is not None:
        columns = list(dataframe[column_names].columns)
    else:
        columns = list(dataframe.columns)

    data = []
    for c in columns:
        v = dataframe[c]

        _mean = _std = _min = _q25 = _q50 = _q75 = _max = z_min = z_max = np.NAN
        if pd.api.types.is_numeric_dtype(v):
            _mean = v.mean()
            _std = v.std()
            _min = v.min()
            _q25 = v.quantile(.25)
            _q50 = v.quantile(.50)
            _q75 = v.quantile(.75)
            _max = v.max()

            # Calculate lower value when Z-Score = -3
            z_min = -3 * _std + _mean

            # Calculate upper value when Z-Score = 3
            z_max = 3 * _std + _mean

        data.append([
            v.name, v.dtype, v.count(), v.isna().sum(), v.nunique(), _mean, _std, z_min, z_max, _min, _q25,
            _q50, _q75, _max
        ])

    print(f"Total number of rows: {len(dataframe)}")
    return pd.DataFrame(columns=["Column", "DType", "NotNull", "Null", "Unique", "Mean", "Std", "Z-Min", "Z-Max", "Min", "25%", "50%", "75%", "Max"], data=data).round(round)
