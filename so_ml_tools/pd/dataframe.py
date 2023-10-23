import pandas as _pd
import numpy as _np

from sklearn.preprocessing import OneHotEncoder

########################################################################################################################
# General
########################################################################################################################


def one_hot_encode_column(*columns: list[any]) -> (OneHotEncoder, list[any]):
    """
    One-Hot encodes all given columns using all values of all the provided columns.

    Example usage:

    encoder, train_labels_one_hot, val_labels_one_hot, test_labels_one_hot = \
    one_hot_encode_column(
        train_df['target'],
        val_df['target'],
        test_df['target']
    )

    If you get the ValueError:

    all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)

    Then that means one of the lists passed in has ha different shape then the others.

    :param columns: one or more columns
    :return: first value is the encoder, all other values are in same order as the columns but encoded
    """
    encoder = OneHotEncoder(sparse_output=False)

    # Preprocess the columns
    preprocessed_columns = []
    for column in columns:
        if isinstance(column, _pd.DataFrame) or isinstance(column, _pd.Series):
            column = column.to_numpy()

        # Make sure the column is two-dimensional
        if column.ndim == 1:
            column = column.reshape(-1, 1)

        preprocessed_columns.append(column)

    # Concatenate all columns and fit the encoder
    encoder.fit(_np.concatenate(preprocessed_columns))

    return_values = []
    # Transform each list now individually
    for preprocessed_column in preprocessed_columns:
        return_values.append(encoder.transform(preprocessed_column))

    return_values.insert(0, encoder)
    return return_values


def convert_column_to_type(dataframe: _pd.DataFrame, columns: list[str], dtype=_np.float64,
                           inplace=True) -> _pd.DataFrame:
    """
    Converts the dtype of the given column to the specified dtype.

    :param dataframe: the pd.DataFrame
    :param columns: A `list` of column names
    :param dtype: the newly to assign dtype.
    :param inplace: return a new instance of the DataFrame or adjust the given DataFrame
    :return: see inplace
    """
    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    for column in columns:
        work_df[column] = work_df[column].astype(dtype)

    return work_df


def delete_null_rows(dataframe: _pd.DataFrame, column_names: list[str], inplace=True) -> _pd.DataFrame:
    """
    Deletes all rows containing a null values inside the given column.

    :param dataframe: the pd.DatFrame
    :param column_names: the names of the columns
    :param inplace: return a new instance of the DataFrame (False) or adjust the given DataFrame
    :return: see inplace
    """
    if not type(column_names) == list and column_names is not None:
        column_names = [column_names]

    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    for c in column_names:
        if c in dataframe:
            work_df.drop(work_df[dataframe[c].isnull()].index, inplace=True)
        else:
            print(f"delete_null_rows: Column '{c}' does not exist in dataframe.")

    return work_df


def delete_rows_where_value_greater_then_z_max(dataframe: _pd.DataFrame, column_names: list[str], inplace=True) -> _pd.DataFrame:
    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    for c in column_names:
        if c in dataframe:
            v = dataframe[c]
            z_max = 3 * v.std() + v.mean()
            work_df.drop(dataframe[dataframe[c] > z_max].index, inplace=True)
        else:
            print(f"delete_rows_where_value_greater_then_z_max: Column '{c}' does not exist in dataframe.")

    return work_df


def fill_nan_with_value(dataframe: _pd.DataFrame, column_values: dict, inplace=True) -> _pd.DataFrame:
    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    for c, v in column_values.items():
        if c in dataframe:
            work_df[c].fillna(value=v, inplace=True)
        else:
            print(f"fill_nan_with_value: Column '{c}' does not exist in dataframe.")

    return work_df


def generate_code_ordinal_encoder(dataframe: _pd.DataFrame, column_names: list[str]) -> None:
    print(f"\n###################################################\nNOTE: The order still needs to be manualy adjusted.\n###################################################\n")
    for c in column_names:
        if c in dataframe:
            try:
                values = dataframe[c].unique()
                categories = "',\n    '".join(values)
                print(f"{c}_encoder = sklearn.preprocessing.OrdinalEncoder(categories=[[\n    '{categories}'\n]])")
            except TypeError as e:
                print(f"Column {c} threw an exception {e}.")
        else:
            print(f"delete_null_rows: Column '{c}' does not exist in dataframe.")


def delete_rows_not_numeric(dataframe: _pd.DataFrame, column_name: str, inplace=True) -> _pd.DataFrame:
    """
    Deletes all rows containing non-number values inside the given column.

    :param dataframe: the pd.DatFrame
    :param column_name: the column
    :param inplace: return a new instance of the DataFrame (False) or adjust the given DataFrame
    :return: see inplace
    """
    return dataframe.drop(dataframe[_pd.to_numeric(dataframe[column_name], errors='coerce').isna()].index,
                          inplace=inplace)


def drop_columns(dataframe: _pd.DataFrame, column_names: list[str]) -> None:
    """
    Removes (inplace) the given column names from the dataframe.

    :param dataframe: the pd.DatFrame
    :param column_names: the names of the columns to remove
    """
    if not type(column_names) == list and column_names is not None:
        column_names = [column_names]

    for c in column_names:
        if c in dataframe:
            dataframe.drop(c, axis=1, inplace=True)
        else:
            print(f"drop_columns: Column '{c}' does not exist in dataframe.")


def describe(dataframe: _pd.DataFrame, column_names: list[str] = None, round=2):
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

        _mean = _std = _min = _q25 = _q50 = _q75 = _max = z_min = z_max = _np.NAN
        if _pd.api.types.is_numeric_dtype(v):
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
    return _pd.DataFrame(
        columns=["Column", "DType", "NotNull", "Null", "Unique", "Mean", "Std", "Z-Min", "Z-Max", "Min", "25%", "50%",
                 "75%", "Max"], data=data).round(round)
