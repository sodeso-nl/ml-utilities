import pandas as _pd
import numpy as _np

from sklearn.preprocessing import OneHotEncoder


########################################################################################################################
# General
########################################################################################################################

def simple_undersample(dataframe: _pd.DataFrame, label_column, value_to_undersample, undersample_to):
    """
    Undersample a class (label_column) with a value (value_to_undersample) to the number of entries (undersample_to)

    For example:

    undersample(dataframe: training_data, column_name='stroke', value_to_undersample=0, undersample_to=2000)

    Args:
        dataframe: the pd.DataFrame
        label_column: the column to match on
        value_to_undersample: the value within the `column` to match on for under-sampling.
        undersample_to: number of entries that need to exist after the under-sampling.

    Returns:
        A new dataframe with the data.
    """
    print('A better solution for under-sampling / over-sampling is to use the soml.imblearn.smote package.')
    unique = dataframe[label_column].unique()

    sections = {}
    for v in unique:
        sections[v] = dataframe[dataframe[label_column] == v]

    sections[value_to_undersample] = sections[value_to_undersample].sample(undersample_to)
    return _pd.concat(sections.values(), axis=0)


def simple_oversample(dataframe: _pd.DataFrame, label_column, value_to_oversample, oversample_to):
    """
    oversample a class (label_column) with a value (value_to_oversample) to the number of entries (oversample_to)

    For example:

    oversample(dataframe: training_data, column_name='stroke', value_to_oversample=1, oversample_to=2000)

    Args:
        dataframe: the pd.DataFrame
        label_column: the column to match on
        value_to_oversample: the value within the `column` to match on for over sampling.
        oversample_to: number of entries that need to exist after the over sampling.

    Returns:
        A new dataframe with the data.
    """
    print('A better solution for under-sampling / over-sampling is to use the soml.imblearn.smote package.')
    unique = dataframe[label_column].unique()

    sections = {}
    for v in unique:
        sections[v] = dataframe[dataframe[label_column] == v]

    sections[value_to_oversample] = sections[value_to_oversample].sample(oversample_to, replace=True)
    return _pd.concat(sections.values(), axis=0)


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

    all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the
    array at index 1 has 2 dimension(s)

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


def column_as_dataframe(dataframe: _pd.DataFrame, column_name: str, drop_after: False) -> _pd.DataFrame:
    column = dataframe[[column_name]]

    if drop_after:
        drop_columns(dataframe=dataframe, column_names=[column_name])

    return column


def convert_column_to_type(dataframe: _pd.DataFrame, columns: list[str], dtype=_np.float64,
                           inplace=True) -> _pd.DataFrame | None:
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

    if inplace:
        return None

    return work_df


def delete_rows_where_columns_have_null_value(dataframe: _pd.DataFrame, column_names: list[str] = None, inplace=True) \
        -> _pd.DataFrame | None:
    """
    Deletes all rows containing a null values inside the given column .

    :param dataframe: the pd.DatFrame
    :param column_names: the names of the columns to check, if None all columns will be checked.
    :param inplace: return a new instance of the DataFrame (False) or adjust the given DataFrame
    :return: see inplace
    """
    if column_names is None:
        column_names = list(dataframe.columns)

    if not type(column_names) is list and column_names is not None:
        column_names = [column_names]

    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    for c in column_names:
        if c in dataframe:
            work_df.drop(work_df[dataframe[c].isnull()].index, inplace=True)
        else:
            print(f"delete_null_rows: Column '{c}' does not exist in dataframe.")

    if inplace:
        return None

    return work_df


def delete_rows_where_value_equal_to(dataframe: _pd.DataFrame, column_name: str, value: object, inplace=True) \
        -> _pd.DataFrame | None:
    """
    Deletes all rows where the specified column has the specified value.

    :param dataframe: the pd.DatFrame
    :param column_name: the names of the column to check
    :param value: the value to compare with
    :param inplace: return a new instance of the DataFrame (False) or adjust the given DataFrame
    :return: see inplace
    """
    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    if column_name in dataframe:
        work_df.drop(dataframe[dataframe[column_name] == value].index, inplace=True)
    else:
        print(f"delete_rows_where_value_equal_to: Column '{column_name}' does not exist in dataframe.")

    if inplace:
        return None

    return work_df


def delete_rows_where_value_smaller_then(dataframe: _pd.DataFrame, column_name: str, value: object, inplace=True) \
        -> _pd.DataFrame | None:
    """
    Deletes all rows where the specified column has the specified value.

    :param dataframe: the pd.DatFrame
    :param column_name: the names of the column to check
    :param value: the value to compare with
    :param inplace: return a new instance of the DataFrame (False) or adjust the given DataFrame
    :return: see inplace
    """
    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    if column_name in dataframe:
        work_df.drop(dataframe[dataframe[column_name] < value].index, inplace=True)
    else:
        print(f"delete_rows_where_value_equal_to: Column '{column_name}' does not exist in dataframe.")

    if inplace:
        return None

    return work_df


def delete_rows_where_value_larger_then(dataframe: _pd.DataFrame, column_name: str, value: object, inplace=True) \
        -> _pd.DataFrame | None:
    """
    Deletes all rows where the specified column has the specified value.

    :param dataframe: the pd.DatFrame
    :param column_name: the names of the column to check
    :param value: the value to compare with
    :param inplace: return a new instance of the DataFrame (False) or adjust the given DataFrame
    :return: see inplace
    """
    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    if column_name in dataframe:
        work_df.drop(dataframe[dataframe[column_name] > value].index, inplace=True)
    else:
        print(f"delete_rows_where_value_equal_to: Column '{column_name}' does not exist in dataframe.")

    if inplace:
        return None

    return work_df


def fill_nan_with_value(dataframe: _pd.DataFrame, column_names: list[str], value, inplace=True, add_indicator=False) \
        -> _pd.DataFrame | None:
    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    if type(column_names) is not list and column_names is not None:
        column_names = [column_names]

    for c in column_names:
        if c in dataframe:
            if work_df[c].isna().any():
                if add_indicator:
                    work_df[c + '_nan'] = work_df[c].isna().astype(int)
                work_df[c].fillna(value=value, inplace=True)
        else:
            print(f"fill_nan_with_value: Column '{c}' does not exist in dataframe.")

    if inplace:
        return None

    return work_df


def fill_nan_with_previous_value(dataframe: _pd.DataFrame, column_names: list[str], inplace=True, add_indicator=False) \
        -> _pd.DataFrame | None:
    """
        Fill in missing values based on the previous value.

        Args:
            dataframe: the pd.DatFrame
            column_names: list of column names with `NaN` values.
            inplace: update the given dataframe or return a new dataframe.
            add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    if type(column_names) is not list and column_names is not None:
        column_names = [column_names]

    for c in column_names:
        if c in dataframe:
            if add_indicator:
                work_df[c + '_nan'] = work_df[c].isna().astype(int)
            work_df[c].fillna(method='ffill', inplace=True)
        else:
            print(f"fill_nan_with_previous_value: Column '{c}' does not exist in dataframe.")

    if inplace:
        return None

    return work_df


def fill_nan_with_next_value(dataframe: _pd.DataFrame, column_names: list[str], inplace=True, add_indicator=False) \
        -> _pd.DataFrame | None:
    """
        Fill in missing values based on the next value.

        Args:
            dataframe: the pd.DatFrame
            column_names: list of column names with `NaN` values.
            inplace: update the given dataframe or return a new dataframe.
            add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    if type(column_names) is not list and column_names is not None:
        column_names = [column_names]

    for c in column_names:
        if c in dataframe:
            if add_indicator:
                work_df[c + '_nan'] = work_df[c].isna().astype(int)
            work_df[c].fillna(method='bfill', inplace=True)
        else:
            print(f"fill_nan_with_next_value: Column '{c}' does not exist in dataframe.")

    if inplace:
        return None

    return work_df


def fill_nan_with_global_mode(dataframe: _pd.DataFrame, column_names: list[str], inplace=True,
                              add_indicator=False) -> _pd.DataFrame | None:
    """
        Fill in missing values based on the mode (most often) value which is calculated on all non `NaN` values
        in the same column.

        Args:
            dataframe: the pd.DatFrame
            column_names: list of column names with `NaN` values.
            inplace: update the given dataframe or return a new dataframe.
            add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_global_func(dataframe=dataframe, column_names=column_names, agg_func='mode', inplace=inplace,
                                     add_indicator=add_indicator)


def fill_nan_with_global_kurt(dataframe: _pd.DataFrame, column_names: list[str], inplace=True,
                              add_indicator=False) -> _pd.DataFrame | None:
    """
        Fill in missing values based on kurt value which is calculated on all non `NaN` values in the same
        column.

        Args:
            dataframe: the pd.DatFrame
            column_names: list of column names with `NaN` values.
            inplace: update the given dataframe or return a new dataframe.
            add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_global_func(dataframe=dataframe, column_names=column_names, agg_func='kurt', inplace=inplace,
                                     add_indicator=add_indicator)


def fill_nan_with_global_skew(dataframe: _pd.DataFrame, column_names: list[str], inplace=True,
                              add_indicator=False) -> _pd.DataFrame | None:
    """
        Fill in missing values based on skew value which is calculated on all non `NaN` values in the same
        column.

        Args:
            dataframe: the pd.DatFrame
            column_names: list of column names with `NaN` values.
            inplace: update the given dataframe or return a new dataframe.
            add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_global_func(dataframe=dataframe, column_names=column_names, agg_func='kurt', inplace=inplace,
                                     add_indicator=add_indicator)


def fill_nan_with_global_median(dataframe: _pd.DataFrame, column_names: list[str], inplace=True,
                                add_indicator=False) -> _pd.DataFrame | None:
    """
        Fill in missing values based on median value which is calculated on all non `NaN` values in the same
        column.

        Args:
            dataframe: the pd.DatFrame
            column_names: list of column names with `NaN` values.
            inplace: update the given dataframe or return a new dataframe.
            add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_global_func(dataframe=dataframe, column_names=column_names, agg_func='median', inplace=inplace,
                                     add_indicator=add_indicator)


def fill_nan_with_global_mean(dataframe: _pd.DataFrame, column_names: list[str], inplace=True,
                              add_indicator=False) -> _pd.DataFrame | None:
    """
        Fill in missing values based on mean value which is calculated on all non `NaN` values in the same
        column.

        Args:
            dataframe: the pd.DatFrame
            column_names: list of column names with `NaN` values.
            inplace: update the given dataframe or return a new dataframe.
            add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_global_func(dataframe=dataframe, column_names=column_names, agg_func='mean', inplace=inplace,
                                     add_indicator=add_indicator)


def fill_nan_with_global_max(dataframe: _pd.DataFrame, column_names: list[str], inplace=True, add_indicator=False) \
        -> _pd.DataFrame | None:
    """
        Fill in missing values based on max value which is calculated on all non `NaN` values in the same
        column.

        Args:
            dataframe: the pd.DatFrame
            column_names: list of column names with `NaN` values.
            inplace: update the given dataframe or return a new dataframe.
            add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_global_func(dataframe=dataframe, column_names=column_names, agg_func='max', inplace=inplace,
                                     add_indicator=add_indicator)


def fill_nan_with_global_min(dataframe: _pd.DataFrame, column_names: list[str], inplace=True,
                             add_indicator=False) -> _pd.DataFrame | None:
    """
        Fill in missing values based on min value which is calculated on all non `NaN` values in the same
        column.

        Args:
            dataframe: the pd.DatFrame
            column_names: list of column names with `NaN` values.
            inplace: update the given dataframe or return a new dataframe.
            add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_global_func(dataframe=dataframe, column_names=column_names, agg_func='min', inplace=inplace,
                                     add_indicator=add_indicator)


def fill_nan_with_global_func(dataframe: _pd.DataFrame, column_names: list[str], agg_func: str, inplace=True,
                              add_indicator=False) -> _pd.DataFrame | None:
    """
        Fill in missing values based on a func value which is calculated on all non `NaN` values in the same
        column.

        Args:
            dataframe: the pd.DatFrame
            column_names: list of column names with `NaN` values.
            agg_func: type of function to use: 'mean', 'min', 'max', 'median', 'skew' or 'kurt'
            inplace: update the given dataframe or return a new dataframe.
            add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    for c in column_names:
        if c in dataframe:
            if add_indicator:
                work_df[c + '_nan'] = work_df[c].isna().astype(int)

            c_value = work_df[c].apply(agg_func)
            work_df[c].fillna(value=c_value, inplace=True)
        else:
            print(f"fill_nan_with_mean: Column '{c}' does not exist in dataframe.")

    if inplace:
        return None

    return work_df


def ffill_nan_with_groupby_mode(dataframe: _pd.DataFrame, column_name: str, group_by_column_name: str,
                                  remainder_agg_func='mode', inplace=True, add_indicator=False) -> _pd.DataFrame | None:
    """
    Fill in missing values based on the mode value (most often) which is calculated on all data having a similar value
    in another column. For example, fill in BMI values based on other BMI values that have the same age.

    If any NaN values are left over then these can be filled in by setting the `fill_remainder_with_global_func` the
    appropriate function.

    Alternatively you can use the soml.sklearn.SimpleGroupByImputer when you want to do this in a ColumnTransformer.

    Args:
        dataframe: the pd.DatFrame
        column_name: name of the column with the `NaN` values.
        group_by_column_name: name of the column to use as the group by column to calculate the mode
            value of column_name.
        remainder_agg_func: fill any remaining `NaN` values with the global function.
        inplace: update the given dataframe or return a new dataframe.
        add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_groupby_func(dataframe=dataframe, column_name=column_name,
                                         group_by_column_name=group_by_column_name, agg_func='mode',
                                         remainder_agg_func=remainder_agg_func,
                                         inplace=inplace, add_indicator=add_indicator)


def fill_nan_with_groupby_kurt(dataframe: _pd.DataFrame, column_name: str, group_by_column_name: str,
                                  remainder_agg_func='kurt', inplace=True, add_indicator=False) -> _pd.DataFrame | None:
    """
    Fill in missing values based on the kurt value which is calculated on all data having a similar value in
    another column. For example, fill in BMI values based on other BMI values that have the same age.

    If any NaN values are left over then these can be filled in by setting the `fill_remainder_with_global_func` the
    appropriate function.

    Alternatively you can use the soml.sklearn.SimpleGroupByImputer when you want to do this in a ColumnTransformer.

    Args:
        dataframe: the pd.DatFrame
        column_name: name of the column with the `NaN` values.
        group_by_column_name: name of the column to use as the group by column to calculate the kurt
            value of column_name.
        remainder_agg_func: fill any remaining `NaN` values with the global function.
        inplace: update the given dataframe or return a new dataframe.
        add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_groupby_func(dataframe=dataframe, column_name=column_name,
                                         group_by_column_name=group_by_column_name, agg_func='kurt',
                                         remainder_agg_func=remainder_agg_func,
                                         inplace=inplace, add_indicator=add_indicator)


def fill_nan_with_groupby_skew(dataframe: _pd.DataFrame, column_name: str, group_by_column_name: str,
                                  remainder_agg_func='skew', inplace=True, add_indicator=False) -> _pd.DataFrame | None:
    """
    Fill in missing values based on the skew value which is calculated on all data having a similar value in
    another column. For example, fill in BMI values based on other BMI values that have the same age.

    If any NaN values are left over then these can be filled in by setting the `fill_remainder_with_global_func` the
    appropriate function.

    Alternatively you can use the soml.sklearn.SimpleGroupByImputer when you want to do this in a ColumnTransformer.

    Args:
        dataframe: the pd.DatFrame
        column_name: name of the column with the `NaN` values.
        group_by_column_name: name of the column to use as the group by column to calculate the skew
            value of column_name.
        remainder_agg_func: fill any remaining `NaN` values with the global function.
        inplace: update the given dataframe or return a new dataframe.
        add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_groupby_func(dataframe=dataframe, column_name=column_name,
                                         group_by_column_name=group_by_column_name, agg_func='skew',
                                         remainder_agg_func=remainder_agg_func,
                                         inplace=inplace, add_indicator=add_indicator)


def fill_nan_with_groupby_median(dataframe: _pd.DataFrame, column_name: str, group_by_column_name: str,
                                    remainder_agg_func='median', inplace=True, add_indicator=False) -> _pd.DataFrame | None:
    """
    Fill in missing values based on the median value which is calculated on all data having a similar value in
    another column. For example, fill in BMI values based on other BMI values that have the same age.

    If any NaN values are left over then these can be filled in by setting the `fill_remainder_with_global_func` the
    appropriate function.

    Alternatively you can use the soml.sklearn.SimpleGroupByImputer when you want to do this in a ColumnTransformer.

    Args:
        dataframe: the pd.DatFrame
        column_name: name of the column with the `NaN` values.
        group_by_column_name: name of the column to use as the group by column to calculate the median
            value of column_name.
        remainder_agg_func: fill any remaining `NaN` values with the global function.
        inplace: update the given dataframe or return a new dataframe.
        add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_groupby_func(dataframe=dataframe, column_name=column_name,
                                         group_by_column_name=group_by_column_name, agg_func='median',
                                         remainder_agg_func=remainder_agg_func,
                                         inplace=inplace, add_indicator=add_indicator)


def fill_nan_with_groupby_mean(dataframe: _pd.DataFrame, column_name: str, group_by_column_name: str,
                                  remainder_agg_func='mean', inplace=True, add_indicator=False) -> _pd.DataFrame | None:
    """
    Fill in missing values based on the mean value which is calculated on all data having a similar value in
    another column. For example, fill in BMI values based on other BMI values that have the same age.

    If any NaN values are left over then these can be filled in by setting the `fill_remainder_with_global_func` the
    appropriate function.

    Alternatively you can use the soml.sklearn.SimpleGroupByImputer when you want to do this in a ColumnTransformer.

    Args:
        dataframe: the pd.DatFrame
        column_name: name of the column with the `NaN` values.
        group_by_column_name: name of the column to use as the group by column to calculate the mean
            value of column_name.
        remainder_agg_func: fill any remaining `NaN` values with the global function.
        inplace: update the given dataframe or return a new dataframe.
        add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_groupby_func(dataframe=dataframe, column_name=column_name,
                                         group_by_column_name=group_by_column_name, agg_func='mean',
                                         remainder_agg_func=remainder_agg_func,
                                         inplace=inplace, add_indicator=add_indicator)


def fill_nan_with_grouped_max(dataframe: _pd.DataFrame, column_name: str, group_by_column_name: str,
                                 remainder_agg_func='max', inplace=True, add_indicator=False) -> _pd.DataFrame | None:
    """
    Fill in missing values based on the max value which is calculated on all data having a similar value in
    another column. For example, fill in BMI values based on other BMI values that have the same age.

    If any NaN values are left over then these can be filled in by setting the `fill_remainder_with_global_func` the
    appropriate function.

    Alternatively you can use the soml.sklearn.SimpleGroupByImputer when you want to do this in a ColumnTransformer.

    Args:
        dataframe: the pd.DatFrame
        column_name: name of the column with the `NaN` values.
        group_by_column_name: name of the column to use as the group by column to calculate the max
            value of column_name.
        remainder_agg_func: fill any remaining `NaN` values with the global function.
        inplace: update the given dataframe or return a new dataframe.
        add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_groupby_func(dataframe=dataframe, column_name=column_name,
                                         group_by_column_name=group_by_column_name, agg_func='max',
                                         remainder_agg_func=remainder_agg_func,
                                         inplace=inplace, add_indicator=add_indicator)


def fill_nan_with_groupby_min(dataframe: _pd.DataFrame, column_name: str, group_by_column_name: str,
                                 remainder_agg_func='min', inplace=True, add_indicator=False) -> _pd.DataFrame | None:
    """
    Fill in missing values based on the min value which is calculated on all data having a similar value in
    another column. For example, fill in BMI values based on other BMI values that have the same age.

    If any NaN values are left over then these can be filled in by setting the `fill_remainder_with_global_func` the
    appropriate function.

    Alternatively you can use the soml.sklearn.SimpleGroupByImputer when you want to do this in a ColumnTransformer.

    Args:
        dataframe: the pd.DatFrame
        column_name: name of the column with the `NaN` values.
        group_by_column_name: name of the column to use as the group by column to calculate the min
            value of column_name.
        remainder_agg_func: fill any remaining `NaN` values with the global function.
        inplace: update the given dataframe or return a new dataframe.
        add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    return fill_nan_with_groupby_func(dataframe=dataframe, column_name=column_name,
                                         group_by_column_name=group_by_column_name, agg_func='min',
                                         remainder_agg_func=remainder_agg_func,
                                         inplace=inplace, add_indicator=add_indicator)


def fill_nan_with_groupby_func(dataframe: _pd.DataFrame, column_name: str, group_by_column_name: str, agg_func: str,
                                  remainder_agg_func=None, inplace=True, add_indicator=False) -> _pd.DataFrame | None:
    """
    Fill in missing values based on a function  which is calculated on all data having a similar value in
    another column. For example, fill in BMI values based on other BMI values that have the same age.

    If any NaN values are left over then these can be filled in by setting the `fill_remainder_with_global_func` the
    appropriate function.

    Alternatively you can use the soml.sklearn.SimpleGroupByImputer when you want to do this in a ColumnTransformer.

    Args:
        dataframe: the pd.DatFrame
        column_name: name of the column with the `NaN` values.
        group_by_column_name: name of the column to use as the group by column to calculate the function
            value of column_name.
        agg_func: type of function to use: 'mean', 'min', 'max', 'median', 'mode', 'skew' or 'kurt'
        remainder_agg_func: fill any remaining `NaN` values with the global function.
        inplace: update the given dataframe or return a new dataframe.
        add_indicator: add an indicator column indicating with a boolean value if the value was NaN
    """
    work_df = dataframe
    if not inplace:
        work_df = dataframe.copy(deep=True)

    assert column_name in dataframe, f"fill_nan_with_func_grouped_by: Column '{column_name}' does not exist in dataframe."
    assert group_by_column_name in dataframe, (f"fill_nan_with_func_grouped_by: Column '{group_by_column_name}' "
                                               f"does not exist in dataframe.")

    if add_indicator:
        work_df[column_name + '_nan'] = work_df[column_name].isna().astype(int)

    mean_by_group = work_df.groupby(group_by_column_name)[column_name].agg(agg_func)
    work_df[column_name].fillna(work_df[group_by_column_name].map(mean_by_group), inplace=True)

    # If any other values are still NaN then check if we need to fill these in a default manner.
    if remainder_agg_func:
        c_value = work_df[column_name].apply(remainder_agg_func)
        work_df[column_name].fillna(value=c_value, inplace=True)

    if inplace:
        return None

    return work_df


def generate_code_ordinal_encoder(dataframe: _pd.DataFrame, column_names: list[str]) -> None:
    print(f"\n###################################################\n"
          f"NOTE: The order still needs to be manualy adjusted.\n"
          f"###################################################\n")
    for c in column_names:
        if c in dataframe:
            try:
                values = dataframe[c].unique()
                categories = "',\n    '".join(values)
                print(f"{c}_encoder = sklearn.preprocessing.OrdinalEncoder(categories=[[\n    '{categories}'\n]])")
            except TypeError as e:
                print(f"Column {c} threw an exception {e}.")
        else:
            print(f"generate_code_ordinal_encoder: Column '{c}' does not exist in dataframe.")


def generate_one_hot_encoder(dataframe: _pd.DataFrame, column_names: list[str]) -> None:
    print(f"\n###################################################\n"
          f"NOTE: The order still needs to be manualy adjusted.\n"
          f"###################################################\n")
    for c in column_names:
        if c in dataframe:
            try:
                values = dataframe[c].unique()
                categories = "',\n    '".join(values)
                print(f"{c}_encoder = sklearn.preprocessing.OneHotEncoder(categories=[[\n    '{categories}'\n]])")
            except TypeError as e:
                print(f"Column {c} threw an exception {e}.")
        else:
            print(f"generate_one_hot_encoder: Column '{c}' does not exist in dataframe.")


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

        _mean = _std = _min = _q25 = _q50 = _q75 = _max = z_score_negative = z_score_positive = _irq = _irq_lower = _irq_upper = _np.NAN
        if _pd.api.types.is_numeric_dtype(v):
            _mean = v.mean()
            _std = v.std()
            _min = v.min()
            _q25 = v.quantile(.25)
            _q50 = v.quantile(.50)
            _q75 = v.quantile(.75)
            _max = v.max()

            # Calculate lower value when Z-Score = -3
            z_score_negative = -3 * _std + _mean

            # Calculate upper value when Z-Score = 3
            z_score_positive = 3 * _std + _mean

            # Calculate Interquartile Range (IRQ) and lower / upper bounds
            _irq = _q75 - _q25
            _irq_lower = _q25 - (1.5 * _irq)
            _irq_upper = _q75 + (1.5 * _irq)

        data.append([
            v.name, v.dtype, v.count(), v.isna().sum(), v.nunique(), _mean, _std, z_score_negative, z_score_positive,
            _min, _q25,
            _q50, _q75, _max, _irq_lower, _irq, _irq_upper
        ])

    data = _pd.DataFrame(
        columns=['Column', 'DType', 'NotNull', 'Null', 'Unique', 'Mean', 'Std', '-3σ', '3σ', 'Min', '25%', '50%',
                 '75%', 'Max', 'IRQ-L', 'IRQ', 'IRQ-U'], data=data)

    def highlight_columns(col):
        if col.name == '-3σ' or col.name == 'IRQ-L':
            css = []

            for idx, val in enumerate(col):

                if data['Unique'][idx] > 2 and val > data['Min'][idx]:
                    style = 'color: blue'
                elif _pd.isnull(val):
                    style = 'color: lightgrey'
                else:
                    style = ''

                css.append(style)

            return css;
        elif col.name == '3σ' or col.name == 'IRQ-U':
            css = []

            for idx, val in enumerate(col):
                if data['Unique'][idx] > 3 and val < data['Max'][idx]:
                    style = 'color: blue'
                elif _pd.isnull(val):
                    style = 'color: lightgrey'
                else:
                    style = ''

                css.append(style)

            return css;
        elif col.name == 'Null':
            return ['color: red' if val > 0 else '' for val in col]

        return ['color: lightgrey' if _pd.isnull(val) else '' for val in col]

    return data.style.apply(highlight_columns).format(precision=round)