import imblearn as _imblearn
import pandas as _pd


def resample(dataframe: _pd.DataFrame, label_column: str, sampling_strategy: str, k_neighbors = 5) -> _pd.DataFrame:
    """
    Splits the given DataFrame into a separate X / y DataFrame, then applies the SMOTE sampling
    strategy and finally merges the X / y DataFrame back to a new dataframe.

    'minority': resample only the minority class;
    'not minority': resample all classes but the minority class;
    'not majority': resample all classes but the majority class;
    'all': resample all classes;
    'auto': equivalent to 'not majority'.

    Args:
        dataframe: dataframe to use as input for oversampling.
        label_column: the column containing the label values.
        sampling_strategy: which sampling strategy should be applied
        k_neighbors: number of neighbors to use

    Returns:
        A new dataframe containing the original data and the oversampled data
    """
    y = dataframe[[label_column]]
    X = dataframe.drop(label_column, axis=1, inplace=False)
    x_s, y_s = resample_xy(X=X, y=y, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)
    return _pd.concat([x_s, y_s], axis=1)


def resample_xy(X: _pd.DataFrame, y: _pd.DataFrame, sampling_strategy: str, k_neighbors=5) -> (_pd.DataFrame, _pd.DataFrame):
    """
    Applies the SMOTE sampling strategy and finally merges the X / y DataFrame back to a new dataframe.

    'minority': resample only the minority class;
    'not minority': resample all classes but the minority class;
    'not majority': resample all classes but the majority class;
    'all': resample all classes;
    'auto': equivalent to 'not majority'.

    Args:
        X: DataFrame containing the features
        y: DataFrame containing the labels
        sampling_strategy: which sampling strategy should be applied
        k_neighbors: number of neighbors to use

    Returns:
        A new dataframe containing the original data and the oversampled data
    """
    smote = _imblearn.over_sampling.SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)
    return smote.fit_resample(X=X, y=y)