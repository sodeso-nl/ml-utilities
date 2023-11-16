import pandas as _pd


def convert_to_dataframe(series: _pd.Series):
    """
    Converts a Pandas Series object into a Pandas DataFrame.
    """
    return series.to_frame()
