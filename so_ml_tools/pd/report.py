import pandas as _pd
from ydata_profiling import ProfileReport as _pr


def profile_report(dataframe: _pd.DataFrame) -> _pr:
    """
    Args:
        dataframe:
    """
    return _pr(df=dataframe)
