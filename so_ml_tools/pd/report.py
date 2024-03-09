import pandas as _pd
import sweetviz as _sv
#from ydata_profiling import ProfileReport as _pr


#def profile_report(dataframe: _pd.DataFrame) -> _pr:
    # """
    # Args:
    #     dataframe:
    # """
    # return _pr(df=dataframe)


def sweetvis_report(dataframe: _pd, target_feat: str = None):
    report = _sv.analyze(dataframe, target_feat=target_feat)
    report.show_html()
