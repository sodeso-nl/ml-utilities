import sklearn as _sklearn
import so_ml_tools as _soml


class FillForwardImputer(_sklearn.base.BaseEstimator, _sklearn.base.TransformerMixin):

    def __init__(self, add_indicator=False):
        self.add_indicator = add_indicator
        self.column_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.column_names_ = list(X.columns)
        _soml.pd.dataframe.fill_nan_with_previous_value(dataframe=X, column_names=self.column_names_, inplace=True,
                                                        add_indicator=self.add_indicator)
        return X

    def get_feature_names_out(self, input_features=None):
        if self.add_indicator:
            feature_names = []
            for item in self.column_names_:
                feature_names.append(item)
                feature_names.append(str(item) + '_nan')
        else:
            feature_names = self.column_names_

        return feature_names
