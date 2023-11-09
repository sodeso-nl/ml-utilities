import sklearn as _sklearn


class BooleanEncoder(_sklearn.base.BaseEstimator, _sklearn.base.TransformerMixin):

    def __init__(self, value_zero: str, value_one: str):
        self.value_zero = value_zero
        self.value_one = value_one
        self.column_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.column_names_ = list(X.columns)

        source_values = [self.value_zero, self.value_one]
        target_values = [0, 1]

        for c in self.column_names_:
            unique_values = X[c].unique()
            if all(item in source_values for item in unique_values):
                X[c] = X[c].replace(source_values, target_values)
            else:
                raise ValueError(f"Column '{c}' contains other values then '{self.value_zero}' and '{self.value_one}'.")

        return X

    def get_feature_names_out(self, input_features=None):
        return self.column_names_
