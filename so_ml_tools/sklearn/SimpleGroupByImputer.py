import sklearn as _sklearn


class SimpleGroupByImputer(_sklearn.base.BaseEstimator, _sklearn.base.TransformerMixin):

    def __init__(self, column_name, group_by_column_name, agg_func, remainder_agg_func):
        self.column_name = column_name
        self.group_by_column_name = group_by_column_name
        self.agg_func = agg_func
        self.remainder_agg_func = remainder_agg_func
        self.mean_by_group_ = None
        self.global_mean_ = None

    def fit(self, X, y=None):
        assert self.group_by_column_name in X, (f"fill_nan_with_func_grouped_by: Column '{self.group_by_column_name}' "
                                                f"does not exist in dataframe, did you add "
                                                f"'{self.group_by_column_name}' to the list of columns?")
        self.mean_by_group_ = X.groupby(self.group_by_column_name)[self.column_name].agg(self.agg_func)
        self.global_mean_ = X[self.column_name].apply(self.remainder_agg_func)
        return self

    def transform(self, X):
        # Fill in all NaN values where we have a calculated mean value from the group by.
        X[self.column_name].fillna(X[self.group_by_column_name].map(self.mean_by_group_), inplace=True)

        if self.remainder_agg_func:
            # Fill in all the NaN values with the global mean value.
            X[self.column_name].fillna(value=self.global_mean_, inplace=True)

        # Drop the group by column, otherwise it will be part of the feature set.
        X.drop(self.group_by_column_name, axis=1, inplace=True)

        return X

    def get_feature_names_out(self, input_features=None):
        return [self.column_name]