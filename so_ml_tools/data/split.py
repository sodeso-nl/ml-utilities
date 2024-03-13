from sklearn import model_selection as _model_selection


def split_train_test_data(*arrays, train_size=.8, test_size=.2, random_state=42, shuffle=True, stratify=None):
    """
    Usage:

    X_train, X_test, y_train, y_test =
        split_train_test_data(X, y)
    """
    assert test_size + train_size == 1., f"The sum of test_size: {test_size} and train_size: {train_size} does not " \
                                         f"add up to 1, not all data is used."

    return _model_selection.train_test_split(*arrays,
                                                    test_size=test_size,
                                                    train_size=train_size,
                                                    random_state=random_state,
                                                    shuffle=shuffle,
                                                    stratify=stratify)
