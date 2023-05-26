import ml_internal as mlint

from sklearn.metrics import accuracy_score


def calculate_accuracy(y_true, y_pred):
    y_true, y_pred = mlint.convert_to_sparse_or_binary(y_true=y_true, y_pred=y_pred)
    return accuracy_score(y_true, y_pred)
