import ml_internal as mlint

from sklearn.metrics import accuracy_score


def calculate_accuract(y_true, y_pred):
    # If y_true or y_pred is not a Numpy array then try to convert it.
    y_true = mlint.convert_to_numpy_array_if_neccesary(y_true)
    y_pred = mlint.convert_to_numpy_array_if_neccesary(y_pred)

    # If the y_true labels are one-hot encoded then convert them to integer encoded labels.
    if mlint.is_multiclass_classification(y_true):
        y_true = mlint.to_ordinal(y_true)

    # Check if we need to convert multi-class classification one-hot encoding to index or
    # if we are dealing with binary classification, then we need to round the numer to either 0 or 1
    if mlint.is_multiclass_classification(y_pred):
        y_pred = mlint.to_ordinal(y_pred)
    elif mlint.is_binary_classification(y_pred):
        y_pred = mlint.to_binary(y_pred)

    return accuracy_score(y_true, y_pred)
