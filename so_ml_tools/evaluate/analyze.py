import pandas as _pd
import tensorflow as _tf
import sklearn as _sklearn
import so_ml_tools as _soml


def determine_outliers_for_multiclass_classification(x, y_true, y_pred=None, y_prob=None, top=10) -> _pd.DataFrame:
    """
    Displays the top X (default 10) outliers between y_true and y_pred.

    Args:
        x: data to display
        y_true: actual labels [3, 3, 2, ..., 4, 4, 1]
        y_pred: predictions [3, 3, 2, ..., 4, 2, 1]
        y_prob: probabilities (multiclass) [[0.30974227, ...], [0.32494593, ...], ..] this field is optional, if not provided the y_prob will be determined by y_pred.
        top: how many outliers to return

    Returns:
        The outliers
    """
    # Check if we need to convert y_prob to numpy.
    _y_prob = y_prob
    if _tf.is_tensor(x=y_prob):
        _y_prob = y_prob.numpy()

    # If y_pred is None then determine the predictions, otherwise check if we need to convert to numpy.
    _y_pred = y_pred
    if y_pred is None:
        _y_pred = _soml.util.label.to_prediction(y_prob=_y_prob)
    elif _tf.is_tensor(x=y_pred):
        _y_pred = y_pred.numpy()

    # Create a matrix containing the y_true, y_pred and y_prob value in columns.
    data = None
    if _soml.util.label.is_multiclass_classification(_y_prob):
        data = [[y_true[i], x, _y_prob[i][x]] for i, x in enumerate(_y_pred)]
    elif _soml.util.label.is_binary_classification(_y_prob):
        data = [[y_true[i], x[0], _y_prob[i][0]] for i, x in enumerate(_y_pred)]

    if data is None:
        raise ValueError('y_prob is not a multiclass or binary classification.')

    # Create a dataframe from the matrix with the appropriate column names
    data_df = _pd.DataFrame(data, columns=['y_true', 'y_pred', 'y_prob'])

    # Add the columns of X to the dataframe.
    all_df = data_df
    if x is not None:
        all_df = _pd.concat([x, data_df], axis=1)

    # From all rows that have in-equal y_true vs y_pred, sort on y_prob in descending and select the :top rows.
    outliers = all_df[all_df['y_true'] != all_df['y_pred']].sort_values('y_prob', ascending=False)[:top]

    # Round the y_prob column to 2 decimals, for that we need to cast the y_prob column to float instead of float16
    outliers['y_prob'] = outliers['y_prob'].astype(float).round(decimals=2)

    return outliers


def determine_outliers_for_binary_classification(x, y_true, y_pred, target_column: str, top=10):
    """
    Displays the top X (default 10) outliers between y_true and y_pred.
    :param x: data to display
    :param y_true: labels
    :param y_pred: predictions
    :param target_column: column on which we can calculate the mean negative / positive.
    :param top: how many rows to display
    :return: dataframe containing the top X.
    """
    x_copy = x.copy()
    x_copy["y_true"] = y_true.round(0)
    x_copy["y_pred"] = y_pred.round(0)

    diff = y_true - y_pred
    x_copy["diff"] = diff.round(0)

    mean_positive = diff[diff[target_column] > 0].mean()[0]
    mean_negative = diff[diff[target_column] < 0].mean()[0]

    positive_outliers = x_copy[x_copy["diff"] > mean_positive]
    negative_outliers = x_copy[x_copy["diff"] < mean_negative]

    positive_outliers_sorted = positive_outliers.sort_values(['diff'], ascending=False, inplace=False)[:top]
    negative_outliers_sorted = negative_outliers.sort_values(['diff'], ascending=True, inplace=False)[:top]

    return _pd.concat([positive_outliers_sorted, negative_outliers_sorted])


def quality_metrics(y_true, y_pred):
    """
    calculates model accuracy, precision, recall and F1-Score
    :param y_true: the truth labels
    :param y_pred: the predictions
    :return: dictionary containing accuracy, precision, recall, f1 score and support
    """
    # Calculate model accuracy
    model_accuracy = _sklearn.metrics.accuracy_score(y_true, y_pred) * 100

    # Calculate precision, recall and F1-score using "weighted" average,
    # weighted will also take the amount of samples for each in mind.
    model_precission, model_recall, model_f1_score, support = \
        _sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = \
        {
            "accuracy": model_accuracy,
            "precision": model_precission,
            "recall": model_recall,
            "f1-score": model_f1_score,
            "support": support
        }
    return dict(sorted(model_results.items()))
