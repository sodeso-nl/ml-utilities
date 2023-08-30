import pandas as pd
import sklearn as sklearn
import tensorflow as tf
import so_ml_utilities as somlu


def determine_outliers(x, y_true, y_pred = None, y_prob = None, top=10) -> pd.DataFrame:
    """
    Displays the top X (default 10) outliers between y_true and y_pred.

    Args:
        x: data to display
        y_true: actual labels [3, 3, 2, ..., 4, 4, 1]
        y_pred: predictions [3, 3, 2, ..., 4, 2, 1]
        y_prob: probabilities (multiclass) [[0.30974227, ...], [0.32494593, ...], ..]
        top: how many outliers to return

    Returns:
        The outliers
    """

    # Check if we need to convert y_prob to numpy.
    _y_prob = y_prob
    if tf.is_tensor(x=y_prob):
        _y_prob = y_prob.numpy()

    # If y_pred is None then determine the predictions, otherwise check if we need to convert to numpy.
    _y_pred = y_pred
    if y_pred is None:
        _y_pred = somlu.util.label.to_prediction(y_prob=_y_prob)
    elif tf.is_tensor(x=y_pred):
        _y_pred = y_pred.numpy()

    # Create a matrix containing the y_true, y_pred and y_prob value in columns.
    data = None
    if somlu.util.label.is_multiclass_classification(_y_prob):
        data = [[y_true[i], x, _y_prob[i][x]] for i, x in enumerate(_y_pred)]
    elif somlu.util.label.is_binary_classification(_y_prob):
        data = [[y_true[i], x[0], _y_prob[i][0]] for i, x in enumerate(_y_pred)]

    if data is None:
        raise ValueError('y_prob is not a multiclass or binary classification.')

    # Create a dataframe from the matrix with the appropriate column names
    data_df = pd.DataFrame(data, columns=['y_true', 'y_pred', 'y_prob'])

    # Add the columns of X to the dataframe.
    all_df = data_df
    if x is not None:
        all_df = pd.concat([x, data_df], axis=1)

    # From all rows that have in-equal y_true vs y_pred, sort on y_prob in descending and select the :top rows.
    outliers = all_df[all_df['y_true'] != all_df['y_pred']].sort_values('y_prob', ascending=False)[:top]

    # Round the y_prob column to 2 decimals, for that we need to cast the y_prob column to float instead of float16
    outliers['y_prob'] = outliers['y_prob'].astype(float).round(decimals=2)

    return outliers


def quality_metrics(y_true, y_pred):
    """
    calculates model accuracy, precision, recall and F1-Score
    :param y_true: the truth labels
    :param y_pred: the predictions
    :return: dictionary containing accuracy, precision, recall, f1 score and support
    """
    # Calculate model accuracy
    model_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred) * 100

    # Calculate precision, recall and F1-score using "weighted" average,
    # weighted will also take the amount of samples for each in mind.
    model_precission, model_recall, model_f1_score, support = \
        sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = \
        {
            "accuracy" : model_accuracy,
            "precision": model_precission,
            "recall": model_recall,
            "f1-score": model_f1_score,
            "support": support
        }
    return dict(sorted(model_results.items()))