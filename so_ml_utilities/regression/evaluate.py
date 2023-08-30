import pandas as pd
import so_ml_utilities as somlu


def determine_outliers(x, y_true, y_pred, target_column: str, top=10):
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

    return pd.concat([positive_outliers_sorted, negative_outliers_sorted])


def quality_metrics(y_true, y_pred):
    """
    calculates model accuracy, precision, recall and F1-Score
    :param y_true: the truth labels
    :param y_pred: the predictions
    :return: dictionary containing accuracy, precision, recall and f1 score.
    """
    return ml_utilities.multiclass.score.quality_metrics(y_true, y_pred)