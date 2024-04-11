import sklearn as _sklearn
import so_ml_tools as _soml
from typing import Union as _Union

import tensorflow as _tf
import numpy as _np
import pandas as _pd
from so_ml_tools.tf.loss.mase import mean_absolute_scaled_error as _mean_absolute_scaled_error


def evaluate_preds(y_true, y_pred, seasonality: int = None) -> dict:
    y_true = _soml.util.types.to_numpy(y_true)
    y_pred = _soml.util.types.to_numpy(y_pred)

    y_true = y_true.astype(dtype=_np.float32).ravel()
    y_pred = y_pred.astype(dtype=_np.float32).ravel()

    mae = _tf.keras.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred).numpy()
    mse = _tf.keras.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred).numpy()
    rmse = _np.sqrt(mse)
    mape = _tf.keras.metrics.mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred).numpy()
    mase = _mean_absolute_scaled_error(y_true=y_true, y_pred=y_pred, seasonality=seasonality)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'mase': mase
    }


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
        _y_pred = _soml.util.prediction.multiclass_probability_to_prediction(y=_y_prob)
    elif _tf.is_tensor(x=y_pred):
        _y_pred = y_pred.numpy()

    # Create a matrix containing the y_true, y_pred and y_prob value in columns.
    data = None
    if _soml.util.prediction.is_multiclass_classification(y=_y_prob):
        data = [[y_true[i], x, _y_prob[i][x]] for i, x in enumerate(_y_pred)]
    elif _soml.util.prediction.is_binary_classification(y=_y_prob):
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


def classification_report(y_true, y_pred=None, y_prob=None) -> None:
    y_true = _soml.util.prediction.probability_to_prediction(y=y_true)

    # If y_pred is not supplied but y_prob is then calculatwe y_pred
    if y_pred is None and y_prob is not None:
        y_pred = _soml.util.prediction.probability_to_prediction(y=y_prob)
    elif y_pred is None and y_prob is None:
        raise "Must specify 'y_pred' or 'y_prob'"

    print(_sklearn.metrics.classification_report(y_true=y_true, y_pred=y_pred, zero_division=0))

# Methods below are less intuitive and informative then using the classification_report.


def quality_metrics(y_true, y_pred=None, y_prob=None) -> _pd.DataFrame:
    """
    calculates model accuracy, precision, recall and F1-Score

    Args:
        y_true: the truth labels
        y_pred: (optional) the predictions (either y_pred or y_prob should be supplied)
        y_prob: (optional) the probabilities (either y_pred or y_prob should be supplied)

    Returns:
        A 'dict' containing accuracy, precision, recall, f1 score and support
    """
    y_true = _soml.util.prediction.probability_to_prediction(y=y_true)

    # If y_pred is not supplied but y_prob is then calculatwe y_pred
    if y_pred is None and y_prob is not None:
        y_pred = _soml.util.probability_to_prediction.probability_to_prediction(y=y_prob)
    elif y_pred is None and y_prob is None:
        raise "Must specify 'y_pred' or 'y_prob'"

    # Calculate model accuracy
    model_accuracy = _sklearn.metrics.accuracy_score(y_true, y_pred) * 100

    # Calculate precision, recall and F1-score using "weighted" average,
    # weighted will also take the amount of samples for each in mind.
    model_precission, model_recall, model_f1_score, _ = \
        _sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = \
        {
            "accuracy": model_accuracy,
            "precision": model_precission,
            "recall": model_recall,
            "f1-score": model_f1_score
        }
    return _pd.DataFrame(data=dict(sorted(model_results.items())), index=[0])


def quality_metrics_diff(metrics_1: _pd.DataFrame, metrics_2: _pd.DataFrame) -> _pd.DataFrame:
    """
    Returns the difference between 'metrics_1' and 'metrics_2'.

    Args:
        metrics_1: the first set of metrics
        metrics_2: the second set of metrics

    Returns
        A new 'pd.DataFrame' with the values of 'metrics_1' and 'metrics_2' and difference between these two.
    """
    c1 = {'metrics': 'metrics_1'}
    c1.update(metrics_1.to_dict('records')[0])
    m1 = _pd.DataFrame(data=c1, index=[0])

    c2 = {'metrics': 'metrics_2'}
    c2.update(metrics_2.to_dict('records')[0])
    m2 = _pd.DataFrame(data=c2, index=[0])

    diff = (m1.iloc[0][1:] - m2.iloc[0][1:]).to_frame().transpose()
    diff.insert(loc=0, column='metrics', value='diff')

    complete = _pd.concat([m1, m2, diff])
    complete.reset_index(drop=True, inplace=True)
    return complete


def quality_metrics_combine(metrics: dict, sort_by: list[str] = None, ascending: _Union[bool, list[bool], tuple[bool, ...]] = False) -> _pd.DataFrame:
    """
    Combines all the given metrics into a single overview, call with a dictionary:

        {
            "baseline": baseline_results,
            "model_1": model_1_results,
            "model_2": model_2_results,
            "model_3": model_3_results
        }

    Args:
        metrics: a dictionary of quality metrics pd.DataFrame objects where the key is the name.
        sort_by: column to sort by, if None specified then the default "accuracy" will be used.
        ascending: a 'list' or 'tuple' or boolean value containing the sort order.

    Returns:
        A 'pd.DataFrame' containing all quality metrics combined.
    """
    if sort_by is None:
        sort_by = ["accuracy"]

    all_metrics_results = _pd.concat(metrics)
    all_metrics_results.reset_index(inplace=True)
    _soml.pd.dataframe.drop_columns(dataframe=all_metrics_results, column_names=['level_1'])
    all_metrics_results.rename(columns={"level_0": "name"}, inplace=True)
    all_metrics_results.sort_values(by=sort_by, inplace=True, ascending=ascending)
    return all_metrics_results
