import tensorflow as _tf
import numpy as _np
import pandas as _pd
import so_ml_tools as _soml
import sklearn as _sk
import matplotlib.pyplot as _plt

from typing import Union as _Union


def permutation_feature_importance_plot(
        model: _Union[_tf.keras.Model, _tf.keras.Sequential],
        x: _Union[_np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor],
        y: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor],
        feature_names: list[str],
        n: int = 10,
        seed: int = 42,
        label_color: str = 'black',
        figsize: tuple = (10, 6)):
    permuted_scores = permutation_feature_importance(
        model=model,
        x=x,
        y=y,
        feature_names=feature_names,
        n=n,
        seed=seed
    )
    _plot(scores=permuted_scores, label_color=label_color, figsize=figsize)


def permutation_feature_importance(
        model: _Union[_tf.keras.Model, _tf.keras.Sequential],
        x: _Union[_np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor],
        y: _Union[list, _np.ndarray, _pd.Series, _pd.DataFrame, _tf.Tensor],
        feature_names: list[str],
        n: int = 10,
        seed: int = 42) -> _pd.DataFrame:
    """
    Calculates permutation feature importance for each feature in the dataset.

    Args:
        model (tf.keras.Sequential, tf.keras.Model): A trained Keras model.
        x (Union[np.ndarray, pd.Series, pd.DataFrame, tf.Tensor]): The input data.
        y (Union[list, np.ndarray, pd.Series, pd.DataFrame, tf.Tensor]): The target data.
        feature_names (list[str]): Names of the features.
        n (int): Number of times to calculate the permutation feature importance
        seed (int): Random seed initialization for reproducible results

    Returns:
        dict: A dictionary containing the permutation importance scores for each feature.

    Raises:
        TypeError: If the input `x` or `y` is not supported.

    Example:
        from so_ml_tools import soml


        # Assuming 'model' is a trained Keras sequential model
        # 'x' is the input data, 'y' is the target data, and 'feature_names' is a list of feature names
        importance_scores = soml.regression.feature_importance.permutation_feature_importance(model, x, y, feature_names)
    """
    if seed is not None:
        _np.random.seed(seed=seed)

    if not isinstance(x, _np.ndarray):
        x = _soml.util.types.to_numpy(value=x)

    if not isinstance(y, _np.ndarray):
        y = _soml.util.types.to_numpy(value=y)

    y_prob_base = model.predict(x, verbose=0)
    base_r2_score = _sk.metrics.r2_score(y, y_prob_base)

    permuted_scores = {}
    for i, feature_name in enumerate(feature_names):
        x_permuted = x.copy()

        permuted_r2_score = 0
        for p_i in range(n):
            x_permuted[:, i] = _np.random.permutation(x[::, i])
            y_prob_permuted = model.predict(x_permuted, verbose=0)
            permuted_r2_score += _sk.metrics.r2_score(y, y_prob_permuted)

        permuted_scores[feature_name] = (base_r2_score / n) - permuted_r2_score

    return _pd.DataFrame(list(permuted_scores.items()), columns=['Feature', 'Importance'])


def _plot(scores: _pd.DataFrame, label_color='black', figsize=(10, 6)):
    fig, ax = _plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)  # Transparant background

    y_pos = range(len(scores))
    ax_scores = ax.barh(y_pos, scores.values.ravel(), align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scores.index.values)

    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance Scores')
    ax.bar_label(ax_scores, fmt='%.2f')

    ax.xaxis.label.set_color(label_color)  # Set color of x-axis label
    ax.tick_params(axis='x', colors=label_color)  # Set color of x-axis ticks.

    ax.yaxis.label.set_color(label_color)  # Set color of y-axis label
    ax.tick_params(axis='y', colors=label_color)  # Set color of y-axis ticks.
    ax.title.set_color(label_color)  # Set color of title

    # Remove anoying white space at top / bottom.
    _plt.margins(y=0, x=.05)

    _plt.show()