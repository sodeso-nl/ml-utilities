import pandas as _pd
import numpy as _np
import matplotlib.pyplot as _plt

from sklearn.feature_selection import SelectKBest as _SelectKBest
from sklearn.feature_selection import mutual_info_classif as _mutual_info_classif
from sklearn.feature_selection import f_classif as _f_classif

from sklearn.feature_selection import mutual_info_regression as _mutual_info_regression


def mutual_information_regression(X: _pd.DataFrame, k='all', y=_pd.DataFrame, seed=42) -> _pd.DataFrame:
    _np.random.seed(seed)
    selector = _SelectKBest(score_func=_mutual_info_regression, k=k)
    selector.fit(X, y.values.ravel())
    return _pd.DataFrame(selector.scores_, index=X.columns, columns=['Score']).sort_values(by='Score', ascending=False)


def mutual_information_regression_plot(X: _pd.DataFrame, y=_pd.DataFrame, seed=42, label_color='black') -> None:
    scores = mutual_information_regression(X=X, y=y, seed=seed)
    _plot(scores=scores, label_color=label_color)


def mutual_information_classification(X: _pd.DataFrame, k='all', y=_pd.DataFrame, seed=42) -> _pd.DataFrame:
    _np.random.seed(seed)
    selector = _SelectKBest(score_func=_mutual_info_classif, k=k)
    selector.fit(X, y.values.ravel())
    return _pd.DataFrame(selector.scores_, index=X.columns, columns=['Score']).sort_values(by='Score', ascending=False)


def mutual_information_classification_plot(X: _pd.DataFrame, y=_pd.DataFrame, seed=42, label_color='black') -> None:
    scores = mutual_information_classification(X=X, y=y, seed=seed)
    _plot(scores=scores, label_color=label_color)


def anova_f_classification(X: _pd.DataFrame, k='all', y=_pd.DataFrame, seed=42) -> _pd.DataFrame:
    _np.random.seed(seed)
    selector = _SelectKBest(score_func=_f_classif, k=k)
    selector.fit(X, y.values.ravel())
    return _pd.DataFrame(selector.scores_, index=X.columns, columns=['Score']).sort_values(by='Score', ascending=False)


def anova_f_classification_plot(X: _pd.DataFrame, y=_pd.DataFrame, seed=42, label_color='black') -> None:
    scores = anova_f_classification(X=X, y=y, seed=seed)
    _plot(scores=scores, label_color=label_color)


def _plot(scores: _pd.DataFrame, label_color):
    fig, ax = _plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0.0)  # Transparant background

    y_pos = range(len(scores))
    scores = ax.barh(y_pos, scores.values.ravel(), align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scores.index.values)

    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance Scores')
    ax.bar_label(scores, fmt='%.2f')

    ax.xaxis.label.set_color(label_color)  # Set color of x-axis label
    ax.tick_params(axis='x', colors=label_color)  # Set color of x-axis ticks.

    ax.yaxis.label.set_color(label_color)  # Set color of y-axis label
    ax.tick_params(axis='y', colors=label_color)  # Set color of y-axis ticks.
    ax.title.set_color(label_color)  # Set color of title

    # Remove anoying white space at top / bottom.
    _plt.margins(y=0, x=.05)

    _plt.show()