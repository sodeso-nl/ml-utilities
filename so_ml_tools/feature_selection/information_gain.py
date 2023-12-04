import matplotlib.pyplot as _plt
import sklearn as _sk
import pandas as _pd


def calculate(X: _pd.DataFrame, y: _pd.DataFrame) -> _pd.DataFrame:
    mutual_info = _sk.feature_selection.mutual_info_classif(X=X, y=y.values.ravel())
    return _pd.DataFrame(mutual_info, index=X.columns, columns=['IG']).sort_values(by='IG', ascending=False)


def plot(X: _pd.DataFrame, y: _pd.DataFrame, label_color='black'):
    mutual_info_df = calculate(X=X, y=y)

    fig, ax = _plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0.0)  # Transparant background

    y_pos = range(len(mutual_info_df))
    scores = ax.barh(y_pos, mutual_info_df.values.ravel(), align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mutual_info_df.index.values)

    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance Scores (Information Gain)')
    ax.bar_label(scores, fmt='%.2f')

    ax.xaxis.label.set_color(label_color)  # Set color of x-axis label
    ax.tick_params(axis='x', colors=label_color)  # Set color of x-axis ticks.

    ax.yaxis.label.set_color(label_color)  # Set color of y-axis label
    ax.tick_params(axis='y', colors=label_color)  # Set color of y-axis ticks.
    ax.title.set_color(label_color)  # Set color of title

    # Remove anoying white space at top / bottom.
    _plt.margins(y=0, x=.05)

    _plt.show()
