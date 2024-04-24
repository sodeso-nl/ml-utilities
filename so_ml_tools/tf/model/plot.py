import matplotlib.pyplot as _plt
import tensorflow as _tf
import numpy as _np
import keras as _ks


_titles = {
    'mse': 'Mean Squared Error',
    'mae': 'Mean Absolute Error',
    'mape': 'Mean Absolute Percentage Error',
    'rmse': 'Root Mean Squared Error',
    'msle': 'Mean Squared Logarithmic Error',
    'loss': 'Loss',
    'lr': 'Learning Rate'}


def plot_history(history: _ks.callbacks.History, figsize=(30, 5)):
    """
    Plots the history of a single Keras History object.

    Args:
        history (_ks.callbacks.History): A Keras History object containing training metrics.
        figsize (tuple, optional): Size of the plot. Defaults to (30, 5).
    """
    plot_consecutive_histories(histories=[history], labels=[], figsize=figsize)


def plot_consecutive_histories(histories: list[_ks.callbacks.History], labels: list[str], reset_epochs=False,
                               figsize=(30, 5)):
    """
    Plots metrics from consecutive Keras History objects.

    Args:
        histories (list[_ks.callbacks.History]): List of Keras History objects.
        labels (list[str]): List of labels for the histories.
        reset_epochs (bool, optional): If True, resets the epoch count. Defaults to False.
        figsize (tuple, optional): Size of the plot. Defaults to (30, 5).
    """
    merged_histories, merged_epochs = _merge_histories(histories=histories)

    if reset_epochs:
        merged_epochs = _np.arange(1, len(merged_epochs), 1)

    keys_processed = []
    for key, value in merged_histories.items():
        if key not in keys_processed:

            # Find the accompanying val_ or non val_ key.
            val_key, val_value = None, None
            if key.startswith('val_'):
                val_key = key
                val_value = value
                key = key[4:]
                value = merged_histories[key]
            elif 'val_' + key in merged_histories:
                val_key = 'val_' + key
                val_value = merged_histories[val_key]

            _plt.figure(figsize=figsize, facecolor='#FFFFFF')
            _plt.plot(value, label=key, linestyle='solid', linewidth=1.5, color='#0000FF')

            if val_key:
                _plt.plot(val_value, label=val_key, linestyle='solid', linewidth=1.5, color='#00FF00')

            if key in _titles:
                title = _titles[key]
            else:
                title = key

            _plt.title(title, size=20)
            _plt.xticks(ticks=merged_epochs, labels=merged_epochs, rotation='vertical')
            _plt.xlabel('Epoch', size=14)
            _plt.margins(x=0)

            # Only plot history boundaries when we have more then one history object.
            if len(histories) > 1:
                for idx, history in enumerate(histories):
                    _plt.axvline(min(history.epoch), label=f'{labels[idx]}')

            # Mark the lowest point with a dot
            min_value = min(value)
            min_index = _tf.argmin(value)
            _plt.plot(min_index, min_value, marker='.', markersize=8, color='#FF0000', label=f'low {min_value}')

            if val_key:
                min_value = min(val_value)
                min_index = _tf.argmin(val_value)
                _plt.plot(min_index, min_value, marker='.', markersize=8, color='#FF0000', label=f'val_low {min_value}')

            _plt.legend()

            keys_processed += [key, val_key]


def _merge_histories(histories: list[_ks.callbacks.History]) -> (dict, list[int]):
    """
        Merges multiple Keras History objects into a single history.

        This method takes a list of Keras History objects and merges their metrics
        and epochs into a single history.

        Args:
            histories (list[_ks.callbacks.History]): A list of Keras History objects.

        Returns:
            tuple: A tuple containing the merged metrics (dict) and epochs (list[int]).

        Example:
            >>> import keras as ks
            >>> # Create a history object
            >>> history1 = ks.callbacks.History()
            >>> history1.history = {'loss': [0.1, 0.2], 'accuracy': [0.9, 0.85]}
            >>> history1.epoch = [0, 1]
            >>> # Create a second history object
            >>> history2 = ks.callbacks.History()
            >>> history2.history = {'loss': [0.15, 0.25], 'accuracy': [0.88, 0.82]}
            >>> history2.epoch = [2, 3]
            >>> # Merge the history objects
            >>> merged_metrics, merged_epochs = _merge_histories([history1, history2])
            >>> print(merged_metrics)
            {'loss': [0.1, 0.2, 0.15, 0.25], 'accuracy': [0.9, 0.85, 0.88, 0.82]}
            >>> print(merged_epochs)
            [0, 1, 2, 3]
        """
    merged_metrics = {}
    merged_epochs = []
    for history in histories:
        for key, value in history.history.items():
            if key in merged_metrics:
                merged_metrics[key] = [*merged_metrics[key], *value]
            else:
                merged_metrics[key] = value
        merged_epochs += history.epoch

    return merged_metrics, merged_epochs
