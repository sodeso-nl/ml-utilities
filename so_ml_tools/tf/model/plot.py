import matplotlib.pyplot as _plt
import tensorflow as _tf


def plot_consecutive_histories(histories: list[_tf.keras.callbacks.History], labels: list[str], figsize=(10, 6)):
    """
    Plots (when available), the validation loss and accuracy, training loss and accuracy and learning rate.

    :param histories: the history objects returned from fitting models.
    :param labels: the labels for each history object for seperating the epochs
    :param figsize: figure size (default: (10, 6))
    :return: two graphs, one for loss and one for accuracy
    """
    all_loss_history = []
    all_val_loss_history = []
    all_accuracy_history = []
    all_val_accuracy_history = []
    all_mae_history = []
    all_val_mae_history = []
    all_lr_history = []

    first_epoch = 10000
    last_epoch = -1
    for history in histories:
        first_epoch = min(first_epoch, min(history.epoch))
        last_epoch = max(last_epoch, max(history.epoch))

        all_loss_history = [*all_loss_history, *history.history['loss']]
        all_val_loss_history = [*all_val_loss_history, *history.history['val_loss']]

        if 'accuracy' in history.history:
            all_accuracy_history = [*all_accuracy_history, *history.history['accuracy']]
            all_val_accuracy_history = [*all_val_accuracy_history, *history.history['val_accuracy']]

        if 'mae' in history.history:
            all_mae_history = [*all_mae_history, *history.history['mae']]
            all_val_mae_history = [*all_val_mae_history, *history.history['val_mae']]

        if 'lr' in history.history:
            all_lr_history = [*all_lr_history, *history.history['lr']]

    epoch_labels = range(first_epoch + 1, last_epoch + 2)
    ticks = range(len(epoch_labels))
    _plt.figure(figsize=figsize, facecolor='#FFFFFF')
    _plot_history_graph_line(all_loss_history, label='Training loss', color='#0000FF')
    _plot_history_graph_line(all_val_loss_history, label='Validation loss', color='#00FF00')
    _plot_history_graph_line(all_lr_history, label='Learning rate', color='#000000', linestyle='dashed')
    _plot_history_ends(histories, labels)
    _plt.title('Loss', size=20)
    _plt.xticks(ticks, epoch_labels)
    _plt.xlabel('Epoch', size=14)
    _plt.legend()

    if all_accuracy_history:
        # Start a new figure
        _plt.figure(figsize=figsize, facecolor='#FFFFFF')
        _plot_history_graph_line(all_accuracy_history, label='Training accuracy', color='#0000FF')
        _plot_history_graph_line(all_val_accuracy_history, label='Validation accuracy', color='#00FF00')
        _plot_history_graph_line(all_lr_history, label='Learning rate', color='#000000', linestyle='dashed')
        _plot_history_ends(histories, labels)
        _plt.title('Accuracy', size=20)
        _plt.xticks(ticks, epoch_labels)
        _plt.xlabel('Epoch', size=14)
        _plt.legend()

    if all_mae_history:
        # Start a new figure
        _plt.figure(figsize=figsize, facecolor='#FFFFFF')
        _plot_history_graph_line(all_mae_history, label='Training mae', color='#0000FF')
        _plot_history_graph_line(all_val_mae_history, label='Validation mae', color='#00FF00')
        _plot_history_graph_line(all_lr_history, label='Learning rate', color='#000000', linestyle='dashed')
        _plot_history_ends(histories, labels)
        _plt.title('Mean Absolute Accuracy', size=20)
        _plt.xticks(ticks, epoch_labels)
        _plt.xlabel('Epoch', size=14)
        _plt.legend()


def plot_history(history: _tf.keras.callbacks.History, figsize=(10, 6)):
    plot_consecutive_histories([history], ["Start history"], figsize=figsize)


def _plot_history_ends(histories: list[_tf.keras.callbacks.History], labels: list[str]) -> None:
    """
    Internal method which will plot a vertical line showing where a histories last epoch is visible.

    :param histories: the history objects returned from fitting models.
    :param labels: the labels for each history object for seperating the epochs
    """
    for idx, history in enumerate(histories):
        _plt.plot([min(history.epoch), min(history.epoch)], _plt.ylim(), label=f'{labels[idx]}')


def _plot_history_graph_line(data, label, color, linestyle='solid') -> None:
    """
    Internal method which will plot the information from the fit histroy.

    :param data: the data to plot
    :param label: the label associated with the data
    :param color: color of the line
    :param linestyle: line-style of the line (default: solid)
    """
    if data:
        _plt.plot(data, label=label, color=color, linestyle=linestyle, linewidth=1.5)