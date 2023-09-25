from tensorflow.python.platform import tf_logging as _logging
from keras.callbacks import Callback as _Callback
from keras.callbacks import TensorBoard as _TensorBoard
from keras.callbacks import LearningRateScheduler as _LearningRateScheduler
from keras.callbacks import ModelCheckpoint as _ModelCheckpoint
from keras.callbacks import EarlyStopping as _EarlyStopping
from keras.callbacks import ReduceLROnPlateau as _ReduceLROnPlateau

import datetime
import os


def reduce_lr_on_plateau_callback(monitor="val_loss",
                                  factor=0.2,
                                  patience=4,
                                  verbose=1,
                                  mode="auto",
                                  min_delta=1e-4,
                                  cooldown=0,
                                  min_lr=1e-7) -> _ReduceLROnPlateau:
    return _ReduceLROnPlateau(monitor=monitor,
                              factor=factor,
                              patience=patience,
                              verbose=verbose,
                              mode=mode,
                              min_delta=min_delta,
                              cooldown=cooldown,
                              min_lr=min_lr)


def early_stopping_callback(monitor='val_loss',
                            min_delta=0,
                            patience=6,
                            verbose=1,
                            mode='auto',
                            baseline=None,
                            restore_best_weights=True,
                            start_from_epoch=0) -> _EarlyStopping:
    return _EarlyStopping(monitor=monitor,
                          min_delta=min_delta,
                          patience=patience,
                          verbose=verbose,
                          mode=mode,
                          baseline=baseline,
                          restore_best_weights=restore_best_weights,
                          start_from_epoch=start_from_epoch)


def model_checkpoint_callback(dir_name='./checkpoints',
                              experiment_name='/experiment',
                              file_name='/epoch-{{epoch:02d}}-{metric}-{{{metric}:.2f}}.hdf5',
                              metric='val_loss',
                              save_weights_only=True,
                              save_best_only=True,
                              save_freq='epoch',
                              verbose=0) -> _ModelCheckpoint:
    log_dir = os.path.join(dir_name, experiment_name, file_name)
    return _ModelCheckpoint(filepath=log_dir,
                            monitor=metric,
                            save_weights_only=save_weights_only,
                            save_best_only=save_best_only,
                            save_freq=save_freq,
                            verbose=verbose)


def learning_rate_scheduler_callback(learning_rate_start=0.001, epochs=50) -> _LearningRateScheduler:
    """
    Creates a LearningRateScheduler which will be pre-configured with a division. The division
    is calculated using find_learning_rate_division.

    :param learning_rate_start: initial starting learning rate
    :param epochs: number of epochs the model will train for
    :return: the pre-configured LearningRateScheduler
    """
    min, max, division = _find_learning_rate_division(learning_rate=learning_rate_start, epochs=epochs)
    print(f"Min learning rate: {min}\nMax learning rate: {max}\nDivision: {division}")
    return _LearningRateScheduler(lambda epoch: learning_rate_start * 10 ** (epoch / division))


def tensorboard_callback(experiment_name: str, dir_name='./logs'):
    log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    print(f"Saving TensorBoard log files to: {log_dir}")
    return _TensorBoard(log_dir=log_dir)


def create_stop_training_on_target_callback(metric="val_loss",
                                            target=0.95,
                                            patience=0,
                                            verbose=0,
                                            start_from_epoch=0,
                                            restore_best_weights=False):
    return StopTrainingOnTarget(metric=metric,
                                target=target,
                                patience=patience,
                                verbose=verbose,
                                start_from_epoch=start_from_epoch,
                                restore_best_weights=restore_best_weights)


def _find_learning_rate_division(learning_rate: float, epochs: int):
    """
    Finds the optimal division which can be used in a learning rate scheduler.

    epochs = 50
    initial_lr = 0.001

    division = find_learning_rate_division(initial_lr, epochs)
    lr_scheduler = LearningRateScheduler(lambda epoch: initial_lr * 10 ** (epoch / division))

    :param learning_rate: initial learning rate
    :param epochs: number of epochs that the model will train for
    :return: the minimum learning rate, maximum learning rate and the division which can be used in the scheduler.
    """
    min_lr = 0.
    max_lr = 0.
    division = 1000
    while max_lr < 0.1:
        min_lr = learning_rate * 10 ** (1 / division)
        max_lr = learning_rate * 10 ** (epochs / division)
        division -= 1
    return min_lr, max_lr, division


class StopTrainingOnTarget(_Callback):

    def __init__(self,
                 metric="val_loss",
                 target=0.95,
                 patience=0,
                 verbose=0,
                 start_from_epoch=0,
                 restore_best_weights=False
                 ):
        super().__init__()
        self.metric = metric
        self.target = target
        self.patience = patience
        self.verbose = verbose
        self.start_from_epoch = start_from_epoch
        self.restore_best_weights = restore_best_weights
        self.stopped_epoch = 0
        self.wait = 0
        self.best_weights = None
        self.best_epoch = 0
        self.best_value = 0

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.stopped_epoch = 0
        self.wait = 0
        self.best_weights = None
        self.best_epoch = 0
        self.best_value = 0

    def on_epoch_end(self, epoch, logs=None):
        current_metric_value = self._get_metric_value(logs)
        if current_metric_value is None or epoch < self.start_from_epoch:
            # If the metric is not part of the logs or still in initial warm-up stage.
            return

        self.wait += 1
        if self._is_metric_reached_or_better(current_metric_value):
            self.best_value = current_metric_value
            self.best_epoch = epoch
            self.wait = 0

        # If our patience has run out, and we have a best performing epoch
        if self.wait >= self.patience and self.best_epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    print(f'Restoring model weights from the end of the best epoch: {self.best_epoch +1})')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Epoch {self.stopped_epoch + 1}: training stopped "
                f"on metric a`{self.metric}` with a target value of {self.target}), "
                f"epoch {self.best_epoch} achieved {round(self.best_value * 100, 2)}% accuracy")

    def _get_metric_value(self, logs):
        logs = logs or {}
        value = logs.get(self.metric)
        if value is None:
            _logging.warning(
                f"Stop training goal condition on metric `{self.target}` "
                f"which is not available. Available metrics are: {','.join(list(logs.keys()))}"
            )
        return value

    def _is_metric_reached_or_better(self, value):
        # If the value is larger than the target and larger than the current known best value
        return self.target < value > self.best_value
