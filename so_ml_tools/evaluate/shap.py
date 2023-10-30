import shap as _sh
import numpy as _np
import tensorflow as _tf

# https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a


def create_kernel_explainer(model: [_tf.keras.Model | _tf.keras.Sequential], x, sample_size=None) -> _sh.Explainer:
    x_sample = x
    if sample_size is not None:
        x_sample = x[:sample_size, :]

    return _sh.KernelExplainer(model, x_sample)


def calculate_shap_values(explainer: _sh.Explainer, x, n_samples: [int | str] = "auto"):
    print(f"Calculating shap values for {len(x)} entries.")
    return explainer.shap_values(x, nsamples=n_samples)


def waterfall_plot(explainer, shap_values: list[_np.array], feature_names: list[str], max_features: [str | int] = 'auto',
                   display_entry: int = 0, display_class: int = 0):
    """
    Plot the breakdown of the

    Args:
      explainer: the explainer object
      shap_values: calculated values, use shap_values[x] for single class.
      feature_names: a `list` containing the feature names.
      max_features: default 'auto' to show the default amount, set to 'max' to show all
      display_entry: (optional) the entry to display when multiple have been calculated
      display_class: (optional) the class to plot

    Returns:
        None
    """
    shap_value_class = shap_values
    if len(shap_values) > 1:
        shap_value_class = shap_values[display_class]

    shap_example_within_class = shap_value_class
    if shap_value_class.ndim > 1:
        shap_example_within_class = shap_value_class[display_entry]

    explanation = _sh.Explanation(
        values=shap_example_within_class,  # Actual values of specific class
        base_values=explainer.expected_value[display_class],  # Mean value: E[f(x)] for specific class.
        feature_names=feature_names)

    if max_features == 'max':
        max_features = len(feature_names)

    _sh.waterfall_plot(shap_values=explanation, max_display=max_features)


def summary_plot(shap_values: list[_np.array], class_names: list[str], feature_names: list[str],
                 plot_type: str = 'dot', display_entry: int = None, display_class: int = None) -> None:
    """
    Summary plot for displaying feature for all entries.

    Args:
      shap_values: calculated values, use shap_values[x] for single class.
      class_names: a `list` containing the class names.
      feature_names: a `list` containing the feature names.
      plot_type: 'bar' for multiclass and single class, 'dot', 'violin' and  for single class.

    Returns:
        None
    """
    assert len(class_names) == len(
        shap_values), f"Number of class_names ({len(class_names)}) does not match the number of classes in shap_values ({shap_values[0].ndim})"
    assert display_class is None or (
                -1 < display_class < len(class_names)), f"Invalid display_class value {display_class}."

    shap_values = _np.array(shap_values)

    # In case the shap_values were calculated for a single set of features we need to add an additional dimension.
    if shap_values[0].ndim == 1:
        shap_values = _np.array(list(map(lambda x: _np.expand_dims(x, axis=0), shap_values)))

    # Filter on a specific entry when applicable.
    if display_entry is not None:
        shap_values = _np.expand_dims(shap_values[:, display_entry], axis=1)

    # Filter on a specific class when applicable.
    if display_class is not None:
        class_names = [class_names[display_class]]
        shap_values = _np.expand_dims(shap_values[display_class, :], axis=0)

    # Convert the numpy array back to a list with arrays as was the original data structure
    shap_values = list(map(lambda x: x, shap_values))

    # Old code that was used when dealing with single / multiple entries
    # if shap_values[0].ndim == 1:
    #     shap_values = list(map(lambda x: np.expand_dims(x, axis=0), shap_values))

    # Do we display a single class
    if len(shap_values) == 1:
        if plot_type is None:
            print("Using plot_type = 'dot', other options are 'violin' and 'bar'")
            plot_type = 'dot'

        shap_values = shap_values[0]
    elif plot_type != 'bar':
        print(f"{'Overriding' if plot_type != 'bar' else 'Using'} plot_type with 'bar' since we have multiple classes.")
        plot_type = 'bar'

    _sh.summary_plot(shap_values=shap_values,
                     class_names=class_names,
                     feature_names=feature_names,
                     plot_type=plot_type, show=True)


def force_plot(shap_values: list[_np.array], explainer: _sh.Explainer, feature_names: list[str],
               display_class: int = 0) -> None:
    """
    Force plot for displaying feature for a single class.

    Args:
      shap_values: calculated values
      explainer: the explainer object
      feature_names: a `list` containing the feature names.
      display_class: (optional) the class to plot

    Returns:
        None
    """
    if display_class is None:
        display_class = 0

    _sh.force_plot(base_value=explainer.expected_value[display_class], shap_values=shap_values[display_class],
                   features=feature_names)
