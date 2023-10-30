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


def waterfall_plot(explainer: _sh.Explainer, shap_values, feature_names: list[str], max_features: [str | int] = 'auto',
              plot_example: int = None, plot_class: int = None):
    """
    Plot the breakdown of the 

    Args:
      explainer: the explainer object
      shap_values: calculated values, use shap_values[x] for single class.
      feature_names: a `list` containing the feature names.
      max_features: maximum number of features to display ordered by most imported to least.
      plot_class: (optional) the class to plot

    Returns:
        None
    """
    assert len(shap_values) > 1 and plot_class is not None, (f"shap_values contains multiple classes ({len(shap_values)}), "
                                                         f"the force_plot can only plot a single class at a time, "
                                                         f"please specify the class to plot.")


    assert shap_values[0].ndim > 1 and plot_example is not None, (f"shap_values contains multiple examples ({len(shap_values[0])}), "
                                                         f"the waterfall_plot can only plot a single example at a time, "
                                                         f"please specify the plot_example.")

    if plot_class is None:
        plot_class = 0

    if plot_example is None:
        plot_example = 1

    shap_example = None
    if shap_values[0].ndim > 1:
        shap_example = shap_values[plot_example]

    explanation = _sh.Explanation(
        values=shap_example[plot_class],  # Actual values of specific class
        base_values=explainer.expected_value[plot_class],  # Mean value: E[f(x)] for specific class.
        feature_names=feature_names)

    if max_features == 'auto':
        max_features = len(feature_names)

    _sh.waterfall_plot(shap_values=explanation, max_display=max_features)


def summary_plot(shap_values: list[_np.array], class_names: list[str], feature_names: list[str],
                 plot_type: str = 'auto') -> None:
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
    assert len(shap_values) > 1 and len(shap_values) == len(class_names), (f"Number of class_names {len(class_names)} "
                                                                           f"does not match the number of classes "
                                                                           f"present in shape_values {len(shap_values)}"
                                                                           f", possible options:\n\nCheck if class_names"
                                                                           f" is a list within list, if so use "
                                                                           f"class_names[0].")

    # In case the shap_values were calculated for a single set of features we need to add an additional dimension.
    if shap_values[0].ndim == 1:
        shap_values[0] = _np.expand_dims(shap_values[0], axis=0)
        shap_values[1] = _np.expand_dims(shap_values[1], axis=0)
        shap_values[2] = _np.expand_dims(shap_values[2], axis=0)

    if len(shap_values) > 1:
        plot_type = 'bar'

    _sh.summary_plot(shap_values=shap_values,
                     class_names=class_names,
                     feature_names=feature_names,
                     plot_type=plot_type)


def force_plot(shap_values: list[_np.array], explainer: _sh.Explainer, feature_names: list[str],
               plot_class: int = None):
    """
    Force plot for displaying feature for a single class.

    Args:
      shap_values: calculated values
      explainer: the explainer object
      feature_names: a `list` containing the feature names.
      plot_class: (optional) the class to plot

    Returns:
        None
    """
    assert len(shap_values) > 1 and plot_class is not None, (f"shap_values contains multiple classes ({len(shap_values)}), "
                                                         f"the force_plot can only plot a single class at a time, "
                                                         f"please specify the class to plot.")

    if plot_class is None:
        plot_class = 0

    _sh.force_plot(base_value=explainer.expected_value[plot_class], shap_values=shap_values[plot_class],
                   features=feature_names)
