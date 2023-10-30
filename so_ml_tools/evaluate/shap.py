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


def waterfall(explainer: _sh.Explainer, shap_values, plot_class: int, feature_names: list[str], max_features: [str | int] = 'auto'):
    """
    Plot the breakdown of the 

    Args:
      shap_values: calculated values, use shap_values[x] for single class.
      class_names: a `list` containing the class names.
      feature_names: a `list` containing the feature names.
      plot_type: 'bar' for multiclass and single class, 'dot', 'violin' and  for single class.

    Returns:
        None
    """
    explanation = _sh.Explanation(
        values=shap_values[plot_class],  # Actual values of specific class
        base_values=explainer.expected_value[plot_class],  # Mean value: E[f(x)] for specific class.
        feature_names=feature_names)

    if max_features == 'auto':
        max_features = len(feature_names)

    _sh.waterfall_plot(shap_values=explanation, max_display=max_features)


def summary_plot(shap_values: _np.array, class_names: list[str], feature_names: list[str], plot_type: str = 'auto') -> None:
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
    assert len(shap_values) > 1 and len(shap_values) == len(class_names), (f"Number of class_names {len(class_names)} does not match "
                                                  f"the number of classes present in shape_values {len(shap_values)}.")

    _sh.summary_plot(shap_values=shap_values,
                     class_names=class_names,
                     feature_names=feature_names,
                     plot_type=plot_type)
