from unittest import TestCase

import tensorflow as _tf
import numpy as _np
import so_ml_tools as _soml


class Test(TestCase):

    def test_determine_outliers_multiclass_no_outliers(self):
        import pandas as pd

        x = pd.DataFrame(data=[["A"], ["B"], ["C"], ["D"], ["E"]], columns=['X'])

        y_true = [1, 4, 2, 4, 0]

        y_prob = _tf.constant([
            [0.20, 0.21, 0.20, 0.19, 0.20], # 1
            [0.20, 0.19, 0.20, 0.20, 0.21], # 4
            [0.20, 0.20, 0.21, 0.19, 0.20], # 2
            [0.19, 0.20, 0.20, 0.20, 0.21], # 4
            [0.21, 0.20, 0.20, 0.20, 0.19]  # 0
        ])

        y_pred = _soml.util.prediction.multiclass_probability_to_prediction(y_probs=y_prob)

        pd = _soml.evaluate.analyze.determine_outliers_for_multiclass_classification(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            top=10
        )

        self.assertTrue(len(pd) == 0)

    def test_determine_outliers_multiclass_with_outliers(self):
        import pandas as pd

        x = pd.DataFrame(data=[
            ["A"],  # Predicted 4, should be 1
            ["B"],
            ["C"],
            ["D"],  # Predicted 1, should be 4
            ["E"]
        ], columns=['X'])

        y_true = [1, 4, 2, 4, 0]

        y_prob = _tf.constant([
            [0.20, 0.20, 0.12, 0.20, 0.28],
            [0.20, 0.19, 0.20, 0.20, 0.21],
            [0.20, 0.20, 0.21, 0.19, 0.20],
            [0.14, 0.26, 0.20, 0.20, 0.20],
            [0.21, 0.20, 0.20, 0.20, 0.19]
        ], dtype=_tf.float16)

        pd = _soml.evaluate.analyze.determine_outliers_for_multiclass_classification(
            x=x,
            y_true=y_true,
            y_prob=y_prob,
            top=10
        )

        result = pd.to_numpy().astype(str)
        compare_to = _np.array([
            ['A', 1, 4, 0.28],
            ['D', 4, 1, 0.26]
        ])
        self.assertTrue(_np.array_equal(result, compare_to))

    def test_determine_outliers_binary_with_outliers(self):
        import pandas as pd

        x = pd.DataFrame(data=[["A"], ["B"], ["C"], ["D"], ["E"]], columns=['X'])

        y_true = [1, 0, 0, 1, 1]

        y_prob = _tf.constant([
            [0.79],
            [0.23],
            [0.78],
            [0.89],
            [0.19]
        ])

        y_pred = _soml.util.prediction.binary_probability_to_prediction(y_probs=y_prob)

        pd = _soml.evaluate.analyze.determine_outliers_for_multiclass_classification(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            top=10
        )

        result = pd.to_numpy().astype(str)
        compare_to = _np.array([
            ['C', 0, 1, 0.78],
            ['E', 1, 0, 0.19]
        ])

        self.assertTrue(_np.array_equal(result, compare_to))
