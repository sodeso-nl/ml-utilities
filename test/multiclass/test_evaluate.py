from unittest import TestCase

import tensorflow as tf
import numpy as np
import ml_utilities as ml


class Test(TestCase):
    def test_determine_outliers_multiclass_no_outliers(self):
        import pandas as pd

        x = pd.DataFrame(data=[["A"], ["B"], ["C"], ["D"], ["E"]], columns=['X'])

        y_true = [1, 3, 2, 4, 0]

        y_prob = tf.constant([
            [0.20, 0.21, 0.20, 0.19, 0.20],
            [0.20, 0.19, 0.20, 0.21, 0.21],
            [0.20, 0.20, 0.21, 0.19, 0.20],
            [0.19, 0.20, 0.20, 0.20, 0.21],
            [0.21, 0.20, 0.20, 0.20, 0.19]
        ])

        y_pred = ml.util.label.to_prediction(y_prob=y_prob)

        pd = ml.multiclass.evaluate.determine_outliers(
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

        y_true = [1, 3, 2, 4, 0]

        y_prob = tf.constant([
            [0.20, 0.20, 0.12, 0.20, 0.28],
            [0.20, 0.19, 0.20, 0.21, 0.21],
            [0.20, 0.20, 0.21, 0.19, 0.20],
            [0.14, 0.26, 0.20, 0.20, 0.20],
            [0.21, 0.20, 0.20, 0.20, 0.19]
        ], dtype=tf.float16)

        pd = ml.multiclass.evaluate.determine_outliers(
            x=x,
            y_true=y_true,
            y_prob=y_prob,
            top=10
        )

        result = pd.to_numpy().astype(str)
        compare_to = np.array([
            ['A', 1, 4, 0.28],
            ['D', 4, 1, 0.26]
        ])
        self.assertTrue(np.array_equal(result, compare_to))

    def test_determine_outliers_binary_with_outliers(self):
        import pandas as pd

        x = pd.DataFrame(data=[["A"], ["B"], ["C"], ["D"], ["E"]], columns=['X'])

        y_true = [1, 0, 0, 1, 1]

        y_prob = tf.constant([
            [0.79],
            [0.23],
            [0.78],
            [0.89],
            [0.19]
        ])

        y_pred = ml.util.label.to_prediction(y_prob=y_prob)

        pd = ml.multiclass.evaluate.determine_outliers(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            top=10
        )

        result = pd.to_numpy().astype(str)
        compare_to = np.array([
            ['C', 0, 1, 0.78],
            ['E', 1, 0, 0.19]
        ])

        self.assertTrue(np.array_equal(result, compare_to))
