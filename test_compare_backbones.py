import unittest

from compare_backbones import metrics
from patch_decisions import LABELS


class ComparisonTests(unittest.TestCase):
    def test_presence_collapses_full_and_partial_but_accuracy_does_not(self):
        a, b, c = LABELS
        result = metrics([a, b, c, c], [b, c, a, c])
        self.assertEqual(result["accuracy"], .25)
        self.assertEqual(result["presence_precision"], .5)
        self.assertEqual(result["presence_recall"], .5)
        self.assertEqual(result["confusion_matrix_truth_rows_predicted_columns"],
                         [[0, 1, 0], [0, 0, 1], [1, 0, 1]])

    def test_undefined_metrics_and_invalid_inputs(self):
        self.assertIsNone(metrics([LABELS[2]], [LABELS[2]])["presence_precision"])
        for actual, predicted in [([], []), ([LABELS[0]], []), (["unknown"], [LABELS[0]])]:
            with self.assertRaises(ValueError):
                metrics(actual, predicted)
