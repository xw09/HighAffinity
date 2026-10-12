import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from revision.scripts.analyze_predictions import (
    all_metrics,
    bootstrap_metrics,
    paired_bootstrap,
)


def prediction_frame(probabilities, affinity_predictions):
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5", "6"],
            "receptor_group": ["a", "a", "b", "b", "c", "c"],
            "negative_type": ["experimental", None, "synthetic", None, None, None],
            "class_true": [0, 1, 0, 1, 0, 1],
            "class_probability": probabilities,
            "regression_eligible": [True] * 6,
            "affinity_true": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "affinity_pred": affinity_predictions,
        }
    )


class RevisionStatisticsTests(unittest.TestCase):
    def test_metrics_are_computed_from_continuous_scores(self):
        frame = prediction_frame(
            [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        metrics = all_metrics(frame, 0.5, True, True)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["tn"], 3)

    def test_cluster_bootstrap_is_reproducible(self):
        frame = prediction_frame(
            [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            [0.1, 0.9, 2.2, 2.8, 4.3, 4.9],
        )
        left = bootstrap_metrics(frame, 0.5, True, True, 100, 0.95, 17)
        right = bootstrap_metrics(frame, 0.5, True, True, 100, 0.95, 17)
        self.assertEqual(left, right)
        self.assertTrue(all(row["n_receptor_groups"] == 3 for row in left))

    def test_paired_bootstrap_preserves_pairing(self):
        frame_a = prediction_frame(
            [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        frame_b = prediction_frame(
            [0.4, 0.6, 0.6, 0.4, 0.7, 0.3],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        rows = paired_bootstrap(
            "toy_a", frame_a, "toy_b", frame_b, 0.5, 100, 0.95, 23
        )
        mae = next(row for row in rows if row["metric"] == "mae")
        accuracy = next(row for row in rows if row["metric"] == "accuracy")
        self.assertLess(mae["delta_a_minus_b"], 0)
        self.assertGreater(mae["advantage_a"], 0)
        self.assertGreater(accuracy["delta_a_minus_b"], 0)


if __name__ == "__main__":
    unittest.main()
