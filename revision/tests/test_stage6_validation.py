import unittest

from revision.scripts.run_stage6_validation import metric_row


class Stage6ValidationTests(unittest.TestCase):
    def test_common_metric_row_handles_disabled_task(self):
        config = {"experiment": "M2", "parameters": {"seed": 1}}
        metrics = {
            "classification": {"status": "task_disabled"},
            "regression": {
                "mae": 0.8,
                "rmse": 1.1,
                "pearson": 0.7,
                "spearman": 0.6,
            },
            "metrics_file": "/tmp/metrics.json",
        }
        row = metric_row("M2_seed1", config, metrics)
        self.assertEqual(row["accuracy"], "")
        self.assertEqual(row["mae"], 0.8)
        self.assertEqual(row["metrics_file"], "/tmp/metrics.json")


if __name__ == "__main__":
    unittest.main()
