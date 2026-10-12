import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from revision.scripts.run_affinity_evaluation import summarize_predictions
from revision.scripts.run_stage5_pipeline import (
    SCREENING_RUNS,
    SUBMISSION_RUNS,
    checkpoint_score,
    select_best_checkpoint,
)


class Stage5PipelineTests(unittest.TestCase):
    def test_experiment_suite_sizes(self):
        self.assertEqual(len(SCREENING_RUNS), 7)
        self.assertEqual(len(SUBMISSION_RUNS), 11)
        self.assertEqual(len(set(SUBMISSION_RUNS)), 11)
        self.assertNotIn("M7_seed1", SCREENING_RUNS)
        self.assertNotIn("M7_seed1", SUBMISSION_RUNS)

    def test_best_checkpoint_is_selected_by_validation_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoints"
            checkpoint_dir.mkdir()
            first = checkpoint_dir / "epoch2-valloss0.3100.ckpt"
            best = checkpoint_dir / "epoch3-valloss0.2400.ckpt"
            last = checkpoint_dir / "last.ckpt"
            for path in (first, best, last):
                path.touch()
            self.assertEqual(checkpoint_score(best), 0.24)
            self.assertEqual(select_best_checkpoint(Path(temporary)), best)

    def test_clean_test_metrics_are_generated_from_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_dir = root / "records"
            records_dir.mkdir()
            truth_rows = [
                ("a", 0, 0.0, "experimental_negative", "g1"),
                ("b", 0, 1.0, "synthetic_decoy", "g2"),
                ("c", 1, 2.0, "", "g3"),
                ("d", 1, 3.0, "", "g4"),
            ]
            probabilities = [0.1, 0.2, 0.8, 0.9]
            predictions = [0.2, 1.2, 1.8, 3.2]
            for (record_id, *_), probability, prediction in zip(
                truth_rows, probabilities, predictions, strict=True
            ):
                directory = records_dir / record_id
                directory.mkdir()
                (directory / f"affinity_{record_id}.json").write_text(
                    json.dumps(
                        {
                            "affinity_pred_value": prediction,
                            "affinity_probability_binary": probability,
                        }
                    )
                )

            manifest = root / "split_manifest.csv"
            fields = [
                "id",
                "receptor_group",
                "negative_type_clean",
                "class_clean",
                "regression_eligible",
                "logAffinity",
            ]
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for record_id, label, affinity, negative_type, group in truth_rows:
                    writer.writerow(
                        {
                            "id": record_id,
                            "receptor_group": group,
                            "negative_type_clean": negative_type,
                            "class_clean": label,
                            "regression_eligible": True,
                            "logAffinity": affinity,
                        }
                    )

            args = SimpleNamespace(
                output_dir=root,
                split_manifest=manifest,
                checkpoint=root / "mock.ckpt",
            )
            config = {
                "experiment": "mock",
                "parameters": {
                    "classification_enabled": True,
                    "regression_enabled": True,
                },
            }
            metrics = summarize_predictions(
                args, config, [row[0] for row in truth_rows]
            )
            self.assertEqual(metrics["test_records"], 4)
            self.assertEqual(metrics["classification"]["accuracy"], 1.0)
            self.assertEqual(metrics["classification"]["roc_auc"], 1.0)
            self.assertEqual(metrics["regression"]["n"], 4)
            self.assertTrue((root / "predictions.csv").is_file())
            self.assertTrue((root / "negative_subgroup_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
