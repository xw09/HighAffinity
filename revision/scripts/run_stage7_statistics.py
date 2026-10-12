#!/usr/bin/env python3
"""Discover completed validation predictions and run all revision statistics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTION_ROOT = ROOT / "revision/predictions/stage6_validation"
ANALYZER = ROOT / "revision/scripts/analyze_predictions.py"
TABLE_BUILDER = ROOT / "revision/scripts/build_revision_table_templates.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "revision/statistics"
    )
    parser.add_argument("--table-output-dir", type=Path)
    parser.add_argument("--split-label", default="fixed validation")
    parser.add_argument("--reference-run", default="M0_seed1")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_predictions(root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        completed = sorted(run_dir.glob("*/EVALUATION_COMPLETE.json"))
        if not completed:
            continue
        if len(completed) > 1:
            raise RuntimeError(
                f"{run_dir.name} has multiple completed evaluations; select one "
                f"explicitly before final statistics: {completed}"
            )
        prediction = completed[0].parent / "predictions.csv"
        if not prediction.is_file():
            raise FileNotFoundError(prediction)
        discovered[run_dir.name] = prediction
    if not discovered:
        raise RuntimeError(f"No completed evaluations found under {root}")
    return discovered


def main() -> None:
    args = parse_args()
    prediction_root = args.prediction_root.expanduser().resolve()
    predictions = discover_predictions(prediction_root)
    if args.reference_run not in predictions:
        raise RuntimeError(
            f"Reference run {args.reference_run} is not complete under "
            f"{prediction_root}"
        )
    command = [
        sys.executable,
        str(ANALYZER),
        "--bootstrap-replicates",
        str(args.bootstrap_replicates),
        "--output-dir",
        str(args.output_dir.expanduser().resolve()),
    ]
    for run_name, path in predictions.items():
        command.extend(["--prediction", f"{run_name}={path}"])
    comparisons = []
    for other in ("M1_seed1", "M2_seed1", "M3_seed1", "M4_seed1", "M5_seed1", "M6_seed1"):
        if args.reference_run in predictions and other in predictions:
            comparisons.append((args.reference_run, other))
    # The repeated central ablation is paired by seed. Do not compare M0 seed 1
    # with M1 seed 2/3, because that needlessly mixes training randomness.
    for seed in (2, 3):
        left = f"M0_seed{seed}"
        right = f"M1_seed{seed}"
        if left in predictions and right in predictions:
            comparisons.append((left, right))
    for left, right in comparisons:
        command.extend(["--comparison", f"{left}:{right}"])
    print(f"Discovered {len(predictions)} completed validation runs:")
    for run_name, path in predictions.items():
        print(f"- {run_name}: {path}")
    if args.dry_run:
        print("Dry run; command not executed.")
        print(" ".join(command))
        return
    subprocess.run(command, check=True)
    table_output = args.table_output_dir
    if table_output is None:
        table_output = ROOT / "revision/tables/templates"
    subprocess.run(
        [
            sys.executable,
            str(TABLE_BUILDER),
            "--statistics-dir",
            str(args.output_dir.expanduser().resolve()),
            "--output-dir",
            str(table_output.expanduser().resolve()),
            "--split-label",
            args.split_label,
        ],
        check=True,
    )
    print("Stage-7 statistics and table templates are complete.")


if __name__ == "__main__":
    main()
