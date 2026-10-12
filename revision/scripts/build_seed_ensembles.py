#!/usr/bin/env python3
"""Build prediction-level M0/M1 three-seed ensembles.

The ensemble averages per-record class probabilities and regression outputs.
It is deliberately different from averaging already-computed metrics.  All
labels and grouping metadata must agree exactly across the three seed files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "revision/predictions/ensembles"
SPLIT_ROOTS = {
    "validation": ROOT / "revision/predictions/stage6_validation",
    "test": ROOT / "revision/predictions/stage5",
}
EXPERIMENTS = ("M0", "M1")
SEEDS = (1, 2, 3)
METADATA_COLUMNS = (
    "receptor_group",
    "negative_type",
    "class_true",
    "regression_eligible",
    "affinity_true",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--split", choices=("validation", "test", "all"), default="all"
    )
    return parser.parse_args()


def discover_prediction(root: Path, run: str) -> Path:
    matches = sorted((root / run).glob("*/predictions.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one completed prediction CSV for {run}, got {matches}"
        )
    return matches[0]


def normalized_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "id",
        *METADATA_COLUMNS,
        "class_probability",
        "affinity_pred",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate IDs")
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    return frame.sort_values("id").reset_index(drop=True)


def assert_metadata_equal(
    reference: pd.DataFrame, candidate: pd.DataFrame, candidate_path: Path
) -> None:
    if not reference["id"].equals(candidate["id"]):
        raise ValueError(f"{candidate_path}: ID set/order differs across seeds")
    for column in METADATA_COLUMNS:
        left = reference[column].fillna("__MISSING__").astype(str)
        right = candidate[column].fillna("__MISSING__").astype(str)
        if not left.equals(right):
            mismatch = int(np.flatnonzero(left.to_numpy() != right.to_numpy())[0])
            raise ValueError(
                f"{candidate_path}: {column} differs for ID "
                f"{reference.loc[mismatch, 'id']}"
            )


def build_one(split: str, experiment: str, output_root: Path) -> dict[str, object]:
    source_root = SPLIT_ROOTS[split]
    paths = [
        discover_prediction(source_root, f"{experiment}_seed{seed}")
        for seed in SEEDS
    ]
    frames = [normalized_frame(path) for path in paths]
    reference = frames[0]
    for frame, path in zip(frames[1:], paths[1:]):
        assert_metadata_equal(reference, frame, path)

    output = reference[["id", *METADATA_COLUMNS]].copy()
    output["class_probability"] = np.mean(
        [frame["class_probability"].to_numpy(dtype=float) for frame in frames],
        axis=0,
    )
    output["class_pred_0.5"] = output["class_probability"].ge(0.5).astype(int)
    output["affinity_pred"] = np.mean(
        [frame["affinity_pred"].to_numpy(dtype=float) for frame in frames],
        axis=0,
    )
    ordered = [
        "id",
        "receptor_group",
        "negative_type",
        "class_true",
        "class_probability",
        "class_pred_0.5",
        "regression_eligible",
        "affinity_true",
        "affinity_pred",
    ]
    output = output[ordered]

    output_dir = output_root / split / f"{experiment}_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.csv"
    output.to_csv(output_path, index=False)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "experiment": experiment,
        "method": "arithmetic mean of per-record probabilities and affinity predictions",
        "classification_threshold": 0.5,
        "seeds": list(SEEDS),
        "sources": [str(path.resolve()) for path in paths],
        "records": int(len(output)),
        "output": str(output_path.resolve()),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    splits = tuple(SPLIT_ROOTS) if args.split == "all" else (args.split,)
    results = [
        build_one(split, experiment, output_root)
        for split in splits
        for experiment in EXPERIMENTS
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
