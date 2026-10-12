#!/usr/bin/env python3
"""Run affinity-only inference on precomputed PEPBI Boltz structures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from run_affinity_evaluation import (
    prediction_path,
    read_ids,
    run_inference,
    valid_prediction,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = (
    ROOT / "revision/external/pepbi/inference/observation_to_inference_mapping.csv"
)
DEFAULT_IDS = ROOT / "revision/external/pepbi/inference/inference_ids.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--boltz-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed")
    return parser.parse_args()


def metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    true = np.asarray([float(row["logAffinity"]) for row in rows])
    pred = np.asarray([float(row["affinity_pred"]) for row in rows])
    return {
        "n": int(len(rows)),
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(math.sqrt(mean_squared_error(true, pred))),
        "pearson": float(pearsonr(true, pred).statistic),
        "spearman": float(spearmanr(true, pred).statistic),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.boltz_dir = args.boltz_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.mapping = args.mapping.expanduser().resolve()
    args.ids = args.ids.expanduser().resolve()
    args.cache = args.cache.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not (args.boltz_dir / "processed/manifest.json").is_file():
        raise FileNotFoundError(args.boltz_dir / "processed/manifest.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = args.output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    ids = read_ids(args.ids)
    pending = [
        record_id
        for record_id in ids
        if not valid_prediction(prediction_path(records_dir, record_id))
    ]
    pending_path = args.output_dir / "pending_ids.txt"
    pending_path.write_text(
        "".join(f"{record_id}\n" for record_id in pending), encoding="utf-8"
    )
    state = {
        "checkpoint": str(args.checkpoint),
        "boltz_dir": str(args.boltz_dir),
        "expected_unique_pairs": len(ids),
        "pending_before_run": len(pending),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "evaluation_state.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    if pending:
        runner_args = SimpleNamespace(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            test_dir=args.boltz_dir,
            cache=args.cache,
            num_workers=args.num_workers,
            batch_size=1,
            precision=args.precision,
        )
        run_inference(runner_args, pending_path)

    missing = [
        record_id
        for record_id in ids
        if not valid_prediction(prediction_path(records_dir, record_id))
    ]
    if missing:
        (args.output_dir / "missing_ids.txt").write_text(
            "".join(f"{record_id}\n" for record_id in missing), encoding="utf-8"
        )
        raise RuntimeError(f"Prediction incomplete: {len(missing)} unique pairs missing")

    unique_rows: list[dict[str, object]] = []
    by_id: dict[str, dict[str, float]] = {}
    for record_id in ids:
        payload = json.loads(prediction_path(records_dir, record_id).read_text())
        values = {
            "affinity_pred": float(payload["affinity_pred_value"]),
            "binder_probability": float(payload["affinity_probability_binary"]),
        }
        by_id[record_id] = values
        unique_rows.append({"inference_id": record_id, **values})
    write_csv(args.output_dir / "unique_predictions.csv", unique_rows)

    with args.mapping.open(newline="", encoding="utf-8") as handle:
        mapping_rows = list(csv.DictReader(handle))
    observation_rows = [
        {**row, **by_id[row["inference_id"]]} for row in mapping_rows
    ]
    write_csv(args.output_dir / "observation_predictions.csv", observation_rows)

    # Also summarize after averaging replicated thermodynamic observations per pair.
    pair_rows: list[dict[str, object]] = []
    for record_id in ids:
        subset = [row for row in observation_rows if row["inference_id"] == record_id]
        pair_rows.append(
            {
                "inference_id": record_id,
                "logAffinity": float(np.mean([float(row["logAffinity"]) for row in subset])),
                "affinity_pred": by_id[record_id]["affinity_pred"],
            }
        )
    result = {
        **state,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "observation_level": metrics(observation_rows),
        "unique_pair_level": metrics(pair_rows),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "EVALUATION_COMPLETE.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
