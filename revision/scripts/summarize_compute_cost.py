#!/usr/bin/env python3
"""Recover auditable compute-cost information from completed run artifacts."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "revision"
RUNS = (
    "M0_seed1",
    "M0_seed2",
    "M0_seed3",
    "M1_seed1",
    "M1_seed2",
    "M1_seed3",
    "M2_seed1",
    "M3_seed1",
    "M4_seed1",
    "M5_seed1",
    "M6_seed1",
)
PARAMETERS = 515_055_240
TRAINABLE = 8_741_896
# These are sampled nvidia-smi observations made while the completed jobs were
# running, not allocator-instrumented maxima. Keep the distinction explicit.
SAMPLED_TRAIN_MEMORY_MIB = 18_768
SAMPLED_INFERENCE_MEMORY_MIB = 7_876


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def best_checkpoint() -> dict[str, dict[str, str]]:
    path = REVISION / "status/stage5_submission_training_summary.csv"
    frame = pd.read_csv(path)
    return {str(row["run"]): row.to_dict() for _, row in frame.iterrows()}


def epoch_durations(log_path: Path) -> dict[int, int]:
    text = log_path.read_bytes().decode("utf-8", errors="ignore").replace("\r", "\n")
    pattern = re.compile(
        r"Epoch\s+(\d+):\s+100%.*?\[(\d+):(\d+):(\d+)<.*?val/loss_affinity="
    )
    durations: dict[int, int] = {}
    for match in pattern.finditer(text):
        epoch = int(match.group(1))
        seconds = (
            int(match.group(2)) * 3600
            + int(match.group(3)) * 60
            + int(match.group(4))
        )
        durations[epoch] = max(durations.get(epoch, 0), seconds)
    if not durations:
        raise RuntimeError(f"No completed epochs found in {log_path}")
    return durations


def format_hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f}"


def main() -> None:
    checkpoint_rows = best_checkpoint()
    evaluation = json.loads(
        (REVISION / "status/stage5_submission.json").read_text(encoding="utf-8")
    )["runs"]
    rows = []
    for run in RUNS:
        experiment = run.split("_", 1)[0]
        durations = epoch_durations(REVISION / f"logs/stage5/{run}.train.log")
        training_seconds = sum(durations.values())
        checkpoint = str(checkpoint_rows[run]["checkpoint"])
        match = re.search(r"epoch(\d+)-valloss([0-9.]+)\.ckpt$", checkpoint)
        if not match:
            raise ValueError(f"Cannot parse checkpoint: {checkpoint}")
        eval_record = evaluation[run]
        inference_seconds = (
            parse_iso(eval_record["evaluation_completed_at"])
            - parse_iso(eval_record["evaluation_started_at"])
        ).total_seconds()
        n_test = 2496
        rows.append(
            {
                "Experiment": experiment,
                "Reported run": run,
                "Trainable parameters": TRAINABLE,
                "Total parameters": PARAMETERS,
                "Trainable fraction (%)": 100 * TRAINABLE / PARAMETERS,
                "Completed epochs": max(durations) + 1,
                "Best epoch (zero-based)": int(match.group(1)),
                "Best validation loss": float(match.group(2)),
                "Training wall time from epoch logs (h)": float(
                    format_hours(training_seconds)
                ),
                "Held-out inference records": n_test,
                "Held-out affinity inference wall time (h)": float(
                    format_hours(inference_seconds)
                ),
                "Affinity inference time per complex (s)": inference_seconds / n_test,
                "GPU": "NVIDIA GeForce RTX 4090 (24 GB)",
                "Precision": "bf16-mixed",
                "Sampled training memory (GiB)": SAMPLED_TRAIN_MEMORY_MIB / 1024,
                "Sampled affinity inference memory (GiB)": (
                    SAMPLED_INFERENCE_MEMORY_MIB / 1024
                ),
                "Memory measurement": "nvidia-smi snapshot; not allocator peak",
                "Inference scope": "affinity head on precomputed Boltz structures",
            }
        )

    output_dir = REVISION / "tables/final_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    output = output_dir / "table_s6_compute_cost.csv"
    frame.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    compact_output = output_dir / "table_s6_compute_cost_compact.csv"
    compact = frame[
        [
            "Reported run",
            "Experiment",
            "Completed epochs",
            "Best epoch (zero-based)",
            "Best validation loss",
            "Training wall time from epoch logs (h)",
            "Held-out affinity inference wall time (h)",
            "Affinity inference time per complex (s)",
        ]
    ].rename(columns={"Reported run": "Run"})
    compact.insert(2, "Seed", compact["Run"].str.extract(r"seed(\d+)").astype(int))
    compact.to_csv(compact_output, index=False, quoting=csv.QUOTE_MINIMAL)
    note = output_dir / "table_s6_compute_cost_notes.md"
    note.write_text(
        "# Compute-cost reporting notes\n\n"
        "- Training wall time is reconstructed by summing the final completed "
        "progress-bar duration for each epoch in the archived Lightning logs.\n"
        "- Inference wall time comes from the recorded start/completion timestamps "
        "for the 2,496-record held-out test evaluation.\n"
        "- Inference uses precomputed Boltz structures and therefore measures the "
        "affinity inference stage, not de novo structure prediction.\n"
        "- Memory values are contemporaneous `nvidia-smi` snapshots and must be "
        "reported as approximate observed usage, not exact peak allocation.\n"
        "- Best epoch numbers are zero-based, matching checkpoint filenames.\n"
        "- The table contains all 11 prespecified runs: three seeds each for "
        "M0 and M1, and seed 1 for M2--M6.\n",
        encoding="utf-8",
    )
    print(output)
    print(compact_output)
    print(note)


if __name__ == "__main__":
    main()
