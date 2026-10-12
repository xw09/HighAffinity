#!/usr/bin/env python3
"""Resumable common-metric evaluation on the fixed validation split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.scripts.run_stage5_pipeline import (
    SCREENING_RUNS,
    StatusStore,
    load_config,
    now,
    query_gpu_free_mib,
    run_dir,
    run_logged_process,
    select_best_checkpoint,
    training_complete,
)


EVALUATION_RUNNER = ROOT / "revision/scripts/run_affinity_evaluation.py"
VALIDATION_DATA = Path(
    os.environ.get("HIGH_AFFINITY_PROCESSED_DIR", "data/processed/train")
)
VALIDATION_IDS = ROOT / "revision/data/splits/validation_ids.txt"
SPLIT_MANIFEST = ROOT / "revision/data/splits/split_manifest_v2.csv"
PREDICTION_ROOT = ROOT / "revision/predictions/stage6_validation"
LOG_DIR = ROOT / "revision/logs/stage6_validation"
STATUS_PATH = ROOT / "revision/status/stage6_validation.json"
SUMMARY_PATH = ROOT / "revision/status/stage6_validation_metrics_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-wait-minutes", type=float, default=10.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=60.0)
    parser.add_argument("--min-free-mib", type=int, default=23000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def output_dir(run_name: str, checkpoint: Path) -> Path:
    return PREDICTION_ROOT / run_name / checkpoint.stem


def validation_complete(run_name: str) -> bool:
    _, config = load_config(run_name)
    checkpoint = select_best_checkpoint(run_dir(config))
    return checkpoint is not None and (
        output_dir(run_name, checkpoint) / "EVALUATION_COMPLETE.json"
    ).is_file()


def wait_for_gpu(gpu: int, args: argparse.Namespace) -> None:
    while True:
        free = query_gpu_free_mib().get(gpu, 0)
        if free >= args.min_free_mib:
            return
        print(
            f"[{now()}] GPU {gpu}: {free} MiB free; waiting for "
            f"{args.min_free_mib} MiB.",
            flush=True,
        )
        time.sleep(args.gpu_poll_seconds)


def validate_one(
    run_name: str,
    gpu: int,
    args: argparse.Namespace,
    status: StatusStore,
) -> bool:
    config_path, config = load_config(run_name)
    if not training_complete(config):
        status.update(run_name, validation="blocked_training_incomplete")
        return False
    checkpoint = select_best_checkpoint(run_dir(config))
    assert checkpoint is not None
    destination = output_dir(run_name, checkpoint)
    marker = destination / "EVALUATION_COMPLETE.json"
    if marker.is_file():
        status.update(
            run_name,
            validation="complete",
            checkpoint=str(checkpoint),
            output_dir=str(destination),
            skipped=True,
        )
        print(f"[{run_name}] validation already complete; skipping.", flush=True)
        return True

    log_path = LOG_DIR / f"{run_name}.validation.log"
    for attempt in range(1, args.max_retries + 2):
        wait_for_gpu(gpu, args)
        command = [
            sys.executable,
            str(EVALUATION_RUNNER),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(destination),
            "--evaluation-split",
            "validation",
            "--test-dir",
            str(VALIDATION_DATA),
            "--test-ids",
            str(VALIDATION_IDS),
            "--split-manifest",
            str(SPLIT_MANIFEST),
        ]
        status.update(
            run_name,
            validation="running",
            gpu=gpu,
            attempt=attempt,
            checkpoint=str(checkpoint),
            output_dir=str(destination),
            log=str(log_path),
            started_at=now(),
        )
        print(
            f"[{run_name}] fixed-validation inference on GPU {gpu}; "
            f"log: {log_path}",
            flush=True,
        )
        returncode = run_logged_process(
            command, log_path, gpu, run_name, attempt
        )
        if returncode == 0 and marker.is_file():
            status.update(
                run_name,
                validation="complete",
                returncode=returncode,
                completed_at=now(),
            )
            return True
        status.update(
            run_name,
            validation=(
                "retry_wait" if attempt <= args.max_retries else "failed"
            ),
            returncode=returncode,
            failed_at=now(),
        )
        if attempt <= args.max_retries:
            time.sleep(args.retry_wait_minutes * 60)
    return False


def execute(
    run_names: list[str],
    gpus: list[int],
    args: argparse.Namespace,
    status: StatusStore,
) -> bool:
    for run_name in run_names:
        status.replace(run_name, validation="queued")
    gpu_queue: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def assigned(run_name: str) -> bool:
        gpu = gpu_queue.get()
        try:
            return validate_one(run_name, gpu, args, status)
        finally:
            gpu_queue.put(gpu)

    workers = min(args.parallel, len(gpus), len(run_names))
    results: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(assigned, run_name): run_name
            for run_name in run_names
        }
        for future in as_completed(futures):
            run_name = futures[future]
            try:
                results[run_name] = future.result()
            except Exception as error:  # noqa: BLE001
                results[run_name] = False
                status.update(
                    run_name, validation="failed", unexpected_error=repr(error)
                )
                print(f"[{run_name}] validation error: {error}", flush=True)
    return all(results.get(run_name, False) for run_name in run_names)


def metric_row(run_name: str, config: dict, metrics: dict) -> dict[str, object]:
    classification = metrics["classification"]
    regression = metrics["regression"]
    return {
        "run": run_name,
        "experiment": config["experiment"],
        "seed": config["parameters"]["seed"],
        "accuracy": classification.get("accuracy", ""),
        "roc_auc": classification.get("roc_auc", ""),
        "average_precision": classification.get("average_precision", ""),
        "precision": classification.get("precision", ""),
        "recall": classification.get("recall", ""),
        "f1": classification.get("f1", ""),
        "specificity": classification.get("specificity", ""),
        "mcc": classification.get("mcc", ""),
        "mae": regression.get("mae", ""),
        "rmse": regression.get("rmse", ""),
        "pearson": regression.get("pearson", ""),
        "spearman": regression.get("spearman", ""),
        "metrics_file": metrics.get("metrics_file", ""),
    }


def write_summary(run_names: list[str]) -> Path:
    rows = []
    for run_name in run_names:
        _, config = load_config(run_name)
        checkpoint = select_best_checkpoint(run_dir(config))
        if checkpoint is None:
            continue
        metrics_path = output_dir(run_name, checkpoint) / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["metrics_file"] = str(metrics_path)
        rows.append(metric_row(run_name, config, metrics))
    if len(rows) != len(run_names):
        raise RuntimeError(
            f"Expected metrics for {len(run_names)} runs, found {len(rows)}."
        )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return SUMMARY_PATH


def main() -> None:
    args = parse_args()
    if args.parallel < 1:
        raise ValueError("--parallel must be at least 1.")
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain unique GPU indices.")
    if args.parallel > len(gpus):
        raise ValueError("--parallel cannot exceed the number of listed GPUs.")

    run_names = list(SCREENING_RUNS)
    incomplete = [
        run_name
        for run_name in run_names
        if not training_complete(load_config(run_name)[1])
    ]
    if incomplete:
        raise SystemExit("Training incomplete: " + ", ".join(incomplete))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    status = StatusStore(STATUS_PATH, "screening", "validation")
    status.retain_runs(run_names)
    print(
        f"stage=6 split=validation runs={len(run_names)} "
        f"gpus={gpus} parallel={args.parallel}"
    )
    for run_name in run_names:
        _, config = load_config(run_name)
        checkpoint = select_best_checkpoint(run_dir(config))
        print(
            f"- {run_name}: checkpoint={checkpoint} "
            f"complete={validation_complete(run_name)}"
        )
    if args.dry_run:
        for run_name in run_names:
            status.replace(
                run_name,
                validation=(
                    "complete" if validation_complete(run_name) else "not_started"
                ),
            )
        print("Dry run complete; final test set was not accessed.")
        return

    if not execute(run_names, gpus, args, status):
        raise SystemExit("At least one validation run failed; see stage-6 logs.")
    summary = write_summary(run_names)
    print(f"Stage-6 validation complete. Summary: {summary}")


if __name__ == "__main__":
    main()
