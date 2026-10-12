#!/usr/bin/env python3
"""Unattended, resumable stage-5 training and final evaluation pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "revision/configs/public"
TRAIN_RUNNER = ROOT / "revision/scripts/run_affinity_training.py"
EVAL_RUNNER = ROOT / "revision/scripts/run_affinity_evaluation.py"
STATUS_DIR = ROOT / "revision/status"
LOG_DIR = ROOT / "revision/logs/stage5"
PREDICTION_ROOT = ROOT / "revision/predictions/stage5"

SCREENING_RUNS = [
    "M0_seed1",
    "M1_seed1",
    "M2_seed1",
    "M3_seed1",
    "M4_seed1",
    "M5_seed1",
    "M6_seed1",
]
SUBMISSION_RUNS = [
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
]

EXPECTED_MAX_EPOCHS = 8
EXPECTED_EARLY_STOPPING_PATIENCE = 2


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=["screening", "submission"], default="screening"
    )
    parser.add_argument(
        "--phase", choices=["train", "evaluate", "all"], default="train"
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated physical GPU indices, e.g. 0,1,2,3.",
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-wait-minutes", type=float, default=10.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=60.0)
    parser.add_argument("--min-free-mib", type=int, default=23000)
    parser.add_argument(
        "--allow-test-after-training",
        action="store_true",
        help="Required for evaluate/all; test inference starts only after all selected training runs complete.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(run_name: str) -> tuple[Path, dict]:
    path = CONFIG_DIR / f"{run_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing configuration: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    parameters = config["parameters"]
    if parameters.get("unfreeze_last_trunk_blocks", 0) != 0:
        raise ValueError(
            f"{run_name} attempts to unfreeze the shared structural trunk; "
            "this is excluded from the final revision plan."
        )
    if parameters.get("max_epochs") != EXPECTED_MAX_EPOCHS:
        raise ValueError(
            f"{run_name} max_epochs must be {EXPECTED_MAX_EPOCHS}, got "
            f"{parameters.get('max_epochs')}. Regenerate the stage-4 configs."
        )
    if (
        parameters.get("early_stopping_patience")
        != EXPECTED_EARLY_STOPPING_PATIENCE
    ):
        raise ValueError(
            f"{run_name} early_stopping_patience must be "
            f"{EXPECTED_EARLY_STOPPING_PATIENCE}, got "
            f"{parameters.get('early_stopping_patience')}."
        )
    return path, config


def run_dir(config: dict) -> Path:
    return Path(config["parameters"]["ckpt_path"]).expanduser().resolve()


def checkpoint_score(path: Path) -> float:
    match = re.search(r"valloss(-?[0-9]+(?:\.[0-9]+)?)", path.name)
    return float(match.group(1)) if match else float("inf")


def select_best_checkpoint(directory: Path) -> Optional[Path]:
    checkpoint_dir = directory / "checkpoints"
    candidates = [
        path
        for path in checkpoint_dir.glob("*.ckpt")
        if path.name != "last.ckpt"
    ]
    if candidates:
        return min(candidates, key=lambda path: (checkpoint_score(path), path.name))
    last = checkpoint_dir / "last.ckpt"
    return last if last.is_file() else None


def last_checkpoint(directory: Path) -> Optional[Path]:
    path = directory / "checkpoints/last.ckpt"
    return path if path.is_file() else None


def training_complete(config: dict) -> bool:
    directory = run_dir(config)
    return (directory / "TRAINING_COMPLETE.json").is_file() and (
        select_best_checkpoint(directory) is not None
    )


def evaluation_dir(run_name: str, checkpoint: Path) -> Path:
    return PREDICTION_ROOT / run_name / checkpoint.stem


def query_gpu_free_mib() -> dict[int, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    free: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, memory = line.split(",", maxsplit=1)
        free[int(index.strip())] = int(memory.strip())
    return free


def wait_for_gpu(gpu: int, min_free_mib: int, poll_seconds: float) -> None:
    while True:
        free = query_gpu_free_mib().get(gpu, 0)
        if free >= min_free_mib:
            return
        print(
            f"[{now()}] GPU {gpu}: {free} MiB free; waiting for "
            f"{min_free_mib} MiB.",
            flush=True,
        )
        time.sleep(poll_seconds)


class StatusStore:
    def __init__(self, path: Path, suite: str, phase: str) -> None:
        self.path = path
        self.lock = threading.Lock()
        if path.is_file():
            self.payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.payload = {"suite": suite, "phase": phase, "runs": {}}
        self.payload["suite"] = suite
        self.payload["phase"] = phase

    def retain_runs(self, run_names: list[str]) -> None:
        """Drop stale status entries from experiments no longer in the suite."""
        with self.lock:
            existing = self.payload.setdefault("runs", {})
            self.payload["runs"] = {
                run_name: existing.get(run_name, {}) for run_name in run_names
            }
            self.payload["updated_at"] = now()
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)

    def update(self, run_name: str, **values: object) -> None:
        with self.lock:
            entry = self.payload.setdefault("runs", {}).setdefault(run_name, {})
            entry.update(values)
            self.payload["updated_at"] = now()
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)

    def replace(self, run_name: str, **values: object) -> None:
        """Replace one run entry, clearing stale fields from an older launch."""
        with self.lock:
            self.payload.setdefault("runs", {})[run_name] = dict(values)
            self.payload["updated_at"] = now()
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)


def child_environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    python_path = str(ROOT / "boltz/src")
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{python_path}:{old_python_path}" if old_python_path else python_path
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def run_logged_process(
    command: list[str], log_path: Path, gpu: int, run_name: str, attempt: int
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n===== {now()} run={run_name} gpu={gpu} attempt={attempt} =====\n"
        )
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=child_environment(gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"===== {now()} returncode={process.returncode} =====\n")
        return process.returncode


def train_one(
    run_name: str,
    gpu: int,
    args: argparse.Namespace,
    status: StatusStore,
) -> bool:
    config_path, config = load_config(run_name)
    directory = run_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    if training_complete(config):
        checkpoint = select_best_checkpoint(directory)
        status.update(
            run_name,
            training="complete",
            checkpoint=str(checkpoint),
            skipped=True,
        )
        print(f"[{run_name}] already complete; skipping training.", flush=True)
        return True

    log_path = LOG_DIR / f"{run_name}.train.log"
    for attempt in range(1, args.max_retries + 2):
        wait_for_gpu(gpu, args.min_free_mib, args.gpu_poll_seconds)
        resume = last_checkpoint(directory)
        command = [
            sys.executable,
            str(TRAIN_RUNNER),
            "--config",
            str(config_path),
        ]
        if resume is not None:
            command.extend(["--resume-ckpt", str(resume)])
        status.update(
            run_name,
            training="running",
            gpu=gpu,
            attempt=attempt,
            resume_checkpoint=str(resume) if resume else None,
            log=str(log_path),
            started_at=now(),
        )
        print(
            f"[{run_name}] training on GPU {gpu}, attempt {attempt}; "
            f"log: {log_path}",
            flush=True,
        )
        returncode = run_logged_process(command, log_path, gpu, run_name, attempt)
        if returncode == 0 and training_complete(config):
            checkpoint = select_best_checkpoint(directory)
            (directory / "best_checkpoint.txt").write_text(
                str(checkpoint) + "\n", encoding="utf-8"
            )
            status.update(
                run_name,
                training="complete",
                checkpoint=str(checkpoint),
                completed_at=now(),
                returncode=returncode,
            )
            print(f"[{run_name}] training complete: {checkpoint}", flush=True)
            return True

        status.update(
            run_name,
            training="retry_wait" if attempt <= args.max_retries else "failed",
            returncode=returncode,
            failed_at=now(),
        )
        if attempt <= args.max_retries:
            wait_seconds = args.retry_wait_minutes * 60
            print(
                f"[{run_name}] failed (return code {returncode}); retrying in "
                f"{args.retry_wait_minutes:g} minutes.",
                flush=True,
            )
            time.sleep(wait_seconds)
    print(f"[{run_name}] failed after all retries; see {log_path}", flush=True)
    return False


def evaluate_one(
    run_name: str,
    gpu: int,
    args: argparse.Namespace,
    status: StatusStore,
) -> bool:
    config_path, config = load_config(run_name)
    checkpoint = select_best_checkpoint(run_dir(config))
    if checkpoint is None:
        status.update(run_name, evaluation="blocked_no_checkpoint")
        return False
    output_dir = evaluation_dir(run_name, checkpoint)
    marker = output_dir / "EVALUATION_COMPLETE.json"
    if marker.is_file():
        status.update(
            run_name,
            evaluation="complete",
            evaluation_dir=str(output_dir),
            skipped=True,
        )
        print(f"[{run_name}] evaluation already complete; skipping.", flush=True)
        return True

    log_path = LOG_DIR / f"{run_name}.evaluate.log"
    for attempt in range(1, args.max_retries + 2):
        wait_for_gpu(gpu, args.min_free_mib, args.gpu_poll_seconds)
        command = [
            sys.executable,
            str(EVAL_RUNNER),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--acknowledge-test-set-is-final",
        ]
        status.update(
            run_name,
            evaluation="running",
            gpu=gpu,
            evaluation_attempt=attempt,
            evaluation_log=str(log_path),
            evaluation_dir=str(output_dir),
            evaluation_started_at=now(),
        )
        print(
            f"[{run_name}] final test evaluation on GPU {gpu}, attempt "
            f"{attempt}; log: {log_path}",
            flush=True,
        )
        returncode = run_logged_process(command, log_path, gpu, run_name, attempt)
        if returncode == 0 and marker.is_file():
            status.update(
                run_name,
                evaluation="complete",
                evaluation_completed_at=now(),
                evaluation_returncode=returncode,
            )
            print(f"[{run_name}] evaluation complete: {output_dir}", flush=True)
            return True
        status.update(
            run_name,
            evaluation=(
                "retry_wait" if attempt <= args.max_retries else "failed"
            ),
            evaluation_returncode=returncode,
            evaluation_failed_at=now(),
        )
        if attempt <= args.max_retries:
            time.sleep(args.retry_wait_minutes * 60)
    print(f"[{run_name}] evaluation failed; see {log_path}", flush=True)
    return False


def execute_phase(
    run_names: list[str],
    gpus: list[int],
    args: argparse.Namespace,
    status: StatusStore,
    worker,
) -> bool:
    status_field = "training" if worker is train_one else "evaluation"
    for run_name in run_names:
        status.replace(run_name, **{status_field: "queued"})

    gpu_queue: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def assigned_worker(run_name: str) -> bool:
        gpu = gpu_queue.get()
        try:
            return worker(run_name, gpu, args, status)
        finally:
            gpu_queue.put(gpu)

    workers = min(args.parallel, len(gpus), len(run_names))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(assigned_worker, run_name): run_name
            for run_name in run_names
        }
        results = {}
        for future in as_completed(futures):
            run_name = futures[future]
            try:
                results[run_name] = future.result()
            except Exception as error:  # noqa: BLE001
                results[run_name] = False
                status.update(run_name, unexpected_error=repr(error), failed_at=now())
                print(f"[{run_name}] unexpected pipeline error: {error}", flush=True)
    return all(results.get(run_name, False) for run_name in run_names)


def print_plan(run_names: list[str], args: argparse.Namespace, gpus: list[int]) -> None:
    print(
        f"suite={args.suite} phase={args.phase} runs={len(run_names)} "
        f"gpus={gpus} parallel={args.parallel}"
    )
    for run_name in run_names:
        _, config = load_config(run_name)
        directory = run_dir(config)
        print(
            f"- {run_name}: complete={training_complete(config)} "
            f"resume={last_checkpoint(directory)} best={select_best_checkpoint(directory)}"
        )


def write_training_summary(run_names: list[str], suite: str) -> Path:
    rows = []
    for run_name in run_names:
        _, config = load_config(run_name)
        directory = run_dir(config)
        checkpoint = select_best_checkpoint(directory)
        rows.append(
            {
                "run": run_name,
                "experiment": config["experiment"],
                "seed": config["parameters"]["seed"],
                "training_complete": training_complete(config),
                "best_val_loss": (
                    checkpoint_score(checkpoint) if checkpoint is not None else ""
                ),
                "checkpoint": str(checkpoint) if checkpoint else "",
            }
        )
    output = STATUS_DIR / f"stage5_{suite}_training_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_evaluation_summary(run_names: list[str], suite: str) -> Path:
    rows = []
    for run_name in run_names:
        _, config = load_config(run_name)
        checkpoint = select_best_checkpoint(run_dir(config))
        if checkpoint is None:
            continue
        metrics_path = evaluation_dir(run_name, checkpoint) / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        classification = metrics["classification"]
        regression = metrics["regression"]
        rows.append(
            {
                "run": run_name,
                "experiment": config["experiment"],
                "seed": config["parameters"]["seed"],
                "accuracy": classification.get("accuracy", ""),
                "roc_auc": classification.get("roc_auc", ""),
                "average_precision": classification.get("average_precision", ""),
                "f1": classification.get("f1", ""),
                "specificity": classification.get("specificity", ""),
                "mcc": classification.get("mcc", ""),
                "mae": regression.get("mae", ""),
                "rmse": regression.get("rmse", ""),
                "pearson": regression.get("pearson", ""),
                "spearman": regression.get("spearman", ""),
                "metrics_file": str(metrics_path),
            }
        )
    output = STATUS_DIR / f"stage5_{suite}_test_metrics_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return output


def main() -> None:
    args = parse_args()
    if args.parallel < 1:
        raise ValueError("--parallel must be at least 1.")
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain unique GPU indices.")
    if args.parallel > len(gpus):
        raise ValueError("--parallel cannot exceed the number of listed GPUs.")
    if args.phase in {"evaluate", "all"} and not args.allow_test_after_training:
        raise SystemExit(
            "Test-set safety gate: add --allow-test-after-training only after "
            "committing to the selected configuration suite."
        )

    run_names = SCREENING_RUNS if args.suite == "screening" else SUBMISSION_RUNS
    for run_name in run_names:
        load_config(run_name)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = STATUS_DIR / f"stage5_{args.suite}.json"
    status = StatusStore(status_path, args.suite, args.phase)
    status.retain_runs(run_names)

    print_plan(run_names, args, gpus)
    if args.dry_run:
        status_field = (
            "training" if args.phase in {"train", "all"} else "evaluation"
        )
        for run_name in run_names:
            _, config = load_config(run_name)
            state = (
                "complete"
                if status_field == "training" and training_complete(config)
                else "not_started"
            )
            status.replace(run_name, **{status_field: state})
        print("Dry run complete; no training or evaluation was started.")
        return

    if args.phase in {"train", "all"}:
        train_ok = execute_phase(run_names, gpus, args, status, train_one)
        if not train_ok:
            raise SystemExit(
                "At least one training run failed. Test evaluation was not started."
            )
        summary = write_training_summary(run_names, args.suite)
        print(f"Training summary: {summary}")

    if args.phase in {"evaluate", "all"}:
        incomplete = [
            run_name
            for run_name in run_names
            if not training_complete(load_config(run_name)[1])
        ]
        if incomplete:
            raise SystemExit(
                "Test evaluation blocked because training is incomplete: "
                + ", ".join(incomplete)
            )
        evaluate_ok = execute_phase(run_names, gpus, args, status, evaluate_one)
        if not evaluate_ok:
            raise SystemExit("At least one final test evaluation failed.")
        summary = write_evaluation_summary(run_names, args.suite)
        print(f"Final test summary: {summary}")

    print(f"Stage-5 pipeline complete. Status: {status_path}")


if __name__ == "__main__":
    main()
