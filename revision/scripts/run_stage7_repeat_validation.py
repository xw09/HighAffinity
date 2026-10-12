#!/usr/bin/env python3
"""Validate M0/M1 repeat seeds as soon as each training run completes.

This is deliberately validation-only.  It never accesses the held-out test
set and can run on a second GPU while the last repeat seed is still training.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision.scripts import run_stage6_validation as stage6
from revision.scripts.run_stage5_pipeline import (
    StatusStore,
    load_config,
    now,
    training_complete,
)


REPEAT_RUNS = ["M0_seed2", "M0_seed3", "M1_seed2", "M1_seed3"]
STATUS_PATH = ROOT / "revision/status/stage7_repeat_validation.json"
SUMMARY_PATH = ROOT / "revision/status/stage7_repeat_validation_metrics_summary.csv"
LOG_DIR = ROOT / "revision/logs/stage7_repeat_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-wait-minutes", type=float, default=10.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=60.0)
    parser.add_argument("--training-poll-seconds", type=float, default=300.0)
    parser.add_argument("--min-free-mib", type=int, default=23000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # run_stage6_validation.execute expects the same namespace field used by
    # its multi-GPU CLI. This launcher intentionally assigns one validation
    # worker to one GPU.
    args.parallel = 1
    if args.training_poll_seconds < 10:
        raise ValueError("--training-poll-seconds must be at least 10 seconds.")
    for run_name in REPEAT_RUNS:
        load_config(run_name)

    stage6.LOG_DIR = LOG_DIR
    stage6.SUMMARY_PATH = SUMMARY_PATH
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stage6.PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    status = StatusStore(STATUS_PATH, "repeat_seeds", "validation")
    status.retain_runs(REPEAT_RUNS)

    print(f"stage=7 repeat_validation gpu={args.gpu}", flush=True)
    for run_name in REPEAT_RUNS:
        ready = training_complete(load_config(run_name)[1])
        complete = stage6.validation_complete(run_name) if ready else False
        print(
            f"- {run_name}: training_complete={ready} "
            f"validation_complete={complete}",
            flush=True,
        )
        status.replace(
            run_name,
            validation=(
                "complete" if complete else "queued" if ready else "waiting_training"
            ),
        )
    if args.dry_run:
        print("Dry run complete; no validation was started.", flush=True)
        return

    pending = list(REPEAT_RUNS)
    while pending:
        completed_now = [
            run_name for run_name in pending if stage6.validation_complete(run_name)
        ]
        for run_name in completed_now:
            pending.remove(run_name)
            status.update(run_name, validation="complete", skipped=True)
        if not pending:
            break

        ready = [
            run_name
            for run_name in pending
            if training_complete(load_config(run_name)[1])
        ]
        if ready:
            ok = stage6.execute(ready, [args.gpu], args, status)
            if not ok:
                raise SystemExit(
                    "At least one repeat-seed validation failed; see stage-7 logs."
                )
            continue

        for run_name in pending:
            status.update(run_name, validation="waiting_training", checked_at=now())
        print(
            f"[{now()}] Waiting for training: {', '.join(pending)}",
            flush=True,
        )
        time.sleep(args.training_poll_seconds)

    summary = stage6.write_summary(REPEAT_RUNS)
    print(f"Repeat-seed validation complete. Summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
