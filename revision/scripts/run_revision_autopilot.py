#!/usr/bin/env python3
"""Finish the computational revision pipeline without manual supervision.

Order of operations:
1. wait for all fixed training and validation jobs;
2. compute validation statistics and tables;
3. run the single gated held-out-test evaluation for the frozen suite;
4. compute held-out-test statistics and tables;
5. clean PEPBI when its official workbook is available locally.

The script never starts new training and uses only one GPU for inference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH = ROOT / "revision/status/revision_autopilot.json"
VALIDATION_ROOT = ROOT / "revision/predictions/stage6_validation"
TEST_ROOT = ROOT / "revision/predictions/stage5"
VALIDATION_STATS = ROOT / "revision/statistics"
TEST_STATS = ROOT / "revision/statistics/final_test"
VALIDATION_TABLES = ROOT / "revision/tables/templates"
TEST_TABLES = ROOT / "revision/tables/final_test"
PEPBI_INPUT = ROOT / "revision/external/pepbi/raw/PEPBI.xlsx"

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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--min-free-mib", type=int, default=23000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class Status:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.payload = {
            "pipeline": "revision_autopilot",
            "started_at": now(),
            "stage": "initializing",
            "stages": {},
        }
        if path.is_file():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
                self.payload["previous_run"] = previous
            except (OSError, json.JSONDecodeError):
                pass
        self.write()

    def update_stage(self, stage: str, state: str, **details: object) -> None:
        self.payload["stage"] = stage
        self.payload["stages"].setdefault(stage, {})
        self.payload["stages"][stage].update(
            {"state": state, "updated_at": now(), **details}
        )
        self.payload["updated_at"] = now()
        self.write()

    def finish(self, state: str) -> None:
        self.payload["state"] = state
        self.payload["completed_at"] = now()
        self.payload["stage"] = "complete"
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def training_complete(run_name: str) -> bool:
    directory = ROOT / "revision/runs" / run_name
    return (directory / "TRAINING_COMPLETE.json").is_file() and any(
        path.name != "last.ckpt"
        for path in (directory / "checkpoints").glob("*.ckpt")
    )


def evaluation_complete(root: Path, run_name: str) -> bool:
    markers = list((root / run_name).glob("*/EVALUATION_COMPLETE.json"))
    return len(markers) == 1


def tmux_session_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def wait_for_prerequisites(args: argparse.Namespace, status: Status) -> None:
    while True:
        missing_training = [run for run in SUBMISSION_RUNS if not training_complete(run)]
        missing_validation = [
            run
            for run in SUBMISSION_RUNS
            if not evaluation_complete(VALIDATION_ROOT, run)
        ]
        if not missing_training and not missing_validation:
            status.update_stage(
                "wait_prerequisites",
                "complete",
                training_runs=len(SUBMISSION_RUNS),
                validation_runs=len(SUBMISSION_RUNS),
            )
            return
        if missing_training and not tmux_session_exists("highaff_stage7"):
            run_command(
                "recover_training",
                [
                    sys.executable,
                    str(ROOT / "revision/scripts/run_stage5_pipeline.py"),
                    "--suite",
                    "submission",
                    "--phase",
                    "train",
                    "--gpus",
                    "0",
                    "--parallel",
                    "1",
                ],
                status,
            )
            continue
        if (
            not missing_training
            and missing_validation
            and not tmux_session_exists("highaff_stage7_validation")
        ):
            run_command(
                "recover_repeat_validation",
                [
                    sys.executable,
                    str(ROOT / "revision/scripts/run_stage7_repeat_validation.py"),
                    "--gpu",
                    str(args.gpu),
                    "--min-free-mib",
                    str(args.min_free_mib),
                ],
                status,
            )
            continue
        status.update_stage(
            "wait_prerequisites",
            "waiting",
            missing_training=missing_training,
            missing_validation=missing_validation,
        )
        print(
            f"[{now()}] waiting: training={missing_training or 'complete'} "
            f"validation={missing_validation or 'complete'}",
            flush=True,
        )
        time.sleep(args.poll_seconds)


def run_command(stage: str, command: list[str], status: Status) -> None:
    status.update_stage(stage, "running", command=command, started_at=now())
    print(f"[{now()}] stage={stage}", flush=True)
    print("command=" + " ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except BaseException as error:
        status.update_stage(stage, "failed", error=repr(error), failed_at=now())
        raise
    status.update_stage(stage, "complete", completed_at=now())


def statistics_command(
    prediction_root: Path,
    output_dir: Path,
    table_dir: Path,
    split_label: str,
    replicates: int,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "revision/scripts/run_stage7_statistics.py"),
        "--prediction-root",
        str(prediction_root),
        "--output-dir",
        str(output_dir),
        "--table-output-dir",
        str(table_dir),
        "--split-label",
        split_label,
        "--bootstrap-replicates",
        str(replicates),
    ]


def final_test_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "revision/scripts/run_stage5_pipeline.py"),
        "--suite",
        "submission",
        "--phase",
        "evaluate",
        "--gpus",
        str(args.gpu),
        "--parallel",
        "1",
        "--min-free-mib",
        str(args.min_free_mib),
        "--allow-test-after-training",
    ]


def rebuild_tables(statistics_dir: Path, output_dir: Path, split_label: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "revision/scripts/build_revision_table_templates.py"),
            "--statistics-dir",
            str(statistics_dir),
            "--output-dir",
            str(output_dir),
            "--split-label",
            split_label,
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 10:
        raise ValueError("--poll-seconds must be at least 10 seconds.")
    status = Status(STATUS_PATH)
    plan = {
        "gpu": args.gpu,
        "runs": SUBMISSION_RUNS,
        "bootstrap_replicates": args.bootstrap_replicates,
        "final_test_authorized": True,
        "pepbi_available": PEPBI_INPUT.is_file(),
    }
    status.update_stage("plan", "complete", **plan)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        status.finish("dry_run")
        return

    wait_for_prerequisites(args, status)
    run_command(
        "validation_statistics",
        statistics_command(
            VALIDATION_ROOT,
            VALIDATION_STATS,
            VALIDATION_TABLES,
            "fixed validation",
            args.bootstrap_replicates,
        ),
        status,
    )

    if all(evaluation_complete(TEST_ROOT, run) for run in SUBMISSION_RUNS):
        status.update_stage("final_test", "complete", skipped=True)
    else:
        run_command("final_test", final_test_command(args), status)

    run_command(
        "final_test_statistics",
        statistics_command(
            TEST_ROOT,
            TEST_STATS,
            TEST_TABLES,
            "held-out test",
            args.bootstrap_replicates,
        ),
        status,
    )

    if PEPBI_INPUT.is_file():
        run_command(
            "pepbi_cleaning",
            [
                sys.executable,
                str(ROOT / "revision/scripts/prepare_pepbi_external.py"),
            ],
            status,
        )
        rebuild_tables(VALIDATION_STATS, VALIDATION_TABLES, "fixed validation")
        rebuild_tables(TEST_STATS, TEST_TABLES, "held-out test")
        status.finish("complete")
    else:
        status.update_stage(
            "pepbi_cleaning",
            "pending_missing_source",
            required_file=str(PEPBI_INPUT),
        )
        status.finish("complete_except_external_data")


if __name__ == "__main__":
    main()
