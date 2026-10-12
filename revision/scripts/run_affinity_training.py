#!/usr/bin/env python3
"""Run one controlled affinity fine-tuning configuration."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from boltz.affinity_train import train_affinity_finetune


UNEXPANDED_ENVIRONMENT_VARIABLE = re.compile(r"\$\{[^}]+\}")


def expand_environment_variables(value: object) -> object:
    """Expand environment variables in nested configuration values."""
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if UNEXPANDED_ENVIRONMENT_VARIABLE.search(expanded):
            raise ValueError(
                f"Unresolved environment variable in configuration value: {value}"
            )
        return expanded
    if isinstance(value, list):
        return [expand_environment_variables(item) for item in value]
    if isinstance(value, dict):
        return {
            key: expand_environment_variables(item) for key, item in value.items()
        }
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Run the fixed validation set without fitting.",
    )
    parser.add_argument(
        "--resume-ckpt",
        type=Path,
        help="Resume from a Lightning checkpoint without editing the YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    parameters = expand_environment_variables(dict(config["parameters"]))
    parameters["validation_only"] = args.validation_only
    if args.resume_ckpt is not None:
        resume_path = args.resume_ckpt.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        parameters["resume_ckpt_path"] = str(resume_path)

    run_dir = Path(parameters["ckpt_path"])
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "source_config": str(config_path),
        **config,
        "parameters": parameters,
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    train_affinity_finetune(**parameters)

    marker_name = (
        "VALIDATION_COMPLETE.json" if args.validation_only else "TRAINING_COMPLETE.json"
    )
    marker = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_config": str(config_path),
        "resume_ckpt_path": parameters.get("resume_ckpt_path"),
    }
    (run_dir / marker_name).write_text(
        json.dumps(marker, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
