#!/usr/bin/env python3
"""Generate the fixed stage-4 training configurations and ablation ID list."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "revision"
SPLITS = REVISION / "data" / "splits"
CONFIG_DIR = REVISION / "configs" / "public"


def write_no_decoy_train_ids() -> Path:
    split_manifest = SPLITS / "split_manifest_v2.csv"
    output = SPLITS / "train_no_synthetic_decoy_ids.txt"
    with split_manifest.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        ids = [
            row["id"]
            for row in rows
            if row["split_clean"] == "train"
            and row["negative_type_clean"] != "synthetic_decoy"
        ]
    output.write_text("".join(f"{record_id}\n" for record_id in ids), encoding="utf-8")
    return output


def common_parameters(seed: int, run_name: str) -> dict:
    return {
        "out_dir": "${HIGH_AFFINITY_PROCESSED_DIR}",
        "ckpt_path": f"revision/runs/{run_name}",
        "cache": "${BOLTZ_CACHE}",
        "pretrained_ckpt": "${BOLTZ_CACHE}/boltz2_aff.ckpt",
        "validation_ids_path": "revision/data/splits/validation_ids.txt",
        "train_ids_path": "revision/data/splits/train_ids.txt",
        # The three historical final runs all completed epochs 0--7.  An
        # eight-epoch ceiling therefore preserves the original training
        # budget, while validation early stopping can still save compute.
        "max_epochs": 8,
        "batch_size": 3,
        "devices": 1,
        "precision": "bf16-mixed",
        "accelerator": "gpu",
        "accumulate_steps": 4,
        "num_workers": 8,
        "seed": seed,
        "base_lr": 0.0001,
        "regression_enabled": True,
        "classification_enabled": True,
        "regression_loss": "huber",
        "classification_loss": "focal",
        "lambda_reg": 1.0,
        "lambda_cls": 1.0,
        "huber_delta": 0.5,
        "focal_gamma": 1.0,
        "focal_alpha": None,
        "reset_affinity_on_load": False,
        "unfreeze_last_trunk_blocks": 0,
        # Historical validation loss occasionally worsened for one epoch and
        # then improved again.  Patience 2 tolerates that single fluctuation.
        "early_stopping_patience": 2,
        "resume_ckpt_path": None,
    }


def build_configs(no_decoy_ids: Path) -> list[Path]:
    experiments: list[tuple[str, str, dict, list[int]]] = [
        (
            "M0",
            "pretrained affinity module; frozen trunk; Huber + focal",
            {},
            [1, 2, 3],
        ),
        (
            "M1",
            "randomly initialized affinity module; frozen trunk; Huber + focal",
            {"reset_affinity_on_load": True},
            [1, 2, 3],
        ),
        (
            "M2",
            "regression-only Huber objective",
            {
                "classification_enabled": False,
                "train_ids_path": "revision/data/splits/train_regression_ids.txt",
                "validation_ids_path": (
                    "revision/data/splits/validation_regression_ids.txt"
                ),
            },
            [1],
        ),
        (
            "M3",
            "classification-only focal objective",
            {"regression_enabled": False},
            [1],
        ),
        (
            "M4",
            "MSE + focal loss",
            {"regression_loss": "mse"},
            [1],
        ),
        (
            "M5",
            "Huber + BCE loss",
            {"classification_loss": "bce"},
            [1],
        ),
        (
            "M6",
            "Huber + focal after excluding synthetic decoys from training only",
            {
                "train_ids_path": (
                    "revision/data/splits/train_no_synthetic_decoy_ids.txt"
                )
            },
            [1],
        ),
    ]

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for code, description, overrides, seeds in experiments:
        for seed in seeds:
            run_name = f"{code}_seed{seed}"
            parameters = common_parameters(seed, run_name)
            parameters.update(overrides)
            payload = {
                "experiment": code,
                "description": description,
                "parameters": parameters,
            }
            output = CONFIG_DIR / f"{run_name}.yaml"
            output.write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            outputs.append(output)
    return outputs


def main() -> None:
    no_decoy_ids = write_no_decoy_train_ids()
    configs = build_configs(no_decoy_ids)
    print(f"Wrote {len(configs)} configs to {CONFIG_DIR}")
    print(f"Wrote no-decoy training IDs to {no_decoy_ids}")


if __name__ == "__main__":
    main()
