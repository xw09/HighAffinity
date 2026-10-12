#!/usr/bin/env python3
"""Run fixed-split affinity inference for one completed experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.module.trainingv2_test import (
    Boltz2InferenceDataModule,
    DataConfig,
    DatasetConfig,
    DefaultUniformSampler,
)
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.write.writer import BoltzAffinityWriter
from boltz.model.models.boltz2_test import Boltz2


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DIR = Path(
    os.environ.get("HIGH_AFFINITY_TEST_DIR", "data/processed/test")
)
DEFAULT_TEST_IDS = ROOT / "revision/data/splits/test_ids.txt"
DEFAULT_VALIDATION_DIR = Path(
    os.environ.get("HIGH_AFFINITY_PROCESSED_DIR", "data/processed/train")
)
DEFAULT_VALIDATION_IDS = ROOT / "revision/data/splits/validation_ids.txt"
DEFAULT_SPLIT_MANIFEST = ROOT / "revision/data/splits/split_manifest_v2.csv"


@dataclass
class Boltz2DiffusionParams:
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True


@dataclass
class PairformerArgsV2:
    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True


@dataclass
class MSAModuleArgs:
    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--evaluation-split",
        choices=["validation", "test"],
        default="test",
        help="Validation is ungated; final test inference requires acknowledgement.",
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--test-ids", type=Path, default=DEFAULT_TEST_IDS)
    parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument(
        "--acknowledge-test-set-is-final",
        action="store_true",
        help="Required guard confirming that checkpoint selection is complete.",
    )
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def valid_prediction(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
        return math.isfinite(float(payload["affinity_pred_value"])) and math.isfinite(
            float(payload["affinity_probability_binary"])
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def prediction_path(records_dir: Path, record_id: str) -> Path:
    return records_dir / record_id / f"affinity_{record_id}.json"


def prepare_data(
    test_dir: Path,
    cache: Path,
    ids_path: Path,
    num_workers: int,
) -> DataConfig:
    processed_dir = test_dir / "processed"
    dataset = DatasetConfig(
        target_dir=str(test_dir / "predictions"),
        msa_dir=str(processed_dir / "msa"),
        mol_dir=str(cache / "mols"),
        num_workers=num_workers,
        sampler=DefaultUniformSampler(),
        constraints_dir=(
            processed_dir / "constraints"
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            processed_dir / "templates"
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            processed_dir / "mols" if (processed_dir / "mols").exists() else None
        ),
        override_method=None,
        affinity=True,
        manifest_path=str(processed_dir / "manifest.json"),
        filters=None,
        split=None,
        ids_path=str(ids_path),
    )
    return DataConfig(
        datasets=[dataset],
        filters=[],
        featurizer=Boltz2Featurizer(),
        tokenizer=Boltz2Tokenizer(),
        val_batch_size=1,
    )


def run_inference(args: argparse.Namespace, pending_ids_path: Path) -> None:
    predict_args = {
        "recycling_steps": 5,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    model = Boltz2.load_from_checkpoint(
        args.checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs()),
        steering_args={"fk_steering": False, "guidance_update": False},
        affinity_mw_correction=False,
        skip_run_structure=True,
        affinity_prediction=True,
    )
    model.eval()
    model.requires_grad_(False)

    data_config = prepare_data(
        args.test_dir, args.cache, pending_ids_path, args.num_workers
    )
    data_module = Boltz2InferenceDataModule(data_config)
    data_module.batchsize = args.batch_size
    data_module.num_workers = args.num_workers

    writer = BoltzAffinityWriter(
        data_dir=args.test_dir / "processed/structures",
        output_dir=args.output_dir / "records",
    )
    trainer = Trainer(
        default_root_dir=args.output_dir,
        strategy="auto",
        accelerator="gpu",
        devices=1,
        logger=False,
        precision=args.precision,
        callbacks=[writer],
    )
    trainer.predict(model, datamodule=data_module, return_predictions=False)


def safe_float(value: str) -> Optional[float]:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def summarize_predictions(
    args: argparse.Namespace,
    config: dict,
    test_ids: list[str],
) -> dict:
    records_dir = args.output_dir / "records"
    with args.split_manifest.open(newline="", encoding="utf-8") as handle:
        truth = {row["id"]: row for row in csv.DictReader(handle)}

    output_rows = []
    for record_id in test_ids:
        payload = json.loads(prediction_path(records_dir, record_id).read_text())
        row = truth[record_id]
        output_rows.append(
            {
                "id": record_id,
                "receptor_group": row["receptor_group"],
                "negative_type": row["negative_type_clean"],
                "class_true": int(float(row["class_clean"])),
                "class_probability": float(
                    payload["affinity_probability_binary"]
                ),
                "class_pred_0.5": int(
                    float(payload["affinity_probability_binary"]) >= 0.5
                ),
                "regression_eligible": row["regression_eligible"],
                "affinity_true": safe_float(row["logAffinity"]),
                "affinity_pred": float(payload["affinity_pred_value"]),
            }
        )

    csv_path = args.output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    parameters = config["parameters"]
    metrics: dict[str, object] = {
        "experiment": config["experiment"],
        "checkpoint": str(args.checkpoint),
        "evaluation_split": getattr(args, "evaluation_split", "test"),
        "classification_threshold": 0.5,
        "evaluation_records": len(output_rows),
        # Retained for compatibility with the original stage-5 output schema.
        "test_records": len(output_rows),
    }
    if parameters["classification_enabled"]:
        y_true = np.asarray([row["class_true"] for row in output_rows])
        y_prob = np.asarray([row["class_probability"] for row in output_rows])
        y_pred = np.asarray([row["class_pred_0.5"] for row in output_rows])
        negatives = y_true == 0
        metrics["classification"] = {
            "n": int(y_true.size),
            "positive": int(y_true.sum()),
            "negative": int(negatives.sum()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "specificity": float(np.mean(y_pred[negatives] == 0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }
        subgroup_rows = []
        negative_types = sorted(
            {row["negative_type"] for row in output_rows if row["class_true"] == 0}
        )
        for negative_type in negative_types:
            subset = [
                row
                for row in output_rows
                if row["class_true"] == 0 and row["negative_type"] == negative_type
            ]
            subgroup_rows.append(
                {
                    "negative_type": negative_type or "unspecified",
                    "n": len(subset),
                    "specificity": float(
                        np.mean([row["class_pred_0.5"] == 0 for row in subset])
                    ),
                    "mean_positive_probability": float(
                        np.mean([row["class_probability"] for row in subset])
                    ),
                }
            )
        with (args.output_dir / "negative_subgroup_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(subgroup_rows[0]))
            writer.writeheader()
            writer.writerows(subgroup_rows)
    else:
        metrics["classification"] = {"status": "task_disabled"}

    if parameters["regression_enabled"]:
        regression_rows = [
            row for row in output_rows if row["affinity_true"] is not None
        ]
        y_true = np.asarray([row["affinity_true"] for row in regression_rows])
        y_pred = np.asarray([row["affinity_pred"] for row in regression_rows])
        metrics["regression"] = {
            "n": int(y_true.size),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
            "pearson": float(pearsonr(y_true, y_pred).statistic),
            "spearman": float(spearmanr(y_true, y_pred).statistic),
        }
    else:
        metrics["regression"] = {"status": "task_disabled"}

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    args = parse_args()
    if (
        args.evaluation_split == "test"
        and not args.acknowledge_test_set_is_final
    ):
        raise SystemExit(
            "Refusing test inference without --acknowledge-test-set-is-final."
        )
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.test_dir = args.test_dir.expanduser().resolve()
    args.test_ids = args.test_ids.expanduser().resolve()
    args.split_manifest = args.split_manifest.expanduser().resolve()
    args.cache = args.cache.expanduser().resolve()
    if args.evaluation_split == "validation":
        if args.test_ids != DEFAULT_VALIDATION_IDS.resolve():
            raise SystemExit(
                "Validation safety check failed: --test-ids must be the fixed "
                f"list at {DEFAULT_VALIDATION_IDS}."
            )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = args.output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    test_ids = read_ids(args.test_ids)
    pending = [
        record_id
        for record_id in test_ids
        if not valid_prediction(prediction_path(records_dir, record_id))
    ]
    pending_path = args.output_dir / "pending_ids.txt"
    pending_path.write_text(
        "".join(f"{record_id}\n" for record_id in pending), encoding="utf-8"
    )

    state = {
        "evaluation_split": args.evaluation_split,
        "checkpoint": str(args.checkpoint),
        "source_config": str(args.config),
        "expected_records": len(test_ids),
        "pending_before_run": len(pending),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "evaluation_state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if pending:
        run_inference(args, pending_path)

    missing = [
        record_id
        for record_id in test_ids
        if not valid_prediction(prediction_path(records_dir, record_id))
    ]
    if missing:
        (args.output_dir / "missing_ids.txt").write_text(
            "".join(f"{record_id}\n" for record_id in missing), encoding="utf-8"
        )
        raise RuntimeError(f"Prediction incomplete: {len(missing)} records missing.")

    metrics = summarize_predictions(args, config, test_ids)
    marker = {
        **state,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "metrics_file": str(args.output_dir / "metrics.json"),
        "metrics": metrics,
    }
    (args.output_dir / "EVALUATION_COMPLETE.json").write_text(
        json.dumps(marker, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
