#!/usr/bin/env python3
"""Create portable run configurations and affinity-head checkpoint bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG_DIR = ROOT / "revision/configs/stage4"
PUBLIC_CONFIG_DIR = ROOT / "revision/configs/public"
RUNS_DIR = ROOT / "revision/runs"
PUBLIC_CHECKPOINT_DIR = ROOT / "revision/checkpoints/affinity_heads"
RUN_NAMES = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Only generate the portable YAML and resolved JSON files.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_checkpoint(run_name: str) -> Path:
    candidates = sorted(
        path
        for path in (RUNS_DIR / run_name / "checkpoints").glob("*.ckpt")
        if path.name != "last.ckpt"
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one selected checkpoint for {run_name}, found {candidates}"
        )
    return candidates[0]


def portable_parameters(run_name: str, parameters: dict[str, object]) -> dict:
    output = dict(parameters)
    output.update(
        {
            "out_dir": "${HIGH_AFFINITY_PROCESSED_DIR}",
            "ckpt_path": f"revision/runs/{run_name}",
            "cache": "${BOLTZ_CACHE}",
            "pretrained_ckpt": "${BOLTZ_CACHE}/boltz2_aff.ckpt",
            "validation_ids_path": str(
                Path(str(parameters["validation_ids_path"])).relative_to(ROOT)
            ),
            "train_ids_path": str(
                Path(str(parameters["train_ids_path"])).relative_to(ROOT)
            ),
            "resume_ckpt_path": None,
        }
    )
    return output


def write_configs() -> None:
    PUBLIC_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for run_name in RUN_NAMES:
        source_path = SOURCE_CONFIG_DIR / f"{run_name}.yaml"
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        source["parameters"] = portable_parameters(
            run_name, source["parameters"]
        )
        destination = PUBLIC_CONFIG_DIR / source_path.name
        destination.write_text(
            yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
        )

        resolved_path = RUNS_DIR / run_name / "resolved_config.json"
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        resolved["source_config"] = f"revision/configs/public/{run_name}.yaml"
        resolved["parameters"] = portable_parameters(
            run_name, resolved["parameters"]
        )
        (PUBLIC_CONFIG_DIR / f"{run_name}.resolved.json").write_text(
            json.dumps(resolved, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def write_checkpoint_bundles() -> None:
    PUBLIC_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for run_name in RUN_NAMES:
        source_path = selected_checkpoint(run_name)
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        affinity_state = {
            key: tensor
            for key, tensor in checkpoint["state_dict"].items()
            if key.startswith(("affinity_module1.", "affinity_module2."))
        }
        if not affinity_state:
            raise RuntimeError(f"No affinity-module tensors found in {source_path}")
        bundle_path = PUBLIC_CHECKPOINT_DIR / f"{run_name}.affinity_head.pt"
        torch.save(
            {
                "format": "highaffinity-affinity-head-v1",
                "run": run_name,
                "source_checkpoint": source_path.name,
                "base_checkpoint": "Boltz2 boltz2_aff.ckpt",
                "state_dict": affinity_state,
            },
            bundle_path,
        )
        manifest_rows.append(
            {
                "run": run_name,
                "bundle": bundle_path.name,
                "source_checkpoint": source_path.name,
                "tensor_count": len(affinity_state),
                "size_bytes": bundle_path.stat().st_size,
                "sha256": sha256(bundle_path),
            }
        )
        del checkpoint, affinity_state
        print(f"Exported {run_name}: {bundle_path}", flush=True)

    (PUBLIC_CHECKPOINT_DIR / "manifest.json").write_text(
        json.dumps(manifest_rows, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    write_configs()
    if not args.skip_checkpoints:
        write_checkpoint_bundles()


if __name__ == "__main__":
    main()
