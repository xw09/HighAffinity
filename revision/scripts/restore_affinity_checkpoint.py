#!/usr/bin/env python3
"""Merge a published HighAffinity head bundle into the Boltz2 base checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--affinity-head", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = args.base_checkpoint.expanduser().resolve()
    head_path = args.affinity_head.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if not head_path.is_file():
        raise FileNotFoundError(head_path)

    checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    bundle = torch.load(head_path, map_location="cpu", weights_only=False)
    if bundle.get("format") != "highaffinity-affinity-head-v1":
        raise ValueError(f"Unsupported affinity-head bundle: {head_path}")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"No Lightning state_dict found in {base_path}")
    missing = sorted(set(bundle["state_dict"]) - set(state_dict))
    if missing:
        raise ValueError(f"Base checkpoint is missing {len(missing)} tensor keys")
    state_dict.update(bundle["state_dict"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Restored {bundle['run']} checkpoint: {output_path}")


if __name__ == "__main__":
    main()
