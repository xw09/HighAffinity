#!/usr/bin/env python3
"""Force every PEPBI Boltz YAML chain into a local single-sequence protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = ROOT / "revision/external/pepbi/inference/inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs_dir = args.inputs_dir.expanduser().resolve()
    paths = sorted([*inputs_dir.glob("*.yaml"), *inputs_dir.glob("*.yml")])
    if not paths:
        raise FileNotFoundError(f"No YAML inputs found in {inputs_dir}")
    chain_count = 0
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for sequence in payload.get("sequences", []):
            protein = sequence.get("protein")
            if protein is not None:
                protein["msa"] = "empty"
                chain_count += 1
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    marker = {
        "protocol": "Boltz2 single-sequence mode for every protein and peptide chain",
        "reason": (
            "No local MMseqs2 sequence database was available, and PEPBI sequences "
            "were not submitted to an external MSA service."
        ),
        "yaml_inputs": len(paths),
        "protein_chains_set_to_empty_msa": chain_count,
        "limitation": (
            "The external inference MSA condition differs from the precomputed "
            "internal benchmark structures and can reduce structure accuracy."
        ),
    }
    marker_path = inputs_dir.parent / "single_sequence_protocol.json"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
