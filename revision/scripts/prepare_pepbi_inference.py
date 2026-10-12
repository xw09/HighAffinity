#!/usr/bin/env python3
"""Prepare deduplicated Boltz inputs for PEPBI external inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "revision/external/pepbi/processed/pepbi_external_pair_disjoint_model_input.csv"
)
DEFAULT_OUT = ROOT / "revision/external/pepbi/inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def short_hash(*values: str, length: int = 16) -> str:
    payload = "\x1f".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    inputs_dir = out_dir / "inputs"
    msa_dir = out_dir / "receptor_msas"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    observations = pd.read_csv(input_path)
    required = {
        "id",
        "binding_group",
        "pepbi_complex_name",
        "pdb_id",
        "Protein_Sequence",
        "Peptide_Sequence",
        "logAffinity",
        "affinity_uM",
    }
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    pair_to_id: dict[tuple[str, str], str] = {}
    protein_to_id: dict[str, str] = {}
    for protein, peptide in observations[
        ["Protein_Sequence", "Peptide_Sequence"]
    ].itertuples(index=False, name=None):
        pair_to_id.setdefault(
            (protein, peptide), f"PEPBI_PAIR_{short_hash(protein, peptide)}"
        )
        protein_to_id.setdefault(protein, f"PEPBI_REC_{short_hash(protein)}")

    mapping_rows: list[dict[str, object]] = []
    for row in observations.to_dict(orient="records"):
        protein = str(row["Protein_Sequence"])
        peptide = str(row["Peptide_Sequence"])
        mapping_rows.append(
            {
                **row,
                "inference_id": pair_to_id[(protein, peptide)],
                "receptor_msa_id": protein_to_id[protein],
            }
        )

    mapping = pd.DataFrame(mapping_rows)
    mapping_path = out_dir / "observation_to_inference_mapping.csv"
    mapping.to_csv(mapping_path, index=False)

    receptor_rows = [
        {
            "receptor_msa_id": msa_id,
            "Protein_Sequence": protein,
            "sequence_length": len(protein),
            "msa_path": str((msa_dir / f"{msa_id}.csv").resolve()),
        }
        for protein, msa_id in sorted(protein_to_id.items(), key=lambda item: item[1])
    ]
    receptor_path = out_dir / "receptor_msa_requests.csv"
    with receptor_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(receptor_rows[0]))
        writer.writeheader()
        writer.writerows(receptor_rows)

    pair_rows: list[dict[str, object]] = []
    for (protein, peptide), inference_id in sorted(
        pair_to_id.items(), key=lambda item: item[1]
    ):
        msa_id = protein_to_id[protein]
        msa_path = (msa_dir / f"{msa_id}.csv").resolve()
        payload = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": protein,
                        "msa": str(msa_path),
                    }
                },
                {
                    "protein": {
                        "id": "B",
                        "sequence": peptide,
                        "msa": "empty",
                    }
                },
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        yaml_path = inputs_dir / f"{inference_id}.yaml"
        yaml_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        pair_rows.append(
            {
                "inference_id": inference_id,
                "receptor_msa_id": msa_id,
                "Protein_Sequence": protein,
                "Peptide_Sequence": peptide,
                "protein_length": len(protein),
                "peptide_length": len(peptide),
                "yaml_path": str(yaml_path.resolve()),
            }
        )

    pair_manifest = out_dir / "unique_pair_manifest.csv"
    with pair_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    ids_path = out_dir / "inference_ids.txt"
    ids_path.write_text(
        "".join(f"{row['inference_id']}\n" for row in pair_rows), encoding="utf-8"
    )

    summary = {
        "source": str(input_path),
        "observations": len(mapping),
        "unique_sequence_pairs": len(pair_rows),
        "unique_receptors": len(receptor_rows),
        "unique_peptides": int(mapping["Peptide_Sequence"].nunique()),
        "binding_groups": int(mapping["binding_group"].nunique()),
        "inputs_dir": str(inputs_dir),
        "mapping": str(mapping_path),
        "pair_manifest": str(pair_manifest),
        "receptor_msa_requests": str(receptor_path),
    }
    (out_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
