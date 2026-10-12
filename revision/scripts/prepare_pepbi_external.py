#!/usr/bin/env python3
"""Clean PEPBI and audit overlap with the internal HighAffinity dataset.

PEPBI contains 329 structural rows, 181 named complexes, and a small number of
named complexes with measurements at multiple temperatures or replicate
affinities.  This script preserves the structural rows and collapses only
exactly repeated thermodynamic observations.  Correlated observations remain
identifiable through ``pepbi_complex_name`` and ``binding_group`` so downstream
confidence intervals can cluster at the appropriate level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "revision/external/pepbi/raw/PEPBI.xlsx"
DEFAULT_MANIFEST = ROOT / "revision/data/splits/split_manifest_v2.csv"
DEFAULT_OUTPUT = ROOT / "revision/external/pepbi/processed"
CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
GAS_CONSTANT_KCAL = 0.00198720425864083
THERMODYNAMIC_CONSISTENCY_TOLERANCE_LOG10 = 0.05


COLUMN_CANDIDATES = {
    "binding_group": ("bindinggroup",),
    "pepbi_complex_name": ("pepbicomplexname", "complexname"),
    "pdb_id": ("pdbid",),
    "crystallographic_unit": ("crystallographicunit",),
    "unit_copy": ("unitcopy",),
    "protein_name": ("proteinname",),
    "protein_sequence": ("proteinsequence",),
    "peptide_name": ("peptidename",),
    "peptide_sequence": ("peptidesequence",),
    "binding_site": ("bindingsite",),
    "kd_m": ("kdm", "kdmolar"),
    "dg_kcal_mol": ("dgkcalmol",),
    "calculated_dg_kcal_mol": ("calculateddgkcalmol",),
    "temperature_k": ("tk", "temperaturek"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--internal-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--internal-scope",
        choices=("all", "development"),
        default="all",
        help="'all' is the conservative primary analysis and also removes "
        "overlap with the internal final test set.",
    )
    parser.add_argument("--receptor-identity-cutoff", type=float, default=0.70)
    return parser.parse_args()


def normalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("δ", "d").replace("∆", "d").replace("Δ", "d")
    text = text.replace("α", "a").replace("β", "b")
    return re.sub(r"[^a-z0-9]+", "", text)


def locate_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    normalized = {normalize_column_name(column): column for column in frame.columns}
    result: dict[str, str | None] = {}
    for target, candidates in COLUMN_CANDIDATES.items():
        result[target] = next(
            (normalized[candidate] for candidate in candidates if candidate in normalized),
            None,
        )
    required = (
        "binding_group",
        "pepbi_complex_name",
        "pdb_id",
        "protein_sequence",
        "peptide_sequence",
        "temperature_k",
    )
    missing = [name for name in required if result[name] is None]
    if missing:
        raise ValueError(
            f"Could not locate required PEPBI columns {missing}. Available columns: "
            f"{list(frame.columns)}"
        )
    if result["kd_m"] is None and result["calculated_dg_kcal_mol"] is None:
        raise ValueError("PEPBI input must provide KD (M) or calculated dG.")
    return result


def read_pepbi(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"PEPBI source file not found: {path}\n"
            "Download PEPBI.xlsx from https://doi.org/10.5061/dryad.wstqjq2wk "
            "and place it at revision/external/pepbi/raw/PEPBI.xlsx."
        )
    if path.suffix.lower() in {".xlsx", ".xls"}:
        # The official workbook's first row contains merged category labels;
        # the second row contains the actual column names.
        frame = pd.read_excel(path, skiprows=1)
    else:
        frame = pd.read_csv(path, low_memory=False)
    frame = frame.dropna(how="all").reset_index(drop=True)
    return frame


def clean_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Za-z]", "", str(value)).upper()


def clean_pdb_id(value: object) -> str:
    if pd.isna(value):
        return ""
    match = re.search(r"[0-9][A-Za-z0-9]{3}", str(value))
    return match.group(0).upper() if match else str(value).strip().upper()


def numeric_column(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = frame[column].copy()
    missing_marker = values.astype(str).str.strip().isin(("X", "x", ""))
    values = values.mask(missing_marker)
    return pd.to_numeric(values, errors="coerce")


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    columns = locate_columns(frame)
    output = pd.DataFrame(index=frame.index)
    text_fields = (
        "binding_group",
        "pepbi_complex_name",
        "pdb_id",
        "crystallographic_unit",
        "unit_copy",
        "protein_name",
        "protein_sequence",
        "peptide_name",
        "peptide_sequence",
        "binding_site",
    )
    for field in text_fields:
        source = columns[field]
        output[field] = frame[source] if source is not None else ""
    output["protein_sequence"] = output["protein_sequence"].map(clean_sequence)
    output["peptide_sequence"] = output["peptide_sequence"].map(clean_sequence)
    output["pdb_id"] = output["pdb_id"].map(clean_pdb_id)
    output["kd_m"] = numeric_column(frame, columns["kd_m"])
    output["dg_kcal_mol_reported"] = numeric_column(frame, columns["dg_kcal_mol"])
    output["dg_kcal_mol_calculated"] = numeric_column(
        frame, columns["calculated_dg_kcal_mol"]
    )
    output["temperature_k"] = numeric_column(frame, columns["temperature_k"])

    valid_kd = output["kd_m"].gt(0)
    output["affinity_source"] = np.where(valid_kd, "KD_M", "calculated_dG")
    output["logAffinity"] = np.nan
    output.loc[valid_kd, "logAffinity"] = np.log10(
        output.loc[valid_kd, "kd_m"] * 1e6
    )
    derive = ~valid_kd & output["dg_kcal_mol_calculated"].notna() & output[
        "temperature_k"
    ].gt(0)
    output.loc[derive, "logAffinity"] = (
        output.loc[derive, "dg_kcal_mol_calculated"]
        / (
            GAS_CONSTANT_KCAL
            * output.loc[derive, "temperature_k"]
            * math.log(10)
        )
        + 6.0
    )
    output["affinity_uM"] = 10 ** output["logAffinity"]

    dg_check = valid_kd & output["temperature_k"].gt(0)
    output["logAffinity_from_kd_minus_dg"] = np.nan
    dg_value = output["dg_kcal_mol_calculated"].where(
        output["dg_kcal_mol_calculated"].notna(),
        output["dg_kcal_mol_reported"],
    )
    dg_check &= dg_value.notna()
    log_from_dg = (
        dg_value[dg_check]
        / (
            GAS_CONSTANT_KCAL
            * output.loc[dg_check, "temperature_k"]
            * math.log(10)
        )
        + 6.0
    )
    output.loc[dg_check, "logAffinity_from_kd_minus_dg"] = (
        output.loc[dg_check, "logAffinity"] - log_from_dg
    )

    output["protein_sequence_valid"] = output["protein_sequence"].map(
        lambda sequence: bool(sequence) and set(sequence) <= CANONICAL_AMINO_ACIDS
    )
    output["peptide_sequence_valid"] = output["peptide_sequence"].map(
        lambda sequence: bool(sequence) and set(sequence) <= CANONICAL_AMINO_ACIDS
    )
    # The official workbook contains one grossly inconsistent KD/Delta-G row.
    # Preserve it in the structural audit, but never silently repair or use it
    # as a regression label. Missing cross-check values remain eligible.
    output["thermodynamic_consistency_valid"] = (
        output["logAffinity_from_kd_minus_dg"].isna()
        | output["logAffinity_from_kd_minus_dg"]
        .abs()
        .le(THERMODYNAMIC_CONSISTENCY_TOLERANCE_LOG10)
    )
    output["affinity_valid"] = np.isfinite(output["logAffinity"]) & output[
        "thermodynamic_consistency_valid"
    ]
    output["source_row"] = np.arange(2, len(output) + 2)
    return output


def make_experiment_key(row: pd.Series) -> str:
    fields = (
        str(row["binding_group"]),
        row["protein_sequence"],
        row["peptide_sequence"],
        str(row["binding_site"]),
        f"{row['temperature_k']:.8g}",
        f"{row['logAffinity']:.10g}",
    )
    digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()[:16]
    return f"PEPBI_EXP_{digest}"


def deduplicate_experimental_complexes(
    structures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = structures[
        structures["protein_sequence_valid"]
        & structures["peptide_sequence_valid"]
        & structures["affinity_valid"]
    ].copy()
    valid["experimental_complex_id"] = valid.apply(make_experiment_key, axis=1)
    counts = valid["experimental_complex_id"].value_counts()
    valid["structure_copy_count"] = valid["experimental_complex_id"].map(counts)
    valid["all_pepbi_complex_names"] = valid.groupby("experimental_complex_id")[
        "pepbi_complex_name"
    ].transform(lambda values: "|".join(sorted(set(map(str, values)))))
    deduplicated = (
        valid.sort_values(
            ["experimental_complex_id", "pepbi_complex_name", "source_row"]
        )
        .drop_duplicates("experimental_complex_id", keep="first")
        .reset_index(drop=True)
    )
    return valid, deduplicated


def normalized_edit_identity(left: str, right: str) -> float:
    try:
        import jellyfish

        distance = jellyfish.levenshtein_distance(left, right)
    except ImportError:
        # Memory-efficient fallback. The project environment normally has the
        # C-backed jellyfish package, so this branch is mainly for portability.
        if len(left) < len(right):
            left, right = right, left
        previous = list(range(len(right) + 1))
        for index, char_left in enumerate(left, start=1):
            current = [index]
            for offset, char_right in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[offset] + 1,
                        previous[offset - 1] + (char_left != char_right),
                    )
                )
            previous = current
        distance = previous[-1]
    return 1.0 - distance / max(len(left), len(right))


def nearest_internal_receptor(
    query: str,
    internal_sequences: list[str],
    cutoff: float,
) -> tuple[float, str]:
    best_identity = -1.0
    best_sequence = ""
    for candidate in internal_sequences:
        # Under normalized global edit identity, a large length mismatch cannot
        # reach the cutoff and can be skipped exactly.
        if min(len(query), len(candidate)) / max(len(query), len(candidate)) < cutoff:
            continue
        identity = normalized_edit_identity(query, candidate)
        if identity > best_identity:
            best_identity = identity
            best_sequence = candidate
            if identity == 1.0:
                break
    return max(best_identity, 0.0), best_sequence


def audit_overlap(
    external: pd.DataFrame,
    internal_manifest: Path,
    scope: str,
    cutoff: float,
) -> pd.DataFrame:
    internal = pd.read_csv(internal_manifest, low_memory=False)
    if scope == "development":
        internal = internal[internal["split_clean"].isin(["train", "validation"])]
    else:
        internal = internal[
            internal["split_clean"].isin(["train", "validation", "test"])
        ]
    internal = internal.copy()
    internal["Protein_Sequence"] = internal["Protein_Sequence"].map(clean_sequence)
    internal["Peptide_Sequence"] = internal["Peptide_Sequence"].map(clean_sequence)
    internal["PDB_ID_clean"] = internal["PDB_ID"].map(clean_pdb_id)
    protein_set = set(internal["Protein_Sequence"].dropna()) - {""}
    peptide_set = set(internal["Peptide_Sequence"].dropna()) - {""}
    pdb_set = set(internal["PDB_ID_clean"].dropna()) - {""}
    pair_set = set(zip(internal["Protein_Sequence"], internal["Peptide_Sequence"]))
    internal_sequences = sorted(protein_set)

    audit = external.copy()
    audit["overlap_exact_pdb"] = audit["pdb_id"].isin(pdb_set)
    audit["overlap_exact_receptor"] = audit["protein_sequence"].isin(protein_set)
    audit["overlap_exact_peptide"] = audit["peptide_sequence"].isin(peptide_set)
    audit["overlap_exact_pair"] = [
        pair in pair_set
        for pair in zip(audit["protein_sequence"], audit["peptide_sequence"])
    ]
    nearest = {}
    for sequence in sorted(audit["protein_sequence"].unique()):
        nearest[sequence] = nearest_internal_receptor(
            sequence, internal_sequences, cutoff
        )
    audit["max_internal_receptor_identity"] = audit["protein_sequence"].map(
        lambda sequence: nearest[sequence][0]
    )
    audit["nearest_internal_receptor_sequence"] = audit["protein_sequence"].map(
        lambda sequence: nearest[sequence][1]
    )
    audit["overlap_receptor_identity_ge_cutoff"] = audit[
        "max_internal_receptor_identity"
    ].ge(cutoff)
    audit["exclude_receptor_cold"] = audit[
        [
            "overlap_exact_pdb",
            "overlap_exact_pair",
            "overlap_receptor_identity_ge_cutoff",
        ]
    ].any(axis=1)
    audit["exclude_strict"] = audit[
        [
            "overlap_exact_pdb",
            "overlap_exact_receptor",
            "overlap_exact_peptide",
            "overlap_exact_pair",
            "overlap_receptor_identity_ge_cutoff",
        ]
    ].any(axis=1)
    # These less restrictive cohorts are useful for transparent sensitivity
    # analyses when a fully receptor-cold cohort is unavailable.  They must not
    # be described as receptor-cold: PEPBI reuses many receptor families and
    # source structures represented in the internal dataset.
    audit["exclude_pair_disjoint"] = audit["overlap_exact_pair"]
    audit["exclude_pdb_disjoint"] = audit[
        ["overlap_exact_pdb", "overlap_exact_pair"]
    ].any(axis=1)
    return audit


def model_input(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[
        [
            "experimental_complex_id",
            "binding_group",
            "pepbi_complex_name",
            "pdb_id",
            "protein_sequence",
            "peptide_sequence",
            "logAffinity",
            "affinity_uM",
            "temperature_k",
            "structure_copy_count",
        ]
    ].copy()
    result = result.rename(
        columns={
            "experimental_complex_id": "id",
            "protein_sequence": "Protein_Sequence",
            "peptide_sequence": "Peptide_Sequence",
        }
    )
    result.insert(1, "source", "PEPBI")
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.internal_manifest = args.internal_manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not 0 < args.receptor_identity_cutoff <= 1:
        raise ValueError("--receptor-identity-cutoff must be in (0, 1].")

    raw = read_pepbi(args.input)
    structures = canonicalize(raw)
    valid_structures, experiments = deduplicate_experimental_complexes(structures)
    audit = audit_overlap(
        experiments,
        args.internal_manifest,
        args.internal_scope,
        args.receptor_identity_cutoff,
    )
    receptor_cold = audit[~audit["exclude_receptor_cold"]].copy()
    strict = audit[~audit["exclude_strict"]].copy()
    pair_disjoint = audit[~audit["exclude_pair_disjoint"]].copy()
    pdb_disjoint = audit[~audit["exclude_pdb_disjoint"]].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "structure_rows": args.output_dir / "pepbi_structure_rows_cleaned.csv",
        "eligible_structure_rows": args.output_dir / "pepbi_structure_rows_eligible.csv",
        "excluded_rows": args.output_dir / "pepbi_rows_excluded_by_qc.csv",
        "experimental_complexes": args.output_dir / "pepbi_experimental_complexes.csv",
        "overlap_audit": args.output_dir / "pepbi_overlap_audit.csv",
        "receptor_cold": args.output_dir / "pepbi_external_receptor_cold.csv",
        "strict": args.output_dir / "pepbi_external_strict.csv",
        "pair_disjoint": args.output_dir / "pepbi_external_pair_disjoint.csv",
        "pdb_disjoint": args.output_dir / "pepbi_external_pdb_disjoint.csv",
        "receptor_cold_model_input": args.output_dir / "pepbi_external_receptor_cold_model_input.csv",
        "strict_model_input": args.output_dir / "pepbi_external_strict_model_input.csv",
        "pair_disjoint_model_input": args.output_dir / "pepbi_external_pair_disjoint_model_input.csv",
        "pdb_disjoint_model_input": args.output_dir / "pepbi_external_pdb_disjoint_model_input.csv",
    }
    structures.to_csv(outputs["structure_rows"], index=False)
    valid_structures.to_csv(outputs["eligible_structure_rows"], index=False)
    structures[~structures["affinity_valid"]].to_csv(
        outputs["excluded_rows"], index=False
    )
    experiments.to_csv(outputs["experimental_complexes"], index=False)
    audit.to_csv(outputs["overlap_audit"], index=False)
    receptor_cold.to_csv(outputs["receptor_cold"], index=False)
    strict.to_csv(outputs["strict"], index=False)
    pair_disjoint.to_csv(outputs["pair_disjoint"], index=False)
    pdb_disjoint.to_csv(outputs["pdb_disjoint"], index=False)
    model_input(receptor_cold).to_csv(outputs["receptor_cold_model_input"], index=False)
    model_input(strict).to_csv(outputs["strict_model_input"], index=False)
    model_input(pair_disjoint).to_csv(outputs["pair_disjoint_model_input"], index=False)
    model_input(pdb_disjoint).to_csv(outputs["pdb_disjoint_model_input"], index=False)

    discrepancy = structures["logAffinity_from_kd_minus_dg"].abs().gt(0.05)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "PEPBI",
            "doi": "10.5061/dryad.wstqjq2wk",
            "paper_doi": "10.1038/s41597-025-05754-7",
            "license": "CC0-1.0",
            "input_path": str(args.input),
            "input_sha256": sha256(args.input),
        },
        "label_definition": "log10(KD / micromolar)",
        "gas_constant_kcal_mol_K": GAS_CONSTANT_KCAL,
        "thermodynamic_consistency_tolerance_log10": (
            THERMODYNAMIC_CONSISTENCY_TOLERANCE_LOG10
        ),
        "internal_overlap_scope": args.internal_scope,
        "receptor_identity_definition": "1 - Levenshtein_distance / max(sequence_lengths)",
        "receptor_identity_cutoff": args.receptor_identity_cutoff,
        "raw_rows": int(len(raw)),
        "cleaned_structure_rows": int(len(structures)),
        "valid_structure_rows": int(len(valid_structures)),
        "invalid_sequence_or_affinity_rows": int(len(structures) - len(valid_structures)),
        "unique_pepbi_complex_names": int(experiments["pepbi_complex_name"].nunique()),
        "deduplicated_thermodynamic_observations": int(len(experiments)),
        # Backward-compatible name retained for existing table scripts.
        "deduplicated_experimental_complexes": int(len(experiments)),
        "receptor_cold_external_complexes": int(len(receptor_cold)),
        "strict_external_complexes": int(len(strict)),
        "pair_disjoint_external_observations": int(len(pair_disjoint)),
        "pair_disjoint_external_complex_names": int(
            pair_disjoint["pepbi_complex_name"].nunique()
        ),
        "pair_disjoint_binding_groups": int(pair_disjoint["binding_group"].nunique()),
        "pdb_disjoint_external_observations": int(len(pdb_disjoint)),
        "pdb_disjoint_external_complex_names": int(
            pdb_disjoint["pepbi_complex_name"].nunique()
        ),
        "pdb_disjoint_binding_groups": int(pdb_disjoint["binding_group"].nunique()),
        "kd_vs_dg_discrepancy_gt_0.05_log_units": int(discrepancy.sum()),
        "overlap_counts": {
            column: int(audit[column].sum())
            for column in (
                "overlap_exact_pdb",
                "overlap_exact_receptor",
                "overlap_exact_peptide",
                "overlap_exact_pair",
                "overlap_receptor_identity_ge_cutoff",
                "exclude_receptor_cold",
                "exclude_strict",
                "exclude_pair_disjoint",
                "exclude_pdb_disjoint",
            )
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    (args.output_dir / "pepbi_cleaning_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
