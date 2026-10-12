#!/usr/bin/env python3
"""Build the audited HighAffinity revision manifest from the frozen split files."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


BINDER_THRESHOLD_UM = 1000.0

ASSAY_RE = re.compile(r"(?i)(IC50|EC50|Kd|Ki|Km)")
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
UNIT_RE = re.compile(r"(?i)(pM|nM|uM|mM|M)\b")


def normalize_affinity_text(value: object) -> str | None:
    """Normalize symbols without changing the meaning of the source string."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return (
        text.replace("μ", "u")
        .replace("µ", "u")
        .replace("≥", ">=")
        .replace("≤", "<=")
        .replace(" ", "")
    )


def unit_multiplier_to_um(unit: str | None) -> float | None:
    if unit is None:
        return None
    return {
        "pm": 1e-6,
        "nm": 1e-3,
        "um": 1.0,
        "mm": 1e3,
        "m": 1e6,
    }.get(unit.lower())


def parse_affinity(value: object) -> dict[str, object]:
    """Parse assay, censoring operator, numeric bound, and unit from Affinity."""
    text = normalize_affinity_text(value)
    result: dict[str, object] = {
        "affinity_normalized": text,
        "assay_type": None,
        "operator": None,
        "bound_value": np.nan,
        "bound_unit": None,
        "affinity_uM_parsed": np.nan,
        "range_low_uM": np.nan,
        "range_high_uM": np.nan,
        "parse_status": "missing",
    }
    if text is None:
        return result

    assay_match = ASSAY_RE.search(text)
    if assay_match:
        result["assay_type"] = assay_match.group(1)

    # Handle expressions such as 4<IC50<6uM.
    range_match = re.fullmatch(
        r"(?P<low>[-+0-9.eE]+)<(?P<assay>IC50|EC50|Kd|Ki|Km)<(?P<high>[-+0-9.eE]+)(?P<unit>pM|nM|uM|mM|M)",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        multiplier = unit_multiplier_to_um(range_match.group("unit"))
        result.update(
            {
                "assay_type": range_match.group("assay"),
                "operator": "range",
                "bound_unit": range_match.group("unit"),
                "range_low_uM": float(range_match.group("low")) * multiplier,
                "range_high_uM": float(range_match.group("high")) * multiplier,
                "parse_status": "parsed_range",
            }
        )
        return result

    operator_match = re.search(r"(>>|>=|<=|>|<|=|~)", text)
    if not operator_match:
        result["parse_status"] = "missing_operator"
        return result

    operator = operator_match.group(1)
    if operator == ">>":
        operator = ">"
    result["operator"] = operator

    suffix = text[operator_match.end() :]
    number_match = NUMBER_RE.search(suffix)
    unit_match = UNIT_RE.search(suffix)
    if not number_match or not unit_match:
        result["parse_status"] = "unparsed_value_or_unit"
        return result

    number = float(number_match.group(0))
    unit = unit_match.group(1)
    multiplier = unit_multiplier_to_um(unit)
    result.update(
        {
            "bound_value": number,
            "bound_unit": unit,
            "affinity_uM_parsed": number * multiplier,
            "parse_status": "parsed",
        }
    )
    return result


def source_from_row(row: pd.Series) -> str:
    if pd.notna(row["Affinity"]):
        return "experimental_affinity"
    if pd.notna(row["PDB_ID"]):
        return "structure_derived"
    return "synthetic_decoy"


def clean_label(row: pd.Series) -> dict[str, object]:
    """Apply conservative 1000-uM classification and regression eligibility rules."""
    original = int(row["class"])
    exact_log = row["pAffinity"]
    source = row["sample_source"]

    if pd.notna(exact_log):
        clean = int(float(exact_log) <= math.log10(BINDER_THRESHOLD_UM))
        return {
            "class_clean": clean,
            "classification_eligible": True,
            "regression_eligible": True,
            "label_status": "exact_numeric",
            "label_action": "unchanged" if clean == original else "relabeled",
        }

    if source == "structure_derived":
        return {
            "class_clean": original,
            "classification_eligible": True,
            "regression_eligible": False,
            "label_status": "structure_derived",
            "label_action": "unchanged",
        }
    if source == "synthetic_decoy":
        return {
            "class_clean": original,
            "classification_eligible": True,
            "regression_eligible": False,
            "label_status": "synthetic_decoy",
            "label_action": "unchanged",
        }

    operator = row["operator"]
    bound_um = row["affinity_uM_parsed"]
    low_um = row["range_low_uM"]
    high_um = row["range_high_uM"]

    if operator in {">", ">="} and pd.notna(bound_um):
        if float(bound_um) >= BINDER_THRESHOLD_UM:
            clean = 0
            status = "unambiguous_lower_censored"
        else:
            clean = np.nan
            status = "ambiguous_lower_censored"
    elif operator in {"<", "<="} and pd.notna(bound_um):
        if float(bound_um) <= BINDER_THRESHOLD_UM:
            clean = 1
            status = "unambiguous_upper_censored"
        else:
            clean = np.nan
            status = "ambiguous_upper_censored"
    elif operator == "~" and pd.notna(bound_um):
        clean = int(float(bound_um) <= BINDER_THRESHOLD_UM)
        status = "approximate_numeric"
    elif operator == "=" and pd.notna(bound_um):
        clean = int(float(bound_um) <= BINDER_THRESHOLD_UM)
        status = "recoverable_exact_missing_logAffinity"
    elif operator == "range" and pd.notna(low_um) and pd.notna(high_um):
        if float(high_um) <= BINDER_THRESHOLD_UM:
            clean = 1
            status = "unambiguous_range"
        elif float(low_um) > BINDER_THRESHOLD_UM:
            clean = 0
            status = "unambiguous_range"
        else:
            clean = np.nan
            status = "ambiguous_range"
    else:
        clean = np.nan
        status = "unparsed_or_unknown_affinity"

    if pd.isna(clean):
        action = "exclude_from_classification"
        eligible = False
    else:
        clean = int(clean)
        action = "unchanged" if clean == original else "relabeled"
        eligible = True

    return {
        "class_clean": clean,
        "classification_eligible": eligible,
        "regression_eligible": False,
        "label_status": status,
        "label_action": action,
    }


def original_negative_type(row: pd.Series) -> str | None:
    if int(row["class_original"]) != 0:
        return None
    return {
        "experimental_affinity": "experimental_negative",
        "structure_derived": "bsa_structure_negative",
        "synthetic_decoy": "synthetic_decoy",
    }[row["sample_source"]]


def clean_negative_type(row: pd.Series) -> str | None:
    if pd.isna(row["class_clean"]) or int(row["class_clean"]) != 0:
        return None
    return {
        "experimental_affinity": "experimental_negative",
        "structure_derived": "bsa_structure_negative",
        "synthetic_decoy": "synthetic_decoy",
    }[row["sample_source"]]


def make_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [("trainval", df[df["split_original"] == "trainval"]), ("test", df[df["split_original"] == "test"]), ("overall", df)]
    for split, part in groups:
        rows.append(
            {
                "split": split,
                "records": len(part),
                "affinity_records": int(part["Affinity_raw"].notna().sum()),
                "exact_regression_records": int(part["regression_eligible"].sum()),
                "nonnumeric_or_censored_affinity_records": int((part["Affinity_raw"].notna() & ~part["regression_eligible"]).sum()),
                "classification_original_records": len(part),
                "classification_clean_records": int(part["classification_eligible"].sum()),
                "positive_original": int((part["class_original"] == 1).sum()),
                "negative_original": int((part["class_original"] == 0).sum()),
                "positive_clean": int((part["class_clean"] == 1).sum()),
                "negative_clean": int((part["class_clean"] == 0).sum()),
                "excluded_ambiguous": int((part["label_action"] == "exclude_from_classification").sum()),
                "relabeled": int((part["label_action"] == "relabeled").sum()),
                "receptor_clusters": int(part["prot_cluster"].nunique()),
                "peptide_clusters": int(part["pep_cluster"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "originals" / "splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for split in ("trainval", "test"):
        frame = pd.read_csv(args.input_dir / f"{split}.csv", low_memory=False)
        frame.insert(1, "split_original", split)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    if df["id"].duplicated().any():
        duplicates = df.loc[df["id"].duplicated(keep=False), "id"].tolist()
        raise ValueError(f"Duplicate IDs across frozen splits: {duplicates[:20]}")

    parsed = pd.DataFrame([parse_affinity(value) for value in df["Affinity"]])
    df = pd.concat([df.reset_index(drop=True), parsed], axis=1)
    df["sample_source"] = df.apply(source_from_row, axis=1)
    cleaned = pd.DataFrame([clean_label(row) for _, row in df.iterrows()])
    df = pd.concat([df, cleaned], axis=1)

    df = df.rename(
        columns={
            "Affinity": "Affinity_raw",
            "pAffinity": "logAffinity",
            "class": "class_original",
        }
    )
    df["negative_type_original"] = df.apply(original_negative_type, axis=1)
    df["negative_type_clean"] = df.apply(clean_negative_type, axis=1)
    df["BSA"] = np.nan

    preferred = [
        "id",
        "split_original",
        "Protein_Sequence",
        "Peptide_Sequence",
        "prot_len",
        "pep_len",
        "PDB_ID",
        "prot_cluster",
        "pep_cluster",
        "Affinity_raw",
        "affinity_normalized",
        "assay_type",
        "operator",
        "bound_value",
        "bound_unit",
        "affinity_uM_parsed",
        "logAffinity",
        "class_original",
        "class_clean",
        "classification_eligible",
        "regression_eligible",
        "label_status",
        "label_action",
        "sample_source",
        "negative_type_original",
        "negative_type_clean",
        "iptm",
        "BSA",
    ]
    remaining = [column for column in df.columns if column not in preferred]
    manifest = df[preferred + remaining]

    manifest.to_csv(args.output_dir / "final_manifest_v2.csv", index=False)

    affinity_audit = manifest[manifest["Affinity_raw"].notna()].copy()
    audit_columns = [
        "id",
        "split_original",
        "Affinity_raw",
        "affinity_normalized",
        "assay_type",
        "operator",
        "bound_value",
        "bound_unit",
        "affinity_uM_parsed",
        "logAffinity",
        "class_original",
        "class_clean",
        "classification_eligible",
        "regression_eligible",
        "label_status",
        "label_action",
    ]
    affinity_audit[audit_columns].to_csv(args.output_dir / "label_audit.csv", index=False)
    affinity_audit[~affinity_audit["regression_eligible"]][audit_columns].to_csv(
        args.output_dir / "censored_label_audit.csv", index=False
    )
    affinity_audit[affinity_audit["label_action"] != "unchanged"][audit_columns].to_csv(
        args.output_dir / "dataset_changes.csv", index=False
    )

    negative_columns = [
        "id",
        "split_original",
        "Protein_Sequence",
        "Peptide_Sequence",
        "PDB_ID",
        "Affinity_raw",
        "logAffinity",
        "class_original",
        "class_clean",
        "label_status",
        "label_action",
        "sample_source",
        "negative_type_original",
        "negative_type_clean",
    ]
    manifest[manifest["class_original"] == 0][negative_columns].to_csv(
        args.output_dir / "negative_type_audit.csv", index=False
    )

    statistics = make_statistics(manifest)
    statistics.to_csv(args.output_dir / "dataset_statistics.csv", index=False)

    decision_summary = (
        manifest.groupby(
            ["split_original", "label_status", "label_action"], dropna=False
        )
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split_original", "label_action", "label_status"])
    )
    decision_summary.to_csv(
        args.output_dir / "cleaning_decision_summary.csv", index=False
    )

    excluded_ids = manifest.loc[
        ~manifest["classification_eligible"], "id"
    ].astype(str)
    (args.output_dir / "excluded_classification_ids.txt").write_text(
        "\n".join(excluded_ids) + "\n", encoding="utf-8"
    )
    relabeled_ids = manifest.loc[
        manifest["label_action"] == "relabeled", "id"
    ].astype(str)
    (args.output_dir / "relabeled_ids.txt").write_text(
        "\n".join(relabeled_ids) + "\n", encoding="utf-8"
    )

    summary = {
        "binder_threshold_uM": BINDER_THRESHOLD_UM,
        "records": int(len(manifest)),
        "affinity_records": int(manifest["Affinity_raw"].notna().sum()),
        "exact_regression_records": int(manifest["regression_eligible"].sum()),
        "censored_or_nonnumeric_records": int(
            (manifest["Affinity_raw"].notna() & ~manifest["regression_eligible"]).sum()
        ),
        "classification_clean_records": int(manifest["classification_eligible"].sum()),
        "excluded_ambiguous_records": int(
            (manifest["label_action"] == "exclude_from_classification").sum()
        ),
        "relabeled_records": int((manifest["label_action"] == "relabeled").sum()),
        "label_status_counts": {
            str(key): int(value) for key, value in manifest["label_status"].value_counts().items()
        },
        "negative_type_original_counts": {
            str(key): int(value)
            for key, value in manifest["negative_type_original"].value_counts(dropna=True).items()
        },
        "negative_type_clean_counts": {
            str(key): int(value)
            for key, value in manifest["negative_type_clean"].value_counts(dropna=True).items()
        },
    }
    (args.output_dir / "data_quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
