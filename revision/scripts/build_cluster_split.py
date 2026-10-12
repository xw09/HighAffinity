#!/usr/bin/env python3
"""Create a fixed receptor-cluster-disjoint train/validation/test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


VALIDATION_FRACTION = 0.10
REGRESSION_QUANTILES = 5
DEFAULT_FIXED_VALIDATION_GROUPS = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "splits"
    / "validation_receptor_groups.txt"
)


def assign_strata(dev: pd.DataFrame) -> tuple[pd.Series, list[float]]:
    strata = pd.Series(index=dev.index, dtype="object")
    regression_mask = dev["regression_eligible"].astype(bool)
    quantile_labels = [f"reg_q{i + 1}" for i in range(REGRESSION_QUANTILES)]
    quantiles, bin_edges = pd.qcut(
        dev.loc[regression_mask, "logAffinity"],
        q=REGRESSION_QUANTILES,
        labels=quantile_labels,
        retbins=True,
        duplicates="drop",
    )
    strata.loc[regression_mask] = quantiles.astype(str)
    strata.loc[~regression_mask & dev["class_clean"].eq(1)] = "nonreg_positive"
    strata.loc[
        ~regression_mask
        & dev["negative_type_clean"].eq("experimental_negative")
    ] = "nonreg_experimental_negative"
    strata.loc[
        ~regression_mask
        & dev["negative_type_clean"].eq("bsa_structure_negative")
    ] = "bsa_structure_negative"
    strata.loc[
        ~regression_mask & dev["negative_type_clean"].eq("synthetic_decoy")
    ] = "synthetic_decoy"
    if strata.isna().any():
        missing = dev.loc[strata.isna(), "id"].tolist()
        raise ValueError(f"Could not assign validation strata for IDs: {missing[:20]}")
    return strata, [float(value) for value in bin_edges]


def merge_clusters_linked_by_exact_sequence(
    dev: pd.DataFrame,
) -> tuple[pd.Series, dict[int, int]]:
    """Merge nominal clusters when an exact receptor sequence links them."""
    clusters = [int(value) for value in dev["prot_cluster"].unique()]
    parent = {cluster: cluster for cluster in clusters}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        lower, upper = sorted((left_root, right_root))
        parent[upper] = lower

    sequence_clusters = dev.groupby("Protein_Sequence")["prot_cluster"].unique()
    for linked in sequence_clusters:
        linked = [int(value) for value in linked]
        for cluster in linked[1:]:
            union(linked[0], cluster)

    mapping = {cluster: find(cluster) for cluster in clusters}
    groups = dev["prot_cluster"].astype(int).map(mapping)
    return groups, mapping


def optimize_validation_clusters(
    dev: pd.DataFrame, strata: pd.Series
) -> tuple[np.ndarray, dict[str, object]]:
    work = dev.copy()
    work["validation_stratum"] = strata
    clusters = np.array(sorted(work["receptor_group"].unique()))
    stratum_names = sorted(work["validation_stratum"].unique())
    counts = (
        pd.crosstab(work["receptor_group"], work["validation_stratum"])
        .reindex(index=clusters, columns=stratum_names, fill_value=0)
        .astype(float)
    )

    n_clusters = len(clusters)
    n_strata = len(stratum_names)
    n_variables = n_clusters + 2 * n_strata
    target_clusters = round(n_clusters * VALIDATION_FRACTION)
    targets = np.array(
        [round(counts[name].sum() * VALIDATION_FRACTION) for name in stratum_names],
        dtype=float,
    )

    # Binary cluster-selection variables followed by positive/negative absolute
    # deviation variables for every stratum.
    objective = np.zeros(n_variables, dtype=float)
    objective[n_clusters : n_clusters + n_strata] = 1.0 / np.maximum(targets, 1.0)
    objective[n_clusters + n_strata :] = 1.0 / np.maximum(targets, 1.0)
    # Deterministic tie-breaker that is negligible relative to any count deviation.
    objective[:n_clusters] = np.arange(n_clusters, dtype=float) * 1e-12

    matrix = lil_matrix((n_strata + 1, n_variables), dtype=float)
    for index in range(n_strata):
        matrix[index, :n_clusters] = counts.iloc[:, index].to_numpy()
        matrix[index, n_clusters + index] = -1.0
        matrix[index, n_clusters + n_strata + index] = 1.0
    matrix[n_strata, :n_clusters] = 1.0
    constraint_values = np.r_[targets, target_clusters]

    result = milp(
        c=objective,
        integrality=np.r_[np.ones(n_clusters), np.zeros(2 * n_strata)],
        bounds=Bounds(
            np.zeros(n_variables),
            np.r_[np.ones(n_clusters), np.full(2 * n_strata, np.inf)],
        ),
        constraints=LinearConstraint(
            matrix.tocsr(), constraint_values, constraint_values
        ),
        options={"time_limit": 60},
    )
    if result.x is None:
        raise RuntimeError(f"Cluster split optimization failed: {result.message}")

    selected = clusters[result.x[:n_clusters] > 0.5]
    selected_rows = work[work["receptor_group"].isin(selected)]
    actual = (
        selected_rows["validation_stratum"]
        .value_counts()
        .reindex(stratum_names, fill_value=0)
    )
    details = {
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "objective_value": float(result.fun),
        "candidate_receptor_groups": int(n_clusters),
        "selected_validation_groups": int(len(selected)),
        "target_validation_groups": int(target_clusters),
        "stratum_targets": {
            name: int(target) for name, target in zip(stratum_names, targets)
        },
        "stratum_actual": {
            name: int(actual[name]) for name in stratum_names
        },
    }
    return selected, details


def write_ids(path: Path, values: pd.Series) -> None:
    path.write_text("\n".join(values.astype(str)) + "\n", encoding="utf-8")


def set_overlap(left: pd.Series, right: pd.Series) -> int:
    left_values = set(left.dropna().astype(str))
    right_values = set(right.dropna().astype(str))
    return len(left_values & right_values)


def make_overlap_audit(split_manifest: pd.DataFrame) -> dict[str, object]:
    clean = split_manifest[split_manifest["split_clean"].isin(["train", "validation", "test"])]
    parts = {
        name: clean[clean["split_clean"] == name]
        for name in ("train", "validation", "test")
    }
    fields = [
        "id",
        "Protein_Sequence",
        "Peptide_Sequence",
        "PDB_ID",
        "prot_cluster",
        "pep_cluster",
    ]
    result: dict[str, object] = {}
    for left_name, right_name in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        left = parts[left_name]
        right = parts[right_name]
        key = f"{left_name}_vs_{right_name}"
        result[key] = {
            field: set_overlap(left[field], right[field]) for field in fields
        }
        left_pairs = set(zip(left["Protein_Sequence"], left["Peptide_Sequence"]))
        right_pairs = set(zip(right["Protein_Sequence"], right["Peptide_Sequence"]))
        result[key]["exact_protein_peptide_pair"] = len(left_pairs & right_pairs)
        result[key]["right_rows_with_peptide_seen_in_left"] = int(
            right["Peptide_Sequence"].isin(set(left["Peptide_Sequence"])).sum()
        )
        result[key]["right_rows_with_PDB_seen_in_left"] = int(
            (
                right["PDB_ID"].notna()
                & right["PDB_ID"].isin(set(left["PDB_ID"].dropna()))
            ).sum()
        )

    original_test = split_manifest[split_manifest["split_original"] == "test"]
    result["original_test_invariance"] = {
        "original_test_records": int(len(original_test)),
        "assigned_clean_test": int((original_test["split_clean"] == "test").sum()),
        "assigned_excluded_test": int(
            (original_test["split_clean"] == "excluded_test").sum()
        ),
        "assigned_train_or_validation": int(
            original_test["split_clean"].isin(["train", "validation"]).sum()
        ),
        "original_test_receptor_clusters": int(original_test["prot_cluster"].nunique()),
        "clean_test_receptor_clusters": int(parts["test"]["prot_cluster"].nunique()),
    }
    return result


def make_split_statistics(split_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name in (
        "train",
        "validation",
        "test",
        "excluded_trainval",
        "excluded_test",
    ):
        part = split_manifest[split_manifest["split_clean"] == split_name]
        rows.append(
            {
                "split": split_name,
                "records": int(len(part)),
                "positive": int(part["class_clean"].eq(1).sum()),
                "negative": int(part["class_clean"].eq(0).sum()),
                "exact_regression": int(part["regression_eligible"].sum()),
                "experimental_negative": int(
                    part["negative_type_clean"].eq("experimental_negative").sum()
                ),
                "bsa_structure_negative": int(
                    part["negative_type_clean"].eq("bsa_structure_negative").sum()
                ),
                "synthetic_decoy": int(
                    part["negative_type_clean"].eq("synthetic_decoy").sum()
                ),
                "receptor_clusters": int(part["prot_cluster"].nunique()),
                "peptide_clusters": int(part["pep_cluster"].nunique()),
                "logAffinity_mean": (
                    float(part.loc[part["regression_eligible"], "logAffinity"].mean())
                    if part["regression_eligible"].any()
                    else None
                ),
                "logAffinity_std": (
                    float(part.loc[part["regression_eligible"], "logAffinity"].std())
                    if part["regression_eligible"].any()
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def write_hashes(output_dir: Path, paths: list[Path]) -> None:
    lines = []
    for path in sorted(paths, key=lambda item: item.name):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "final_manifest_v2.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "splits",
    )
    parser.add_argument(
        "--fixed-validation-groups",
        type=Path,
        default=DEFAULT_FIXED_VALIDATION_GROUPS,
        help=(
            "Published receptor-group list to replay. This is the default "
            "because MILP solvers may choose different equivalent optima."
        ),
    )
    parser.add_argument(
        "--reoptimize-validation-groups",
        action="store_true",
        help="Solve the stratified group-selection problem again instead of replaying the fixed split.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest, low_memory=False)
    if manifest["id"].duplicated().any():
        raise ValueError("The audited manifest contains duplicate IDs")

    development = manifest[
        manifest["split_original"].eq("trainval")
        & manifest["classification_eligible"].astype(bool)
    ].copy()
    development["receptor_group"], receptor_group_mapping = (
        merge_clusters_linked_by_exact_sequence(development)
    )
    strata, bin_edges = assign_strata(development)
    if args.reoptimize_validation_groups:
        validation_groups, optimization = optimize_validation_clusters(
            development, strata
        )
        optimization["selection_mode"] = "reoptimized"
    else:
        fixed_path = args.fixed_validation_groups.expanduser().resolve()
        if not fixed_path.is_file():
            raise FileNotFoundError(
                "The fixed validation-group list is required unless "
                "--reoptimize-validation-groups is supplied: "
                f"{fixed_path}"
            )
        validation_groups = np.asarray(
            [
                int(line.strip())
                for line in fixed_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ],
            dtype=int,
        )
        unknown = sorted(
            set(validation_groups) - set(development["receptor_group"].unique())
        )
        if unknown:
            raise ValueError(
                f"Fixed validation list contains unknown receptor groups: {unknown}"
            )
        optimization = {
            "selection_mode": "fixed_published_groups",
            "fixed_validation_groups_file": str(args.fixed_validation_groups),
            "selected_validation_groups": int(len(validation_groups)),
        }

    manifest["receptor_group"] = ""
    trainval_mask = manifest["split_original"].eq("trainval")
    manifest.loc[trainval_mask, "receptor_group"] = (
        manifest.loc[trainval_mask, "prot_cluster"]
        .astype(int)
        .map(receptor_group_mapping)
        .astype("Int64")
        .astype(str)
    )
    manifest.loc[manifest["split_original"].eq("test"), "receptor_group"] = (
        "test:"
        + manifest.loc[manifest["split_original"].eq("test"), "prot_cluster"]
        .astype(int)
        .astype(str)
    )

    manifest["split_clean"] = ""
    original_test = manifest["split_original"].eq("test")
    eligible = manifest["classification_eligible"].astype(bool)
    validation = (
        manifest["split_original"].eq("trainval")
        & eligible
        & manifest["receptor_group"].isin(
            {str(int(value)) for value in validation_groups}
        )
    )
    train = manifest["split_original"].eq("trainval") & eligible & ~validation
    manifest.loc[train, "split_clean"] = "train"
    manifest.loc[validation, "split_clean"] = "validation"
    manifest.loc[original_test & eligible, "split_clean"] = "test"
    manifest.loc[
        manifest["split_original"].eq("trainval") & ~eligible, "split_clean"
    ] = "excluded_trainval"
    manifest.loc[original_test & ~eligible, "split_clean"] = "excluded_test"
    if manifest["split_clean"].eq("").any():
        raise ValueError("Some records were not assigned to a clean split")

    development_strata = pd.Series(strata.values, index=development["id"].values)
    manifest["validation_stratum"] = manifest["id"].map(development_strata)

    output_files: list[Path] = []
    split_manifest_path = args.output_dir / "split_manifest_v2.csv"
    manifest.to_csv(split_manifest_path, index=False)
    output_files.append(split_manifest_path)

    for split_name, filename in (
        ("train", "train_ids.txt"),
        ("validation", "validation_ids.txt"),
        ("test", "test_ids.txt"),
        ("excluded_trainval", "excluded_trainval_ids.txt"),
        ("excluded_test", "excluded_test_ids.txt"),
    ):
        path = args.output_dir / filename
        write_ids(path, manifest.loc[manifest["split_clean"] == split_name, "id"])
        output_files.append(path)

    original_test_ids_path = args.output_dir / "test_ids_all_original.txt"
    write_ids(
        original_test_ids_path,
        manifest.loc[manifest["split_original"] == "test", "id"],
    )
    output_files.append(original_test_ids_path)

    for split_name in ("train", "validation", "test"):
        path = args.output_dir / f"{split_name}_regression_ids.txt"
        write_ids(
            path,
            manifest.loc[
                manifest["split_clean"].eq(split_name)
                & manifest["regression_eligible"].astype(bool),
                "id",
            ],
        )
        output_files.append(path)

    validation_cluster_values = sorted(
        manifest.loc[manifest["split_clean"].eq("validation"), "prot_cluster"].unique()
    )
    cluster_path = args.output_dir / "validation_receptor_clusters.txt"
    write_ids(cluster_path, pd.Series(validation_cluster_values))
    output_files.append(cluster_path)

    group_path = args.output_dir / "validation_receptor_groups.txt"
    write_ids(group_path, pd.Series(sorted(validation_groups)))
    output_files.append(group_path)

    statistics = make_split_statistics(manifest)
    statistics_path = args.output_dir / "split_statistics.csv"
    statistics.to_csv(statistics_path, index=False)
    output_files.append(statistics_path)

    overlap = make_overlap_audit(manifest)
    overlap_path = args.output_dir / "split_overlap_audit.json"
    overlap_path.write_text(
        json.dumps(overlap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_files.append(overlap_path)

    strategy = {
        "input_manifest": str(args.manifest),
        "validation_fraction": VALIDATION_FRACTION,
        "group_field": "receptor_group",
        "group_definition": "connected components of prot_cluster values linked by an identical Protein_Sequence",
        "nominal_trainval_receptor_clusters": int(
            manifest.loc[trainval_mask, "prot_cluster"].nunique()
        ),
        "effective_trainval_receptor_groups": int(
            development["receptor_group"].nunique()
        ),
        "nominal_clusters_merged_by_exact_sequence": int(
            sum(cluster != group for cluster, group in receptor_group_mapping.items())
        ),
        "regression_quantiles": REGRESSION_QUANTILES,
        "regression_quantile_edges": bin_edges,
        "uses_model_predictions": False,
        "uses_test_records_for_validation_selection": False,
        "optimization": optimization,
    }
    strategy_path = args.output_dir / "split_strategy.json"
    strategy_path.write_text(
        json.dumps(strategy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_files.append(strategy_path)

    write_hashes(args.output_dir, output_files)


if __name__ == "__main__":
    main()
