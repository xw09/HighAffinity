#!/usr/bin/env python3
"""Build revision table templates, prefilled with currently available results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "revision"
EXPERIMENTS = [f"M{index}" for index in range(7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REVISION / "tables/templates",
    )
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        default=REVISION / "statistics",
    )
    parser.add_argument("--split-label", default="fixed validation")
    return parser.parse_args()


def metric_text(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "TBD"
    return f"{number:.{digits}f}" if np.isfinite(number) else "TBD"


def dataset_table() -> pd.DataFrame:
    source = pd.read_csv(REVISION / "data/splits/split_statistics.csv")
    source = source[source["split"].isin(["train", "validation", "test"])].copy()
    source["classification_records"] = source["positive"] + source["negative"]
    return source[
        [
            "split",
            "records",
            "classification_records",
            "positive",
            "negative",
            "exact_regression",
            "experimental_negative",
            "bsa_structure_negative",
            "synthetic_decoy",
            "receptor_clusters",
            "peptide_clusters",
        ]
    ].rename(
        columns={
            "split": "Dataset split",
            "records": "Records",
            "classification_records": "Classification eligible",
            "positive": "Positive",
            "negative": "Negative",
            "exact_regression": "Exact regression labels",
            "experimental_negative": "Experimental negatives",
            "bsa_structure_negative": "BSA/structure negatives",
            "synthetic_decoy": "Synthetic decoys",
            "receptor_clusters": "Receptor clusters",
            "peptide_clusters": "Peptide clusters",
        }
    )


def config_descriptions() -> dict[str, str]:
    descriptions = {}
    for experiment in EXPERIMENTS:
        path = REVISION / f"configs/public/{experiment}_seed1.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        descriptions[experiment] = config["description"]
    return descriptions


def validation_metrics(statistics_dir: Path) -> pd.DataFrame:
    per_run_path = statistics_dir / "per_run_metrics.csv"
    if per_run_path.is_file():
        source = pd.read_csv(per_run_path)
        source = source[source["scope"].eq("overall")].copy()
        source["experiment"] = source["run"].str.replace(
            r"_seed\d+$", "", regex=True
        )
        source["seed"] = pd.to_numeric(
            source["run"].str.extract(r"_seed(\d+)$")[0], errors="coerce"
        )
        return source[source["seed"].notna()].copy()
    path = REVISION / "status/stage6_validation_metrics_summary.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def seed_summary_lookup(statistics_dir: Path) -> dict[tuple[str, str], str]:
    path = statistics_dir / "seed_summary.csv"
    if not path.is_file():
        return {}
    source = pd.read_csv(path)
    result = {}
    for row in source.itertuples(index=False):
        if int(row.n_seeds) > 1 and np.isfinite(row.std):
            result[(row.experiment, row.metric)] = f"{row.mean:.4f} ± {row.std:.4f}"
        else:
            result[(row.experiment, row.metric)] = f"{row.mean:.4f}"
    return result


def ablation_table(statistics_dir: Path, split_label: str) -> pd.DataFrame:
    descriptions = config_descriptions()
    metrics = validation_metrics(statistics_dir)
    seed_lookup = seed_summary_lookup(statistics_dir)
    metric_names = [
        "accuracy",
        "roc_auc",
        "average_precision",
        "f1",
        "specificity",
        "mcc",
        "mae",
        "rmse",
        "pearson",
        "spearman",
    ]
    rows = []
    for experiment in EXPERIMENTS:
        current = metrics[metrics["experiment"].eq(experiment)]
        row = {
            "Experiment": experiment,
            "Change relative to M0": descriptions[experiment],
            "Split": split_label,
            "Seeds represented": (
                int(current["seed"].nunique()) if not current.empty else 0
            ),
        }
        for metric in metric_names:
            key = (experiment, metric)
            classification_metric = metric in {
                "accuracy", "roc_auc", "average_precision", "f1", "specificity", "mcc"
            }
            disabled = (experiment == "M2" and classification_metric) or (
                experiment == "M3" and not classification_metric
            )
            if disabled:
                row[metric] = "N/A"
            elif key in seed_lookup:
                row[metric] = seed_lookup[key]
            elif not current.empty and metric in current:
                row[metric] = metric_text(current.iloc[0][metric])
            else:
                row[metric] = "TBD"
        represented = int(current["seed"].nunique()) if not current.empty else 0
        row["Final table status"] = (
            "complete"
            if experiment not in {"M0", "M1"} or represented >= 3
            else "awaiting seeds 2-3"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def locate_subgroup_files() -> list[Path]:
    return sorted(
        (REVISION / "predictions/stage6_validation").glob(
            "M*_seed1/*/negative_subgroup_metrics.csv"
        )
    )


def negative_table(statistics_dir: Path) -> pd.DataFrame:
    statistical_path = statistics_dir / "negative_subgroup_metrics.csv"
    if statistical_path.is_file():
        source = pd.read_csv(statistical_path)
        if not source.empty:
            rows = []
            for record in source.to_dict("records"):
                rows.append(
                    {
                        "Run": record["run"],
                        "Negative subset": record["negative_type"],
                        "N": int(record["n"]),
                        "Receptor groups": int(record["n_receptor_groups"]),
                        "Specificity at 0.5": metric_text(record["specificity"]),
                        "Specificity 95% cluster-bootstrap CI": (
                            f"{record['specificity_ci_low']:.4f}–"
                            f"{record['specificity_ci_high']:.4f}"
                        ),
                        "Mean predicted binder probability": metric_text(
                            record["mean_positive_probability"]
                        ),
                    }
                )
            return pd.DataFrame(rows)
    rows = []
    for path in locate_subgroup_files():
        run = path.parents[1].name
        source = pd.read_csv(path)
        for record in source.to_dict("records"):
            rows.append(
                {
                    "Run": run,
                    "Negative subset": record["negative_type"],
                    "N": int(record["n"]),
                    "Receptor groups": "TBD",
                    "Specificity at 0.5": metric_text(record["specificity"]),
                    "Mean predicted binder probability": metric_text(
                        record["mean_positive_probability"]
                    ),
                    "Specificity 95% cluster-bootstrap CI": "TBD",
                }
            )
    return pd.DataFrame(rows)


def comparison_table(statistics_dir: Path) -> pd.DataFrame:
    path = statistics_dir / "paired_comparisons.csv"
    if path.is_file():
        source = pd.read_csv(path)
        if not source.empty:
            return source
    rows = []
    for other in EXPERIMENTS[1:]:
        classification_metrics = ("roc_auc", "average_precision", "mcc")
        regression_metrics = ("mae", "rmse", "pearson", "spearman")
        if other == "M2":
            metrics = regression_metrics
        elif other == "M3":
            metrics = classification_metrics
        else:
            metrics = (*classification_metrics, *regression_metrics)
        for metric in metrics:
            rows.append(
                {
                    "run_a": "M0_seed1",
                    "run_b": f"{other}_seed1",
                    "metric": metric,
                    "delta_a_minus_b": "TBD",
                    "ci_low_delta": "TBD",
                    "ci_high_delta": "TBD",
                    "p_two_sided_bootstrap_sign": "TBD",
                    "decision": "TBD",
                }
            )
    return pd.DataFrame(rows)


def external_table() -> pd.DataFrame:
    summary_path = REVISION / "external/pepbi/processed/pepbi_cleaning_summary.json"
    cohorts: list[tuple[str, int | str, str]] = [
        (
            "PEPBI strict receptor-cold non-overlap",
            "TBD after PEPBI cleaning",
            "Primary only if non-empty",
        ),
        (
            "PEPBI exact sequence-pair-disjoint",
            "TBD after PEPBI cleaning",
            "Sensitivity analysis; receptor overlap remains",
        ),
        (
            "PEPBI PDB-ID-disjoint",
            "TBD after PEPBI cleaning",
            "Sensitivity analysis; receptor overlap remains",
        ),
    ]
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cohorts = [
            (
                "PEPBI strict receptor-cold non-overlap",
                summary["strict_external_complexes"],
                "Unavailable as an evaluation cohort when N=0",
            ),
            (
                "PEPBI exact sequence-pair-disjoint",
                summary["pair_disjoint_external_observations"],
                "Sensitivity analysis; receptor overlap remains",
            ),
            (
                "PEPBI PDB-ID-disjoint",
                summary["pdb_disjoint_external_observations"],
                "Sensitivity analysis; receptor overlap remains",
            ),
        ]
    rows = []
    for cohort, sample_count, interpretation in cohorts:
        for model in (
            "HighAffinity (M0 seed ensemble)",
            "PEPBI Rosetta RIA dG baseline",
        ):
            rows.append(
                {
                    "External cohort": cohort,
                    "Model": model,
                    "N thermodynamic observations": sample_count,
                    "Interpretation": interpretation,
                    "MAE": "TBD",
                    "MAE 95% CI": "TBD",
                    "RMSE": "TBD",
                    "RMSE 95% CI": "TBD",
                    "Pearson r": "TBD",
                    "Pearson 95% CI": "TBD",
                    "Spearman rho": "TBD",
                    "Spearman 95% CI": "TBD",
                }
            )
    return pd.DataFrame(rows)


def compute_table() -> pd.DataFrame:
    rows = []
    for experiment in EXPERIMENTS:
        rows.append(
            {
                "Experiment": experiment,
                "Trainable parameters": "8,741,896",
                "Total parameters": "515,055,240",
                "Peak GPU memory (GiB)": "TBD from logs/profiler",
                "Best epoch": "TBD auto-fill",
                "Wall-clock training time": "TBD from logs",
                "Inference time per complex": "TBD",
                "GPU": "TBD",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "table_s1_dataset_split.csv": dataset_table(),
        "table_s2_ablation_validation.csv": ablation_table(
            args.statistics_dir, args.split_label
        ),
        "table_s3_negative_subgroups.csv": negative_table(args.statistics_dir),
        "table_s4_paired_statistics.csv": comparison_table(args.statistics_dir),
        "table_s5_external_validation.csv": external_table(),
        "table_s6_compute_cost.csv": compute_table(),
    }
    for filename, table in tables.items():
        table.to_csv(args.output_dir / filename, index=False)
    print(f"Wrote {len(tables)} table templates to {args.output_dir}")


if __name__ == "__main__":
    main()
