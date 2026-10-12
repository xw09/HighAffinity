#!/usr/bin/env python3
"""Reproducible clustered-bootstrap analysis for affinity predictions.

The unit resampled by the bootstrap is the receptor group, not the row.  This
is important because many rows can belong to one receptor and are therefore
not independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "revision/statistics"
DEFAULT_CONFIG_DIR = ROOT / "revision/configs/public"

CLASSIFICATION_METRICS = (
    "accuracy",
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1",
    "specificity",
    "mcc",
)
REGRESSION_METRICS = ("mae", "rmse", "pearson", "spearman")
LOWER_IS_BETTER = {"mae", "rmse"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute common metrics, receptor-cluster bootstrap CIs, "
        "paired comparisons, target-macro metrics, and seed summaries."
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="RUN=CSV",
        help="Repeat once per run, e.g. M0_seed1=.../predictions.csv.",
    )
    parser.add_argument(
        "--reference-run",
        help="Compare this run with every other compatible run.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="RUN_A:RUN_B",
        help="Additional paired comparison; can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected RUN=CSV, got: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"Missing or duplicate run name: {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0"})
    if not valid.all():
        bad = sorted(normalized[~valid].unique())[:10]
        raise ValueError(f"Invalid boolean values: {bad}")
    return normalized.isin({"true", "1"})


def task_flags(run_name: str) -> tuple[bool, bool]:
    config_path = DEFAULT_CONFIG_DIR / f"{run_name}.yaml"
    if not config_path.is_file():
        return True, True
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parameters = config.get("parameters", {})
    return (
        bool(parameters.get("classification_enabled", True)),
        bool(parameters.get("regression_enabled", True)),
    )


def read_predictions(run_name: str, path: Path) -> pd.DataFrame:
    required = {
        "id",
        "receptor_group",
        "class_true",
        "class_probability",
        "regression_eligible",
        "affinity_true",
        "affinity_pred",
    }
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{run_name}: missing columns {missing}")
    if frame["id"].duplicated().any():
        duplicate = frame.loc[frame["id"].duplicated(), "id"].iloc[0]
        raise ValueError(f"{run_name}: duplicate id {duplicate}")
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    frame["receptor_group"] = frame["receptor_group"].astype("object")
    missing_group = frame["receptor_group"].isna() | frame[
        "receptor_group"
    ].astype(str).str.strip().eq("")
    frame.loc[missing_group, "receptor_group"] = (
        "missing_group:" + frame.loc[missing_group, "id"]
    )
    frame["receptor_group"] = frame["receptor_group"].astype(str)
    frame["regression_eligible"] = as_boolean(frame["regression_eligible"])
    for column in (
        "class_true",
        "class_probability",
        "affinity_true",
        "affinity_pred",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    class_values = set(frame["class_true"].dropna().unique())
    if not class_values <= {0, 1}:
        raise ValueError(f"{run_name}: class_true must contain only 0/1")
    frame.attrs["source_path"] = str(path)
    return frame


def safe_correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        return math.nan
    return float(function(truth, prediction).statistic)


def classification_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    valid = frame[["class_true", "class_probability"]].notna().all(axis=1)
    y_true = frame.loc[valid, "class_true"].astype(int).to_numpy()
    y_prob = frame.loc[valid, "class_probability"].astype(float).to_numpy()
    if y_true.size == 0:
        return {metric: math.nan for metric in CLASSIFICATION_METRICS}
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = np.unique(y_true).size == 2
    result = {
        "n_classification": int(y_true.size),
        "positive": int(y_true.sum()),
        "negative": int((y_true == 0).sum()),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else math.nan,
        "average_precision": (
            float(average_precision_score(y_true, y_prob))
            if both_classes
            else math.nan
        ),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if both_classes else math.nan,
    }
    return result


def regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    eligible = frame["regression_eligible"] & frame[
        ["affinity_true", "affinity_pred"]
    ].notna().all(axis=1)
    truth = frame.loc[eligible, "affinity_true"].astype(float).to_numpy()
    prediction = frame.loc[eligible, "affinity_pred"].astype(float).to_numpy()
    if truth.size == 0:
        return {metric: math.nan for metric in REGRESSION_METRICS}
    return {
        "n_regression": int(truth.size),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(truth, prediction))),
        "pearson": safe_correlation(pearsonr, truth, prediction),
        "spearman": safe_correlation(spearmanr, truth, prediction),
    }


def all_metrics(
    frame: pd.DataFrame,
    threshold: float,
    classification_enabled: bool,
    regression_enabled: bool,
) -> dict[str, float]:
    result: dict[str, float] = {"n_rows": int(len(frame))}
    result["n_receptor_groups"] = int(frame["receptor_group"].nunique())
    if classification_enabled:
        result.update(classification_metrics(frame, threshold))
    if regression_enabled:
        result.update(regression_metrics(frame))
    return result


def target_macro_metrics(
    run_name: str,
    frame: pd.DataFrame,
    threshold: float,
    classification_enabled: bool,
    regression_enabled: bool,
) -> list[dict[str, object]]:
    per_group = []
    for group, part in frame.groupby("receptor_group", sort=True):
        values = all_metrics(
            part, threshold, classification_enabled, regression_enabled
        )
        values["receptor_group"] = group
        per_group.append(values)
    group_frame = pd.DataFrame(per_group)
    output: list[dict[str, object]] = []
    for task, metrics in (
        ("classification", CLASSIFICATION_METRICS),
        ("regression", REGRESSION_METRICS),
    ):
        enabled = classification_enabled if task == "classification" else regression_enabled
        if not enabled:
            continue
        for metric in metrics:
            values = pd.to_numeric(group_frame.get(metric), errors="coerce").dropna()
            output.append(
                {
                    "run": run_name,
                    "task": task,
                    "metric": metric,
                    "macro_mean": float(values.mean()) if len(values) else math.nan,
                    "macro_std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    "valid_receptor_groups": int(len(values)),
                    "total_receptor_groups": int(len(group_frame)),
                }
            )
    return output


def bootstrap_metrics(
    frame: pd.DataFrame,
    threshold: float,
    classification_enabled: bool,
    regression_enabled: bool,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, object]]:
    groups = sorted(frame["receptor_group"].unique())
    positions = {
        group: np.flatnonzero(frame["receptor_group"].to_numpy() == group)
        for group in groups
    }
    rng = np.random.default_rng(seed)
    metric_names = []
    if classification_enabled:
        metric_names.extend(("classification", item) for item in CLASSIFICATION_METRICS)
    if regression_enabled:
        metric_names.extend(("regression", item) for item in REGRESSION_METRICS)
    samples = {metric: [] for _, metric in metric_names}
    for _ in range(replicates):
        selected = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([positions[group] for group in selected])
        values = all_metrics(
            frame.iloc[indices], threshold, classification_enabled, regression_enabled
        )
        for _, metric in metric_names:
            value = values.get(metric, math.nan)
            if math.isfinite(value):
                samples[metric].append(float(value))
    point = all_metrics(
        frame, threshold, classification_enabled, regression_enabled
    )
    alpha = 1.0 - confidence_level
    rows = []
    for task, metric in metric_names:
        values = np.asarray(samples[metric], dtype=float)
        rows.append(
            {
                "task": task,
                "metric": metric,
                "estimate": point.get(metric, math.nan),
                "ci_low": (
                    float(np.quantile(values, alpha / 2)) if values.size else math.nan
                ),
                "ci_high": (
                    float(np.quantile(values, 1 - alpha / 2))
                    if values.size
                    else math.nan
                ),
                "valid_bootstrap_replicates": int(values.size),
                "requested_bootstrap_replicates": int(replicates),
                "confidence_level": confidence_level,
                "n_rows": int(len(frame)),
                "n_receptor_groups": int(len(groups)),
            }
        )
    return rows


def align_pair(
    run_a: str,
    frame_a: pd.DataFrame,
    run_b: str,
    frame_b: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids_a = set(frame_a["id"])
    ids_b = set(frame_b["id"])
    if ids_a != ids_b:
        raise ValueError(
            f"Cannot pair {run_a} and {run_b}: ID sets differ "
            f"({len(ids_a - ids_b)} only A, {len(ids_b - ids_a)} only B)."
        )
    a = frame_a.set_index("id").sort_index()
    b = frame_b.set_index("id").sort_index()
    for column in ("class_true", "affinity_true", "receptor_group"):
        left = a[column].fillna("missing").astype(str)
        right = b[column].fillna("missing").astype(str)
        if not left.equals(right):
            raise ValueError(f"{run_a} vs {run_b}: mismatched {column}")
    return a.reset_index(), b.reset_index()


def paired_bootstrap(
    run_a: str,
    frame_a: pd.DataFrame,
    run_b: str,
    frame_b: pd.DataFrame,
    threshold: float,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, object]]:
    a, b = align_pair(run_a, frame_a, run_b, frame_b)
    class_a, reg_a = task_flags(run_a)
    class_b, reg_b = task_flags(run_b)
    classification_enabled = class_a and class_b
    regression_enabled = reg_a and reg_b
    point_a = all_metrics(a, threshold, classification_enabled, regression_enabled)
    point_b = all_metrics(b, threshold, classification_enabled, regression_enabled)
    metric_names = []
    if classification_enabled:
        metric_names.extend(("classification", item) for item in CLASSIFICATION_METRICS)
    if regression_enabled:
        metric_names.extend(("regression", item) for item in REGRESSION_METRICS)
    groups = sorted(a["receptor_group"].unique())
    group_array = a["receptor_group"].to_numpy()
    positions = {group: np.flatnonzero(group_array == group) for group in groups}
    rng = np.random.default_rng(seed)
    differences = {metric: [] for _, metric in metric_names}
    for _ in range(replicates):
        selected = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([positions[group] for group in selected])
        metrics_a = all_metrics(
            a.iloc[indices], threshold, classification_enabled, regression_enabled
        )
        metrics_b = all_metrics(
            b.iloc[indices], threshold, classification_enabled, regression_enabled
        )
        for _, metric in metric_names:
            left = metrics_a.get(metric, math.nan)
            right = metrics_b.get(metric, math.nan)
            if math.isfinite(left) and math.isfinite(right):
                differences[metric].append(float(left - right))
    alpha = 1.0 - confidence_level
    rows = []
    for task, metric in metric_names:
        values = np.asarray(differences[metric], dtype=float)
        raw_delta = point_a.get(metric, math.nan) - point_b.get(metric, math.nan)
        advantage = -raw_delta if metric in LOWER_IS_BETTER else raw_delta
        if values.size:
            less_equal = (np.count_nonzero(values <= 0) + 1) / (values.size + 1)
            greater_equal = (np.count_nonzero(values >= 0) + 1) / (values.size + 1)
            p_value = min(1.0, 2 * min(less_equal, greater_equal))
            ci_low = float(np.quantile(values, alpha / 2))
            ci_high = float(np.quantile(values, 1 - alpha / 2))
        else:
            p_value = ci_low = ci_high = math.nan
        rows.append(
            {
                "run_a": run_a,
                "run_b": run_b,
                "task": task,
                "metric": metric,
                "estimate_a": point_a.get(metric, math.nan),
                "estimate_b": point_b.get(metric, math.nan),
                "delta_a_minus_b": raw_delta,
                "ci_low_delta": ci_low,
                "ci_high_delta": ci_high,
                "advantage_a": advantage,
                "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                "p_two_sided_bootstrap_sign": p_value,
                "valid_bootstrap_replicates": int(values.size),
                "n_rows": int(len(a)),
                "n_receptor_groups": int(len(groups)),
            }
        )
    return rows


def negative_subgroups(
    run_name: str,
    frame: pd.DataFrame,
    threshold: float,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, object]]:
    if "negative_type" not in frame.columns:
        return []
    negatives = frame[frame["class_true"].eq(0)].copy()
    negatives["negative_type"] = negatives["negative_type"].fillna("unspecified")
    rows = []
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence_level
    for negative_type, part in negatives.groupby("negative_type", sort=True):
        probability = part["class_probability"].dropna().astype(float)
        predicted_negative = probability < threshold
        part = part.loc[probability.index]
        groups = sorted(part["receptor_group"].unique())
        group_array = part["receptor_group"].to_numpy()
        positions = {
            group: np.flatnonzero(group_array == group) for group in groups
        }
        bootstrap_specificity = []
        bootstrap_mean_probability = []
        for _ in range(replicates):
            selected = rng.choice(groups, size=len(groups), replace=True)
            indices = np.concatenate([positions[group] for group in selected])
            sampled_probability = part.iloc[indices]["class_probability"].to_numpy(
                dtype=float
            )
            bootstrap_specificity.append(
                float(np.mean(sampled_probability < threshold))
            )
            bootstrap_mean_probability.append(float(np.mean(sampled_probability)))
        rows.append(
            {
                "run": run_name,
                "negative_type": negative_type,
                "n": int(len(probability)),
                "n_receptor_groups": int(len(groups)),
                "specificity": float(predicted_negative.mean()) if len(probability) else math.nan,
                "specificity_ci_low": float(
                    np.quantile(bootstrap_specificity, alpha / 2)
                ),
                "specificity_ci_high": float(
                    np.quantile(bootstrap_specificity, 1 - alpha / 2)
                ),
                "mean_positive_probability": float(probability.mean()) if len(probability) else math.nan,
                "mean_positive_probability_ci_low": float(
                    np.quantile(bootstrap_mean_probability, alpha / 2)
                ),
                "mean_positive_probability_ci_high": float(
                    np.quantile(bootstrap_mean_probability, 1 - alpha / 2)
                ),
                "median_positive_probability": float(probability.median()) if len(probability) else math.nan,
            }
        )
    return rows


def make_seed_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    overall = per_run[per_run["scope"].eq("overall")].copy()
    overall["experiment"] = overall["run"].str.replace(
        r"_seed\d+$", "", regex=True
    )
    metric_columns = [
        column
        for column in (*CLASSIFICATION_METRICS, *REGRESSION_METRICS)
        if column in overall.columns
    ]
    rows = []
    for experiment, part in overall.groupby("experiment", sort=True):
        for metric in metric_columns:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "n_seeds": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def make_paired_seed_summary(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    pattern = re.compile(r"^(?P<experiment>.+)_seed(?P<seed>\d+)$")
    rows = []
    work = []
    for record in paired.to_dict("records"):
        left = pattern.match(str(record["run_a"]))
        right = pattern.match(str(record["run_b"]))
        if not left or not right or left.group("seed") != right.group("seed"):
            continue
        record = dict(record)
        record["experiment_a"] = left.group("experiment")
        record["experiment_b"] = right.group("experiment")
        record["seed"] = int(left.group("seed"))
        work.append(record)
    if not work:
        return pd.DataFrame()
    frame = pd.DataFrame(work)
    keys = ["experiment_a", "experiment_b", "task", "metric", "direction"]
    for key, part in frame.groupby(keys, sort=True, dropna=False):
        delta = pd.to_numeric(part["delta_a_minus_b"], errors="coerce").dropna()
        advantage = pd.to_numeric(part["advantage_a"], errors="coerce").dropna()
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_matched_seeds": int(len(delta)),
                "mean_delta_a_minus_b": float(delta.mean()),
                "sd_delta_a_minus_b": (
                    float(delta.std(ddof=1)) if len(delta) > 1 else math.nan
                ),
                "mean_advantage_a": float(advantage.mean()),
                "seeds": "|".join(map(str, sorted(part["seed"].unique()))),
            }
        )
    return pd.DataFrame(rows)


def parse_comparisons(
    args: argparse.Namespace, names: list[str]
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if args.reference_run:
        if args.reference_run not in names:
            raise ValueError(f"Unknown reference run: {args.reference_run}")
        pairs.extend((args.reference_run, name) for name in names if name != args.reference_run)
    for value in args.comparison:
        if ":" not in value:
            raise ValueError(f"Expected RUN_A:RUN_B, got: {value}")
        left, right = value.split(":", 1)
        if left not in names or right not in names:
            raise ValueError(f"Unknown run in comparison: {value}")
        pairs.append((left, right))
    return list(dict.fromkeys(pairs))


def write_csv(path: Path, rows: list[dict[str, object]] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    if not 0 < args.confidence_level < 1:
        raise ValueError("--confidence-level must be between 0 and 1.")
    paths = parse_named_paths(args.prediction)
    frames = {name: read_predictions(name, path) for name, path in paths.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows = []
    bootstrap_rows = []
    macro_rows = []
    subgroup_rows = []
    for run_name, frame in frames.items():
        class_enabled, reg_enabled = task_flags(run_name)
        largest_group = frame["receptor_group"].value_counts().index[0]
        scopes = {
            "overall": frame,
            "without_largest_receptor": frame[
                frame["receptor_group"].ne(largest_group)
            ],
        }
        for scope, subset in scopes.items():
            row = {
                "run": run_name,
                "scope": scope,
                "source_path": frame.attrs["source_path"],
                "classification_enabled": class_enabled,
                "regression_enabled": reg_enabled,
                "excluded_receptor_group": largest_group if scope != "overall" else "",
            }
            row.update(
                all_metrics(subset, args.threshold, class_enabled, reg_enabled)
            )
            per_run_rows.append(row)
        run_bootstrap = bootstrap_metrics(
            frame,
            args.threshold,
            class_enabled,
            reg_enabled,
            args.bootstrap_replicates,
            args.confidence_level,
            stable_seed(args.seed, f"bootstrap:{run_name}"),
        )
        for row in run_bootstrap:
            row["run"] = run_name
        bootstrap_rows.extend(run_bootstrap)
        macro_rows.extend(
            target_macro_metrics(
                run_name, frame, args.threshold, class_enabled, reg_enabled
            )
        )
        if class_enabled:
            subgroup_rows.extend(
                negative_subgroups(
                    run_name,
                    frame,
                    args.threshold,
                    args.bootstrap_replicates,
                    args.confidence_level,
                    stable_seed(args.seed, f"negative-subgroups:{run_name}"),
                )
            )

    paired_rows = []
    for run_a, run_b in parse_comparisons(args, list(frames)):
        paired_rows.extend(
            paired_bootstrap(
                run_a,
                frames[run_a],
                run_b,
                frames[run_b],
                args.threshold,
                args.bootstrap_replicates,
                args.confidence_level,
                stable_seed(args.seed, f"paired:{run_a}:{run_b}"),
            )
        )

    per_run = pd.DataFrame(per_run_rows)
    write_csv(args.output_dir / "per_run_metrics.csv", per_run)
    write_csv(args.output_dir / "cluster_bootstrap_ci.csv", bootstrap_rows)
    paired_frame = pd.DataFrame(paired_rows)
    write_csv(args.output_dir / "paired_comparisons.csv", paired_frame)
    write_csv(
        args.output_dir / "paired_seed_summary.csv",
        make_paired_seed_summary(paired_frame),
    )
    write_csv(args.output_dir / "target_macro_metrics.csv", macro_rows)
    write_csv(args.output_dir / "negative_subgroup_metrics.csv", subgroup_rows)
    write_csv(args.output_dir / "seed_summary.csv", make_seed_summary(per_run))
    print(f"Statistical outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
