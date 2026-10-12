#!/usr/bin/env python3
"""Build the three-seed PEPBI ensemble and binding-group bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = (
    ROOT / "revision/external/pepbi/inference/observation_to_inference_mapping.csv"
)
DEFAULT_PDB_SUBSET = (
    ROOT
    / "revision/external/pepbi/processed/pepbi_external_pdb_disjoint_model_input.csv"
)
DEFAULT_OUT = ROOT / "revision/external/pepbi/results/M0_ensemble"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-input", action="append", required=True, type=Path)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--pdb-subset", type=Path, default=DEFAULT_PDB_SUBSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20261006)
    return parser.parse_args()


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    true = frame["logAffinity"].to_numpy(float)
    pred = frame["affinity_pred"].to_numpy(float)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(math.sqrt(mean_squared_error(true, pred))),
        "pearson": float(pearsonr(true, pred).statistic),
        "spearman": float(spearmanr(true, pred).statistic),
    }


def clustered_bootstrap(
    frame: pd.DataFrame, n_bootstrap: int, seed: int
) -> tuple[dict[str, float], list[dict[str, object]]]:
    point = metric_values(frame)
    groups = frame["binding_group"].drop_duplicates().tolist()
    grouped = {name: data for name, data in frame.groupby("binding_group", sort=False)}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(n_bootstrap):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([grouped[name] for name in selected], ignore_index=True)
        values = metric_values(sample)
        for name, value in values.items():
            if math.isfinite(value):
                draws[name].append(value)
    rows: list[dict[str, object]] = []
    for name, value in point.items():
        values = np.asarray(draws[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "estimate": value,
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "valid_bootstrap_draws": int(values.size),
            }
        )
    return point, rows


def summarize_subset(
    name: str, frame: pd.DataFrame, n_bootstrap: int, seed: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    point, ci_rows = clustered_bootstrap(frame, n_bootstrap, seed)
    summary = {
        "subset": name,
        "observations": int(len(frame)),
        "unique_pairs": int(frame["inference_id"].nunique()),
        "binding_groups": int(frame["binding_group"].nunique()),
        **point,
    }
    return summary, [{"subset": name, **row} for row in ci_rows]


def main() -> None:
    args = parse_args()
    if len(args.seed_input) < 2:
        raise ValueError("At least two seed inputs are required for an ensemble")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_frames = []
    seed_names = []
    for index, path in enumerate(args.seed_input, start=1):
        path = path.expanduser().resolve()
        frame = pd.read_csv(path)
        needed = {"inference_id", "affinity_pred", "binder_probability"}
        if not needed.issubset(frame.columns):
            raise ValueError(f"{path} lacks {sorted(needed - set(frame.columns))}")
        frame = frame.sort_values("inference_id").reset_index(drop=True)
        if seed_frames and not frame["inference_id"].equals(
            seed_frames[0]["inference_id"]
        ):
            raise ValueError(f"Inference IDs do not align in {path}")
        seed_frames.append(frame)
        seed_names.append(path.parent.name or f"seed{index}")

    unique = seed_frames[0][["inference_id"]].copy()
    unique["affinity_pred"] = np.mean(
        np.vstack([frame["affinity_pred"].to_numpy(float) for frame in seed_frames]),
        axis=0,
    )
    unique["binder_probability"] = np.mean(
        np.vstack(
            [frame["binder_probability"].to_numpy(float) for frame in seed_frames]
        ),
        axis=0,
    )
    unique.to_csv(output_dir / "unique_predictions.csv", index=False)

    mapping = pd.read_csv(args.mapping.expanduser().resolve())
    observations = mapping.merge(unique, on="inference_id", how="left", validate="many_to_one")
    if observations[["affinity_pred", "binder_probability"]].isna().any().any():
        raise RuntimeError("Missing ensemble predictions after mapping to observations")
    observations.to_csv(output_dir / "observation_predictions.csv", index=False)

    pdb_ids = set(pd.read_csv(args.pdb_subset.expanduser().resolve())["id"].astype(str))
    subsets = {
        "PEPBI pair-disjoint": observations,
        "PEPBI PDB-disjoint": observations[observations["id"].astype(str).isin(pdb_ids)],
    }
    summaries: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for offset, (name, frame) in enumerate(subsets.items()):
        summary, ci = summarize_subset(
            name, frame.copy(), args.bootstrap, args.random_seed + offset
        )
        summaries.append(summary)
        bootstrap_rows.extend(ci)

    pd.DataFrame(summaries).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "binding_group_bootstrap_ci.csv", index=False
    )

    ci_lookup = {
        (row["subset"], row["metric"]): row for row in bootstrap_rows
    }
    table_rows = []
    for summary in summaries:
        row: dict[str, object] = {
            "Model": "HighAffinity M0 three-seed ensemble",
            "External subset": summary["subset"],
            "Observations": summary["observations"],
            "Unique sequence pairs": summary["unique_pairs"],
            "Binding groups": summary["binding_groups"],
        }
        for metric, label in [
            ("mae", "MAE (95% CI)"),
            ("rmse", "RMSE (95% CI)"),
            ("pearson", "Pearson r (95% CI)"),
            ("spearman", "Spearman rho (95% CI)"),
        ]:
            ci = ci_lookup[(summary["subset"], metric)]
            row[label] = (
                f"{summary[metric]:.4f} ({ci['ci_low']:.4f}–{ci['ci_high']:.4f})"
            )
        table_rows.append(row)
    table_path = ROOT / "revision/tables/final_test/table_s5_external_validation.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    provenance = {
        "seed_inputs": [str(path.expanduser().resolve()) for path in args.seed_input],
        "seed_labels": seed_names,
        "ensemble": "arithmetic mean of per-pair predictions",
        "bootstrap_unit": "PEPBI binding_group",
        "bootstrap_replicates": args.bootstrap,
        "random_seed": args.random_seed,
        "table": str(table_path),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "ENSEMBLE_COMPLETE.json").write_text(
        json.dumps({"summaries": summaries, **provenance}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summaries": summaries, **provenance}, indent=2))


if __name__ == "__main__":
    main()
