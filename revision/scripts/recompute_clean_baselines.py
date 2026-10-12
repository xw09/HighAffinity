#!/usr/bin/env python3
"""Recompute legacy baselines on the audited 2,496/516 held-out test rows."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "revision/data/splits/split_manifest_v2.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "revision/tables/final_test/table_baselines_clean_test.csv",
    )
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, low_memory=False)
    test = manifest[
        manifest["split_clean"].eq("test")
        & as_bool(manifest["classification_eligible"])
    ].copy()
    test["id"] = test["id"].astype(str)

    specifications = [
        (
            "Boltz2 zero-shot",
            ROOT / "revision/originals/results/boltz2_native.csv",
            "affinity_probability_binary",
            "affinity_pred_value",
            "official pretrained checkpoint",
        ),
        (
            "PPAP",
            ROOT / "revision/originals/results/ppap_with_Kd_uM_log10_class.csv",
            None,
            "log10_Kd_uM",
            "official pretrained checkpoint; classification score=-log10(Kd/uM)",
        ),
        (
            "ProAffinity-GNN",
            ROOT / "revision/originals/results/proaffinity-gnn_parsed.csv",
            None,
            "log10_K_uM",
            "official pretrained checkpoint; classification score=-log10(K/uM)",
        ),
    ]
    rows = []
    for name, path, probability_column, prediction_column, protocol in specifications:
        predictions = pd.read_csv(path)
        predictions["id"] = predictions["id"].astype(str)
        columns = ["id", prediction_column]
        if probability_column:
            columns.append(probability_column)
        frame = test.merge(predictions[columns], on="id", how="inner", validate="one_to_one")
        if len(frame) != len(test):
            raise RuntimeError(f"{name}: expected {len(test)} aligned rows, got {len(frame)}")
        true_class = frame["class_clean"].astype(int).to_numpy()
        affinity_prediction = frame[prediction_column].to_numpy(float)
        if probability_column:
            class_score = frame[probability_column].to_numpy(float)
        else:
            # Monotonic conversion: 0.5 corresponds to logAffinity=3 and does
            # not alter ROC-AUC or AP ranking.
            class_score = expit(3.0 - affinity_prediction)
        class_prediction = (class_score >= 0.5).astype(int)
        negatives = true_class == 0
        regression = frame[
            as_bool(frame["regression_eligible"])
            & frame["logAffinity"].notna()
            & frame[prediction_column].notna()
        ]
        true_affinity = regression["logAffinity"].to_numpy(float)
        predicted_affinity = regression[prediction_column].to_numpy(float)
        rows.append(
            {
                "Model": name,
                "Evaluation protocol": protocol,
                "N classification": len(frame),
                "Accuracy": accuracy_score(true_class, class_prediction),
                "ROC-AUC": roc_auc_score(true_class, class_score),
                "Average precision": average_precision_score(true_class, class_score),
                "F1": f1_score(true_class, class_prediction),
                "Specificity": np.mean(class_prediction[negatives] == 0),
                "MCC": matthews_corrcoef(true_class, class_prediction),
                "N regression": len(regression),
                "MAE": mean_absolute_error(true_affinity, predicted_affinity),
                "RMSE": math.sqrt(mean_squared_error(true_affinity, predicted_affinity)),
                "Pearson r": pearsonr(true_affinity, predicted_affinity).statistic,
                "Spearman rho": spearmanr(true_affinity, predicted_affinity).statistic,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
