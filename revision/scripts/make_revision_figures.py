#!/usr/bin/env python3
"""Generate revision-ready main and supplementary figures.

The script reads only audited/final outputs under ``revision/`` and writes
600-dpi PNG plus vector PDF/SVG files under ``revision/figures/``.  It also
exports the plotted source data so every panel can be independently checked.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision"
OUT = REV / "figures"
MAIN = OUT / "main"
SUPP = OUT / "supplementary"
SOURCE = OUT / "source_data"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PINK = "#CC79A7"
GRAY = "#6B7280"
LIGHT_BLUE = "#DCEEF8"
LIGHT_ORANGE = "#FCE8C3"
LIGHT_GREEN = "#DDF3E9"
LIGHT_GRAY = "#EEF1F4"
RED = "#D55E00"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.1,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 600}),
    ):
        fig.savefig(
            directory / f"{stem}{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axes(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.65)
        ax.set_axisbelow(True)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#4B5563",
    fontsize: float = 8.0,
    weight: str = "normal",
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.25,
    )
    return box


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#4B5563",
    style: str = "-|>",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=1.1,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(REV / "data/splits/split_manifest_v2.csv", low_memory=False)


def make_figure1() -> None:
    manifest = load_manifest()
    class_ok = as_bool(manifest["classification_eligible"])
    reg_ok = as_bool(manifest["regression_eligible"])
    counts = []
    for split in ("train", "validation", "test"):
        frame = manifest[manifest["split_clean"].eq(split)]
        counts.append(
            {
                "split": split,
                "classification": int(as_bool(frame["classification_eligible"]).sum()),
                "regression": int(as_bool(frame["regression_eligible"]).sum()),
                "receptor_groups": int(frame["receptor_group"].nunique()),
            }
        )
    neg = (
        manifest[class_ok & manifest["class_clean"].eq(0)]["negative_type_clean"]
        .value_counts()
        .rename_axis("negative_type")
        .reset_index(name="n")
    )
    SOURCE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts).to_csv(SOURCE / "figure1_split_counts.csv", index=False)
    neg.to_csv(SOURCE / "figure1_negative_counts.csv", index=False)

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.25),
        gridspec_kw={"width_ratios": [0.93, 1.62], "wspace": 0.08},
    )
    for ax in (ax_a, ax_b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    panel_label(ax_a, "A", x=0.0, y=1.0)
    panel_label(ax_b, "B", x=0.0, y=1.0)
    ax_a.text(0.50, 0.97, "Audited data and fixed split", ha="center", va="top", fontweight="bold", fontsize=9.5)
    rounded_box(ax_a, (0.12, 0.80), 0.76, 0.11, "Integrated records\n20,950", LIGHT_GRAY, weight="bold")
    arrow(ax_a, (0.50, 0.80), (0.50, 0.73))
    rounded_box(ax_a, (0.06, 0.60), 0.40, 0.12, "Class-eligible\nN = 20,846", LIGHT_BLUE, BLUE, fontsize=7.8, weight="bold")
    rounded_box(ax_a, (0.54, 0.60), 0.40, 0.12, "Exact-label regr.\nN = 5,016", LIGHT_ORANGE, ORANGE, fontsize=7.8, weight="bold")
    ax_a.text(0.50, 0.54, "Receptor-cluster-disjoint split", ha="center", va="center", fontsize=8.3, fontweight="bold")
    split_y = [0.39, 0.24, 0.09]
    split_colors = ["#E8F1FB", "#E7F5EF", "#FCEFD7"]
    for row, y, fc in zip(counts, split_y, split_colors):
        name = "Validation" if row["split"] == "validation" else row["split"].title()
        rounded_box(
            ax_a,
            (0.08, y),
            0.84,
            0.105,
            f"{name}: {row['classification']:,} class. | {row['regression']:,} regr.\n{row['receptor_groups']:,} receptor groups",
            fc,
            fontsize=6.4,
        )

    ax_b.text(0.50, 0.97, "HighAffinity transfer-learning workflow", ha="center", va="top", fontweight="bold", fontsize=9.5)
    rounded_box(ax_b, (0.02, 0.67), 0.20, 0.15, "Protein + peptide\nsequences", LIGHT_GRAY, fontsize=7.5, weight="bold")
    rounded_box(ax_b, (0.29, 0.64), 0.25, 0.21, "Frozen Boltz2 trunk\n506.31 M", "#E5E7EB", GRAY, fontsize=7.8, weight="bold")
    rounded_box(ax_b, (0.61, 0.67), 0.17, 0.15, "Interface\nfeatures", "#E8F1FB", BLUE, fontsize=7.8, weight="bold")
    rounded_box(ax_b, (0.81, 0.62), 0.18, 0.25, "Trainable\naffinity module\n8.74 M\n(1.697%)", LIGHT_ORANGE, ORANGE, fontsize=7.4, weight="bold")
    arrow(ax_b, (0.22, 0.745), (0.29, 0.745))
    arrow(ax_b, (0.54, 0.745), (0.61, 0.745))
    arrow(ax_b, (0.78, 0.745), (0.81, 0.745))
    rounded_box(ax_b, (0.18, 0.34), 0.27, 0.16, "Binder probability\nFocal loss", LIGHT_BLUE, BLUE, fontsize=7.0, weight="bold")
    rounded_box(ax_b, (0.55, 0.34), 0.27, 0.16, "logAffinity\nHuber loss", LIGHT_GREEN, GREEN, fontsize=7.0, weight="bold")
    arrow(ax_b, (0.90, 0.62), (0.69, 0.50))
    arrow(ax_b, (0.90, 0.62), (0.31, 0.50))
    rounded_box(
        ax_b,
        (0.18, 0.10),
        0.64,
        0.13,
        "Selection: validation set\nEvaluation: held-out test + external PEPBI",
        "#F7F7F8",
        fontsize=7.0,
    )
    arrow(ax_b, (0.50, 0.34), (0.50, 0.23))
    ax_b.text(0.50, 0.03, "The structural trunk remains frozen during affinity fine-tuning.", ha="center", va="bottom", fontsize=7.2, color="#374151")

    save_figure(fig, MAIN, "figure1_revised_pipeline")


def aligned_baseline_scores(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    test = manifest[
        manifest["split_clean"].eq("test") & as_bool(manifest["classification_eligible"])
    ].copy()
    test["id"] = test["id"].astype(str)
    specs = {
        "Boltz2 zero-shot": (
            REV / "originals/results/boltz2_native.csv",
            "affinity_probability_binary",
            "affinity_pred_value",
        ),
        "PPAP": (
            REV / "originals/results/ppap_with_Kd_uM_log10_class.csv",
            None,
            "log10_Kd_uM",
        ),
        "ProAffinity-GNN": (
            REV / "originals/results/proaffinity-gnn_parsed.csv",
            None,
            "log10_K_uM",
        ),
    }
    out: dict[str, pd.DataFrame] = {}
    for name, (path, score_col, pred_col) in specs.items():
        pred = pd.read_csv(path)
        pred["id"] = pred["id"].astype(str)
        cols = ["id", pred_col] + ([score_col] if score_col else [])
        frame = test.merge(pred[cols], on="id", how="inner", validate="one_to_one")
        if len(frame) != len(test):
            raise RuntimeError(f"{name}: {len(frame)} aligned rows; expected {len(test)}")
        frame["class_true"] = frame["class_clean"].astype(int)
        frame["affinity_pred"] = pd.to_numeric(frame[pred_col], errors="coerce")
        frame["class_score"] = (
            pd.to_numeric(frame[score_col], errors="coerce")
            if score_col
            else expit(3.0 - frame["affinity_pred"].to_numpy(float))
        )
        out[name] = frame
    return out


def make_figure2() -> None:
    manifest = load_manifest()
    model_files = {
        "HighAffinity (M0 ensemble)": REV / "predictions/ensembles/test/M0_ensemble/predictions.csv",
        "Random affinity head (M1 ensemble)": REV / "predictions/ensembles/test/M1_ensemble/predictions.csv",
    }
    colors = {
        "HighAffinity (M0 ensemble)": BLUE,
        "Random affinity head (M1 ensemble)": ORANGE,
        "Boltz2 zero-shot": GRAY,
        "PPAP": GREEN,
        "ProAffinity-GNN": PINK,
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, path in model_files.items():
        frame = pd.read_csv(path)
        frame["id"] = frame["id"].astype(str)
        frames[name] = frame
    frames.update(aligned_baseline_scores(manifest))

    scatter_models = [
        "Boltz2 zero-shot",
        "Random affinity head (M1 ensemble)",
        "HighAffinity (M0 ensemble)",
    ]
    fig = plt.figure(figsize=(7.2, 5.75))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.05], hspace=0.43, wspace=0.68)
    scatter_axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])]
    ax_roc = fig.add_subplot(gs[1, 0:3])
    ax_pr = fig.add_subplot(gs[1, 3:6])

    scatter_source = []
    for label, name, ax in zip(("A", "B", "C"), scatter_models, scatter_axes):
        frame = frames[name]
        if name == "Boltz2 zero-shot":
            eligible = as_bool(frame["regression_eligible"]) & frame["logAffinity"].notna() & frame["affinity_pred"].notna()
            truth = frame.loc[eligible, "logAffinity"].to_numpy(float)
            pred = frame.loc[eligible, "affinity_pred"].to_numpy(float)
            ids = frame.loc[eligible, "id"].astype(str).to_numpy()
        else:
            eligible = as_bool(frame["regression_eligible"]) & frame["affinity_true"].notna() & frame["affinity_pred"].notna()
            truth = frame.loc[eligible, "affinity_true"].to_numpy(float)
            pred = frame.loc[eligible, "affinity_pred"].to_numpy(float)
            ids = frame.loc[eligible, "id"].astype(str).to_numpy()
        if len(truth) != 516:
            raise RuntimeError(f"{name}: expected 516 regression rows, got {len(truth)}")
        rmse = math.sqrt(mean_squared_error(truth, pred))
        r = pearsonr(truth, pred).statistic
        ax.scatter(truth, pred, s=9, alpha=0.42, color=colors[name], edgecolors="none", rasterized=True)
        ax.plot([-5.6, 5.6], [-5.6, 5.6], linestyle="--", color="#111827", linewidth=0.9, label="Identity")
        ax.set_xlim(-5.6, 5.6)
        ax.set_ylim(-5.6, 5.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(name.replace(" (M1 ensemble)", "\n(M1 ensemble)").replace(" (M0 ensemble)", "\n(M0 ensemble)"), pad=3)
        ax.set_xlabel("Observed logAffinity")
        ax.set_ylabel("Predicted logAffinity")
        ax.text(
            0.04,
            0.95,
            f"N = {len(truth)}\nRMSE = {rmse:.3f}\nr = {r:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D1D5DB", alpha=0.90),
        )
        panel_label(ax, label, x=-0.16, y=1.12)
        clean_axes(ax, grid=False)
        scatter_source.append(pd.DataFrame({"model": name, "id": ids, "observed_logAffinity": truth, "predicted_logAffinity": pred}))

    curve_source = []
    curve_order = [
        "HighAffinity (M0 ensemble)",
        "Random affinity head (M1 ensemble)",
        "Boltz2 zero-shot",
        "PPAP",
        "ProAffinity-GNN",
    ]
    for name in curve_order:
        frame = frames[name]
        if name in model_files:
            truth = frame["class_true"].astype(int).to_numpy()
            score = frame["class_probability"].to_numpy(float)
            ids = frame["id"].astype(str).to_numpy()
        else:
            truth = frame["class_true"].astype(int).to_numpy()
            score = frame["class_score"].to_numpy(float)
            ids = frame["id"].astype(str).to_numpy()
        fpr, tpr, _ = roc_curve(truth, score)
        precision, recall, _ = precision_recall_curve(truth, score)
        auc = roc_auc_score(truth, score)
        ap = average_precision_score(truth, score)
        short_name = {
            "HighAffinity (M0 ensemble)": "HighAffinity M0",
            "Random affinity head (M1 ensemble)": "Random-head M1",
        }.get(name, name)
        ax_roc.plot(fpr, tpr, color=colors[name], label=f"{short_name} ({auc:.3f})")
        ax_pr.plot(recall, precision, color=colors[name], label=f"{short_name} ({ap:.3f})")
        curve_source.append(pd.DataFrame({"model": name, "id": ids, "class_true": truth, "class_score": score}))
    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="#9CA3AF")
    prevalence = frames["HighAffinity (M0 ensemble)"]["class_true"].astype(int).mean()
    ax_pr.axhline(prevalence, linestyle="--", linewidth=0.8, color="#9CA3AF")
    for label, ax, xlabel, ylabel, title in (
        ("D", ax_roc, "False-positive rate", "True-positive rate", "ROC curves (N = 2,496)"),
        ("E", ax_pr, "Recall", "Precision", "Precision–recall curves (N = 2,496)"),
    ):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=4)
        ax.legend(frameon=False, loc="lower right" if label == "D" else "lower left", title="Model (AUC)" if label == "D" else "Model (AP)", title_fontsize=6.7, fontsize=6.6)
        panel_label(ax, label, x=-0.13, y=1.08)
        clean_axes(ax, grid=False)
    fig.text(0.50, 0.005, "logAffinity = log10(Kd/µM); lower values indicate stronger binding.", ha="center", fontsize=7.5, color="#374151")

    SOURCE.mkdir(parents=True, exist_ok=True)
    pd.concat(scatter_source, ignore_index=True).to_csv(SOURCE / "figure2_regression_scatter.csv", index=False)
    pd.concat(curve_source, ignore_index=True).to_csv(SOURCE / "figure2_classification_scores.csv", index=False)
    save_figure(fig, MAIN, "figure2_internal_performance")


def heatmap_with_values(
    ax: plt.Axes,
    frame: pd.DataFrame,
    row_col: str,
    metrics: list[str],
    display: list[str],
    title: str,
    cmap: str,
) -> None:
    data = frame.set_index(row_col)[metrics].astype(float)
    normalized = data.copy()
    for column in metrics:
        lo, hi = data[column].min(), data[column].max()
        normalized[column] = 0.5 if math.isclose(lo, hi) else (data[column] - lo) / (hi - lo)
    ax.imshow(normalized.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metrics)), display)
    ax.set_yticks(np.arange(len(data)), data.index)
    ax.set_title(title, pad=5)
    for i in range(len(data)):
        for j, metric in enumerate(metrics):
            value = data.iloc[i, j]
            text_color = "white" if normalized.iloc[i, j] > 0.65 else "#111827"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7.3, color=text_color, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure3() -> None:
    per_run = pd.read_csv(REV / "statistics/final_test/per_run_metrics.csv")
    per_run = per_run[per_run["scope"].eq("overall")].copy()
    selected = ["M0_seed1", "M1_seed1", "M2_seed1", "M3_seed1", "M4_seed1", "M5_seed1", "M6_seed1"]
    per_run = per_run[per_run["run"].isin(selected)].copy()
    per_run["model"] = per_run["run"].str.replace("_seed1", "", regex=False)

    negative = pd.read_csv(REV / "statistics/ensembles/test/negative_subgroup_metrics.csv")
    paired = pd.read_csv(REV / "statistics/ensembles/test/paired_comparisons.csv")
    selected_metrics = ["roc_auc", "average_precision", "rmse", "pearson", "spearman"]
    paired = paired[paired["metric"].isin(selected_metrics)].copy()
    paired["advantage_low"] = np.where(
        paired["direction"].eq("lower_is_better"),
        -paired["ci_high_delta"],
        paired["ci_low_delta"],
    )
    paired["advantage_high"] = np.where(
        paired["direction"].eq("lower_is_better"),
        -paired["ci_low_delta"],
        paired["ci_high_delta"],
    )
    paired["advantage"] = paired["advantage_a"]

    SOURCE.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(SOURCE / "figure3_ablation_seed1.csv", index=False)
    negative.to_csv(SOURCE / "figure3_negative_subgroups.csv", index=False)
    paired.to_csv(SOURCE / "figure3_paired_ensemble_comparison.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), gridspec_kw={"hspace": 0.50, "wspace": 0.37})
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    class_frame = per_run[per_run["classification_enabled"].astype(str).str.lower().eq("true")]
    heatmap_with_values(
        ax_a,
        class_frame,
        "model",
        ["roc_auc", "average_precision", "specificity", "mcc"],
        ["ROC-AUC", "AP", "Specificity", "MCC"],
        "Classification ablations (seed 1)",
        "Blues",
    )
    panel_label(ax_a, "A", x=-0.10, y=1.11)

    reg_frame = per_run[per_run["regression_enabled"].astype(str).str.lower().eq("true")].copy()
    reg_frame["negative_rmse"] = -reg_frame["rmse"]
    heatmap_with_values(
        ax_b,
        reg_frame,
        "model",
        ["negative_rmse", "pearson", "spearman"],
        ["RMSE↓", "Pearson r", "Spearman ρ"],
        "Regression ablations (seed 1)",
        "YlOrBr",
    )
    # Replace the displayed negative RMSE numbers with the actual RMSE values.
    for text in ax_b.texts:
        x, y = text.get_position()
        if int(round(x)) == 0 and 0 <= int(round(y)) < len(reg_frame):
            text.set_text(f"{reg_frame.iloc[int(round(y))]['rmse']:.3f}")
    panel_label(ax_b, "B", x=-0.10, y=1.11)

    subgroup_order = ["synthetic_decoy", "bsa_structure_negative", "experimental_negative"]
    subgroup_labels = ["Synthetic decoy\n(N=861)", "BSA-derived\n(N=111)", "Experimental\n(N=81)"]
    x = np.arange(len(subgroup_order))
    width = 0.25
    for offset, run, color, label in ((-width / 1.8, "M0_ensemble", BLUE, "M0 ensemble"), (width / 1.8, "M1_ensemble", ORANGE, "M1 ensemble")):
        mask = negative["run"].eq(run)
        sub = negative[mask].set_index("negative_type").loc[subgroup_order]
        values = sub["specificity"].to_numpy(float)
        lower = values - sub["specificity_ci_low"].to_numpy(float)
        upper = sub["specificity_ci_high"].to_numpy(float) - values
        ax_c.errorbar(x + offset, values, yerr=[lower, upper], fmt="o", color=color, markersize=5.0, capsize=2.5, linewidth=1.2, label=label)
    ax_c.set_xticks(x, subgroup_labels)
    ax_c.set_ylim(0, 1.02)
    ax_c.set_ylabel("Specificity at threshold 0.5")
    ax_c.set_title("Performance on negative subgroups", pad=5)
    ax_c.legend(frameon=False, loc="upper right")
    ax_c.text(0.02, 0.04, "95% receptor-cluster bootstrap CIs", transform=ax_c.transAxes, fontsize=7.0, color="#4B5563")
    clean_axes(ax_c)
    panel_label(ax_c, "C", x=-0.10, y=1.11)

    metric_order = ["roc_auc", "average_precision", "rmse", "pearson", "spearman"]
    metric_labels = ["ROC-AUC", "Average precision", "RMSE", "Pearson r", "Spearman ρ"]
    plot = paired.set_index("metric").loc[metric_order].reset_index()
    y = np.arange(len(plot))[::-1]
    values = plot["advantage"].to_numpy(float)
    lower = values - plot["advantage_low"].to_numpy(float)
    upper = plot["advantage_high"].to_numpy(float) - values
    point_colors = [BLUE if lo > 0 else GRAY for lo in plot["advantage_low"]]
    for yi, value, lo, hi, color in zip(y, values, lower, upper, point_colors):
        ax_d.errorbar(value, yi, xerr=[[lo], [hi]], fmt="o", color=color, capsize=2.5, markersize=5.0, linewidth=1.2)
    ax_d.axvline(0, color="#111827", linewidth=0.9, linestyle="--")
    ax_d.set_yticks(y, metric_labels)
    ax_d.set_xlabel("M0 advantage over M1")
    ax_d.set_title("Paired ensemble comparison", pad=5)
    clean_axes(ax_d, grid=False)
    panel_label(ax_d, "D", x=-0.10, y=1.11)
    fig.text(0.50, 0.005, "Panels A–B are exploratory single-seed ablations; panels C–D use three-seed prediction ensembles.", ha="center", fontsize=7.4, color="#374151")
    save_figure(fig, MAIN, "figure3_ablation_and_negative_validation")


def make_figure4() -> None:
    obs = pd.read_csv(REV / "external/pepbi/results/M0_ensemble/observation_predictions.csv")
    pdb_subset = pd.read_csv(REV / "external/pepbi/processed/pepbi_external_pdb_disjoint.csv")
    pdb_ids = set(pdb_subset["experimental_complex_id"].astype(str))
    pdb_obs = obs[obs["id"].astype(str).isin(pdb_ids)].copy()
    if len(obs) != 177 or len(pdb_obs) != 69:
        raise RuntimeError(f"Unexpected PEPBI sizes: pair={len(obs)}, PDB={len(pdb_obs)}")
    ci = pd.read_csv(REV / "external/pepbi/results/M0_ensemble/binding_group_bootstrap_ci.csv")
    obs = obs.copy()
    obs["residual"] = obs["affinity_pred"] - obs["logAffinity"]
    pdb_obs["residual"] = pdb_obs["affinity_pred"] - pdb_obs["logAffinity"]
    SOURCE.mkdir(parents=True, exist_ok=True)
    obs.to_csv(SOURCE / "figure4_pepbi_pair_disjoint_predictions.csv", index=False)
    pdb_obs.to_csv(SOURCE / "figure4_pepbi_pdb_disjoint_predictions.csv", index=False)
    ci.to_csv(SOURCE / "figure4_pepbi_bootstrap_ci.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), gridspec_kw={"hspace": 0.45, "wspace": 0.36})
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    for label, ax, frame, title, n_groups in (
        ("A", ax_a, obs, "PEPBI pair-disjoint", 31),
        ("B", ax_b, pdb_obs, "PEPBI PDB-disjoint", 4),
    ):
        truth = frame["logAffinity"].to_numpy(float)
        pred = frame["affinity_pred"].to_numpy(float)
        rmse = math.sqrt(mean_squared_error(truth, pred))
        r = pearsonr(truth, pred).statistic
        ax.scatter(truth, pred, s=17, alpha=0.55, color=BLUE, edgecolors="white", linewidths=0.25, rasterized=True)
        limits = (-2.6, 3.2)
        ax.plot(limits, limits, linestyle="--", color="#111827", linewidth=0.9)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed logAffinity")
        ax.set_ylabel("Predicted logAffinity")
        ax.set_title(title, pad=4)
        ax.text(
            0.04,
            0.95,
            f"N = {len(frame)}; groups = {n_groups}\nRMSE = {rmse:.3f}; r = {r:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.3,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D1D5DB", alpha=0.90),
        )
        clean_axes(ax, grid=False)
        panel_label(ax, label, x=-0.12, y=1.10)

    ax_c.scatter(obs["logAffinity"], obs["residual"], s=17, alpha=0.50, color=GREEN, edgecolors="none", rasterized=True)
    ax_c.axhline(0, linestyle="--", color="#111827", linewidth=0.9)
    # A simple binned median trend avoids imposing a linear relationship.
    bins = pd.qcut(obs["logAffinity"], q=6, duplicates="drop")
    trend = obs.assign(bin=bins).groupby("bin", observed=True).agg(x=("logAffinity", "median"), y=("residual", "median"))
    ax_c.plot(trend["x"], trend["y"], color=RED, marker="o", markersize=3.5, linewidth=1.4, label="Binned median")
    ax_c.set_xlabel("Observed logAffinity")
    ax_c.set_ylabel("Residual (predicted − observed)")
    ax_c.set_title("Pair-disjoint residuals", pad=4)
    ax_c.legend(frameon=False, loc="upper left")
    clean_axes(ax_c)
    panel_label(ax_c, "C", x=-0.12, y=1.10)

    corr = ci[ci["metric"].isin(["pearson", "spearman"])].copy()
    corr["label"] = corr["subset"].str.replace("PEPBI ", "", regex=False) + " | " + corr["metric"].map({"pearson": "Pearson r", "spearman": "Spearman ρ"})
    corr_order = [
        "pair-disjoint | Pearson r",
        "pair-disjoint | Spearman ρ",
        "PDB-disjoint | Pearson r",
        "PDB-disjoint | Spearman ρ",
    ]
    corr = corr.set_index("label").loc[corr_order].reset_index()
    y = np.arange(len(corr))[::-1]
    values = corr["estimate"].to_numpy(float)
    lower = values - corr["ci_low"].to_numpy(float)
    upper = corr["ci_high"].to_numpy(float) - values
    colors = [BLUE, BLUE, GRAY, GRAY]
    for yi, value, lo, hi, color in zip(y, values, lower, upper, colors):
        ax_d.errorbar(value, yi, xerr=[[lo], [hi]], fmt="o", color=color, capsize=2.5, markersize=5.0, linewidth=1.2)
    ax_d.axvline(0, color="#111827", linewidth=0.9, linestyle="--")
    ax_d.set_xlim(-1.05, 1.05)
    ax_d.set_yticks(y, ["Pair | Pearson r", "Pair | Spearman ρ", "PDB | Pearson r", "PDB | Spearman ρ"])
    ax_d.set_xlabel("Correlation (95% binding-group bootstrap CI)")
    ax_d.set_title("External-validation uncertainty", pad=4)
    clean_axes(ax_d, grid=False)
    panel_label(ax_d, "D", x=-0.12, y=1.10)
    fig.text(0.50, 0.005, "logAffinity = log10(Kd/µM); PEPBI was not used for training or model selection.", ha="center", fontsize=7.4, color="#374151")
    save_figure(fig, MAIN, "figure4_pepbi_external_validation")


def make_supplementary_dataset_figure() -> None:
    manifest = load_manifest()
    class_ok = as_bool(manifest["classification_eligible"])
    reg_ok = as_bool(manifest["regression_eligible"])
    split_order = ["train", "validation", "test"]
    split_labels = ["Train", "Validation", "Test"]
    split_colors = [BLUE, GREEN, ORANGE]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), gridspec_kw={"hspace": 0.48, "wspace": 0.38})
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    ax_a.axis("off")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    rounded_box(ax_a, (0.05, 0.68), 0.90, 0.18, "Integrated records\n20,950", LIGHT_GRAY, weight="bold")
    arrow(ax_a, (0.50, 0.68), (0.50, 0.56))
    rounded_box(ax_a, (0.02, 0.30), 0.45, 0.22, "Classification-eligible\nN = 20,846", LIGHT_BLUE, BLUE, fontsize=7.1, weight="bold")
    rounded_box(ax_a, (0.53, 0.30), 0.45, 0.22, "Exact regression\nN = 5,016", LIGHT_ORANGE, ORANGE, fontsize=7.1, weight="bold")
    ax_a.set_title("Label audit", pad=4)
    panel_label(ax_a, "A", x=-0.03, y=1.10)

    class_counts = []
    reg_counts = []
    for split in split_order:
        sub = manifest[manifest["split_clean"].eq(split)]
        class_counts.append(as_bool(sub["classification_eligible"]).sum())
        reg_counts.append(as_bool(sub["regression_eligible"]).sum())
    x = np.arange(3)
    ax_b.bar(x - 0.18, class_counts, 0.36, color=BLUE, label="Classification")
    ax_b.bar(x + 0.18, reg_counts, 0.36, color=ORANGE, label="Exact regression")
    for i, (c, r) in enumerate(zip(class_counts, reg_counts)):
        ax_b.text(i - 0.18, c + 250, f"{c:,}", ha="center", va="bottom", fontsize=7.0)
        ax_b.text(i + 0.18, r + 250, f"{r:,}", ha="center", va="bottom", fontsize=7.0)
    ax_b.set_xticks(x, split_labels)
    ax_b.set_ylabel("Number of observations")
    ax_b.set_title("Fixed receptor-cluster split", pad=4)
    ax_b.legend(frameon=False)
    clean_axes(ax_b)
    panel_label(ax_b, "B", x=-0.12, y=1.10)

    bins = np.linspace(-5.5, 5.5, 28)
    for split, label, color in zip(split_order, split_labels, split_colors):
        values = manifest[manifest["split_clean"].eq(split) & reg_ok]["logAffinity"].dropna().to_numpy(float)
        ax_c.hist(values, bins=bins, histtype="step", density=True, linewidth=1.6, color=color, label=f"{label} (N={len(values):,})")
    ax_c.set_xlabel("Observed logAffinity")
    ax_c.set_ylabel("Density")
    ax_c.set_title("Exact-label affinity distributions", pad=4)
    ax_c.legend(frameon=False)
    clean_axes(ax_c)
    panel_label(ax_c, "C", x=-0.12, y=1.10)

    neg_types = ["synthetic_decoy", "bsa_structure_negative", "experimental_negative"]
    neg_labels = ["Synthetic decoy", "BSA-derived", "Experimental"]
    bottom = np.zeros(3)
    for neg_type, label, color in zip(neg_types, neg_labels, [BLUE, GREEN, ORANGE]):
        values = []
        for split in split_order:
            sub = manifest[manifest["split_clean"].eq(split) & class_ok & manifest["class_clean"].eq(0)]
            values.append(int(sub["negative_type_clean"].eq(neg_type).sum()))
        ax_d.bar(x, values, bottom=bottom, color=color, label=label)
        bottom += np.asarray(values)
    ax_d.set_xticks(x, split_labels)
    ax_d.set_ylabel("Number of negative observations")
    ax_d.set_title("Negative-source composition", pad=4)
    ax_d.legend(frameon=False, fontsize=7.0)
    clean_axes(ax_d)
    panel_label(ax_d, "D", x=-0.12, y=1.10)
    fig.text(0.50, 0.005, "Exact regression labels exclude censored, range-only, structure-derived, and negative-only records.", ha="center", fontsize=7.4, color="#374151")
    save_figure(fig, SUPP, "figure_s1_dataset_audit")


def main() -> None:
    configure_style()
    for directory in (MAIN, SUPP, SOURCE):
        directory.mkdir(parents=True, exist_ok=True)
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_supplementary_dataset_figure()
    print(f"Figures written to: {OUT}")


if __name__ == "__main__":
    main()
