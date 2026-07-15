#!/usr/bin/env python3
"""Create statistical tables and publication-quality figures from JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import wilcoxon


METHOD_LABELS = {
    "frequency_only": "Frequency only",
    "padim_diag": "PaDiM-Diag",
    "patchcore_lite": "PatchCore-Lite",
    "freqpatch_lite": "CalibFreq-Patch (ours)",
    "padim_freq_gate": "PaDiM-Diag + gate",
    "patchcore_wr50_compact": "PatchCore-WR50-256",
}


def clustered_ci95(
    group: pd.DataFrame, metric: str, repetitions: int, seed: int
) -> tuple[float, float]:
    """Bootstrap categories, retaining all seed repetitions within a draw."""
    category_means = group.groupby("category")[metric].mean().dropna().to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(category_means), size=(repetitions, len(category_means)))
    boot = category_means[sampled].mean(axis=1)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def paired_effect_ci95(
    differences: np.ndarray, repetitions: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(differences), size=(repetitions, len(differences)))
    boot = differences[sampled].mean(axis=1)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--input", type=Path, default=Path("results/raw/experiments.jsonl"))
    parser.add_argument("--strong-baseline", type=Path, default=Path("results/raw/strong_baseline.jsonl"))
    parser.add_argument("--tables", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/mvtec/samples.json"))
    args = parser.parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.strong_baseline.exists():
        rows.extend(
            json.loads(line)
            for line in args.strong_baseline.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    data = pd.DataFrame(rows)
    main = data[data.experiment.isin(["main", "strong_baseline"])].copy()

    metrics = [
        "image_auroc", "image_ap", "pixel_auroc", "pixel_ap",
        "image_f1_conformal", "image_precision_conformal", "image_recall_conformal",
        "image_specificity_conformal", "image_balanced_accuracy_conformal",
        "normal_fpr_conformal", "false_alarms_per_1000_normals",
    ]
    summary_rows = []
    for method, group in main.groupby("method"):
        row = {"method": METHOD_LABELS.get(method, method)}
        for metric in metrics:
            low, high = clustered_ci95(
                group, metric, int(cfg["bootstrap_repetitions"]), int(cfg["seed"])
            )
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("image_auroc_mean", ascending=False)
    summary.to_csv(args.tables / "main_summary.csv", index=False)
    category = main.pivot_table(index="category", columns="method", values="image_auroc", aggfunc="mean")
    category.to_csv(args.tables / "category_image_auroc.csv")
    category_metrics = main.pivot_table(
        index="category", columns="method", values=metrics, aggfunc="mean"
    )
    category_metrics.to_csv(args.tables / "category_metrics.csv")
    main.sort_values(["category", "seed", "method"]).to_csv(
        args.tables / "main_per_category_seed.csv", index=False
    )

    test_rows = []
    for metric in ("image_auroc", "pixel_auroc", "pixel_ap"):
        paired = main.pivot_table(index="category", columns="method", values=metric, aggfunc="mean")
        differences = paired["freqpatch_lite"] - paired["patchcore_lite"]
        ci_low, ci_high = paired_effect_ci95(
            differences.to_numpy(float), int(cfg["bootstrap_repetitions"]), int(cfg["seed"])
        )
        if np.allclose(differences, 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(
                paired["freqpatch_lite"], paired["patchcore_lite"],
                alternative="two-sided", zero_method="wilcox", correction=False,
                method="approx",
            )
        test_rows.append({
            "metric": metric,
            "n_category_pairs": int(len(paired)),
            "statistic": float(statistic),
            "p_value_raw": float(p_value),
            "mean_difference": float(differences.mean()),
            "median_difference": float(differences.median()),
            "mean_difference_ci_low": float(ci_low),
            "mean_difference_ci_high": float(ci_high),
            "categories_improved": int((differences > 0).sum()),
            "categories_tied": int(np.isclose(differences, 0).sum()),
        })
    order = np.argsort([row["p_value_raw"] for row in test_rows])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, test_rows[index]["p_value_raw"] * (len(test_rows) - rank))
        running = max(running, adjusted)
        test_rows[index]["p_value_holm"] = running
    stats = {
        "test": f"two-sided Wilcoxon signed-rank on {main.category.nunique()} category means",
        "comparison": "CalibFreq-Patch versus PatchCore-Lite",
        "zero_and_tie_handling": (
            "SciPy zero_method='wilcox' discards exact zero differences; "
            "method='approx' uses the normal approximation with tie correction and no continuity correction"
        ),
        "multiplicity": "Holm correction across three prespecified ranking/localization metrics",
        "tests": test_rows,
    }
    (args.tables / "significance.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["samples"]
    manifest_rows = [
        {
            "category": row["category"]["label"],
            "split": row["split"],
            "normal": row["defect"]["label"] == "good",
        }
        for row in manifest
    ]
    sample_frame = pd.DataFrame(manifest_rows)
    dataset_table = []
    for category_name, group in sample_frame.groupby("category"):
        dataset_table.append({
            "category": category_name,
            "train_normal": int(((group.split == "train") & group.normal).sum()),
            "test_normal": int(((group.split == "test") & group.normal).sum()),
            "test_anomalous": int(((group.split == "test") & ~group.normal).sum()),
        })
    pd.DataFrame(dataset_table).sort_values("category").to_csv(args.tables / "dataset_counts.csv", index=False)

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    order = [
        "frequency_only", "padim_diag", "padim_freq_gate",
        "patchcore_wr50_compact", "patchcore_lite", "freqpatch_lite",
    ]
    order = [method for method in order if method in set(main.method)]
    grayscale = ["0.86", "0.72", "0.60", "0.48", "0.34", "0.16"][:len(order)]
    hatches = ["", "//", "xx", "++", "..", "\\\\"][:len(order)]
    figure, axes = plt.subplots(1, 3, figsize=(4.85, 2.35), constrained_layout=True)
    plotted = [("image_auroc", "Image AUROC"), ("pixel_auroc", "Pixel AUROC"), ("pixel_ap", "Pixel AP")]
    for axis, (metric, title) in zip(axes, plotted):
        values, lows, highs = [], [], []
        for method in order:
            group = main[main.method == method]
            mean = float(group[metric].mean())
            low, high = clustered_ci95(
                group, metric, int(cfg["bootstrap_repetitions"]), int(cfg["seed"])
            )
            values.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)
        bars = axis.bar(
            range(len(order)), values, yerr=np.asarray([lows, highs]), color=grayscale,
            edgecolor="black", linewidth=0.7, capsize=2,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        axis.set_xticks(
            range(len(order)), [str(index + 1) for index in range(len(order))],
            rotation=0, ha="center", fontsize=8,
        )
        axis.set_xlabel("")
        axis.set_ylabel(title)
        axis.set_ylim(max(0.0, main[metric].min() - 0.08), 1.0)
    figure.savefig(args.figures / "main_performance.pdf", dpi=300, bbox_inches="tight")
    figure.savefig(args.figures / "main_performance.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    few = data[data.experiment == "few_shot"].copy()
    if not few.empty:
        requested_counts = list(cfg["few_shot_counts"])
        few["requested_fit_images"] = (
            few.groupby(["category", "method"], sort=False).cumcount()
            .map(dict(enumerate(requested_counts)))
        )
        few.groupby(["method", "requested_fit_images"])[metrics].agg(["mean", "std"]).to_csv(
            args.tables / "few_shot.csv"
        )
        few.groupby(["requested_fit_images", "category"])[
            ["fit_images", "branch_calibration_images", "threshold_calibration_images", "total_normal_images"]
        ].first().to_csv(args.tables / "few_shot_split_counts.csv")
        figure, axis = plt.subplots(figsize=(4.4, 3.1), constrained_layout=True)
        sns.lineplot(data=few, x="requested_fit_images", y="image_auroc", hue="method", marker="o", errorbar=("ci", 95), ax=axis)
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles, [METHOD_LABELS.get(x, x) for x in labels], frameon=True)
        axis.set(xlabel="Requested fit-normal count", ylabel="Image-level AUROC", ylim=(max(0.5, few.image_auroc.min() - 0.05), 1.0))
        figure.savefig(args.figures / "few_shot.pdf", dpi=300, bbox_inches="tight")
        figure.savefig(args.figures / "few_shot.png", dpi=300, bbox_inches="tight")
        plt.close(figure)

    total_budget = data[data.experiment == "total_budget"].copy()
    if not total_budget.empty:
        total_budget.groupby(["method", "normal_budget"])[metrics].agg(["mean", "std"]).to_csv(
            args.tables / "total_normal_budget.csv"
        )
        split_counts = total_budget.groupby(
            ["normal_budget", "category"]
        )[["fit_images", "branch_calibration_images", "threshold_calibration_images"]].first()
        split_counts.to_csv(args.tables / "total_normal_budget_splits.csv")
        figure, axis = plt.subplots(figsize=(4.4, 3.1), constrained_layout=True)
        for method, marker, linestyle, color in (
            ("patchcore_lite", "s", "--", "0.45"),
            ("freqpatch_lite", "o", "-", "0.05"),
        ):
            grouped = total_budget[total_budget.method == method].groupby("normal_budget")["pixel_ap"]
            axis.plot(
                grouped.mean().index, grouped.mean().values, marker=marker,
                linestyle=linestyle, color=color, label=METHOD_LABELS[method],
            )
        axis.set(xlabel="Total normal-image budget", ylabel="Pixel AP")
        axis.legend(frameon=True)
        figure.savefig(args.figures / "total_normal_budget.pdf", bbox_inches="tight")
        figure.savefig(args.figures / "total_normal_budget.png", dpi=300, bbox_inches="tight")
        plt.close(figure)

    ablation = data[data.experiment == "ablation_weight"].copy()
    ablation.groupby("weight")[metrics].agg(["mean", "std"]).to_csv(args.tables / "ablation_weight.csv")
    fusion = data[data.experiment == "ablation_fusion"].copy()
    if not fusion.empty:
        fusion.groupby("method")[metrics].agg(["mean", "std"]).to_csv(
            args.tables / "ablation_fusion.csv"
        )
    upper = data[data.experiment == "ablation_upper_quantile"].copy()
    if not upper.empty:
        upper.groupby("upper_quantile")[metrics].agg(["mean", "std"]).to_csv(
            args.tables / "ablation_upper_quantile.csv"
        )
    fraction = data[data.experiment == "ablation_calibration_fraction"].copy()
    if not fraction.empty:
        fraction.groupby("branch_calibration_fraction")[metrics].agg(["mean", "std"]).to_csv(
            args.tables / "ablation_calibration_fraction.csv"
        )
    robustness = data[data.experiment == "robustness"].copy()
    if not robustness.empty:
        robustness.groupby(["method", "translation_px", "brightness"])[metrics].mean().to_csv(args.tables / "robustness.csv")

    operating_rows = []
    for method, group in main.groupby("method"):
        category_level = group.groupby("category").agg(
            normal_fpr=("normal_fpr_conformal", "mean"),
            false_alarms_per_1000=("false_alarms_per_1000_normals", "mean"),
        )
        fpr_low, fpr_high = clustered_ci95(
            group, "normal_fpr_conformal", int(cfg["bootstrap_repetitions"]), int(cfg["seed"])
        )
        operating_rows.append({
            "method": METHOD_LABELS.get(method, method),
            "macro_fpr": float(group["normal_fpr_conformal"].mean()),
            "category_cluster_ci_low": float(fpr_low),
            "category_cluster_ci_high": float(fpr_high),
            "category_fpr_min": float(category_level["normal_fpr"].min()),
            "category_fpr_q1": float(category_level["normal_fpr"].quantile(0.25)),
            "category_fpr_median": float(category_level["normal_fpr"].median()),
            "category_fpr_q3": float(category_level["normal_fpr"].quantile(0.75)),
            "category_fpr_max": float(category_level["normal_fpr"].max()),
            "macro_false_alarms_per_1000": float(group["false_alarms_per_1000_normals"].mean()),
            "pooled_test_false_positive": int(group["test_false_positive"].sum()),
            "pooled_test_true_negative": int(group["test_true_negative"].sum()),
            "pooled_fpr": float(
                group["test_false_positive"].sum()
                / (group["test_false_positive"].sum() + group["test_true_negative"].sum())
            ),
        })
    pd.DataFrame(operating_rows).to_csv(args.tables / "operating_point_uncertainty.csv", index=False)
    print(summary.to_string(index=False))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
