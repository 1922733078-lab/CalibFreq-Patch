#!/usr/bin/env python3
"""Create statistical tables and publication-quality figures from JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import wilcoxon


METHOD_LABELS = {
    "frequency_only": "Frequency only",
    "padim_diag": "PaDiM-Diag",
    "patchcore_lite": "Compact R18-PM",
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


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Exact two-sided sign-flip randomization p value for category effects."""
    values = np.asarray(differences, dtype=float)
    if len(values) > 20:
        raise ValueError("Exact sign-flip enumeration is limited to at most 20 pairs")
    observed = abs(float(values.mean()))
    assignments = np.arange(1 << len(values), dtype=np.uint32)[:, None]
    bits = (assignments >> np.arange(len(values), dtype=np.uint32)) & 1
    signs = bits.astype(np.float64) * 2.0 - 1.0
    permuted = np.abs((signs * values).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def holm_adjust(records: list[dict], key: str = "p_value_raw") -> None:
    """Attach monotone Holm-adjusted p values to a prespecified family."""
    order = np.argsort([record[key] for record in records])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, records[index][key] * (len(records) - rank))
        running = max(running, adjusted)
        records[index]["p_value_holm"] = running


def paired_control_family(
    frame: pd.DataFrame,
    control_column: str,
    baseline,
    alternatives,
    metrics,
    repetitions: int,
    seed: int,
) -> list[dict]:
    records = []
    for metric in metrics:
        paired = frame.pivot_table(
            index=["category", "seed"], columns=control_column, values=metric
        ).groupby("category").mean()
        for alternative in alternatives:
            differences = paired[alternative] - paired[baseline]
            low, high = paired_effect_ci95(differences.to_numpy(float), repetitions, seed)
            if np.allclose(differences, 0):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = wilcoxon(
                    paired[alternative], paired[baseline], alternative="two-sided",
                    zero_method="wilcox", correction=False, method="approx",
                )
            records.append({
                "metric": metric, "baseline": baseline, "alternative": alternative,
                "n_categories": int(len(differences)),
                "n_nonzero_differences": int((~np.isclose(differences, 0)).sum()),
                "n_zero_differences": int(np.isclose(differences, 0).sum()),
                "mean_difference": float(differences.mean()),
                "ci95_low": float(low), "ci95_high": float(high),
                "wilcoxon_statistic": float(statistic), "p_value_raw": float(p_value),
                "sign_flip_p_value_raw": exact_sign_flip_pvalue(differences.to_numpy(float)),
            })
    holm_adjust(records)
    holm_adjust(records, key="sign_flip_p_value_raw")
    for record in records:
        record["sign_flip_p_value_holm"] = record.pop("p_value_holm")
    holm_adjust(records)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--input", type=Path, default=Path("results/raw/experiments.jsonl"))
    parser.add_argument("--strong-baseline", type=Path, default=Path("results/raw/strong_baseline.jsonl"))
    parser.add_argument(
        "--multiseed-ablation", type=Path,
        default=Path("results/raw/multiseed_ablation.jsonl"),
    )
    parser.add_argument(
        "--shift-diagnostics", type=Path,
        default=Path("results/raw/shift_diagnostics.jsonl"),
    )
    parser.add_argument(
        "--wr50-gate-control", type=Path,
        default=Path("results/raw/wr50_gate_control.jsonl"),
    )
    parser.add_argument(
        "--threshold-priority", type=Path,
        default=Path("results/raw/threshold_priority.jsonl"),
    )
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
    ablation_rows = []
    if args.multiseed_ablation.exists():
        ablation_rows = [
            json.loads(line)
            for line in args.multiseed_ablation.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows.extend(ablation_rows)
    if args.shift_diagnostics.exists():
        rows.extend(
            json.loads(line)
            for line in args.shift_diagnostics.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if args.wr50_gate_control.exists():
        rows.extend(
            json.loads(line)
            for line in args.wr50_gate_control.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if args.threshold_priority.exists():
        rows.extend(
            json.loads(line)
            for line in args.threshold_priority.read_text(encoding="utf-8").splitlines()
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
            "n_nonzero_differences": int((~np.isclose(differences, 0)).sum()),
            "n_zero_differences": int(np.isclose(differences, 0).sum()),
            "statistic": float(statistic),
            "p_value_raw": float(p_value),
            "sign_flip_p_value_raw": exact_sign_flip_pvalue(differences.to_numpy(float)),
            "mean_difference": float(differences.mean()),
            "median_difference": float(differences.median()),
            "mean_difference_ci_low": float(ci_low),
            "mean_difference_ci_high": float(ci_high),
            "categories_improved": int((differences > 0).sum()),
            "categories_tied": int(np.isclose(differences, 0).sum()),
        })
    holm_adjust(test_rows)
    holm_adjust(test_rows, key="sign_flip_p_value_raw")
    for row in test_rows:
        row["sign_flip_p_value_holm"] = row.pop("p_value_holm")
    holm_adjust(test_rows)
    stats = {
        "test": f"two-sided Wilcoxon signed-rank on {main.category.nunique()} category means",
        "comparison": "CalibFreq-Patch versus Compact R18-PM",
        "zero_and_tie_handling": (
            "SciPy zero_method='wilcox' discards exact zero differences; "
            "method='approx' uses the normal approximation with tie correction and no continuity correction"
        ),
        "robustness_test": (
            "Exact two-sided sign-flip randomization over all 2^15 category-level sign assignments; "
            "zero differences are retained and contribute zero under every assignment"
        ),
        "multiplicity": "Holm correction across three prespecified ranking/localization metrics",
        "tests": test_rows,
    }
    (args.tables / "significance.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    development_categories = {"bottle", "capsule", "pill", "grid", "leather", "tile"}
    development_split_rows = []
    main_pairs = main.pivot_table(
        index="category", columns="method",
        values=["image_auroc", "pixel_auroc", "pixel_ap"], aggfunc="mean",
    )
    for category_name in main_pairs.index:
        for metric in ("image_auroc", "pixel_auroc", "pixel_ap"):
            development_split_rows.append({
                "category": category_name,
                "historical_role": "development" if category_name in development_categories else "held_out_confirmation",
                "metric": metric,
                "gate_minus_patchcore": float(
                    main_pairs.loc[category_name, (metric, "freqpatch_lite")]
                    - main_pairs.loc[category_name, (metric, "patchcore_lite")]
                ),
            })
    development_frame = pd.DataFrame(development_split_rows)
    development_frame.to_csv(args.tables / "development_confirmation_split.csv", index=False)
    split_inference = []
    for role, role_group in development_frame.groupby("historical_role"):
        for metric, metric_group in role_group.groupby("metric"):
            differences = metric_group["gate_minus_patchcore"].to_numpy(float)
            low, high = paired_effect_ci95(
                differences, int(cfg["bootstrap_repetitions"]), int(cfg["seed"])
            )
            if np.allclose(differences, 0):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = wilcoxon(
                    differences, alternative="two-sided", zero_method="wilcox",
                    correction=False, method="approx",
                )
            split_inference.append({
                "historical_role": role, "metric": metric,
                "n_categories": int(len(differences)), "mean_difference": float(differences.mean()),
                "ci95_low": float(low), "ci95_high": float(high),
                "wilcoxon_statistic": float(statistic), "p_value_raw": float(p_value),
            })
    (args.tables / "development_confirmation_inference.json").write_text(
        json.dumps(split_inference, indent=2), encoding="utf-8"
    )

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
    figure, axes = plt.subplots(1, 3, figsize=(6.8, 2.45), constrained_layout=True)
    plotted = [
        ("image_auroc", "Image AUROC"),
        ("pixel_auroc", "Pixel AUROC"),
        ("pixel_ap", "Pixel AP"),
    ]
    for axis, (metric, title) in zip(axes, plotted):
        paired = main.pivot_table(
            index="category", columns="method", values=metric, aggfunc="mean"
        )
        differences = (paired["freqpatch_lite"] - paired["patchcore_lite"]).sort_index()
        low, high = paired_effect_ci95(
            differences.to_numpy(float), int(cfg["bootstrap_repetitions"]), int(cfg["seed"])
        )
        jitter = np.linspace(-0.20, 0.20, len(differences))
        axis.axvline(0.0, color="0.30", linewidth=0.8, linestyle="--")
        axis.scatter(
            differences.to_numpy(float), jitter, s=17, facecolor="0.72",
            edgecolor="black", linewidth=0.4, zorder=3,
        )
        mean = float(differences.mean())
        axis.errorbar(
            mean, 0.42, xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="D", color="black", capsize=3, markersize=4.5, linewidth=1.1,
            label="Mean and 95% category CI",
        )
        axis.set_title(title, fontsize=8.5)
        axis.set_ylim(-0.30, 0.58)
        axis.set_yticks([])
        axis.set_xlabel("Gate - compact baseline")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.tick_params(axis="x", labelsize=7)
    axes[0].legend(frameon=True, fontsize=6.5, loc="upper left")
    figure.savefig(args.figures / "main_performance.pdf", bbox_inches="tight")
    figure.savefig(args.figures / "main_performance.png", dpi=800, bbox_inches="tight")
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
        figure.savefig(args.figures / "few_shot.pdf", bbox_inches="tight")
        figure.savefig(args.figures / "few_shot.png", dpi=800, bbox_inches="tight")
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

    threshold_priority = data[data.experiment == "threshold_priority"].copy()
    if not threshold_priority.empty:
        summary_rows = []
        for (method, budget, strategy), group in threshold_priority.groupby(
            ["method", "normal_budget_requested", "allocation_strategy"]
        ):
            row = {
                "method": method,
                "normal_budget_requested": int(budget),
                "allocation_strategy": strategy,
                "finite_categories": int(
                    group.groupby("category")["threshold_is_finite"].all().sum()
                ),
                "category_seed_rows": int(len(group)),
            }
            for metric in (
                "image_auroc", "pixel_ap", "normal_fpr_conformal",
                "image_recall_conformal",
            ):
                low, high = clustered_ci95(
                    group, metric, int(cfg["bootstrap_repetitions"]), int(cfg["seed"])
                )
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_ci_low"] = float(low)
                row[f"{metric}_ci_high"] = float(high)
            summary_rows.append(row)
        priority_summary = pd.DataFrame(summary_rows).sort_values(
            ["normal_budget_requested", "allocation_strategy", "method"]
        )
        priority_summary.to_csv(args.tables / "threshold_priority_summary.csv", index=False)
        threshold_priority.sort_values(
            ["normal_budget_requested", "allocation_strategy", "category", "seed", "method"]
        ).to_csv(args.tables / "threshold_priority_per_category_seed.csv", index=False)
        threshold_priority.groupby(
            ["normal_budget_requested", "allocation_strategy", "category"]
        )[[
            "normal_budget_achieved", "fit_images", "branch_calibration_images",
            "threshold_calibration_images", "priority_achieved",
        ]].first().to_csv(args.tables / "threshold_priority_splits.csv")

        figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.75), constrained_layout=True)
        style = {
            ("patchcore_lite", "proportional"): ("s", "--", "0.55", "Compact, 70/15/15"),
            ("freqpatch_lite", "proportional"): ("o", "-", "0.15", "Gate, 70/15/15"),
            ("patchcore_lite", "threshold_prioritized"): ("^", ":", "0.55", "Compact, threshold-first"),
            ("freqpatch_lite", "threshold_prioritized"): ("D", "-.", "0.15", "Gate, threshold-first"),
        }
        for (method, strategy), (marker, linestyle, color, label) in style.items():
            subset = priority_summary[
                (priority_summary.method == method)
                & (priority_summary.allocation_strategy == strategy)
            ].sort_values("normal_budget_requested")
            x = subset.normal_budget_requested.to_numpy(float)
            y = subset.pixel_ap_mean.to_numpy(float)
            axes[0].errorbar(
                x, y,
                yerr=np.asarray([
                    y - subset.pixel_ap_ci_low.to_numpy(float),
                    subset.pixel_ap_ci_high.to_numpy(float) - y,
                ]),
                marker=marker, linestyle=linestyle, color=color, capsize=2,
                linewidth=1.0, markersize=4, label=label,
            )
        axes[0].set(xlabel="Total normal-image budget", ylabel="Pixel AP")
        axes[0].set_title("(a) Ranking with category 95% CI")
        axes[0].legend(frameon=True, fontsize=6.2)

        op_styles = {
            ("normal_fpr_conformal", "proportional"): ("o", "-", "0.15", "FPR, 70/15/15"),
            ("normal_fpr_conformal", "threshold_prioritized"): ("D", "-.", "0.15", "FPR, threshold-first"),
            ("image_recall_conformal", "proportional"): ("s", "--", "0.58", "Recall, 70/15/15"),
            ("image_recall_conformal", "threshold_prioritized"): ("^", ":", "0.58", "Recall, threshold-first"),
        }
        gate = priority_summary[priority_summary.method == "freqpatch_lite"]
        for (metric, strategy), (marker, linestyle, color, label) in op_styles.items():
            subset = gate[gate.allocation_strategy == strategy].sort_values(
                "normal_budget_requested"
            )
            axes[1].plot(
                subset.normal_budget_requested, subset[f"{metric}_mean"],
                marker=marker, linestyle=linestyle, color=color, linewidth=1.0,
                markersize=4, label=label,
            )
        axes[1].set(
            xlabel="Total normal-image budget", ylabel="Macro rate", ylim=(-0.02, 1.02)
        )
        axes[1].set_title("(b) Frozen-threshold operation")
        axes[1].legend(frameon=True, fontsize=6.2)
        figure.savefig(args.figures / "total_normal_budget.pdf", bbox_inches="tight")
        figure.savefig(args.figures / "total_normal_budget.png", dpi=800, bbox_inches="tight")
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
    multiseed_fusion = data[data.experiment == "multiseed_fusion"].copy()
    if not multiseed_fusion.empty:
        multiseed_fusion.groupby("method")[metrics].agg(["mean", "std"]).to_csv(
            args.tables / "multiseed_fusion.csv"
        )
        comparisons = paired_control_family(
            multiseed_fusion, "method", "proposed",
            (
                "raw_weighted_sum", "calibrated_weighted_sum", "calibrated_max",
                "calibrated_min", "calibrated_product", "unbounded_agreement",
                "no_upper_tail", "frequency_tail_gate",
            ),
            ("image_auroc", "pixel_auroc", "pixel_ap"),
            int(cfg["bootstrap_repetitions"]), int(cfg["seed"]),
        )
        (args.tables / "multiseed_fusion_comparisons.json").write_text(
            json.dumps(comparisons, indent=2), encoding="utf-8"
        )
    multiseed_weight = data[data.experiment == "multiseed_weight"].copy()
    if not multiseed_weight.empty:
        multiseed_weight.groupby("weight")[metrics].agg(["mean", "std"]).to_csv(
            args.tables / "multiseed_weight.csv"
        )
        weight_comparisons = paired_control_family(
            multiseed_weight, "weight", 0.25, (0.0,),
            ("image_auroc", "pixel_auroc", "pixel_ap"),
            int(cfg["bootstrap_repetitions"]), int(cfg["seed"]),
        )
        (args.tables / "multiseed_weight_comparisons.json").write_text(
            json.dumps(weight_comparisons, indent=2), encoding="utf-8"
        )
    multiseed_upper = data[data.experiment == "multiseed_upper_quantile"].copy()
    if not multiseed_upper.empty:
        multiseed_upper.groupby("upper_quantile")[metrics].agg(["mean", "std"]).to_csv(
            args.tables / "multiseed_upper_quantile.csv"
        )
        upper_comparisons = paired_control_family(
            multiseed_upper, "upper_quantile", 0.995, (0.99, 0.999),
            ("image_auroc", "pixel_auroc", "pixel_ap"),
            int(cfg["bootstrap_repetitions"]), int(cfg["seed"]),
        )
        (args.tables / "multiseed_upper_quantile_comparisons.json").write_text(
            json.dumps(upper_comparisons, indent=2), encoding="utf-8"
        )
    robustness = data[data.experiment == "robustness"].copy()
    if not robustness.empty:
        robustness.groupby(["method", "translation_px", "brightness"])[metrics].mean().to_csv(args.tables / "robustness.csv")
    shift_diagnostic = data[data.experiment == "shift_diagnostic"].copy()
    if not shift_diagnostic.empty:
        shift_metrics = [
            "image_auroc", "image_ap", "normal_fpr_conformal",
            "interior_image_auroc", "interior_image_ap", "interior_normal_fpr",
            "valid_pixel_fraction",
        ]
        shift_diagnostic.groupby(["method", "condition"])[shift_metrics].agg(
            ["mean", "std"]
        ).to_csv(args.tables / "shift_diagnostics.csv")
        figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), constrained_layout=True)
        direction_order = [
            "east_constant", "west_constant", "south_constant", "north_constant",
            "southeast_constant", "southwest_constant", "northeast_constant", "northwest_constant",
        ]
        direction_labels = ["E", "W", "S", "N", "SE", "SW", "NE", "NW"]
        gated = shift_diagnostic[
            (shift_diagnostic.method == "freqpatch_lite")
            & shift_diagnostic.condition.isin(direction_order)
        ]
        grouped = gated.groupby("condition")
        x = np.arange(len(direction_order))
        full_values = [grouped.get_group(name).normal_fpr_conformal.mean() for name in direction_order]
        interior_values = [grouped.get_group(name).interior_normal_fpr.mean() for name in direction_order]
        axes[0].plot(x, full_values, "o-", color="0.1", label="Full image")
        axes[0].plot(x, interior_values, "s--", color="0.55", label="Valid interior")
        axes[0].set_xticks(x, direction_labels)
        axes[0].set_ylim(0, 1.02)
        axes[0].set_ylabel("Normal FPR")
        axes[0].set_title("(a) Direction, gated method")
        axes[0].legend(frameon=True, fontsize=7)

        boundary_order = [
            "southeast_constant", "southeast_reflect", "southeast_replicate",
            "southeast_reflect_inverse_registered",
        ]
        boundary_labels = ["Gray", "Reflect", "Replicate", "Inverse reg."]
        width = 0.36
        for offset, method, color, hatch, label in (
            (-width / 2, "patchcore_lite", "0.72", "//", "Compact R18-PM"),
            (width / 2, "freqpatch_lite", "0.22", "..", "CalibFreq-Patch"),
        ):
            subset = shift_diagnostic[
                (shift_diagnostic.method == method)
                & shift_diagnostic.condition.isin(boundary_order)
            ].groupby("condition").normal_fpr_conformal.mean()
            bars = axes[1].bar(
                np.arange(len(boundary_order)) + offset,
                [subset[name] for name in boundary_order], width,
                color=color, edgecolor="black", linewidth=0.6, label=label,
            )
            for bar in bars:
                bar.set_hatch(hatch)
        axes[1].set_xticks(np.arange(len(boundary_order)), boundary_labels, rotation=20, ha="right")
        axes[1].set_ylim(0, 1.02)
        axes[1].set_ylabel("Normal FPR")
        axes[1].set_title("(b) Boundary and oracle control")
        axes[1].legend(frameon=True, fontsize=7)
        figure.savefig(args.figures / "shift_diagnostics.pdf", bbox_inches="tight")
        figure.savefig(args.figures / "shift_diagnostics.png", dpi=800, bbox_inches="tight")
        plt.close(figure)
    wr50_gate = data[data.experiment == "wr50_gate_control"].copy()
    if not wr50_gate.empty:
        wr50_gate.groupby("method")[metrics].agg(["mean", "std"]).to_csv(
            args.tables / "wr50_gate_control.csv"
        )
        wr50_comparisons = paired_control_family(
            wr50_gate, "method", "patchcore_wr50_compact", ("freqpatch_wr50_compact",),
            ("image_auroc", "pixel_auroc", "pixel_ap"),
            int(cfg["bootstrap_repetitions"]), int(cfg["seed"]),
        )
        (args.tables / "wr50_gate_comparisons.json").write_text(
            json.dumps(wr50_comparisons, indent=2), encoding="utf-8"
        )

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
