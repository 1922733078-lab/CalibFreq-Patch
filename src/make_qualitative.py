#!/usr/bin/env python3
"""Generate deterministic, median-case localization examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from freqpatch import image_scores, load_manifest, resize_maps
from run_experiments import evaluate_bundle, split_training


def overlay(image, heatmap, maximum):
    colors = plt.get_cmap("inferno")(np.clip(heatmap / max(maximum, 1e-8), 0, 1))[..., :3]
    strength = np.clip(heatmap / max(maximum, 1e-8), 0, 1)[..., None] * 0.62
    return image * (1 - strength) + colors * strength


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/features"))
    parser.add_argument("--output", type=Path, default=Path("figures/qualitative_localization.png"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(min(8, torch.get_num_threads()))
    groups = load_manifest(args.data_root, cfg["categories"])
    selected_categories = ["bottle", "capsule", "grid", "tile"]
    examples, selection_rows = [], []

    for category in selected_categories:
        samples = groups[category]
        train_samples = [sample for sample in samples if sample.split == "train"]
        test_samples = [sample for sample in samples if sample.split == "test"]
        with np.load(args.cache / f"{category}_train.npz") as payload:
            full_train = {key: payload[key] for key in payload.files}
        with np.load(args.cache / f"{category}_test.npz") as payload:
            test = {key: payload[key] for key in payload.files}
        index = {sample.path: i for i, sample in enumerate(train_samples)}
        fit_samples, branch_samples, threshold_samples = split_training(
            train_samples,
            cfg["seed"],
            branch_fraction=cfg["branch_calibration_fraction"],
            threshold_fraction=cfg["threshold_calibration_fraction"],
        )
        fit_idx = [index[sample.path] for sample in fit_samples]
        branch_idx = [index[sample.path] for sample in branch_samples]
        threshold_idx = [index[sample.path] for sample in threshold_samples]
        fit = {key: value[fit_idx] for key, value in full_train.items()}
        branch = {key: value[branch_idx] for key, value in full_train.items()}
        threshold = {key: value[threshold_idx] for key, value in full_train.items()}
        outputs, _ = evaluate_bundle(
            fit, branch, threshold, test, cfg, cfg["seed"],
            weights=(cfg["frequency_weight"],),
        )
        patch_map = outputs["patchcore_lite"][0]
        proposed_map = outputs[f"freqpatch_lite_w{cfg['frequency_weight']:.2f}"][0]
        proposed_scores = image_scores(proposed_map, cfg["score_quantile"])
        anomaly_idx = np.flatnonzero(test["labels"] == 1)
        median = np.median(proposed_scores[anomaly_idx])
        chosen = int(anomaly_idx[np.argmin(np.abs(proposed_scores[anomaly_idx] - median))])
        sample = test_samples[chosen]
        image = np.asarray(
            Image.open(sample.path).convert("RGB").resize(
                (cfg["image_size"], cfg["image_size"]), Image.Resampling.BILINEAR
            ),
            dtype=float,
        ) / 255.0
        patch_high = resize_maps(patch_map[chosen : chosen + 1], cfg["image_size"])[0]
        proposed_high = resize_maps(proposed_map[chosen : chosen + 1], cfg["image_size"])[0]
        examples.append((category, sample.defect, image, test["masks"][chosen], patch_high, proposed_high))
        selection_rows.append({
            "category": category,
            "defect": sample.defect,
            "test_index": chosen,
            "relative_path": str(sample.path.relative_to(args.data_root)),
            "rule": "anomalous sample closest to the category median CalibFreq-Patch image score",
        })

    fig, axes = plt.subplots(len(examples), 4, figsize=(4.85, 5.25), constrained_layout=True)
    titles = ["Input", "Ground truth", "PatchCore-Lite", "CalibFreq-Patch"]
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=7, pad=2)
    for row, (category, defect, image, mask, patch_map, proposed_map) in enumerate(examples):
        maximum = float(np.quantile(np.concatenate([patch_map.ravel(), proposed_map.ravel()]), 0.995))
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(image)
        axes[row, 1].contour(mask, levels=[0.5], colors=["black"], linewidths=1.5)
        axes[row, 1].contour(mask, levels=[0.5], colors=["white"], linewidths=0.7, linestyles="dashed")
        axes[row, 2].imshow(overlay(image, patch_map, maximum))
        axes[row, 3].imshow(overlay(image, proposed_map, maximum))
        axes[row, 0].set_ylabel(f"{category}\n{defect.replace('_', ' ')}", fontsize=6.5)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#455A64")
    fig.text(0.5, 0.004, "Median-score anomalous example per category; double black/white contour denotes the reference mask.", ha="center", fontsize=5.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    args.output.with_name("qualitative_selection.json").write_text(json.dumps(selection_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
