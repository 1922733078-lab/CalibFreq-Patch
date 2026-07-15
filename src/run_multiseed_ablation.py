#!/usr/bin/env python3
"""Run the decisive gate-form, weight, and tail-quantile controls at three seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from freqpatch import load_manifest, metrics
from run_experiments import evaluate_bundle, split_training, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/features"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/multiseed_ablation.jsonl"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    groups = load_manifest(args.data_root, cfg["categories"])
    rows: list[dict] = []
    variants = (
        "proposed",
        "raw_weighted_sum",
        "calibrated_weighted_sum",
        "calibrated_max",
        "calibrated_min",
        "calibrated_product",
        "unbounded_agreement",
        "no_upper_tail",
        "frequency_tail_gate",
    )
    weights = (0.0, 0.10, 0.25, 0.50, 0.75)

    for category in cfg["categories"]:
        train_samples = [sample for sample in groups[category] if sample.split == "train"]
        test_samples = [sample for sample in groups[category] if sample.split == "test"]
        with np.load(args.cache_dir / f"{category}_train.npz") as payload:
            full_train = {key: payload[key] for key in payload.files}
        with np.load(args.cache_dir / f"{category}_test.npz") as payload:
            test = {key: payload[key] for key in payload.files}
        index_by_path = {sample.path: index for index, sample in enumerate(train_samples)}

        for seed in cfg["seeds"]:
            fit_samples, branch_samples, threshold_samples = split_training(
                train_samples,
                seed,
                branch_fraction=cfg["branch_calibration_fraction"],
                threshold_fraction=cfg["threshold_calibration_fraction"],
            )

            def subset(samples):
                indices = [index_by_path[sample.path] for sample in samples]
                return {key: value[indices] for key, value in full_train.items()}

            fit, branch, threshold = map(subset, (fit_samples, branch_samples, threshold_samples))

            def summarize(maps, threshold_images):
                return metrics(
                    test["labels"], test["masks"], maps, threshold_images,
                    cfg["score_quantile"], cfg["image_size"], cfg["threshold_alpha"],
                )[0]

            outputs, aux = evaluate_bundle(
                fit, branch, threshold, test, cfg, seed,
                weights=weights, fusion_variants=variants,
            )
            for weight in weights:
                maps, threshold_images = outputs[f"freqpatch_lite_w{weight:.2f}"]
                rows.append({
                    "experiment": "multiseed_weight", "category": category, "seed": seed,
                    "method": "freqpatch_lite", "weight": weight, **aux,
                    **summarize(maps, threshold_images),
                })
            for variant in variants:
                maps, threshold_images = outputs[f"fusion_{variant}"]
                rows.append({
                    "experiment": "multiseed_fusion", "category": category, "seed": seed,
                    "method": variant, **aux, **summarize(maps, threshold_images),
                })
            for upper_quantile in cfg["calibration_upper_quantiles"]:
                outputs, aux = evaluate_bundle(
                    fit, branch, threshold, test, cfg, seed,
                    weights=(cfg["frequency_weight"],), upper_quantile=upper_quantile,
                )
                maps, threshold_images = outputs[
                    f"freqpatch_lite_w{cfg['frequency_weight']:.2f}"
                ]
                rows.append({
                    "experiment": "multiseed_upper_quantile", "category": category,
                    "seed": seed, "method": "freqpatch_lite",
                    "upper_quantile": upper_quantile, **aux,
                    **summarize(maps, threshold_images),
                })
        write_jsonl(args.output, rows)
        print(f"[checkpoint] {category}: {len(rows)} rows", flush=True)

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
