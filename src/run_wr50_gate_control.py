#!/usr/bin/env python3
"""Evaluate the bounded gate on the cached compact WR50-256 capacity control."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/wr50_features"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/wr50_gate_control.jsonl"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    groups = load_manifest(args.data_root, cfg["categories"])
    rows: list[dict] = []

    for category in cfg["categories"]:
        train_samples = [sample for sample in groups[category] if sample.split == "train"]
        with np.load(args.cache_dir / f"{category}_train.npz") as payload:
            train = {key: payload[key] for key in payload.files}
        with np.load(args.cache_dir / f"{category}_test.npz") as payload:
            test = {key: payload[key] for key in payload.files}
        index_by_path = {sample.path: index for index, sample in enumerate(train_samples)}

        for seed in cfg["seeds"]:
            fit_samples, branch_samples, threshold_samples = split_training(
                train_samples, seed,
                branch_fraction=cfg["branch_calibration_fraction"],
                threshold_fraction=cfg["threshold_calibration_fraction"],
            )

            def subset(samples):
                indices = [index_by_path[sample.path] for sample in samples]
                return {key: value[indices] for key, value in train.items()}

            fit, branch, threshold = map(subset, (fit_samples, branch_samples, threshold_samples))
            outputs, aux = evaluate_bundle(
                fit, branch, threshold, test, cfg, seed,
                weights=(cfg["frequency_weight"],),
            )
            for output_name, method in (
                ("patchcore_lite", "patchcore_wr50_compact"),
                (f"freqpatch_lite_w{cfg['frequency_weight']:.2f}", "freqpatch_wr50_compact"),
            ):
                maps, threshold_images = outputs[output_name]
                result = metrics(
                    test["labels"], test["masks"], maps, threshold_images,
                    cfg["score_quantile"], cfg["image_size"], cfg["threshold_alpha"],
                )[0]
                rows.append({
                    "experiment": "wr50_gate_control", "category": category,
                    "seed": seed, "method": method, "descriptor_channels": 256,
                    **aux, **result,
                })
        write_jsonl(args.output, rows)
        print(f"[checkpoint] {category}: {len(rows)} rows", flush=True)

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
