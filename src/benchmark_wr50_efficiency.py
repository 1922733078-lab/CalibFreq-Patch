#!/usr/bin/env python3
"""Measure the higher-capacity WR50 control under the same CPU timing protocol."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml

from freqpatch import (
    build_memory,
    calibrate,
    calibration_parameters,
    image_scores,
    load_image,
    load_manifest,
    patchcore_score,
)
from run_experiments import split_training
from run_strong_baseline import WideResNetPatchExtractor


def summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "median_ms": float(np.median(values) * 1000),
        "p95_ms": float(np.quantile(values, 0.95) * 1000),
        "mean_ms": float(np.mean(values) * 1000),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/wr50_features"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/efficiency_wr50.json"))
    parser.add_argument("--images-per-category", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--channels", type=int, default=256)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    groups = load_manifest(args.data_root, cfg["categories"])
    extractor = WideResNetPatchExtractor(
        cfg["feature_grid"], args.channels, cfg["seed"]
    )
    rows = []

    for category in cfg["categories"]:
        train_samples = [s for s in groups[category] if s.split == "train"]
        test_samples = [s for s in groups[category] if s.split == "test"]
        with np.load(args.cache / f"{category}_train.npz") as payload:
            train = {key: payload[key] for key in payload.files}
        fit_samples, branch_samples, _ = split_training(
            train_samples,
            cfg["seed"],
            branch_fraction=cfg["branch_calibration_fraction"],
            threshold_fraction=cfg["threshold_calibration_fraction"],
        )
        index = {sample.path: i for i, sample in enumerate(train_samples)}
        fit_idx = [index[s.path] for s in fit_samples]
        branch_idx = [index[s.path] for s in branch_samples]
        memory = build_memory(
            train["features"][fit_idx], cfg["memory_max"], cfg["memory_ratio"], cfg["seed"]
        )
        parameters = calibration_parameters(
            patchcore_score(train["features"][branch_idx], memory, cfg["knn_k"]),
            cfg["score_quantile"],
        )

        load_times, feature_times, score_times, e2e_times = [], [], [], []
        selected = test_samples[: min(args.images_per_category + 2, len(test_samples))]
        for round_index in range(args.rounds):
            for position, sample in enumerate(selected):
                started = time.perf_counter()
                tensor = load_image(sample, cfg["image_size"])[None]
                loaded = time.perf_counter()
                features_t = extractor.semantic_features(tensor)
                extracted = time.perf_counter()
                features = features_t.permute(0, 2, 3, 1).numpy()
                maps = calibrate(
                    patchcore_score(features, memory, cfg["knn_k"]), parameters
                )
                _ = image_scores(maps, cfg["score_quantile"])
                finished = time.perf_counter()
                if round_index > 0 or position >= 2:
                    load_times.append(loaded - started)
                    feature_times.append(extracted - loaded)
                    score_times.append(finished - extracted)
                    e2e_times.append(finished - started)

        rows.append({
            "category": category,
            "timed_observations": len(e2e_times),
            "memory_vectors": int(memory.shape[0]),
            "memory_mib": float(memory.nbytes / 1024**2),
            "timings": {
                "load_preprocess": summary(load_times),
                "feature_extraction": summary(feature_times),
                "patchcore_scoring": summary(score_times),
                "end_to_end": summary(e2e_times),
            },
        })

    parameter_bytes = sum(p.nelement() * p.element_size() for p in extractor.model.parameters())
    aggregate = {
        name: {
            "median_of_category_medians_ms": float(np.median([
                row["timings"][name]["median_ms"] for row in rows
            ])),
            "maximum_category_p95_ms": float(max(
                row["timings"][name]["p95_ms"] for row in rows
            )),
        }
        for name in rows[0]["timings"]
    }
    result = {
        "device": "Apple M4 CPU",
        "torch_threads": torch.get_num_threads(),
        "timing_rounds": args.rounds,
        "descriptor_channels": args.channels,
        "model_parameters": int(sum(p.nelement() for p in extractor.model.parameters())),
        "model_parameter_mib": float(parameter_bytes / 1024**2),
        "rss_mib_after_benchmark": float(psutil.Process().memory_info().rss / 1024**2),
        "categories": rows,
        "aggregate": aggregate,
        "derived_rate_from_median_images_per_second": 1000.0 / aggregate["end_to_end"]["median_of_category_medians_ms"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
