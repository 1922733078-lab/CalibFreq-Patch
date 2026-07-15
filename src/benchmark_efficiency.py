#!/usr/bin/env python3
"""Measure end-to-end and scoring latency on the declared CPU device."""

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
    FrozenFeatureExtractor,
    build_memory,
    calibrate,
    calibration_parameters,
    frequency_score,
    fuse_scores,
    image_scores,
    load_image,
    load_manifest,
    padim_score,
    padim_stats,
    patchcore_score,
    robust_frequency_stats,
)
from run_experiments import split_training


def percentile_summary(values):
    array = np.asarray(values, dtype=float)
    return {
        "median_ms": float(np.median(array) * 1000),
        "q1_ms": float(np.quantile(array, 0.25) * 1000),
        "q3_ms": float(np.quantile(array, 0.75) * 1000),
        "minimum_ms": float(np.min(array) * 1000),
        "maximum_ms": float(np.max(array) * 1000),
        "p95_ms": float(np.quantile(array, 0.95) * 1000),
        "mean_ms": float(np.mean(array) * 1000),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/features"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/efficiency.json"))
    parser.add_argument("--images-per-category", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    groups = load_manifest(args.data_root, cfg["categories"])
    extractor = FrozenFeatureExtractor(cfg["projection_dim"], cfg["feature_grid"], cfg["seed"], "cpu")
    rows = []

    for category in cfg["categories"]:
        train_samples = [sample for sample in groups[category] if sample.split == "train"]
        test_samples = [sample for sample in groups[category] if sample.split == "test"]
        with np.load(args.cache / f"{category}_train.npz") as payload:
            full_train = {key: payload[key] for key in payload.files}
        fit_samples, branch_samples, threshold_samples = split_training(
            train_samples,
            cfg["seed"],
            branch_fraction=cfg["branch_calibration_fraction"],
            threshold_fraction=cfg["threshold_calibration_fraction"],
        )
        index = {sample.path: i for i, sample in enumerate(train_samples)}
        fit_idx = [index[sample.path] for sample in fit_samples]
        branch_idx = [index[sample.path] for sample in branch_samples]
        fit = {key: value[fit_idx] for key, value in full_train.items()}
        branch = {key: value[branch_idx] for key, value in full_train.items()}

        memory = build_memory(fit["features"], cfg["memory_max"], cfg["memory_ratio"], cfg["seed"])
        freq_median, freq_scale = robust_frequency_stats(fit["frequency"])
        padim_mean, padim_std = padim_stats(fit["features"])
        deep_parameters = calibration_parameters(
            patchcore_score(branch["features"], memory, cfg["knn_k"]), cfg["score_quantile"]
        )
        freq_parameters = calibration_parameters(
            frequency_score(branch["frequency"], freq_median, freq_scale), cfg["score_quantile"]
        )
        padim_parameters = calibration_parameters(
            padim_score(branch["features"], padim_mean, padim_std), cfg["score_quantile"]
        )

        timings = {
            "load_preprocess": [], "feature_extraction": [], "patchcore_scoring": [],
            "frequency_scoring": [], "padim_scoring": [], "agreement_gate": [],
            "paired_freqpatch_increment": [],
            "patchcore_end_to_end": [], "freqpatch_end_to_end": [],
            "padim_end_to_end": [], "padim_gate_end_to_end": [],
        }
        selected = test_samples[: min(args.images_per_category + 2, len(test_samples))]
        for round_index in range(args.rounds):
            for position, sample in enumerate(selected):
                start = time.perf_counter()
                tensor = load_image(sample, cfg["image_size"])[None]
                loaded = time.perf_counter()
                features_t, frequency_t = extractor(tensor)
                extracted = time.perf_counter()
                features = features_t.permute(0, 2, 3, 1).numpy()
                frequency = frequency_t.permute(0, 2, 3, 1).numpy()

                part = time.perf_counter()
                deep = calibrate(patchcore_score(features, memory, cfg["knn_k"]), deep_parameters)
                _ = image_scores(deep, cfg["score_quantile"])
                patchcore_done = time.perf_counter()

                freq = calibrate(frequency_score(frequency, freq_median, freq_scale), freq_parameters)
                frequency_done = time.perf_counter()

                padim = calibrate(padim_score(features, padim_mean, padim_std), padim_parameters)
                _ = image_scores(padim, cfg["score_quantile"])
                padim_done = time.perf_counter()

                fused = fuse_scores(deep, freq, cfg["frequency_weight"])
                _ = image_scores(fused, cfg["score_quantile"])
                gate_done = time.perf_counter()
                padim_fused = fuse_scores(padim, freq, cfg["frequency_weight"])
                _ = image_scores(padim_fused, cfg["score_quantile"])
                padim_gate_done = time.perf_counter()

                if round_index > 0 or position >= 2:
                    load_time = loaded - start
                    feature_time = extracted - loaded
                    patch_time = patchcore_done - part
                    freq_time = frequency_done - patchcore_done
                    padim_time = padim_done - frequency_done
                    gate_time = gate_done - padim_done
                    padim_gate_time = padim_gate_done - gate_done
                    timings["load_preprocess"].append(load_time)
                    timings["feature_extraction"].append(feature_time)
                    timings["patchcore_scoring"].append(patch_time)
                    timings["frequency_scoring"].append(freq_time)
                    timings["padim_scoring"].append(padim_time)
                    timings["agreement_gate"].append(gate_time)
                    timings["paired_freqpatch_increment"].append(freq_time + gate_time)
                    timings["patchcore_end_to_end"].append(load_time + feature_time + patch_time)
                    timings["freqpatch_end_to_end"].append(load_time + feature_time + patch_time + freq_time + gate_time)
                    timings["padim_end_to_end"].append(load_time + feature_time + padim_time)
                    timings["padim_gate_end_to_end"].append(
                        load_time + feature_time + padim_time + freq_time + padim_gate_time
                    )

        rows.append({
            "category": category,
            "timed_observations": len(timings["freqpatch_end_to_end"]),
            "memory_vectors": int(memory.shape[0]),
            "memory_mib": float(memory.nbytes / 1024**2),
            "frequency_statistics_mib": float((freq_median.nbytes + freq_scale.nbytes) / 1024**2),
            "padim_statistics_mib": float((padim_mean.nbytes + padim_std.nbytes) / 1024**2),
            "timings": {name: percentile_summary(values) for name, values in timings.items()},
        })

    parameter_bytes = sum(parameter.nelement() * parameter.element_size() for parameter in extractor.model.parameters())
    result = {
        "device": "Apple M4 CPU",
        "torch_threads": torch.get_num_threads(),
        "timing_rounds": args.rounds,
        "model_parameters": int(sum(parameter.nelement() for parameter in extractor.model.parameters())),
        "model_parameter_mib": float(parameter_bytes / 1024**2),
        "rss_mib_after_benchmark": float(psutil.Process().memory_info().rss / 1024**2),
        "categories": rows,
        "aggregate": {
            name: {
                "median_of_category_medians_ms": float(np.median([row["timings"][name]["median_ms"] for row in rows])),
                "maximum_category_p95_ms": float(max(row["timings"][name]["p95_ms"] for row in rows)),
            }
            for name in rows[0]["timings"]
        },
    }
    increment_medians = [
        row["timings"]["paired_freqpatch_increment"]["median_ms"] for row in rows
    ]
    result["paired_increment_category_medians_ms"] = {
        "median": float(np.median(increment_medians)),
        "q1": float(np.quantile(increment_medians, 0.25)),
        "q3": float(np.quantile(increment_medians, 0.75)),
        "minimum": float(np.min(increment_medians)),
        "maximum": float(np.max(increment_medians)),
        "observations_per_category": int(rows[0]["timed_observations"]),
    }
    result["derived_rate_from_freqpatch_median_images_per_second"] = (
        1000.0 / result["aggregate"]["freqpatch_end_to_end"]["median_of_category_medians_ms"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
