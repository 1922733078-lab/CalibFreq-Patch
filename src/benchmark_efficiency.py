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


def bootstrap_median_ci(values, repetitions: int = 20000, seed: int = 20260714):
    """Percentile CI for the median paired difference across categories."""
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.median(
        array[rng.integers(0, len(array), size=(repetitions, len(array)))], axis=1
    )
    return {
        "estimate_ms": float(np.median(array)),
        "ci95_low_ms": float(np.quantile(draws, 0.025)),
        "ci95_high_ms": float(np.quantile(draws, 0.975)),
        "bootstrap_unit": "category median paired difference",
        "bootstrap_statistic": "median",
        "bootstrap_repetitions": repetitions,
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
        deep_parameters = calibration_parameters(
            patchcore_score(branch["features"], memory, cfg["knn_k"]), cfg["score_quantile"]
        )
        freq_parameters = calibration_parameters(
            frequency_score(branch["frequency"], freq_median, freq_scale), cfg["score_quantile"]
        )

        timings = {
            "load_preprocess": [],
            "semantic_extraction_baseline": [],
            "semantic_extraction_full": [],
            "frequency_construction": [],
            "patchcore_scoring_baseline": [],
            "patchcore_scoring_full": [],
            "frequency_scoring": [],
            "agreement_gate": [],
            "paired_full_frequency_increment": [],
            "patchcore_end_to_end": [],
            "freqpatch_end_to_end": [],
        }
        selected = test_samples[: min(args.images_per_category + 2, len(test_samples))]
        order_rng = np.random.default_rng(int(cfg["seed"]) + sum(map(ord, category)))
        retained_count = args.rounds * len(selected) - 2
        if retained_count % 2:
            raise ValueError("Retained paired observations must be even for balanced AB/BA order")
        retained_orders = np.repeat([0, 1], retained_count // 2)
        order_rng.shuffle(retained_orders)
        retained_index = 0
        paired_observations = []
        for round_index in range(args.rounds):
            for position, sample in enumerate(selected):
                start = time.perf_counter()
                tensor = load_image(sample, cfg["image_size"])[None]
                loaded = time.perf_counter()

                def semantic_pipeline():
                    pipeline_start = time.perf_counter()
                    features_t = extractor.semantic_features(tensor)
                    semantic_done = time.perf_counter()
                    features = features_t.permute(0, 2, 3, 1).numpy()
                    deep = calibrate(
                        patchcore_score(features, memory, cfg["knn_k"]), deep_parameters
                    )
                    _ = image_scores(deep, cfg["score_quantile"])
                    scoring_done = time.perf_counter()
                    return {
                        "semantic": semantic_done - pipeline_start,
                        "patchcore": scoring_done - semantic_done,
                        "total": scoring_done - pipeline_start,
                    }

                def full_pipeline():
                    pipeline_start = time.perf_counter()
                    features_t = extractor.semantic_features(tensor)
                    semantic_done = time.perf_counter()
                    frequency_t = extractor.frequency_features(tensor)
                    construction_done = time.perf_counter()
                    features = features_t.permute(0, 2, 3, 1).numpy()
                    frequency = frequency_t.permute(0, 2, 3, 1).numpy()
                    deep = calibrate(
                        patchcore_score(features, memory, cfg["knn_k"]), deep_parameters
                    )
                    patchcore_done = time.perf_counter()
                    freq = calibrate(
                        frequency_score(frequency, freq_median, freq_scale), freq_parameters
                    )
                    frequency_done = time.perf_counter()
                    fused = fuse_scores(deep, freq, cfg["frequency_weight"])
                    _ = image_scores(fused, cfg["score_quantile"])
                    gate_done = time.perf_counter()
                    return {
                        "semantic": semantic_done - pipeline_start,
                        "construction": construction_done - semantic_done,
                        "patchcore": patchcore_done - construction_done,
                        "frequency": frequency_done - patchcore_done,
                        "gate": gate_done - frequency_done,
                        "total": gate_done - pipeline_start,
                    }

                # Randomized interleaving prevents one method from always
                # receiving the warmer cache or later thermal state.
                retained = round_index > 0 or position >= 2
                order_code = int(retained_orders[retained_index]) if retained else position % 2
                if order_code == 0:
                    baseline_result = semantic_pipeline()
                    full_result = full_pipeline()
                else:
                    full_result = full_pipeline()
                    baseline_result = semantic_pipeline()

                if retained:
                    load_time = loaded - start
                    timings["load_preprocess"].append(load_time)
                    timings["semantic_extraction_baseline"].append(baseline_result["semantic"])
                    timings["semantic_extraction_full"].append(full_result["semantic"])
                    timings["frequency_construction"].append(full_result["construction"])
                    timings["patchcore_scoring_baseline"].append(baseline_result["patchcore"])
                    timings["patchcore_scoring_full"].append(full_result["patchcore"])
                    timings["frequency_scoring"].append(full_result["frequency"])
                    timings["agreement_gate"].append(full_result["gate"])
                    timings["paired_full_frequency_increment"].append(
                        full_result["total"] - baseline_result["total"]
                    )
                    timings["patchcore_end_to_end"].append(load_time + baseline_result["total"])
                    timings["freqpatch_end_to_end"].append(load_time + full_result["total"])
                    paired_observations.append({
                        "sample": sample.path.name,
                        "round": round_index,
                        "order": "semantic-first" if order_code == 0 else "full-first",
                        "semantic_ms": baseline_result["total"] * 1000,
                        "full_ms": full_result["total"] * 1000,
                        "delta_ms": (full_result["total"] - baseline_result["total"]) * 1000,
                    })
                    retained_index += 1

        rows.append({
            "category": category,
            "timed_observations": len(timings["freqpatch_end_to_end"]),
            "memory_vectors": int(memory.shape[0]),
            "memory_mib": float(memory.nbytes / 1024**2),
            "frequency_statistics_mib": float((freq_median.nbytes + freq_scale.nbytes) / 1024**2),
            "timings": {name: percentile_summary(values) for name, values in timings.items()},
            "paired_observations": paired_observations,
            "order_counts": {
                "semantic-first": int(sum(row["order"] == "semantic-first" for row in paired_observations)),
                "full-first": int(sum(row["order"] == "full-first" for row in paired_observations)),
            },
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
        row["timings"]["paired_full_frequency_increment"]["median_ms"] for row in rows
    ]
    result["paired_full_frequency_increment_category_medians_ms"] = {
        "median": float(np.median(increment_medians)),
        "q1": float(np.quantile(increment_medians, 0.25)),
        "q3": float(np.quantile(increment_medians, 0.75)),
        "minimum": float(np.min(increment_medians)),
        "maximum": float(np.max(increment_medians)),
        "observations_per_category": int(rows[0]["timed_observations"]),
        **bootstrap_median_ci(increment_medians),
    }
    relative_overhead = [
        100.0 * row["timings"]["paired_full_frequency_increment"]["median_ms"]
        / row["timings"]["patchcore_end_to_end"]["median_ms"]
        for row in rows
    ]
    relative_ci = bootstrap_median_ci(relative_overhead)
    result["paired_relative_overhead_category_medians_percent"] = {
        "median": float(np.median(relative_overhead)),
        "q1": float(np.quantile(relative_overhead, 0.25)),
        "q3": float(np.quantile(relative_overhead, 0.75)),
        "ci95_low": relative_ci["ci95_low_ms"],
        "ci95_high": relative_ci["ci95_high_ms"],
        "bootstrap_unit": relative_ci["bootstrap_unit"],
        "bootstrap_statistic": "median",
        "bootstrap_repetitions": relative_ci["bootstrap_repetitions"],
    }
    result["derived_rate_from_freqpatch_median_images_per_second"] = (
        1000.0 / result["aggregate"]["freqpatch_end_to_end"]["median_of_category_medians_ms"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
