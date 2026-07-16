#!/usr/bin/env python3
"""Run main, ablation, few-shot, robustness, and efficiency experiments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
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
    extract_samples,
    frequency_score,
    fuse_scores,
    fuse_scores_variant,
    image_scores,
    load_manifest,
    metrics,
    padim_score,
    padim_stats,
    patchcore_score,
    robust_frequency_stats,
)


def split_training(
    samples,
    seed: int,
    fit_count: int | None = None,
    total_budget: int | None = None,
    branch_fraction: float = 0.15,
    threshold_fraction: float = 0.15,
    threshold_min_count: int | None = None,
):
    """Create disjoint fit, branch-calibration, and threshold-calibration sets.

    ``threshold_min_count`` activates a threshold-prioritized allocation for a
    strict total budget.  The requested threshold count is guaranteed whenever
    at least four fitting and two branch-calibration images can still be kept;
    otherwise the original proportional allocation is retained and the caller
    can report that the priority target was infeasible for that budget.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    if total_budget is not None:
        indices = indices[: min(total_budget, len(indices))]
        minimum = 2
    else:
        minimum = 8
    branch_count = max(minimum, int(round(branch_fraction * len(indices))))
    threshold_count = max(minimum, int(round(threshold_fraction * len(indices))))
    if total_budget is not None and threshold_min_count is not None:
        requested = int(threshold_min_count)
        if requested < 2:
            raise ValueError("threshold_min_count must be at least 2")
        if len(indices) >= requested + 6:
            threshold_count = requested
            branch_count = min(branch_count, len(indices) - threshold_count - 4)
            branch_count = max(2, branch_count)
    if branch_count + threshold_count > len(indices) - 4:
        branch_count = max(2, (len(indices) - 4) // 2)
        threshold_count = max(2, len(indices) - 4 - branch_count)
    branch_idx = indices[:branch_count]
    threshold_idx = indices[branch_count:branch_count + threshold_count]
    fit_pool = indices[branch_count + threshold_count:]
    if fit_count is not None:
        fit_pool = fit_pool[: min(fit_count, len(fit_pool))]
    return (
        [samples[i] for i in fit_pool],
        [samples[i] for i in branch_idx],
        [samples[i] for i in threshold_idx],
    )


def evaluate_bundle(
    train,
    branch_calibration,
    threshold_calibration,
    test,
    cfg,
    seed: int,
    weights=(0.25,),
    upper_quantile: float | None = None,
    fusion_variants=(),
    include_padim_gate: bool = False,
):
    upper_quantile = float(upper_quantile or cfg["score_quantile"])
    freq_median, freq_scale = robust_frequency_stats(train["frequency"])
    padim_mean, padim_std = padim_stats(train["features"])
    memory = build_memory(train["features"], cfg["memory_max"], cfg["memory_ratio"], seed)

    raw = {}
    raw["frequency_only"] = (
        frequency_score(branch_calibration["frequency"], freq_median, freq_scale),
        frequency_score(threshold_calibration["frequency"], freq_median, freq_scale),
        frequency_score(test["frequency"], freq_median, freq_scale),
    )
    raw["padim_diag"] = (
        padim_score(branch_calibration["features"], padim_mean, padim_std),
        padim_score(threshold_calibration["features"], padim_mean, padim_std),
        padim_score(test["features"], padim_mean, padim_std),
    )
    score_start = time.perf_counter()
    pc_cal = patchcore_score(branch_calibration["features"], memory, cfg["knn_k"])
    pc_threshold = patchcore_score(threshold_calibration["features"], memory, cfg["knn_k"])
    pc_test = patchcore_score(test["features"], memory, cfg["knn_k"])
    patchcore_seconds = time.perf_counter() - score_start
    raw["patchcore_lite"] = (pc_cal, pc_threshold, pc_test)

    calibrated = {}
    for method, (cal_map, threshold_map, test_map) in raw.items():
        parameters = calibration_parameters(cal_map, upper_quantile)
        calibrated[method] = (
            calibrate(cal_map, parameters),
            calibrate(threshold_map, parameters),
            calibrate(test_map, parameters),
        )

    outputs = {}
    for method in ("frequency_only", "padim_diag", "patchcore_lite"):
        _, threshold_map, test_map = calibrated[method]
        threshold_image = image_scores(threshold_map, cfg["score_quantile"])
        outputs[method] = (test_map, threshold_image)

    _, deep_threshold, deep_test = calibrated["patchcore_lite"]
    _, freq_threshold, freq_test = calibrated["frequency_only"]
    for weight in weights:
        fused_threshold = fuse_scores(deep_threshold, freq_threshold, weight)
        fused_test = fuse_scores(deep_test, freq_test, weight)
        outputs[f"freqpatch_lite_w{weight:.2f}"] = (
            fused_test,
            image_scores(fused_threshold, cfg["score_quantile"]),
        )

    if include_padim_gate:
        _, padim_threshold, padim_test = calibrated["padim_diag"]
        fused_threshold = fuse_scores(padim_threshold, freq_threshold, cfg["frequency_weight"])
        fused_test = fuse_scores(padim_test, freq_test, cfg["frequency_weight"])
        outputs["padim_freq_gate"] = (
            fused_test,
            image_scores(fused_threshold, cfg["score_quantile"]),
        )

    for variant in fusion_variants:
        if variant == "raw_weighted_sum":
            _, raw_deep_threshold, raw_deep_test = raw["patchcore_lite"]
            _, raw_freq_threshold, raw_freq_test = raw["frequency_only"]
            fused_threshold = raw_deep_threshold + cfg["frequency_weight"] * raw_freq_threshold
            fused_test = raw_deep_test + cfg["frequency_weight"] * raw_freq_test
        else:
            fused_threshold = fuse_scores_variant(
                deep_threshold, freq_threshold, cfg["frequency_weight"], variant
            )
            fused_test = fuse_scores_variant(
                deep_test, freq_test, cfg["frequency_weight"], variant
            )
        outputs[f"fusion_{variant}"] = (
            fused_test.astype(np.float32),
            image_scores(fused_threshold, cfg["score_quantile"]),
        )

    aux = {
        "memory_vectors": int(memory.shape[0]),
        "memory_bytes": int(memory.nbytes),
        "patchcore_scoring_seconds": patchcore_seconds,
        "branch_upper_quantile": upper_quantile,
    }
    return outputs, aux


def cache_or_extract(cache_path: Path, samples, extractor, cfg):
    if cache_path.exists():
        with np.load(cache_path) as data:
            return {key: data[key] for key in data.files}
    payload = extract_samples(
        samples,
        extractor,
        cfg["image_size"],
        translation=cfg.get("translation", 0),
        brightness=cfg.get("brightness", 1.0),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **payload)
    return payload


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomically checkpoint rows so a long CPU experiment is restart-safe."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/experiments.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/features"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true", help="Run one seed and skip robustness/few-shot")
    parser.add_argument("--fresh", action="store_true", help="Discard an existing output instead of resuming complete categories")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    groups = load_manifest(args.data_root, cfg["categories"])
    extractor = FrozenFeatureExtractor(cfg["projection_dim"], cfg["feature_grid"], cfg["seed"], args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir
    rows = []
    expected_per_category = 19 if args.quick else (
        len(cfg["seeds"]) * 5
        + 5 + 9
        + 2 * len(cfg["few_shot_counts"])
        + 2 * len(cfg["total_normal_budgets"])
        + len(cfg["calibration_upper_quantiles"])
        + len(cfg["calibration_fraction_ablation"])
        + 2 * len(cfg["robustness"]["translations"]) * len(cfg["robustness"]["brightness"])
    )
    completed_categories = set()
    if args.output.exists() and not args.fresh:
        existing = [
            json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for category in cfg["categories"]:
            category_rows = [row for row in existing if row.get("category") == category]
            if len(category_rows) == expected_per_category:
                rows.extend(category_rows)
                completed_categories.add(category)
        print(
            f"[resume] retained {len(rows)} rows across {len(completed_categories)} complete categories; "
            f"expected {expected_per_category} rows/category",
            flush=True,
        )

    hardware = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "cpu_threads": torch.get_num_threads(),
    }
    seeds = cfg["seeds"][:1] if args.quick else cfg["seeds"]

    for category in cfg["categories"]:
        if category in completed_categories:
            print(f"[resume] skip complete category: {category}", flush=True)
            continue
        all_samples = groups[category]
        train_samples = [sample for sample in all_samples if sample.split == "train"]
        test_samples = [sample for sample in all_samples if sample.split == "test"]
        full_train = cache_or_extract(cache_dir / f"{category}_train.npz", train_samples, extractor, cfg)
        test = cache_or_extract(cache_dir / f"{category}_test.npz", test_samples, extractor, cfg)
        index_by_path = {sample.path: i for i, sample in enumerate(train_samples)}

        def make_split(seed, fit_count=None, total_budget=None, branch_fraction=None):
            fit_samples, branch_samples, threshold_samples = split_training(
                train_samples,
                seed,
                fit_count=fit_count,
                total_budget=total_budget,
                branch_fraction=float(branch_fraction or cfg["branch_calibration_fraction"]),
                threshold_fraction=float(cfg["threshold_calibration_fraction"]),
            )
            groups_local = (fit_samples, branch_samples, threshold_samples)
            payloads = []
            counts = []
            for samples_local in groups_local:
                indices_local = [index_by_path[s.path] for s in samples_local]
                payloads.append({key: value[indices_local] for key, value in full_train.items()})
                counts.append(len(indices_local))
            return (*payloads, *counts)

        def score_result(maps, threshold_images):
            return metrics(
                test["labels"],
                test["masks"],
                maps,
                threshold_images,
                cfg["score_quantile"],
                cfg["image_size"],
                cfg["threshold_alpha"],
            )[0]

        for seed in seeds:
            fit, branch_cal, threshold_cal, fit_count, branch_count, threshold_count = make_split(seed)
            start = time.perf_counter()
            outputs, aux = evaluate_bundle(
                fit,
                branch_cal,
                threshold_cal,
                test,
                cfg,
                seed,
                weights=(cfg["frequency_weight"],),
                include_padim_gate=True,
            )
            fit_seconds = time.perf_counter() - start
            for name, (maps, cal_images) in outputs.items():
                method = "freqpatch_lite" if name.startswith("freqpatch") else name
                result = score_result(maps, cal_images)
                rows.append({
                    "experiment": "main",
                    "category": category,
                    "seed": seed,
                    "method": method,
                    "fit_images": fit_count,
                    "branch_calibration_images": branch_count,
                    "threshold_calibration_images": threshold_count,
                    "total_normal_images": fit_count + branch_count + threshold_count,
                    "fit_and_score_seconds": fit_seconds,
                    "rss_mb": psutil.Process().memory_info().rss / 1024**2,
                    **aux,
                    **result,
                })
                print(f"[main] {category} seed={seed} {method}: I-AUROC={result['image_auroc']:.4f}", flush=True)

        # Fusion weight ablation uses the primary seed and cached features.
        seed = cfg["seed"]
        fit, branch_cal, threshold_cal, fit_count, branch_count, threshold_count = make_split(seed)
        weights = (0.0, 0.10, 0.25, 0.50, 0.75)
        outputs, aux = evaluate_bundle(fit, branch_cal, threshold_cal, test, cfg, seed, weights=weights)
        for weight in weights:
            maps, cal_images = outputs[f"freqpatch_lite_w{weight:.2f}"]
            result = score_result(maps, cal_images)
            rows.append({
                "experiment": "ablation_weight", "category": category, "seed": seed,
                "method": "freqpatch_lite", "weight": weight,
                "fit_images": fit_count, "branch_calibration_images": branch_count,
                "threshold_calibration_images": threshold_count, **aux, **result,
            })

        # Fusion-form ablation isolates calibration, agreement, and bounding.
        variants = (
            "proposed", "raw_weighted_sum", "calibrated_weighted_sum",
            "calibrated_max", "calibrated_min", "calibrated_product",
            "unbounded_agreement", "no_upper_tail", "frequency_tail_gate",
        )
        outputs, aux = evaluate_bundle(
            fit, branch_cal, threshold_cal, test, cfg, seed,
            weights=(), fusion_variants=variants,
        )
        for variant in variants:
            maps, threshold_images = outputs[f"fusion_{variant}"]
            rows.append({
                "experiment": "ablation_fusion", "category": category, "seed": seed,
                "method": variant, "fit_images": fit_count,
                "branch_calibration_images": branch_count,
                "threshold_calibration_images": threshold_count,
                **aux, **score_result(maps, threshold_images),
            })

        if not args.quick:
            for count in cfg["few_shot_counts"]:
                few_fit, few_branch, few_threshold, n_fit, n_branch, n_threshold = make_split(
                    seed, fit_count=count
                )
                outputs, aux = evaluate_bundle(
                    few_fit, few_branch, few_threshold, test, cfg, seed,
                    weights=(cfg["frequency_weight"],),
                )
                for name in ("patchcore_lite", f"freqpatch_lite_w{cfg['frequency_weight']:.2f}"):
                    maps, cal_images = outputs[name]
                    rows.append({
                        "experiment": "few_shot", "category": category, "seed": seed,
                        "method": "freqpatch_lite" if name.startswith("freqpatch") else name,
                        "fit_images": n_fit, "branch_calibration_images": n_branch,
                        "threshold_calibration_images": n_threshold,
                        "total_normal_images": n_fit + n_branch + n_threshold,
                        **aux, **score_result(maps, cal_images),
                    })

            # Strict few-normal-sample study: fit and both calibration sets share one budget.
            for budget in cfg["total_normal_budgets"]:
                b_fit, b_branch, b_threshold, n_fit, n_branch, n_threshold = make_split(
                    seed, total_budget=budget
                )
                outputs, aux = evaluate_bundle(
                    b_fit, b_branch, b_threshold, test, cfg, seed,
                    weights=(cfg["frequency_weight"],),
                )
                for name in ("patchcore_lite", f"freqpatch_lite_w{cfg['frequency_weight']:.2f}"):
                    maps, threshold_images = outputs[name]
                    rows.append({
                        "experiment": "total_budget", "category": category, "seed": seed,
                        "method": "freqpatch_lite" if name.startswith("freqpatch") else name,
                        "normal_budget": budget, "fit_images": n_fit,
                        "branch_calibration_images": n_branch,
                        "threshold_calibration_images": n_threshold,
                        "total_normal_images": n_fit + n_branch + n_threshold,
                        **aux, **score_result(maps, threshold_images),
                    })

            for upper_quantile in cfg["calibration_upper_quantiles"]:
                outputs, aux = evaluate_bundle(
                    fit, branch_cal, threshold_cal, test, cfg, seed,
                    weights=(cfg["frequency_weight"],), upper_quantile=upper_quantile,
                )
                maps, threshold_images = outputs[f"freqpatch_lite_w{cfg['frequency_weight']:.2f}"]
                rows.append({
                    "experiment": "ablation_upper_quantile", "category": category,
                    "seed": seed, "method": "freqpatch_lite",
                    "upper_quantile": upper_quantile, **aux,
                    **score_result(maps, threshold_images),
                })

            for branch_fraction in cfg["calibration_fraction_ablation"]:
                c_fit, c_branch, c_threshold, n_fit, n_branch, n_threshold = make_split(
                    seed, branch_fraction=branch_fraction
                )
                outputs, aux = evaluate_bundle(
                    c_fit, c_branch, c_threshold, test, cfg, seed,
                    weights=(cfg["frequency_weight"],),
                )
                maps, threshold_images = outputs[f"freqpatch_lite_w{cfg['frequency_weight']:.2f}"]
                rows.append({
                    "experiment": "ablation_calibration_fraction", "category": category,
                    "seed": seed, "method": "freqpatch_lite",
                    "branch_calibration_fraction": branch_fraction,
                    "fit_images": n_fit, "branch_calibration_images": n_branch,
                    "threshold_calibration_images": n_threshold, **aux,
                    **score_result(maps, threshold_images),
                })

            # Stress tests perturb only test images; the nominal model is unchanged.
            for translation in cfg["robustness"]["translations"]:
                for brightness in cfg["robustness"]["brightness"]:
                    if translation == 0 and brightness == 1.0:
                        perturbed = test
                    else:
                        brightness_tag = f"{brightness:.2f}".replace(".", "p")
                        perturbed = cache_or_extract(
                            cache_dir / f"{category}_test_t{translation}_b{brightness_tag}.npz",
                            test_samples,
                            extractor,
                            {**cfg, "translation": translation, "brightness": brightness},
                        )
                    outputs, aux = evaluate_bundle(
                        fit, branch_cal, threshold_cal, perturbed, cfg, seed,
                        weights=(cfg["frequency_weight"],),
                    )
                    for name in ("patchcore_lite", f"freqpatch_lite_w{cfg['frequency_weight']:.2f}"):
                        maps, cal_images = outputs[name]
                        result, _, _ = metrics(
                            perturbed["labels"], perturbed["masks"], maps, cal_images,
                            cfg["score_quantile"], cfg["image_size"], cfg["threshold_alpha"],
                        )
                        rows.append({
                            "experiment": "robustness", "category": category, "seed": seed,
                            "method": "freqpatch_lite" if name.startswith("freqpatch") else name,
                            "translation_px": translation, "brightness": brightness, **aux, **result,
                        })

        write_jsonl(args.output, rows)
        print(f"[checkpoint] {category}: {len(rows)} rows", flush=True)

    write_jsonl(args.output, rows)
    with args.output.with_name("hardware.json").open("w", encoding="utf-8") as handle:
        json.dump(hardware, handle, indent=2)
    print(f"Wrote {len(rows)} experiment rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
