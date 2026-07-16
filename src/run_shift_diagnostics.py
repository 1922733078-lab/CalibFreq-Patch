#!/usr/bin/env python3
"""Diagnose translation sensitivity across direction, boundary, and registration controls."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from freqpatch import (
    FrozenFeatureExtractor,
    average_precision,
    classification_metrics,
    conformal_upper_threshold,
    extract_samples,
    image_scores,
    load_manifest,
    safe_auc,
)
from run_experiments import evaluate_bundle, split_training, write_jsonl


CONDITIONS = (
    ("east_constant", (4, 0), "constant", (0, 0), 0),
    ("west_constant", (-4, 0), "constant", (0, 0), 0),
    ("south_constant", (0, 4), "constant", (0, 0), 0),
    ("north_constant", (0, -4), "constant", (0, 0), 0),
    ("southeast_constant", (4, 4), "constant", (0, 0), 0),
    ("southwest_constant", (-4, 4), "constant", (0, 0), 0),
    ("northeast_constant", (4, -4), "constant", (0, 0), 0),
    ("northwest_constant", (-4, -4), "constant", (0, 0), 0),
    ("southeast_reflect", (4, 4), "reflect", (0, 0), 0),
    ("southeast_replicate", (4, 4), "replicate", (0, 0), 0),
    # A deliberately simple known-shift registration control: inverse the
    # synthetic displacement before scoring and exclude a four-pixel rim in
    # the interior-only diagnostic.
    ("southeast_reflect_inverse_registered", (4, 4), "reflect", (-4, -4), 4),
)


def valid_mask(image_size: int, dx: int, dy: int, extra_margin: int = 0) -> np.ndarray:
    valid = np.ones((image_size, image_size), dtype=bool)
    if dx > 0:
        valid[:, :dx] = False
    elif dx < 0:
        valid[:, image_size + dx:] = False
    if dy > 0:
        valid[:dy, :] = False
    elif dy < 0:
        valid[image_size + dy:, :] = False
    if extra_margin:
        valid[:extra_margin, :] = False
        valid[-extra_margin:, :] = False
        valid[:, :extra_margin] = False
        valid[:, -extra_margin:] = False
    return valid


def interior_metrics(labels, masks, maps, threshold_maps, cfg, dx, dy, extra_margin):
    grid = maps.shape[-1]
    cells_x = int(math.ceil(abs(dx) * grid / cfg["image_size"]))
    cells_y = int(math.ceil(abs(dy) * grid / cfg["image_size"]))
    x0, x1 = (cells_x if dx > 0 else 0), (grid - cells_x if dx < 0 else grid)
    y0, y1 = (cells_y if dy > 0 else 0), (grid - cells_y if dy < 0 else grid)
    if extra_margin:
        margin_cells = int(math.ceil(extra_margin * grid / cfg["image_size"]))
        x0, x1 = max(x0, margin_cells), min(x1, grid - margin_cells)
        y0, y1 = max(y0, margin_cells), min(y1, grid - margin_cells)
    interior_images = image_scores(maps[:, y0:y1, x0:x1], cfg["score_quantile"])
    interior_threshold_images = image_scores(
        threshold_maps[:, y0:y1, x0:x1], cfg["score_quantile"]
    )
    threshold, _ = conformal_upper_threshold(
        interior_threshold_images, cfg["threshold_alpha"]
    )
    classification = classification_metrics(labels, interior_images > threshold)
    # The score is evaluated on the native anomaly-map grid.  A four-pixel
    # displacement at 224/28 geometry therefore removes one complete grid
    # cell (an eight-input-pixel-equivalent conservative crop).  Report the
    # valid fraction from that exact scoring region rather than from a
    # different full-resolution mask.
    valid_grid_cells = (x1 - x0) * (y1 - y0)
    valid_grid_fraction = valid_grid_cells / float(grid * grid)
    return {
        "interior_image_auroc": safe_auc(labels, interior_images),
        "interior_image_ap": average_precision(labels, interior_images),
        "interior_normal_fpr": classification["normal_fpr_conformal"],
        "interior_recall": classification["image_recall_conformal"],
        "valid_grid_fraction": float(valid_grid_fraction),
        # Backward-compatible alias retained for v1.3.0 consumers.  Despite
        # its historical name, this is the fraction of native anomaly-grid
        # cells scored, not a fraction of 224 x 224 input pixels.
        "valid_pixel_fraction": float(valid_grid_fraction),
        "interior_grid_x0": int(x0),
        "interior_grid_x1": int(x1),
        "interior_grid_y0": int(y0),
        "interior_grid_y1": int(y1),
        "interior_grid_valid_cells": int(valid_grid_cells),
        "interior_grid_total_cells": int(grid * grid),
        "interior_equivalent_pixel_x0": float(x0 * cfg["image_size"] / grid),
        "interior_equivalent_pixel_x1": float((grid - x1) * cfg["image_size"] / grid),
        "interior_equivalent_pixel_y0": float(y0 * cfg["image_size"] / grid),
        "interior_equivalent_pixel_y1": float((grid - y1) * cfg["image_size"] / grid),
    }


def image_only_metrics(labels, maps, threshold_images, cfg):
    scores = image_scores(maps, cfg["score_quantile"])
    threshold, _ = conformal_upper_threshold(threshold_images, cfg["threshold_alpha"])
    classification = classification_metrics(labels, scores > threshold)
    return {
        "image_auroc": safe_auc(labels, scores),
        "image_ap": average_precision(labels, scores),
        **classification,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec"))
    parser.add_argument("--nominal-cache", type=Path, default=Path("data/processed/features"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/shift_diagnostics"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/shift_diagnostics.jsonl"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    groups = load_manifest(args.data_root, cfg["categories"])
    extractor = FrozenFeatureExtractor(cfg["projection_dim"], cfg["feature_grid"], cfg["seed"], "cpu")
    rows: list[dict] = []

    for category in cfg["categories"]:
        train_samples = [sample for sample in groups[category] if sample.split == "train"]
        test_samples = [sample for sample in groups[category] if sample.split == "test"]
        with np.load(args.nominal_cache / f"{category}_train.npz") as payload:
            train = {key: payload[key] for key in payload.files}
        index_by_path = {sample.path: index for index, sample in enumerate(train_samples)}

        for condition, translation, border_mode, registration, extra_margin in CONDITIONS:
            cache_path = args.cache_dir / f"{category}_{condition}.npz"
            if cache_path.exists():
                with np.load(cache_path) as payload:
                    test = {key: payload[key] for key in payload.files}
            else:
                test = extract_samples(
                    test_samples, extractor, cfg["image_size"],
                    translation=translation, border_mode=border_mode,
                    registration_translation=registration,
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path, **test)

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
                nominal_threshold_outputs, _ = evaluate_bundle(
                    fit, branch, threshold, threshold, cfg, seed,
                    weights=(cfg["frequency_weight"],),
                )
                dx, dy = translation
                for output_name in (
                    "patchcore_lite", f"freqpatch_lite_w{cfg['frequency_weight']:.2f}"
                ):
                    maps, threshold_images = outputs[output_name]
                    nominal_threshold_maps, _ = nominal_threshold_outputs[output_name]
                    overall = image_only_metrics(test["labels"], maps, threshold_images, cfg)
                    rows.append({
                        "experiment": "shift_diagnostic", "category": category, "seed": seed,
                        "method": "freqpatch_lite" if output_name.startswith("freqpatch") else output_name,
                        "condition": condition, "translation_dx": dx, "translation_dy": dy,
                        "border_mode": border_mode,
                        "registration_dx": registration[0], "registration_dy": registration[1],
                        **aux, **overall,
                        **interior_metrics(
                            test["labels"], test["masks"], maps, nominal_threshold_maps,
                            cfg, dx, dy, extra_margin,
                        ),
                    })
            write_jsonl(args.output, rows)
            print(f"[checkpoint] {category} {condition}: {len(rows)} rows", flush=True)

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
