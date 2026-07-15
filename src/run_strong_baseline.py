#!/usr/bin/env python3
"""Run a higher-capacity WideResNet50 PatchCore control on the same CPU protocol."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import functional as TF

from freqpatch import (
    build_memory,
    calibrate,
    calibration_parameters,
    extract_samples,
    image_scores,
    load_manifest,
    metrics,
    patchcore_score,
)
from run_experiments import split_training, write_jsonl


class WideResNetPatchExtractor:
    """Frozen WRN50 layer-2/3 descriptors with deterministic channel selection."""

    def __init__(self, grid: int, channels: int, seed: int) -> None:
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        self.model = wide_resnet50_2(weights=weights).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.grid = grid
        rng = np.random.default_rng(seed)
        first = rng.choice(512, size=min(96, channels), replace=False)
        second = 512 + rng.choice(1024, size=channels - len(first), replace=False)
        self.indices = torch.as_tensor(np.sort(np.r_[first, second]), dtype=torch.long)
        self.mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
        self.std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (images - self.mean) / self.std
        model = self.model
        x = model.maxpool(model.relu(model.bn1(model.conv1(normalized))))
        layer1 = model.layer1(x)
        layer2 = model.layer2(layer1)
        layer3 = model.layer3(layer2)
        features = torch.cat(
            [
                F.adaptive_avg_pool2d(layer2, (self.grid, self.grid)),
                F.interpolate(layer3, size=(self.grid, self.grid), mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        features = F.normalize(features.index_select(1, self.indices), p=2, dim=1)

        luminance = 0.299 * images[:, :1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        blur1 = TF.gaussian_blur(luminance, kernel_size=[9, 9], sigma=[1.0, 1.0])
        blur2 = TF.gaussian_blur(luminance, kernel_size=[9, 9], sigma=[2.0, 2.0])
        dx = F.pad(luminance[:, :, :, 2:] - luminance[:, :, :, :-2], (1, 1, 0, 0))
        dy = F.pad(luminance[:, :, 2:, :] - luminance[:, :, :-2, :], (0, 0, 1, 1))
        frequency = torch.cat(
            [(luminance - blur1).abs(), (luminance - blur2).abs(), torch.sqrt(dx.square() + dy.square() + 1e-12)],
            dim=1,
        )
        frequency = F.adaptive_avg_pool2d(frequency, (self.grid, self.grid))
        return features.cpu(), frequency.cpu()


def cache_or_extract(path: Path, samples, extractor, cfg):
    if path.exists():
        with np.load(path) as payload:
            return {key: payload[key] for key in payload.files}
    result = extract_samples(samples, extractor, cfg["image_size"], batch_size=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/strong_baseline.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/wr50_features"))
    parser.add_argument("--channels", type=int, default=256)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    extractor = WideResNetPatchExtractor(cfg["feature_grid"], args.channels, cfg["seed"])
    groups = load_manifest(args.data_root, cfg["categories"])
    rows = []
    completed = set()
    if args.output.exists():
        existing = [
            json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for category in cfg["categories"]:
            category_rows = [row for row in existing if row.get("category") == category]
            if len(category_rows) == len(cfg["seeds"]):
                rows.extend(category_rows)
                completed.add(category)

    for category in cfg["categories"]:
        if category in completed:
            print(f"[resume] skip complete WR50 category: {category}", flush=True)
            continue
        samples = groups[category]
        train_samples = [sample for sample in samples if sample.split == "train"]
        test_samples = [sample for sample in samples if sample.split == "test"]
        train = cache_or_extract(args.cache_dir / f"{category}_train.npz", train_samples, extractor, cfg)
        test = cache_or_extract(args.cache_dir / f"{category}_test.npz", test_samples, extractor, cfg)
        index = {sample.path: position for position, sample in enumerate(train_samples)}

        for seed in cfg["seeds"]:
            fit_samples, branch_samples, threshold_samples = split_training(
                train_samples,
                seed,
                branch_fraction=cfg["branch_calibration_fraction"],
                threshold_fraction=cfg["threshold_calibration_fraction"],
            )
            fit_idx = [index[sample.path] for sample in fit_samples]
            branch_idx = [index[sample.path] for sample in branch_samples]
            threshold_idx = [index[sample.path] for sample in threshold_samples]
            start = time.perf_counter()
            memory = build_memory(
                train["features"][fit_idx], cfg["memory_max"], cfg["memory_ratio"], seed
            )
            parameters = calibration_parameters(
                patchcore_score(train["features"][branch_idx], memory, cfg["knn_k"]),
                cfg["score_quantile"],
            )
            threshold_maps = calibrate(
                patchcore_score(train["features"][threshold_idx], memory, cfg["knn_k"]), parameters
            )
            test_maps = calibrate(patchcore_score(test["features"], memory, cfg["knn_k"]), parameters)
            threshold_images = image_scores(threshold_maps, cfg["score_quantile"])
            result = metrics(
                test["labels"], test["masks"], test_maps, threshold_images,
                cfg["score_quantile"], cfg["image_size"], cfg["threshold_alpha"],
            )[0]
            rows.append({
                "experiment": "strong_baseline", "category": category, "seed": seed,
                "method": "patchcore_wr50_compact", "descriptor_channels": args.channels,
                "fit_images": len(fit_idx), "branch_calibration_images": len(branch_idx),
                "threshold_calibration_images": len(threshold_idx),
                "memory_vectors": int(memory.shape[0]), "memory_bytes": int(memory.nbytes),
                "fit_and_score_seconds": time.perf_counter() - start, **result,
            })
            print(f"[WR50] {category} seed={seed} I-AUROC={result['image_auroc']:.4f}", flush=True)
        write_jsonl(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
