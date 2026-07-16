#!/usr/bin/env python3
"""Compare proportional and threshold-prioritized strict-normal allocations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from freqpatch import load_manifest, metrics
from run_experiments import evaluate_bundle, split_training, write_jsonl


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/features"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/raw/threshold_priority.jsonl"),
    )
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    groups = load_manifest(args.data_root, cfg["categories"])
    budgets = list(cfg["total_normal_budgets"])
    strategies = (("proportional", None), ("threshold_prioritized", 19))
    expected_per_category = len(cfg["seeds"]) * len(budgets) * len(strategies) * 2

    rows: list[dict] = []
    complete_categories: set[str] = set()
    if args.output.exists() and not args.fresh:
        existing = [
            json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for category in cfg["categories"]:
            category_rows = [row for row in existing if row.get("category") == category]
            if len(category_rows) == expected_per_category:
                rows.extend(category_rows)
                complete_categories.add(category)
        print(
            f"[resume] retained {len(rows)} rows across {len(complete_categories)} categories",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for category in cfg["categories"]:
        if category in complete_categories:
            continue
        samples = groups[category]
        train_samples = [sample for sample in samples if sample.split == "train"]
        train = load_npz(args.cache_dir / f"{category}_train.npz")
        test = load_npz(args.cache_dir / f"{category}_test.npz")
        index = {sample.path: position for position, sample in enumerate(train_samples)}

        def subset(selected):
            positions = [index[sample.path] for sample in selected]
            return {key: values[positions] for key, values in train.items()}

        for seed in cfg["seeds"]:
            for budget in budgets:
                for strategy, threshold_min_count in strategies:
                    fit_samples, branch_samples, threshold_samples = split_training(
                        train_samples,
                        int(seed),
                        total_budget=int(budget),
                        branch_fraction=float(cfg["branch_calibration_fraction"]),
                        threshold_fraction=float(cfg["threshold_calibration_fraction"]),
                        threshold_min_count=threshold_min_count,
                    )
                    fit, branch, threshold = map(
                        subset, (fit_samples, branch_samples, threshold_samples)
                    )
                    outputs, aux = evaluate_bundle(
                        fit, branch, threshold, test, cfg, int(seed),
                        weights=(float(cfg["frequency_weight"]),),
                    )
                    output_names = (
                        ("patchcore_lite", "patchcore_lite"),
                        (f"freqpatch_lite_w{float(cfg['frequency_weight']):.2f}", "freqpatch_lite"),
                    )
                    for output_name, method in output_names:
                        maps, threshold_images = outputs[output_name]
                        result = metrics(
                            test["labels"], test["masks"], maps, threshold_images,
                            float(cfg["score_quantile"]), int(cfg["image_size"]),
                            float(cfg["threshold_alpha"]),
                        )[0]
                        rows.append({
                            "experiment": "threshold_priority",
                            "category": category,
                            "seed": int(seed),
                            "method": method,
                            "normal_budget_requested": int(budget),
                            "normal_budget_achieved": int(
                                len(fit_samples) + len(branch_samples) + len(threshold_samples)
                            ),
                            "allocation_strategy": strategy,
                            "priority_target": 19 if threshold_min_count is not None else None,
                            "priority_achieved": bool(len(threshold_samples) >= 19),
                            "fit_images": len(fit_samples),
                            "branch_calibration_images": len(branch_samples),
                            "threshold_calibration_images": len(threshold_samples),
                            **aux,
                            **result,
                        })
                    print(
                        f"[{category}] seed={seed} budget={budget} {strategy} "
                        f"split={len(fit_samples)}/{len(branch_samples)}/{len(threshold_samples)}",
                        flush=True,
                    )
        write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
