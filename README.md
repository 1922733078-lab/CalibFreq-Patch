# CalibFreq-Patch: Reproducible CPU Study

This repository is the public experiment artifact for **Does Frequency Gating
Improve Compact Patch-Memory Anomaly Detection? A Controlled CPU-Only Study**.
It evaluates a normal-calibrated, bounded frequency gate on all 15 MVTec AD
categories. The artifact contains the implementation, fixed configurations,
raw per-run results, derived tables, grayscale/vector figures, tests, and
hardware timing records. Dataset images and pretrained weights are not
redistributed.

## What Is Reproduced

- five ResNet-18 controls plus a 256-channel WideResNet-50 control;
- three fixed seeds for every main category-method combination;
- disjoint fit, branch-calibration, and threshold-calibration normal sets;
- split-conformal operating thresholds and false-alarm metrics;
- fusion-form, gate-weight, tail-quantile, and calibration-size ablations;
- requested fit-count and strict total-normal-budget studies;
- brightness/translation stress tests for PatchCore-Lite and the proposed gate;
- three-round CPU latency, component time, parameter, memory-bank, and RSS data.

## Environment Used for the Paper

- Apple M4, 10 CPU cores, 16 GB unified memory; no CUDA
- Python 3.12
- PyTorch 2.7.1 and torchvision 0.22.1
- at most eight PyTorch CPU threads

Create an environment and install the packages listed in `requirements.txt`.
The official MVTec AD files must be placed under `data/raw/mvtec`, or another
root can be supplied with `--data-root`.

## Reproduce

```bash
pytest -q
PYTHONPATH=src python src/run_experiments.py \
  --config configs/main.yaml \
  --data-root /path/to/mvtec \
  --output results/raw/experiments.jsonl --fresh
PYTHONPATH=src python src/run_strong_baseline.py \
  --config configs/main.yaml \
  --data-root /path/to/mvtec \
  --output results/raw/strong_baseline.jsonl
PYTHONPATH=src python src/analyze_results.py \
  --config configs/main.yaml \
  --input results/raw/experiments.jsonl \
  --strong-baseline results/raw/strong_baseline.jsonl
```

Run `src/benchmark_efficiency.py` and `src/benchmark_wr50_efficiency.py` alone
on an otherwise idle device because their outputs are hardware-dependent.
JSONL outputs are checkpointed by category, and feature caches can be placed on
a local volume without changing sample order.

## Data and Interpretation

MVTec AD is licensed CC BY-NC-SA 4.0. Follow its official terms when obtaining
or using the dataset. The study is a public-benchmark experiment, not a
production-line validation. The complete 15-category evidence does **not**
establish a statistically reliable advantage for the proposed gate; this
negative result and the severe false-alarm failure under translation are
intentional parts of the reported evidence.
