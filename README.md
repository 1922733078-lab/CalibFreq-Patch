# CalibFreq-Patch: Reproducible CPU Study

This repository is the public experiment artifact for **Auditing a Bounded
Frequency Gate in One Compact Patch-Memory Detector: A CPU-Only MVTec AD Case Study**.
It evaluates a normal-calibrated, bounded frequency gate on all 15 MVTec AD
categories. The artifact contains the implementation, fixed configurations,
raw per-run results, derived tables, grayscale/vector figures, tests, and
hardware timing records. Dataset images and pretrained weights are not
redistributed.

## What Is Reproduced

- five ResNet-18 controls plus patch-only and gated 256-channel WideResNet-50 controls;
- three fixed seeds for every main category-method combination;
- disjoint fit, branch-calibration, and threshold-calibration normal sets;
- split-conformal operating thresholds and false-alarm metrics, using an
  infinite threshold when the requested rank exceeds the calibration count;
- three-seed fusion-form, gate-weight, and tail-quantile ablations, plus calibration-size sensitivity;
- requested fit-count and strict total-normal-budget studies, including proportional and threshold-prioritized allocation;
- category-level Wilcoxon and exact sign-flip inference with total/nonzero/zero-pair counts;
- brightness stress tests and three-seed translation direction/boundary/interior/registration diagnostics;
- three-round independent-path CPU latency with balanced AB/BA pairing, raw pair records,
  component time, category-bootstrap uncertainty, parameter, memory-bank, and RSS data.

## Environment Used for the Paper

- Apple M4, 10 CPU cores, 16 GB unified memory; no CUDA
- Python 3.12
- PyTorch 2.7.1 and torchvision 0.22.1
- at most eight PyTorch CPU threads

Create an environment and install the direct packages listed in `requirements.txt`;
`environment-lock.txt` records the complete resolved paper environment.
The official MVTec AD files must be placed under `data/raw/mvtec`, or another
root can be supplied with `--data-root`.

## Reproduce

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python src/run_experiments.py \
  --config configs/main.yaml \
  --data-root /path/to/mvtec \
  --output results/raw/experiments.jsonl --fresh
PYTHONPATH=src python src/run_strong_baseline.py \
  --config configs/main.yaml \
  --data-root /path/to/mvtec \
  --output results/raw/strong_baseline.jsonl
PYTHONPATH=src python src/run_multiseed_ablation.py \
  --config configs/main.yaml \
  --data-root /path/to/mvtec \
  --output results/raw/multiseed_ablation.jsonl
PYTHONPATH=src python src/run_shift_diagnostics.py \
  --config configs/main.yaml \
  --data-root /path/to/mvtec \
  --output results/raw/shift_diagnostics.jsonl
PYTHONPATH=src python src/run_wr50_gate_control.py \
  --config configs/main.yaml \
  --data-root /path/to/mvtec \
  --output results/raw/wr50_gate_control.jsonl
PYTHONPATH=src python src/run_threshold_priority.py \
  --config configs/main.yaml \
  --data-root /path/to/mvtec \
  --output results/raw/threshold_priority.jsonl --fresh
python src/make_dataset_checksums.py --data-root /path/to/mvtec
PYTHONPATH=src python src/analyze_results.py \
  --config configs/main.yaml \
  --input results/raw/experiments.jsonl \
  --strong-baseline results/raw/strong_baseline.jsonl
```

Run `src/benchmark_efficiency.py` and `src/benchmark_wr50_efficiency.py` alone
on an otherwise idle device because their outputs are hardware-dependent.
JSONL outputs are checkpointed by category, and feature caches can be placed on
a local volume without changing sample order.

At `threshold_alpha: 0.05`, at least 19 threshold-calibration normals are
required for a finite deterministic threshold. Smaller sets are recorded with
`threshold: null`, `threshold_is_finite: false`, and zero predicted alarms;
the code does not replace an unattainable conformal rank with the calibration
maximum. Boundary tests cover n = 2, 5, 9, 10, 18, and 19.

The strict-budget sensitivity compares the original 70/15/15 allocation with
a threshold-prioritized allocation. When feasible, the latter reserves 19
threshold-calibration images while retaining at least four detector-fitting
and two branch-calibration images. It therefore tests allocation, not only
total normal supply.

The corrected timing audit compares an actual semantic-only path against the
complete high-pass path. On the declared Apple M4 run, complete high-pass
construction, frequency scoring, and gating added a 1.784 ms median paired
increment (95% category-bootstrap CI 1.642--1.901 ms), or 9.20% relative
median overhead against the load-inclusive semantic baseline. The earlier frequency-score-plus-gate-only quantity is not
used as the complete branch cost.

## Data and Interpretation

MVTec AD is licensed CC BY-NC-SA 4.0. Follow its official terms when obtaining
or using the dataset. The study is a public-benchmark experiment, not a
production-line validation. The expanded all-15-category MVTec AD evidence does **not**
establish a statistically reliable advantage for the proposed gate; this
negative result and the severe false-alarm failure under translation are
intentional parts of the reported evidence.

The gate form and weight were informed by exploratory results on six MVTec AD
categories before the expanded run; those categories remain in the final 15.
The final experiment is therefore an expanded full-MVTec-AD evaluation, not an
independent preregistered confirmation. Its conclusions apply only to the
tested 224-pixel, 96-channel, 1,800-vector compact configuration.

No MVTec image is redistributed in the manuscript or artifact. The downloader
pins the Voxel51 mirror revision and produces SHA-256 manifests for every image
and mask used. Those hashes identify the mirror bytes used; they do not claim
byte identity with the separately licensed official MVTec archive. Figure 4 is
a derived numerical translation diagnostic rather than a dataset-image panel.
The delivered checksum list is `data/mvtec_mirror_SHA256SUMS.txt`, with revision,
counts, and manifest digests in `data/mvtec_mirror_metadata.json`.

## License

The experiment software is released under the MIT License; see `LICENSE`.
MVTec AD images and pretrained model weights are not included and remain under
their respective terms.

## Integrity Verification

`SHA256SUMS.txt` covers versioned release files only and intentionally excludes
itself, Git metadata, ignored caches, and bytecode. Verify a clone or extracted
release from the repository root with `shasum -a 256 -c SHA256SUMS.txt`.
