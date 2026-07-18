# Artifact Identity and Manuscript Detail Map

This page is the stable location for release, integrity, environment, and file-map
details that are useful for reproduction but too granular for the manuscript
narrative.

## Frozen Paper Release

The experiment artifact used for the paper is frozen as follows:

| Field | Value |
|---|---|
| Release | `v1.3.2` |
| Release date | 2026-07-16 |
| Tag target | `375801c18be02f967f28f0e8f3e14bc89641ae53` |
| Asset | `CalibFreq-Patch-v1.3.2.zip` |
| Asset size | 2,510,730 bytes |
| Asset SHA-256 | `5e0b77eccd3a2b015586acb283e8fc5fcdb97281a14bdced55b6aa4855ac191b` |
| Release URL | <https://github.com/1922733078-lab/CalibFreq-Patch/releases/tag/v1.3.2> |

The submission copy named `CalibFreq-Patch_supplementary_artifact.zip` was
verified as byte-identical to the public release asset. The frozen tag is not
retargeted when documentation on the default branch is clarified.

## Environment and Timing Identity

- Apple M4, 10 CPU cores, 16 GB unified memory; no CUDA
- Python 3.12
- PyTorch 2.7.1
- torchvision 0.22.1
- exactly eight PyTorch CPU threads during the timing audit
- complete resolved environment: [`environment-lock.txt`](environment-lock.txt)
- hardware record: [`results/raw/hardware.json`](results/raw/hardware.json)
- complete-path timing summary: [`results/tables/efficiency.json`](results/tables/efficiency.json)
- WideResNet timing summary: [`results/tables/efficiency_wr50.json`](results/tables/efficiency_wr50.json)

The timing audit compares two independently executed paths from the same
preloaded tensor: a semantic-only path that never constructs high-pass
features, and a complete path that additionally constructs luminance,
residuals, gradients, frequency scores, and the gate. Raw pairing and
category-level uncertainty remain in the versioned artifact.

## Dataset Byte Identity

Dataset images are not redistributed. The paper run used the public Voxel51
mirror at revision:

`30a183a3b96e3aef953f230784b123b719b09d97`

The exact mirror manifest identity, sample counts, verification scope, and
manifest digests are recorded in
[`data/mvtec_mirror_metadata.json`](data/mvtec_mirror_metadata.json).
Per-file image and mask hashes are in
[`data/mvtec_mirror_SHA256SUMS.txt`](data/mvtec_mirror_SHA256SUMS.txt).

These hashes identify the mirror bytes used. They do not assert byte identity
with the separately licensed official MVTec archive.

## Pretrained-Weight Identity

Pretrained weights are not redistributed. Model enum names, official URLs,
cache filenames, byte counts, torchvision version, and expected SHA-256 values
are recorded in
[`data/pretrained_weights_metadata.json`](data/pretrained_weights_metadata.json).

## Raw and Derived Evidence Map

| Evidence | Repository location |
|---|---|
| Main category/seed runs | `results/raw/experiments.jsonl` |
| Strong baseline | `results/raw/strong_baseline.jsonl` |
| Multi-seed fusion, weight, and tail controls | `results/raw/multiseed_ablation.jsonl` |
| Direction, boundary, interior, and registration diagnostics | `results/raw/shift_diagnostics.jsonl` |
| Threshold-prioritized allocation | `results/raw/threshold_priority.jsonl` |
| WideResNet gate control | `results/raw/wr50_gate_control.jsonl` |
| Derived statistical tables | `results/tables/` |
| Vector and raster figures | `figures/` |
| Fixed experiment configurations | `configs/` |
| Protocol and geometry tests | `tests/` |
| Versioned-file integrity manifest | `SHA256SUMS.txt` |

The versioned manifest contains 76 entries and excludes itself, Git metadata,
ignored caches, and bytecode by design.

## Native-Grid Validity Field

Translation diagnostics operate on the native 28 × 28 anomaly map. The field
`valid_grid_fraction` records the retained native-grid fraction. The historical
field `valid_pixel_fraction` is retained only as a deprecated,
value-identical compatibility alias; it is not computed from a 224 × 224
input-pixel mask. The equality and diagonal-shift geometry are covered by
`tests/test_core.py`.

## Details Deliberately Kept Out of the Manuscript Narrative

The manuscript uses the repository or release citation instead of repeating:

- the full Git commit identifier and archive SHA-256;
- release asset size and the complete versioned-file list;
- the pinned mirror revision and every image/mask checksum;
- pretrained-weight filenames, sizes, URLs, and hashes;
- raw JSONL/derived CSV directory listings;
- compatibility-field history and low-level schema notes.

These details remain public, versioned, and auditable here without interrupting
the paper's argument.
