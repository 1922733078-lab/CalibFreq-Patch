# Artifact Identity and Manuscript Detail Map

## Frozen PRL release

The reproducible artifact used by the Pattern Recognition Letters manuscript
is frozen at release `v1.4.0` (2026-07-27):

| Field | Frozen value |
|---|---|
| Git tag | `v1.4.0` |
| Tag target | `a609800ddd84d18a0874bc3eb11ec527815a47df` |
| Release asset | `CalibFreq-Patch-v1.4.0.zip` |
| Asset size | 2,515,341 bytes |
| Asset SHA-256 | `5b0422ba2323beab48d334c8f312cfdf4e024a02597eb6164468d5f4cc1bcc9b` |
| GitHub release | <https://github.com/1922733078-lab/CalibFreq-Patch/releases/tag/v1.4.0> |
| Version DOI | <https://doi.org/10.5281/zenodo.21628584> |
| Concept DOI | <https://doi.org/10.5281/zenodo.21628583> |

The Zenodo record contains the byte-identical release asset. The immutable
tag is not retargeted by this or later documentation-only changes.

The release-specific configuration, result-table hashes, qualitative
selection record, and redistribution boundary are documented in
[`PRL_RELEASE.md`](PRL_RELEASE.md). Full file integrity is covered by
[`SHA256SUMS.txt`](SHA256SUMS.txt).

## Environment and evidence map

- Apple M4 CPU, 10 cores, 16 GB unified memory; no CUDA
- Python 3.12, PyTorch 2.7.1, torchvision 0.22.1
- exactly eight PyTorch CPU threads during timing
- complete environment: `environment-lock.txt`
- fixed configuration: `configs/main.yaml` and `configs/prl_v1.4.0.yaml`
- main raw records: `results/raw/experiments.jsonl`
- multi-seed ablations: `results/raw/multiseed_ablation.jsonl`
- shift diagnostics: `results/raw/shift_diagnostics.jsonl`
- derived tables: `results/tables/`
- qualitative selection metadata: `figures/qualitative_selection.json`

## Dataset and pretrained-weight identity

MVTec AD images and masks are not redistributed. The run used the Voxel51
mirror revision `30a183a3b96e3aef953f230784b123b719b09d97`; exact mirror
metadata and per-file hashes are stored in `data/mvtec_mirror_metadata.json`
and `data/mvtec_mirror_SHA256SUMS.txt`. These identify the mirror bytes used
and do not claim byte identity with the separately licensed official archive.

Pretrained weights are not redistributed. Enum names, official URLs, cache
filenames, byte counts, torchvision version, and expected SHA-256 values are
recorded in `data/pretrained_weights_metadata.json`.

## Native-grid validity field

Translation diagnostics operate on the native 28 x 28 anomaly grid.
`valid_grid_fraction` is the canonical retained-grid fraction;
`valid_pixel_fraction` is a deprecated, value-identical compatibility alias.
