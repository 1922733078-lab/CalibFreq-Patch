# Artifact Identity and Manuscript Detail Map

## Frozen PRL release

The reproducible artifact used by the Pattern Recognition Letters manuscript
is frozen at release `v1.4.0` (2026-07-27). The immutable tag target, release
asset size, SHA-256, and any archival DOI are recorded on the GitHub release
page and, after publication, in the documentation-only commit on the default
branch. The frozen tag is not retargeted by later documentation changes.

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
