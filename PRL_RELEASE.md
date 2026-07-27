# Pattern Recognition Letters Freeze (v1.4.0)

This tag is the frozen reproducible artifact for the manuscript **A
Calibration-Aware Reliability Audit of Frequency Gating for Industrial
Anomaly Detection**.

## Immutable archive identity

| Field | Value |
|---|---|
| Tag target | `a609800ddd84d18a0874bc3eb11ec527815a47df` |
| Release asset | `CalibFreq-Patch-v1.4.0.zip` |
| Asset size | 2,515,341 bytes |
| Asset SHA-256 | `5b0422ba2323beab48d334c8f312cfdf4e024a02597eb6164468d5f4cc1bcc9b` |
| GitHub release | <https://github.com/1922733078-lab/CalibFreq-Patch/releases/tag/v1.4.0> |
| Zenodo version DOI | <https://doi.org/10.5281/zenodo.21628584> |
| Zenodo concept DOI | <https://doi.org/10.5281/zenodo.21628583> |

The GitHub and Zenodo archives contain the same 2,515,341-byte ZIP file.

## Frozen evidence

The tag fixes the complete repository state, including code, the primary
configuration, raw records, derived result tables, numerical figures, tests,
environment metadata, and dataset/pretrained-weight identity records. The PRL
submission uses the following primary evidence files:

| Evidence | Repository path | SHA-256 |
|---|---|---|
| Main configuration | `configs/main.yaml` | `355597a4ab731d554fa1b37e80acdcd2367ff3ccdff17d5e29ccb7124a97fcd4` |
| PRL configuration copy | `configs/prl_v1.4.0.yaml` | `355597a4ab731d554fa1b37e80acdcd2367ff3ccdff17d5e29ccb7124a97fcd4` |
| Main summary | `results/tables/main_summary.csv` | `d81f44dbb8e488978bb7094b30ea5c2aa35a5bd0fe7197c1b17b8f6b174606c6` |
| Multi-seed fusion study | `results/tables/multiseed_fusion.csv` | `f99fab3852b44c211d98d4edff403a15c4a3aa0bc00bc2452abd9f271627297c` |
| Threshold-allocation summary | `results/tables/threshold_priority_summary.csv` | `e2d1f7ad0d2b674d5e610031c5ad4b1ab42da9b4a7b5441e6d57e96a0bdce033` |
| Shift diagnostics | `results/tables/shift_diagnostics.csv` | `299c8abbff12fb02e7e141957d6d226694d6be9272a14fbfd1605075fc2c7918` |
| Qualitative selection record | `figures/qualitative_selection.json` | `7bbf60854a0d58dfe8d58d972ba58c7170b5c1861f3cbaf8bab6392601ab7f35` |

`SHA256SUMS.txt` covers every versioned release file except itself. The
qualitative selection JSON records the deterministic case type, category,
defect label, dataset-relative image/mask paths, unnormalized image score,
conformal threshold, pixel metrics, translation, and boundary mode. It does
not contain image pixels.

## Dataset redistribution boundary

No MVTec AD image or defect-mask pixel array is stored in this repository or
either archive. `data/raw/mvtec/samples.json` contains metadata and relative paths
only; the checksum manifest identifies externally obtained bytes without
redistributing them. Repository figures are numerical/derived plots. The PRL
manuscript may display a small attributed qualitative panel under the dataset
terms, but those third-party image panels are deliberately absent from this
public artifact.

## Interpretation boundary

The release supports a negative reliability audit of one compact
configuration. It does not establish that frequency information is generally
ineffective, nor does the synthetic shift diagnostic represent physical
camera tolerances or production-line alarm rates.
