"""Core implementation of FreqPatch-Lite and its experiment baselines."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class Sample:
    path: Path
    mask_path: Path | None
    category: str
    defect: str
    split: str

    @property
    def label(self) -> int:
        return int(self.defect != "good")


def load_manifest(root: Path, categories: Iterable[str]) -> dict[str, list[Sample]]:
    wanted = set(categories)
    with (root / "samples.json").open("r", encoding="utf-8") as handle:
        raw = json.load(handle)["samples"]
    grouped: dict[str, list[Sample]] = {category: [] for category in wanted}
    for row in raw:
        category = row["category"]["label"]
        if category not in wanted:
            continue
        grouped[category].append(
            Sample(
                path=root / row["filepath"],
                mask_path=root / row["defect_mask"]["mask_path"] if "defect_mask" in row else None,
                category=category,
                defect=row["defect"]["label"],
                split=row["split"],
            )
        )
    return grouped


def deterministic_channel_indices(total_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Preserve all three semantic scales rather than letting random sampling
    # overrepresent the deepest (largest) block.
    blocks = [(0, 64, 24), (64, 192, 32), (192, total_dim, output_dim - 56)]
    indices = [rng.choice(np.arange(start, end), size=count, replace=False) for start, end, count in blocks]
    return np.sort(np.concatenate(indices)).astype(np.int64)


class FrozenFeatureExtractor:
    def __init__(self, projection_dim: int, grid: int, seed: int, device: str = "cpu") -> None:
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=weights).eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = torch.device(device)
        self.grid = grid
        self.indices = torch.as_tensor(
            deterministic_channel_indices(448, projection_dim, seed), dtype=torch.long, device=self.device
        )
        self.mean = torch.tensor(weights.transforms().mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(weights.transforms().std, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device)
        normalized = (images - self.mean) / self.std
        model = self.model
        x = model.maxpool(model.relu(model.bn1(model.conv1(normalized))))
        layer1 = model.layer1(x)
        layer2 = model.layer2(layer1)
        layer3 = model.layer3(layer2)
        features = torch.cat(
            [
                F.adaptive_avg_pool2d(layer1, (self.grid, self.grid)),
                F.adaptive_avg_pool2d(layer2, (self.grid, self.grid)),
                F.interpolate(layer3, size=(self.grid, self.grid), mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        features = features.index_select(1, self.indices)
        features = F.normalize(features, p=2, dim=1)

        luminance = 0.299 * images[:, :1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        blur1 = TF.gaussian_blur(luminance, kernel_size=[9, 9], sigma=[1.0, 1.0])
        blur2 = TF.gaussian_blur(luminance, kernel_size=[9, 9], sigma=[2.0, 2.0])
        residual1 = (luminance - blur1).abs()
        residual2 = (luminance - blur2).abs()
        dx = luminance[:, :, :, 2:] - luminance[:, :, :, :-2]
        dy = luminance[:, :, 2:, :] - luminance[:, :, :-2, :]
        dx = F.pad(dx, (1, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 1, 1))
        gradient = torch.sqrt(dx.square() + dy.square() + 1e-12)
        frequency = torch.cat([residual1, residual2, gradient], dim=1)
        frequency = F.adaptive_avg_pool2d(frequency, (self.grid, self.grid))
        return features.cpu(), frequency.cpu()


def load_image(sample: Sample, image_size: int, translation: int = 0, brightness: float = 1.0) -> torch.Tensor:
    with Image.open(sample.path) as image:
        image = image.convert("RGB")
        image = TF.resize(image, [image_size, image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        tensor = TF.to_tensor(image)
    if translation:
        tensor = TF.affine(
            tensor,
            angle=0.0,
            translate=[translation, translation],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.5,
        )
    return tensor


def load_mask(sample: Sample, image_size: int, translation: int = 0) -> np.ndarray:
    if sample.mask_path is None:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    with Image.open(sample.mask_path) as mask:
        mask = mask.convert("L")
        mask = TF.resize(mask, [image_size, image_size], interpolation=InterpolationMode.NEAREST)
        tensor = TF.pil_to_tensor(mask).float() / 255.0
    if translation:
        tensor = TF.affine(
            tensor,
            angle=0.0,
            translate=[translation, translation],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )
    return (tensor.squeeze(0).numpy() > 0.5).astype(np.uint8)


def extract_samples(
    samples: list[Sample],
    extractor: FrozenFeatureExtractor,
    image_size: int,
    batch_size: int = 8,
    translation: int = 0,
    brightness: float = 1.0,
) -> dict[str, np.ndarray]:
    feature_batches: list[np.ndarray] = []
    frequency_batches: list[np.ndarray] = []
    for start in range(0, len(samples), batch_size):
        batch = torch.stack(
            [load_image(s, image_size, translation=translation, brightness=brightness) for s in samples[start : start + batch_size]]
        )
        features, frequency = extractor(batch)
        feature_batches.append(features.permute(0, 2, 3, 1).numpy().astype(np.float32))
        frequency_batches.append(frequency.permute(0, 2, 3, 1).numpy().astype(np.float32))
    return {
        "features": np.concatenate(feature_batches),
        "frequency": np.concatenate(frequency_batches),
        "labels": np.asarray([s.label for s in samples], dtype=np.uint8),
        "masks": np.stack([load_mask(s, image_size, translation=translation) for s in samples]),
    }


def robust_frequency_stats(frequency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(frequency, axis=0)
    mad = np.median(np.abs(frequency - median[None]), axis=0)
    floor = np.quantile(mad, 0.1, axis=(0, 1), keepdims=True)
    return median.astype(np.float32), np.maximum(1.4826 * mad, floor + 1e-6).astype(np.float32)


def frequency_score(frequency: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = np.abs((frequency - median[None]) / (scale[None] + 1e-6))
    return np.sqrt(np.mean(z * z, axis=-1)).astype(np.float32)


def padim_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    floor = np.quantile(std, 0.1, axis=(0, 1), keepdims=True)
    return mean.astype(np.float32), np.maximum(std, floor + 1e-4).astype(np.float32)


def padim_score(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (features - mean[None]) / (std[None] + 1e-6)
    return np.sqrt(np.mean(z * z, axis=-1)).astype(np.float32)


def build_memory(features: np.ndarray, maximum: int, ratio: float, seed: int) -> np.ndarray:
    flattened = features.reshape(-1, features.shape[-1])
    count = min(maximum, max(256, int(math.ceil(len(flattened) * ratio))))
    rng = np.random.default_rng(seed)
    # Stratify across spatial cells, then fill any remainder globally.
    n, h, w, d = features.shape
    per_location = max(1, count // (h * w))
    parts = []
    for row in range(h):
        for col in range(w):
            idx = rng.choice(n, size=min(per_location, n), replace=False)
            parts.append(features[idx, row, col])
    memory = np.concatenate(parts)
    if len(memory) < count:
        idx = rng.choice(len(flattened), size=count - len(memory), replace=False)
        memory = np.concatenate([memory, flattened[idx]])
    if len(memory) > count:
        memory = memory[rng.choice(len(memory), size=count, replace=False)]
    norm = np.linalg.norm(memory, axis=1, keepdims=True)
    return (memory / np.maximum(norm, 1e-8)).astype(np.float32)


def patchcore_score(features: np.ndarray, memory: np.ndarray, k: int = 3, chunk: int = 4096) -> np.ndarray:
    flat = torch.from_numpy(features.reshape(-1, features.shape[-1]))
    bank = torch.from_numpy(memory).T.contiguous()
    output = []
    for start in range(0, len(flat), chunk):
        similarities = flat[start : start + chunk] @ bank
        nearest = torch.topk(similarities, k=min(k, memory.shape[0]), dim=1).values
        output.append((1.0 - nearest).mean(dim=1))
    scores = torch.cat(output).numpy()
    return scores.reshape(features.shape[:3]).astype(np.float32)


def calibration_parameters(scores: np.ndarray, upper_quantile: float = 0.995) -> tuple[float, float]:
    median = float(np.median(scores))
    upper = float(np.quantile(scores, upper_quantile))
    return median, max(upper - median, 1e-6)


def calibrate(scores: np.ndarray, parameters: tuple[float, float]) -> np.ndarray:
    median, scale = parameters
    return np.maximum((scores - median) / scale, 0.0).astype(np.float32)


def fuse_scores(deep: np.ndarray, frequency: np.ndarray, weight: float) -> np.ndarray:
    """Apply a bounded frequency bonus only where both branches agree.

    Both inputs are independently calibrated on held-out normal images.  The
    The gate opens only beyond both branches' calibrated q99.5 normal tails,
    while ``tanh`` limits the multiplier to ``[1, 1 + weight]``. Consequently,
    the frequency branch cannot replace a semantic anomaly score or create an
    unbounded response on its own.
    """
    # A calibrated value of one is the branch's q99.5 normal tail.  Requiring
    # both branches to exceed that reference avoids reordering ordinary
    # background responses and turns the frequency cue into an extreme-tail
    # confirmation signal.
    agreement = np.maximum(np.minimum(deep, frequency) - 1.0, 0.0)
    multiplier = 1.0 + weight * np.tanh(agreement)
    return (deep * multiplier).astype(np.float32)


def fuse_scores_variant(
    deep: np.ndarray,
    frequency: np.ndarray,
    weight: float,
    variant: str,
) -> np.ndarray:
    """Fusion controls used to isolate calibration, agreement, and bounding."""
    if variant == "proposed":
        return fuse_scores(deep, frequency, weight)
    if variant == "calibrated_weighted_sum":
        return ((deep + weight * frequency) / (1.0 + weight)).astype(np.float32)
    if variant == "calibrated_max":
        return np.maximum(deep, frequency).astype(np.float32)
    if variant == "calibrated_min":
        return np.minimum(deep, frequency).astype(np.float32)
    if variant == "calibrated_product":
        return (deep * frequency).astype(np.float32)
    agreement = np.maximum(np.minimum(deep, frequency) - 1.0, 0.0)
    if variant == "unbounded_agreement":
        return (deep * (1.0 + weight * agreement)).astype(np.float32)
    if variant == "no_upper_tail":
        return (deep * (1.0 + weight * np.tanh(np.minimum(deep, frequency)))).astype(np.float32)
    if variant == "frequency_tail_gate":
        cue = np.maximum(frequency - 1.0, 0.0)
        return (deep * (1.0 + weight * np.tanh(cue))).astype(np.float32)
    raise ValueError(f"Unknown fusion variant: {variant}")


def image_scores(pixel_scores: np.ndarray, quantile: float) -> np.ndarray:
    flat = pixel_scores.reshape(len(pixel_scores), -1)
    count = max(1, int(math.ceil(flat.shape[1] * (1.0 - quantile))))
    return np.partition(flat, flat.shape[1] - count, axis=1)[:, -count:].mean(axis=1)


def resize_maps(maps: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(maps[:, None])
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized[:, 0].numpy().astype(np.float32)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")[::-1]
    y = labels[order]
    sorted_scores = scores[order]
    distinct = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(scores)] - 1
    tps = np.cumsum(y, dtype=np.int64)[distinct]
    fps = (distinct + 1) - tps
    tpr = np.r_[0.0, tps / positives, 1.0]
    fpr = np.r_[0.0, fps / negatives, 1.0]
    return float(np.trapezoid(tpr, fpr))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")[::-1]
    y = labels[order]
    sorted_scores = scores[order]
    distinct = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(scores)] - 1
    tps = np.cumsum(y, dtype=np.int64)[distinct]
    precision = tps / (distinct + 1)
    recall = tps / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8)
    predictions = np.asarray(predictions, dtype=np.uint8)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def conformal_upper_threshold(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    """Finite-sample split-conformal upper quantile for anomaly scores."""
    ordered = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    if len(ordered) == 0:
        raise ValueError("Threshold calibration scores must not be empty")
    rank = min(len(ordered), int(math.ceil((len(ordered) + 1) * (1.0 - alpha))))
    return float(ordered[rank - 1]), rank


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.uint8).reshape(-1)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (tn + fp) if tn + fp else 0.0
    f1 = binary_f1(labels, predictions)
    return {
        "image_f1_conformal": f1,
        "image_f1_normal95": f1,
        "image_precision_conformal": float(precision),
        "image_recall_conformal": float(recall),
        "image_specificity_conformal": float(specificity),
        "image_balanced_accuracy_conformal": float((recall + specificity) / 2.0),
        "normal_fpr_conformal": float(fpr),
        "false_alarms_per_1000_normals": float(1000.0 * fpr),
        "test_true_positive": tp,
        "test_false_positive": fp,
        "test_true_negative": tn,
        "test_false_negative": fn,
    }


def metrics(
    labels: np.ndarray,
    masks: np.ndarray,
    lowres_scores: np.ndarray,
    calibration_image_scores: np.ndarray,
    quantile: float,
    image_size: int,
    threshold_alpha: float = 0.05,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    per_image = image_scores(lowres_scores, quantile)
    pixel_scores = resize_maps(lowres_scores, image_size)
    threshold, threshold_rank = conformal_upper_threshold(calibration_image_scores, threshold_alpha)
    predictions = (per_image > threshold).astype(np.uint8)
    result = {
        "image_auroc": safe_auc(labels, per_image),
        "image_ap": average_precision(labels, per_image),
        "pixel_auroc": safe_auc(masks.reshape(-1), pixel_scores.reshape(-1)),
        "pixel_ap": average_precision(masks.reshape(-1), pixel_scores.reshape(-1)),
        "threshold": threshold,
        "threshold_alpha": float(threshold_alpha),
        "threshold_calibration_count": int(len(calibration_image_scores)),
        "threshold_conformal_rank": int(threshold_rank),
        **classification_metrics(labels, predictions),
    }
    return result, per_image, pixel_scores


def timed_score(function, *args, repeats: int = 3, **kwargs):
    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        durations.append(time.perf_counter() - start)
    return result, float(np.median(durations))
