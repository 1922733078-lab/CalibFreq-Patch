import unittest
from unittest.mock import patch

import numpy as np

from freqpatch import (
    FrozenFeatureExtractor,
    classification_metrics,
    conformal_upper_threshold,
    fuse_scores,
    fuse_scores_variant,
    translate_tensor,
)
import torch
from run_experiments import split_training
from run_shift_diagnostics import interior_metrics, valid_mask
from analyze_results import exact_sign_flip_pvalue


class ProtocolTests(unittest.TestCase):
    def test_three_way_split_is_disjoint_and_complete(self):
        samples = list(range(100))
        fit, branch, threshold = split_training(
            samples, 7, branch_fraction=0.15, threshold_fraction=0.15
        )
        self.assertEqual(len(fit), 70)
        self.assertEqual(len(branch), 15)
        self.assertEqual(len(threshold), 15)
        self.assertEqual(set(fit) & set(branch), set())
        self.assertEqual(set(fit) & set(threshold), set())
        self.assertEqual(set(branch) & set(threshold), set())
        self.assertEqual(set(fit + branch + threshold), set(samples))

    def test_strict_total_budget_is_respected(self):
        fit, branch, threshold = split_training(list(range(100)), 11, total_budget=16)
        self.assertEqual(len(fit) + len(branch) + len(threshold), 16)
        self.assertGreaterEqual(len(fit), 4)

    def test_threshold_prioritized_split_guarantees_19_when_feasible(self):
        fit, branch, threshold = split_training(
            list(range(100)), 11, total_budget=32, threshold_min_count=19
        )
        self.assertEqual((len(fit), len(branch), len(threshold)), (8, 5, 19))
        self.assertEqual(len(fit) + len(branch) + len(threshold), 32)
        self.assertEqual(set(fit) & set(branch), set())
        self.assertEqual(set(fit) & set(threshold), set())
        self.assertEqual(set(branch) & set(threshold), set())

    def test_threshold_priority_falls_back_when_budget_is_infeasible(self):
        fit, branch, threshold = split_training(
            list(range(100)), 11, total_budget=16, threshold_min_count=19
        )
        self.assertEqual(len(fit) + len(branch) + len(threshold), 16)
        self.assertLess(len(threshold), 19)
        self.assertGreaterEqual(len(fit), 4)

    def test_exact_sign_flip_retains_zero_pairs(self):
        self.assertAlmostEqual(
            exact_sign_flip_pvalue(np.asarray([1.0, 1.0, 1.0, 0.0])), 0.25
        )

    def test_conformal_quantile_rank(self):
        threshold, rank = conformal_upper_threshold(np.arange(19), alpha=0.10)
        self.assertEqual(rank, 18)
        self.assertEqual(threshold, 17.0)

    def test_conformal_alpha_005_small_sample_boundary(self):
        for count in (2, 5, 9, 10, 18):
            with self.subTest(count=count):
                threshold, rank = conformal_upper_threshold(np.arange(count), alpha=0.05)
                self.assertGreater(rank, count)
                self.assertTrue(np.isinf(threshold))
        threshold, rank = conformal_upper_threshold(np.arange(19), alpha=0.05)
        self.assertEqual(rank, 19)
        self.assertEqual(threshold, 18.0)

    def test_conformal_rejects_invalid_alpha(self):
        for alpha in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    conformal_upper_threshold(np.arange(20), alpha=alpha)

    def test_infinite_threshold_produces_declared_abstention_metrics(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        threshold, _ = conformal_upper_threshold(np.arange(9), alpha=0.05)
        result = classification_metrics(labels, scores > threshold)
        self.assertTrue(np.isinf(threshold))
        self.assertEqual(result["test_false_positive"], 0)
        self.assertEqual(result["test_true_positive"], 0)
        self.assertEqual(result["normal_fpr_conformal"], 0.0)
        self.assertEqual(result["image_precision_conformal"], 0.0)
        self.assertEqual(result["image_recall_conformal"], 0.0)

    def test_integer_translation_boundary_modes(self):
        tensor = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
        shifted = translate_tensor(tensor, (1, 0), border_mode="constant", fill=-1.0)
        self.assertTrue(torch.equal(shifted[0, :, 0], torch.full((3,), -1.0)))
        self.assertTrue(torch.equal(shifted[0, :, 1:], tensor[0, :, :-1]))
        reflected = translate_tensor(tensor, (-1, 0), border_mode="reflect")
        self.assertEqual(reflected.shape, tensor.shape)
        self.assertTrue(torch.equal(reflected[0, :, :2], tensor[0, :, 1:]))

    def test_frequency_branch_constructs_two_gaussian_residuals(self):
        extractor = FrozenFeatureExtractor.__new__(FrozenFeatureExtractor)
        extractor.device = torch.device("cpu")
        extractor.grid = 4
        images = torch.rand(1, 3, 16, 16)
        from torchvision.transforms import functional as transform_functional
        with patch(
            "freqpatch.TF.gaussian_blur", wraps=transform_functional.gaussian_blur
        ) as blur:
            output = extractor.frequency_features(images)
        self.assertEqual(blur.call_count, 2)
        self.assertEqual(tuple(output.shape), (1, 3, 4, 4))

    def test_valid_interior_uses_matching_cropped_calibration_statistic(self):
        cfg = {"image_size": 8, "score_quantile": 0.75, "threshold_alpha": 0.10}
        labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        masks = np.zeros((4, 8, 8), dtype=np.uint8)
        maps = np.zeros((4, 4, 4), dtype=np.float32)
        maps[2:, 1:3, 1:3] = 2.0
        threshold_maps = np.zeros((19, 4, 4), dtype=np.float32)
        # A large value exists only in the newly exposed left boundary.  The
        # valid-interior threshold must crop it just as the test statistic is cropped.
        threshold_maps[:, :, 0] = 100.0
        result = interior_metrics(
            labels, masks, maps, threshold_maps, cfg, dx=2, dy=0, extra_margin=0
        )
        self.assertEqual(result["valid_grid_fraction"], 0.75)
        self.assertEqual(result["valid_pixel_fraction"], result["valid_grid_fraction"])
        self.assertEqual(result["interior_normal_fpr"], 0.0)
        self.assertEqual(result["interior_recall"], 1.0)
        self.assertTrue(valid_mask(8, 2, 0)[:, :2].sum() == 0)

    def test_valid_interior_reports_actual_224_28_grid_geometry(self):
        cfg = {"image_size": 224, "score_quantile": 0.995, "threshold_alpha": 0.10}
        labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        masks = np.zeros((4, 224, 224), dtype=np.uint8)
        maps = np.zeros((4, 28, 28), dtype=np.float32)
        threshold_maps = np.zeros((19, 28, 28), dtype=np.float32)
        result = interior_metrics(
            labels, masks, maps, threshold_maps, cfg, dx=4, dy=4, extra_margin=0
        )
        self.assertEqual(result["interior_grid_valid_cells"], 27 * 27)
        self.assertEqual(result["interior_grid_total_cells"], 28 * 28)
        self.assertAlmostEqual(result["valid_grid_fraction"], 729 / 784)
        self.assertEqual(result["valid_pixel_fraction"], result["valid_grid_fraction"])
        self.assertEqual(result["interior_equivalent_pixel_x0"], 8.0)
        self.assertEqual(result["interior_equivalent_pixel_y0"], 8.0)

    def test_proposed_gate_is_bounded_and_backbone_preserving(self):
        deep = np.asarray([0.0, 0.8, 2.0, 4.0], dtype=np.float32)
        freq = np.asarray([8.0, 5.0, 3.0, 9.0], dtype=np.float32)
        fused = fuse_scores(deep, freq, 0.25)
        self.assertTrue(np.all(fused >= deep))
        self.assertTrue(np.all(fused <= deep * 1.25 + 1e-6))
        self.assertEqual(fused[0], 0.0)

    def test_fusion_controls_return_finite_maps(self):
        deep = np.asarray([0.2, 1.5, 3.0], dtype=np.float32)
        freq = np.asarray([4.0, 0.5, 2.0], dtype=np.float32)
        variants = (
            "proposed", "calibrated_weighted_sum", "calibrated_max",
            "calibrated_min", "calibrated_product", "unbounded_agreement",
            "no_upper_tail", "frequency_tail_gate",
        )
        for variant in variants:
            output = fuse_scores_variant(deep, freq, 0.25, variant)
            self.assertTrue(np.isfinite(output).all(), variant)


if __name__ == "__main__":
    unittest.main()
