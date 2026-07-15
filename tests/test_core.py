import unittest

import numpy as np

from freqpatch import conformal_upper_threshold, fuse_scores, fuse_scores_variant
from run_experiments import split_training


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
