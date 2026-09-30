from __future__ import annotations

import unittest

import pandas as pd

from src.tou_analysis import mean_preserving_tou_profile


class TouAnalysisTests(unittest.TestCase):
    def test_k_one_returns_original_profile(self) -> None:
        tou = pd.DataFrame({"hour": range(24), "tou_sgd_per_kwh": [0.2] * 8 + [0.3] * 8 + [0.5] * 8})

        adjusted = mean_preserving_tou_profile(tou, 1.0)

        self.assertEqual(adjusted["hour"].astype(int).tolist(), list(range(24)))
        self.assertEqual(adjusted["tou_sgd_per_kwh"].astype(float).tolist(), tou["tou_sgd_per_kwh"].tolist())

    def test_k_zero_flattens_to_mean(self) -> None:
        tou = pd.DataFrame({"hour": range(24), "tou_sgd_per_kwh": [0.2] * 8 + [0.3] * 8 + [0.5] * 8})
        expected_mean = float(tou["tou_sgd_per_kwh"].mean())

        adjusted = mean_preserving_tou_profile(tou, 0.0)

        self.assertTrue(all(abs(float(value) - expected_mean) < 1e-12 for value in adjusted["tou_sgd_per_kwh"]))

    def test_k_above_one_preserves_mean_and_expands_spread(self) -> None:
        tou = pd.DataFrame({"hour": range(24), "tou_sgd_per_kwh": [0.2] * 8 + [0.3] * 8 + [0.5] * 8})
        base_values = tou["tou_sgd_per_kwh"].astype(float)

        adjusted = mean_preserving_tou_profile(tou, 1.25)
        adjusted_values = adjusted["tou_sgd_per_kwh"].astype(float)

        self.assertAlmostEqual(float(adjusted_values.mean()), float(base_values.mean()), places=12)
        self.assertGreater(
            float(adjusted_values.max() - adjusted_values.min()),
            float(base_values.max() - base_values.min()),
        )

    def test_negative_adjusted_tariff_raises(self) -> None:
        tou = pd.DataFrame({"hour": range(24), "tou_sgd_per_kwh": [0.01] * 23 + [1.0]})

        with self.assertRaisesRegex(ValueError, "negative tariff"):
            mean_preserving_tou_profile(tou, 2.0)


if __name__ == "__main__":
    unittest.main()
