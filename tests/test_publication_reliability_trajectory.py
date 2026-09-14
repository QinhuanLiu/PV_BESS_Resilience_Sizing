from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.figures.reliability_trajectory import (
    build_reliability_trajectory_source,
    make_reliability_trajectory_publication_figure,
)


def _candidates() -> pd.DataFrame:
    rows = []
    for pv_kw, bess_kwh, ccp, cost in [
        (1000.0, 1000.0, 0.50, 1_000_000.0),
        (1000.0, 2000.0, 0.70, 1_100_000.0),
        (2000.0, 2000.0, 0.85, 1_180_000.0),
        (2000.0, 3000.0, 0.90, 1_250_000.0),
        (3000.0, 4000.0, 0.95, 1_500_000.0),
    ]:
        rows.append(
            {
                "pv_capacity_kw": pv_kw,
                "battery_energy_kwh": bess_kwh,
                "system_total_cost_sgd_per_year": cost,
                "annualized_pv_cost_sgd_per_year": pv_kw * 80.0,
                "annualized_battery_cost_sgd_per_year": bess_kwh * 40.0,
                "battery_om_cost_sgd_per_year": bess_kwh * 2.0,
                "mean_grid_cost_sgd_per_year": cost * 0.55,
                "mean_unserved_energy_cost_sgd_per_year": (1.0 - ccp) * 100_000.0,
                "mean_UE_kwh_per_year": (1.0 - ccp) * 10_000.0,
                "mean_scenario_LOLP": 1.0 - ccp,
                "chance_constraint_probability": ccp,
            }
        )
    return pd.DataFrame(rows)


def _design_space() -> pd.DataFrame:
    rows = []
    for pv_mw in [1.0, 2.0, 3.0]:
        for bess_mwh in [1.0, 2.0, 3.0, 4.0]:
            rows.append(
                {
                    "pv_capacity_mw": pv_mw,
                    "battery_energy_mwh": bess_mwh,
                    "system_total_cost_million_sgd_per_year": 0.8 + pv_mw * 0.12 + bess_mwh * 0.08,
                    "mean_UE_mwh_per_year": max(0.0, 8.0 - pv_mw - bess_mwh),
                    "energy_based_LOLP": max(0.0, 0.6 - pv_mw * 0.08 - bess_mwh * 0.09),
                    "chance_constraint_probability": min(1.0, 0.35 + pv_mw * 0.08 + bess_mwh * 0.11),
                    "lolp_threshold": 0.1,
                }
            )
    return pd.DataFrame(rows)


class PublicationReliabilityTrajectoryTests(unittest.TestCase):
    def test_build_source_selects_minimum_cost_feasible_design_and_unavailable_target(self) -> None:
        source = build_reliability_trajectory_source(_candidates(), targets=(0.50, 0.80, 0.90, 0.99))
        self.assertEqual(source["available"].tolist(), [True, True, True, False])
        row_80 = source[source["target_CCP"] == 0.80].iloc[0]
        self.assertEqual(row_80["pv_capacity_mw"], 2.0)
        self.assertEqual(row_80["battery_energy_mwh"], 2.0)
        self.assertAlmostEqual(row_80["system_total_cost_million_sgd_per_year"], 1.18)
        self.assertAlmostEqual(row_80["mean_UE_mwh_per_year"], 1.5)
        self.assertAlmostEqual(row_80["energy_based_LOLP"], 0.15)
        row_99 = source[source["target_CCP"] == 0.99].iloc[0]
        self.assertFalse(bool(row_99["available"]))
        self.assertTrue(pd.isna(row_99["pv_capacity_mw"]))

    def test_knee_point_is_optional_when_fewer_than_three_available_targets(self) -> None:
        source = build_reliability_trajectory_source(_candidates(), targets=(0.90, 0.99))
        self.assertEqual(int(source["is_knee_point"].sum()), 0)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_publication_figure_exports_all_formats_and_source_data(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_reliability_trajectory_publication_figure(
                Path(tmp),
                "SG_UTown",
                _candidates(),
                _design_space(),
            )
            self.assertIn("source_data", paths)
            self.assertTrue(paths["source_data"].exists())
            self.assertIn("SG_UTown_Figure_4_Reliability_Driven_Sizing_Trajectory_png", paths)
            self.assertIn("SG_UTown_Figure_4_Reliability_Driven_Sizing_Trajectory_svg", paths)
            self.assertIn("SG_UTown_Figure_4_Reliability_Driven_Sizing_Trajectory_pdf", paths)
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            svg_text = paths["SG_UTown_Figure_4_Reliability_Driven_Sizing_Trajectory_svg"].read_text()
            self.assertIn("PV-BESS sizing trajectory", svg_text)


if __name__ == "__main__":
    unittest.main()
