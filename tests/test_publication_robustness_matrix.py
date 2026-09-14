from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.figures.robustness_matrix import (
    build_robustness_matrix_source,
    example_robustness_matrix_source,
    make_robustness_matrix_publication_figure,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pv_capacity_kw": 1000.0,
                "battery_energy_kwh": 1000.0,
                "system_total_cost_sgd_per_year": 1_000_000.0,
                "chance_constraint_probability": 0.50,
            },
            {
                "pv_capacity_kw": 1500.0,
                "battery_energy_kwh": 2000.0,
                "system_total_cost_sgd_per_year": 1_150_000.0,
                "chance_constraint_probability": 0.75,
            },
            {
                "pv_capacity_kw": 2200.0,
                "battery_energy_kwh": 3400.0,
                "system_total_cost_sgd_per_year": 1_420_000.0,
                "chance_constraint_probability": 0.92,
            },
            {
                "pv_capacity_kw": 3000.0,
                "battery_energy_kwh": 5000.0,
                "system_total_cost_sgd_per_year": 1_800_000.0,
                "chance_constraint_probability": 0.97,
            },
        ]
    )


def _scenario_results() -> pd.DataFrame:
    rows = []
    designs = [
        (1000.0, 1000.0, 1.00),
        (1500.0, 2000.0, 0.72),
        (2200.0, 3400.0, 0.40),
        (3000.0, 5000.0, 0.18),
    ]
    for scenario_id, load in enumerate([1000.0, 1200.0, 1500.0, 1800.0, 2200.0, 2600.0], start=1):
        for pv_kw, bess_kwh, risk_factor in designs:
            ue = max(0.0, load * risk_factor - bess_kwh * 0.14 - pv_kw * 0.035)
            lolp = ue / load
            rows.append(
                {
                    "frontier_scenario_id": scenario_id,
                    "pv_capacity_kw": pv_kw,
                    "battery_energy_kwh": bess_kwh,
                    "outage_weighted_load_kwh": load,
                    "UE_kwh_per_year": ue,
                    "scenario_LOLP": lolp,
                    "outage_load_served_ratio": 1.0 - lolp,
                }
            )
    return pd.DataFrame(rows)


class PublicationRobustnessMatrixTests(unittest.TestCase):
    def test_build_source_from_candidate_and_scenario_results(self) -> None:
        source = build_robustness_matrix_source(
            _candidates(),
            _scenario_results(),
            design_targets=(0.50, 0.70, 0.90),
            actual_quantiles=(0.50, 0.80, 0.95),
            lolp_threshold=0.10,
        )
        self.assertEqual(len(source), 9)
        self.assertEqual(source["design_target_CCP"].nunique(), 3)
        self.assertEqual(source["actual_severity_quantile"].nunique(), 3)
        p90 = source[source["design_target_CCP"] == 0.90].iloc[0]
        self.assertAlmostEqual(p90["pv_capacity_mw"], 2.2)
        self.assertAlmostEqual(p90["battery_energy_mwh"], 3.4)
        self.assertTrue(source["scenario_count"].min() >= 1)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_publication_figure_exports_all_formats_and_source_data(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_robustness_matrix_publication_figure(
                Path(tmp),
                "SG_UTown",
                example_robustness_matrix_source(),
            )
            expected = {
                "source_data",
                "SG_UTown_Figure_9_Design_Realization_Robustness_Matrix_png",
                "SG_UTown_Figure_9_Design_Realization_Robustness_Matrix_svg",
                "SG_UTown_Figure_9_Design_Realization_Robustness_Matrix_pdf",
            }
            self.assertTrue(expected.issubset(paths.keys()))
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            svg_text = paths["SG_UTown_Figure_9_Design_Realization_Robustness_Matrix_svg"].read_text()
            self.assertIn("Failure region map", svg_text)
            self.assertIn("Achieved CCP", svg_text)


if __name__ == "__main__":
    unittest.main()
