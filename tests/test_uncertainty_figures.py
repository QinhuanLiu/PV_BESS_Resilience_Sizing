from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.figures.uncertainty import (
    build_uncertainty_baseline_figure_source,
    make_uncertainty_baseline_figures,
    validate_uncertainty_baseline_figure_source,
)


def _uncertainty_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_id, (pv, battery, cost, ue, load) in enumerate(
        [
            (22000.0, 5000.0, 9_000_000.0, 0.0, 10_000.0),
            (22600.0, 7000.0, 9_001_000.0, 100.0, 10_000.0),
            (23000.0, 9000.0, 9_002_000.0, 250.0, 10_000.0),
            (24000.0, 12000.0, 9_003_000.0, 0.0, 0.0),
        ],
        start=1,
    ):
        rows.append(
            {
                "uncertainty_scenario_id": scenario_id,
                "outage_seed": 1000 + scenario_id,
                "pv_capacity_kw": pv,
                "battery_energy_kwh": battery,
                "pv_bess_system_annual_cost_sgd": cost,
                "unserved_energy_kwh_per_year": ue,
                "outage_weighted_load_kwh": load,
            }
        )
    return pd.DataFrame(rows)


class UncertaintyFigureTests(unittest.TestCase):
    def test_build_source_derives_units_and_lolp_without_design_case(self) -> None:
        source = build_uncertainty_baseline_figure_source(_uncertainty_rows())

        self.assertEqual(len(source), 4)
        self.assertNotIn("design_case", source.columns)
        self.assertAlmostEqual(source.loc[0, "pv_capacity_mw"], 22.0)
        self.assertAlmostEqual(source.loc[1, "battery_energy_mwh"], 7.0)
        self.assertAlmostEqual(source.loc[1, "system_total_cost_million_sgd_per_year"], 9.001)
        self.assertAlmostEqual(source.loc[2, "UE_mwh_per_year"], 0.25)
        self.assertAlmostEqual(source.loc[2, "scenario_LOLP"], 0.025)
        self.assertAlmostEqual(source.loc[3, "scenario_LOLP"], 0.0)

    def test_build_source_rejects_bad_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required column"):
            build_uncertainty_baseline_figure_source(_uncertainty_rows().drop(columns=["pv_capacity_kw"]))

        bad = _uncertainty_rows()
        bad["pv_capacity_kw"] = bad["pv_capacity_kw"].astype(object)
        bad.loc[0, "pv_capacity_kw"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "non-numeric required values"):
            build_uncertainty_baseline_figure_source(bad)

    def test_bubble_size_scaling_expands_small_cost_differences(self) -> None:
        source = build_uncertainty_baseline_figure_source(_uncertainty_rows())

        self.assertGreater(float(source["bubble_size_pt2"].max() - source["bubble_size_pt2"].min()), 100.0)
        self.assertTrue(source["bubble_cost_normalized"].between(0.0, 1.0).all())
        validate_uncertainty_baseline_figure_source(source)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_make_uncertainty_baseline_figures_exports_all_formats_and_source_data(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_uncertainty_baseline_figures(Path(tmp), "SG_UTown", _uncertainty_rows())

            expected = {
                "uncertainty_baseline_distribution_source_data",
                "uncertainty_baseline_capacity_bubble_source_data",
                "SG_UTown_Uncertainty_Baseline_Distribution_Violins_png",
                "SG_UTown_Uncertainty_Baseline_Distribution_Violins_svg",
                "SG_UTown_Uncertainty_Baseline_Distribution_Violins_pdf",
                "SG_UTown_Uncertainty_Baseline_Capacity_Cost_Bubble_png",
                "SG_UTown_Uncertainty_Baseline_Capacity_Cost_Bubble_svg",
                "SG_UTown_Uncertainty_Baseline_Capacity_Cost_Bubble_pdf",
            }
            self.assertEqual(set(paths), expected)
            for path in paths.values():
                self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            svg_text = paths["SG_UTown_Uncertainty_Baseline_Capacity_Cost_Bubble_svg"].read_text()
            self.assertIn("Scenario-level optimal capacity and cost", svg_text)
            self.assertIn("Total cost", svg_text)
            self.assertNotIn("Total cost (million SGD/year)", svg_text)


if __name__ == "__main__":
    unittest.main()
