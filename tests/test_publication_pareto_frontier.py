from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.figures.pareto import (
    build_pareto_plot_source,
    compute_marginal_resilience_cost,
    identify_knee_point,
    identify_pareto_efficient,
    make_pareto_frontier_publication_figure,
    validate_pareto_source,
)
from src.figures.core import _matplotlib, _style


def _synthetic_pareto_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pv_capacity_kw": 1000.0,
                "battery_energy_kwh": 0.0,
                "system_total_cost_sgd_per_year": 1_000_000.0,
                "mean_UE_kwh_per_year": 5000.0,
                "mean_scenario_LOLP": 0.50,
                "chance_constraint_probability": 0.20,
                "lolp_threshold": 0.10,
            },
            {
                "pv_capacity_kw": 1500.0,
                "battery_energy_kwh": 1000.0,
                "system_total_cost_sgd_per_year": 1_120_000.0,
                "mean_UE_kwh_per_year": 2400.0,
                "mean_scenario_LOLP": 0.24,
                "chance_constraint_probability": 0.70,
                "lolp_threshold": 0.10,
            },
            {
                "pv_capacity_kw": 2000.0,
                "battery_energy_kwh": 2000.0,
                "system_total_cost_sgd_per_year": 1_250_000.0,
                "mean_UE_kwh_per_year": 1200.0,
                "mean_scenario_LOLP": 0.12,
                "chance_constraint_probability": 0.88,
                "lolp_threshold": 0.10,
            },
            {
                "pv_capacity_kw": 2600.0,
                "battery_energy_kwh": 3500.0,
                "system_total_cost_sgd_per_year": 1_480_000.0,
                "mean_UE_kwh_per_year": 200.0,
                "mean_scenario_LOLP": 0.02,
                "chance_constraint_probability": 0.96,
                "lolp_threshold": 0.10,
            },
            {
                "pv_capacity_kw": 1900.0,
                "battery_energy_kwh": 1000.0,
                "system_total_cost_sgd_per_year": 1_320_000.0,
                "mean_UE_kwh_per_year": 2600.0,
                "mean_scenario_LOLP": 0.26,
                "chance_constraint_probability": 0.65,
                "lolp_threshold": 0.10,
            },
        ]
    )


class PublicationParetoFrontierTests(unittest.TestCase):
    def test_validate_pareto_source_reports_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required column"):
            validate_pareto_source(pd.DataFrame({"pv_capacity_mw": [1.0]}))

    def test_identify_pareto_efficient_excludes_dominated_candidate(self) -> None:
        source = validate_pareto_source(_synthetic_pareto_candidates())
        efficient = identify_pareto_efficient(source)
        dominated = source["system_total_cost_million_sgd_per_year"].eq(1.32)
        self.assertFalse(bool(efficient[dominated].iloc[0]))
        self.assertEqual(int(efficient.sum()), 4)

    def test_build_source_marks_one_knee_when_three_or_more_pareto_points(self) -> None:
        source = build_pareto_plot_source(_synthetic_pareto_candidates())
        self.assertEqual(int(source["is_knee_point"].sum()), 1)
        self.assertIsNotNone(identify_knee_point(source))
        self.assertEqual(int(source["is_minimum_total_cost_design"].sum()), 1)
        self.assertEqual(int(source["is_resilience_optimum"].sum()), 1)

    def test_marginal_cost_fields_are_present_and_safe_for_flat_ue_steps(self) -> None:
        candidates = _synthetic_pareto_candidates()
        candidates.loc[1, "mean_UE_kwh_per_year"] = candidates.loc[0, "mean_UE_kwh_per_year"]
        source = compute_marginal_resilience_cost(build_pareto_plot_source(candidates))
        self.assertIn("marginal_cost_per_mwh_ue_reduction_million_sgd_per_mwh", source.columns)
        finite_or_nan = source["marginal_cost_per_mwh_ue_reduction_million_sgd_per_mwh"].dropna()
        self.assertTrue((finite_or_nan >= 0.0).all())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_make_pareto_frontier_figure_exports_all_formats_and_source(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_pareto_frontier_publication_figure(
                Path(tmp),
                "SG_UTown",
                _synthetic_pareto_candidates(),
            )
            self.assertIn("source_data", paths)
            self.assertIn("SG_UTown_Figure_5_Cost_Resilience_Pareto_Frontier_png", paths)
            self.assertIn("SG_UTown_Figure_5_Cost_Resilience_Pareto_Frontier_svg", paths)
            self.assertIn("SG_UTown_Figure_5_Cost_Resilience_Pareto_Frontier_pdf", paths)
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_svg_text_remains_editable(self) -> None:
        with TemporaryDirectory() as tmp:
            mpl, _ = _matplotlib(Path(tmp))
            _style(mpl)
            self.assertEqual(mpl.rcParams["svg.fonttype"], "none")


if __name__ == "__main__":
    unittest.main()
