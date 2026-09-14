from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.figures.robust_uncertainty import (
    _figure_8b_cost_colorbar_ticks,
    build_figure_8_uncertainty_baseline_source,
    build_robust_sensitivity_band_source,
    build_robust_sensitivity_mechanism_source,
    build_tail_robustness_source,
    make_figure_8_uncertainty_baseline,
    make_figure_9_robust_sensitivity_bands,
    make_figure_10_robust_sensitivity_mechanism,
    make_figure_11_tail_robustness,
)


def _uncertainty_baseline_rows() -> pd.DataFrame:
    rows = []
    for scenario_id, ue in enumerate([0.0, 80.0, 160.0, 320.0, 640.0], start=1):
        rows.append(
            {
                "uncertainty_scenario_id": scenario_id,
                "outage_seed": 1000 + scenario_id,
                "pv_capacity_kw": 1800.0 + 40.0 * scenario_id,
                "battery_energy_kwh": 2400.0 + 120.0 * scenario_id,
                "pv_bess_system_annual_cost_sgd": 3_000_000.0 + 20_000.0 * scenario_id,
                "unserved_energy_kwh_per_year": ue,
                "unserved_energy_cost_sgd_per_year": ue * 50.0,
                "outage_weighted_load_kwh": 10_000.0,
            }
        )
    return pd.DataFrame(rows)


def _uncertainty_sensitivity_rows() -> pd.DataFrame:
    rows = []
    parameter_values = {
        "battery_capex_sgd_per_kwh": [350.0, 450.0, 550.0],
        "pv_capex_sgd_per_kw": [900.0, 1200.0, 1500.0],
        "saidi_min_per_year": [0.0, 60.0, 120.0],
        "saifi_per_year": [0.0, 1.0, 2.0],
        "voll_sgd_per_kwh": [10.0, 25.0, 40.0],
    }
    for p_index, (parameter, values) in enumerate(parameter_values.items(), start=1):
        for v_index, value in enumerate(values, start=1):
            for scenario_id in range(1, 6):
                stress = p_index * 0.2 + v_index * 0.4 + scenario_id * 0.1
                rows.append(
                    {
                        "sensitivity_parameter": parameter,
                        "sensitivity_value": value,
                        "uncertainty_scenario_id": scenario_id,
                        "pv_capacity_kw": 1200.0 + 120.0 * stress,
                        "battery_energy_kwh": 1800.0 + 260.0 * stress,
                        "pv_bess_system_annual_cost_sgd": 2_500_000.0 + 90_000.0 * stress,
                        "pv_bess_grid_cost_sgd": 700_000.0 - 10_000.0 * stress,
                        "annualized_pv_cost_sgd": 220_000.0 + 4_000.0 * stress,
                        "pv_om_cost_sgd": 20_000.0 + 400.0 * stress,
                        "annualized_battery_cost_sgd": 260_000.0 + 7_000.0 * stress,
                        "battery_om_cost_sgd": 18_000.0 + 700.0 * stress,
                        "unserved_energy_kwh_per_year": 100.0 * stress,
                        "unserved_energy_cost_sgd_per_year": 5_000.0 * stress,
                        "outage_weighted_load_kwh": 12_000.0,
                    }
                )
    return pd.DataFrame(rows)


class RobustUncertaintyFigureTests(unittest.TestCase):
    def test_figure_8_source_reuses_uncertainty_baseline_units(self) -> None:
        source = build_figure_8_uncertainty_baseline_source(_uncertainty_baseline_rows())

        self.assertIn("system_total_cost_million_sgd_per_year", source.columns)
        self.assertIn("scenario_LOLP", source.columns)
        self.assertAlmostEqual(source.loc[0, "pv_capacity_mw"], 1.84)
        self.assertAlmostEqual(source.loc[1, "UE_mwh_per_year"], 0.08)

    def test_figure_8b_cost_colorbar_ticks_bound_actual_cost_range(self) -> None:
        ticks = _figure_8b_cost_colorbar_ticks(pd.Series([8.548, 8.77, 10.33]).to_numpy())

        self.assertEqual(ticks.tolist(), [8.5, 9.0, 9.5, 10.0, 10.5])
        self.assertLessEqual(float(ticks[0]), 8.548)
        self.assertGreaterEqual(float(ticks[-1]), 10.33)

    def test_figure_9_source_summarizes_median_and_p10_p90(self) -> None:
        source = build_robust_sensitivity_band_source(_uncertainty_sensitivity_rows())
        row = source[
            (source["sensitivity_parameter"] == "battery_capex_sgd_per_kwh")
            & (source["sensitivity_value"] == 350.0)
            & (source["metric_key"] == "UE_mwh_per_year")
        ].iloc[0]

        self.assertEqual(row["scenario_count"], 5)
        self.assertLessEqual(row["p10"], row["median"])
        self.assertLessEqual(row["median"], row["p90"])
        self.assertEqual(
            source["sensitivity_parameter"].drop_duplicates().tolist(),
            [
                "battery_capex_sgd_per_kwh",
                "pv_capex_sgd_per_kw",
                "saidi_min_per_year",
                "saifi_per_year",
                "voll_sgd_per_kwh",
            ],
        )

    def test_figure_10_source_requires_cost_components(self) -> None:
        source = build_robust_sensitivity_mechanism_source(_uncertainty_sensitivity_rows())

        self.assertIn("pv_annual_cost_million_sgd_per_year_median", source.columns)
        self.assertIn("battery_annual_cost_million_sgd_per_year_median", source.columns)

        with self.assertRaisesRegex(ValueError, "cost component"):
            build_robust_sensitivity_mechanism_source(
                _uncertainty_sensitivity_rows().drop(columns=["pv_bess_grid_cost_sgd"])
            )

    def test_figure_11_source_contains_baseline_exceedance_and_tail_summary(self) -> None:
        source = build_tail_robustness_source(_uncertainty_baseline_rows(), _uncertainty_sensitivity_rows())

        self.assertIn("baseline_exceedance", source["source_type"].tolist())
        self.assertIn("tail_summary", source["source_type"].tolist())
        baseline = source[source["source_type"] == "baseline_exceedance"]
        self.assertTrue(baseline["exceedance_probability"].between(0.0, 1.0).all())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_robust_uncertainty_figures_export_all_formats_and_source_data(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = {}
            paths.update(make_figure_8_uncertainty_baseline(out, "SG_UTown", _uncertainty_baseline_rows()))
            paths.update(make_figure_9_robust_sensitivity_bands(out, "SG_UTown", _uncertainty_sensitivity_rows()))
            paths.update(make_figure_10_robust_sensitivity_mechanism(out, "SG_UTown", _uncertainty_sensitivity_rows()))
            tail_source = build_tail_robustness_source(_uncertainty_baseline_rows(), _uncertainty_sensitivity_rows())
            paths.update(make_figure_11_tail_robustness(out, "SG_UTown", tail_source))

            expected_stems = [
                "SG_UTown_Figure_8A_Uncertainty_Baseline_Distributions",
                "SG_UTown_Figure_8B_Uncertainty_Design_Cost_Bubble",
                "SG_UTown_Figure_9_Robust_Sensitivity_Bands",
                "SG_UTown_Figure_10A_Robust_Design_Trajectories",
                "SG_UTown_Figure_10B_Robust_Cost_Composition",
                "SG_UTown_Figure_11_Tail_Robustness_Resilience",
            ]
            for stem in expected_stems:
                for suffix in ("png", "svg", "pdf"):
                    self.assertTrue((out / f"{stem}.{suffix}").exists())
            for key, path in paths.items():
                self.assertTrue(path.exists(), key)
                self.assertGreater(path.stat().st_size, 0, key)


if __name__ == "__main__":
    unittest.main()
