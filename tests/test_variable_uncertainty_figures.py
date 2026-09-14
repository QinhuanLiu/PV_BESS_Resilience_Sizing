from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.figures.variable_uncertainty import (
    VARIABLE_UNCERTAINTY_HISTOGRAM_SPECS,
    _cdf_smooth_points,
    _draw_histogram_cdf_axes,
    build_variable_uncertainty_histogram_source,
    make_variable_uncertainty_distribution_figures,
)


def _scenario_optima() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_id in range(1, 11):
        rows.append(
            {
                "uncertainty_scenario_id": scenario_id,
                "outage_seed": 1000 + scenario_id,
                "available": True,
                "pv_bess_system_annual_cost_sgd": 1000.0 + scenario_id * 10.0,
                "pv_capacity_kw": 3.0 + 0.1 * scenario_id,
                "battery_energy_kwh": 4.0 + 0.2 * scenario_id,
                "unserved_energy_kwh_per_year": float(scenario_id % 4),
                "scenario_LOLP": min(0.1 * scenario_id, 1.0),
            }
        )
    rows.append(
        {
            "uncertainty_scenario_id": 11,
            "outage_seed": 1011,
            "available": False,
            "pv_bess_system_annual_cost_sgd": np.nan,
            "pv_capacity_kw": np.nan,
            "battery_energy_kwh": np.nan,
            "unserved_energy_kwh_per_year": np.nan,
            "scenario_LOLP": np.nan,
        }
    )
    return pd.DataFrame(rows)


def _selected_designs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "design_type": "minimum_total_cost_design",
                "system_total_cost_sgd_per_year": 1050.0,
                "pv_capacity_kw": 3.5,
                "battery_energy_kwh": 5.0,
                "mean_UE_kwh_per_year": 1.5,
                "mean_scenario_LOLP": 0.45,
            }
        ]
    )


class VariableUncertaintyFigureTests(unittest.TestCase):
    def test_histogram_source_has_frequency_cdf_and_filters_unavailable(self) -> None:
        source, references = build_variable_uncertainty_histogram_source(
            _scenario_optima(),
            selected_designs=_selected_designs(),
            bins=5,
        )

        self.assertEqual(set(source["metric_key"]), {spec.metric_key for spec in VARIABLE_UNCERTAINTY_HISTOGRAM_SPECS})
        self.assertEqual(source["sample_size"].astype(int).unique().tolist(), [10])
        self.assertAlmostEqual(references["system_total_cost"], 1050.0)
        for _, group in source.groupby("metric_key", sort=False):
            self.assertAlmostEqual(float(group["cdf_start"].astype(float).iloc[0]), 0.0)
            self.assertTrue(group["cdf"].astype(float).is_monotonic_increasing)
            self.assertAlmostEqual(float(group["cdf"].astype(float).iloc[-1]), 1.0)
            self.assertAlmostEqual(float(group["frequency"].astype(float).sum()), 1.0)
            self.assertTrue(pd.to_numeric(group["bin_center"], errors="coerce").notna().all())

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is required for smooth CDF rendering")
    def test_smooth_cdf_points_are_monotonic_and_bounded(self) -> None:
        source, _ = build_variable_uncertainty_histogram_source(
            _scenario_optima(),
            selected_designs=_selected_designs(),
            bins=5,
        )

        for spec in VARIABLE_UNCERTAINTY_HISTOGRAM_SPECS:
            group = source[source["metric_key"] == spec.metric_key].reset_index(drop=True)
            x_smooth, y_smooth = _cdf_smooth_points(group)

            self.assertGreater(len(x_smooth), len(group))
            self.assertAlmostEqual(float(y_smooth[0]), 0.0)
            self.assertAlmostEqual(float(y_smooth[-1]), 1.0)
            self.assertTrue(np.all(np.diff(y_smooth) >= -1e-12))
            self.assertTrue(np.all((y_smooth >= -1e-12) & (y_smooth <= 1.0 + 1e-12)))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is required for smooth CDF rendering")
    def test_histogram_bars_use_full_bin_width_without_gaps(self) -> None:
        from src.figures.core import apply_nature_style, matplotlib_context

        source, _ = build_variable_uncertainty_histogram_source(
            _scenario_optima(),
            selected_designs=_selected_designs(),
            bins=5,
        )
        spec = VARIABLE_UNCERTAINTY_HISTOGRAM_SPECS[0]
        group = source[source["metric_key"] == spec.metric_key].reset_index(drop=True)
        mpl, plt = matplotlib_context()
        apply_nature_style(mpl)
        fig, ax = plt.subplots()
        cdf_ax = ax.twinx()
        try:
            _draw_histogram_cdf_axes(ax, cdf_ax, group, spec, show_ylabel=True, show_legend=False)
            patch_widths = [patch.get_width() for patch in ax.patches]
            expected_widths = group["bin_width"].astype(float).tolist()
            self.assertEqual(len(patch_widths), len(expected_widths))
            for actual, expected in zip(patch_widths, expected_widths, strict=True):
                self.assertAlmostEqual(float(actual), float(expected))
            cdf_lines = cdf_ax.get_lines()
            self.assertEqual(len(cdf_lines), 2)
            self.assertTrue(np.all(np.diff(cdf_lines[0].get_ydata()) >= -1e-12))
        finally:
            plt.close(fig)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_make_distribution_figures_exports_single_and_panel_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_variable_uncertainty_distribution_figures(
                Path(tmp),
                "Jakarta_SingleFamily",
                _scenario_optima(),
                selected_designs=_selected_designs(),
                bins=5,
            )

            for spec in VARIABLE_UNCERTAINTY_HISTOGRAM_SPECS:
                base = f"Jakarta_SingleFamily_Variable_Uncertainty_Distribution_{spec.figure_suffix}"
                self.assertIn(f"{base}_png", paths)
                self.assertIn(f"{base}_svg", paths)
                self.assertIn(f"{base}_pdf", paths)
                self.assertIn(f"variable_uncertainty_distribution_{spec.metric_key}_source_data", paths)

            panel_base = "Jakarta_SingleFamily_Variable_Uncertainty_Distribution_All_Metrics"
            self.assertIn(f"{panel_base}_png", paths)
            self.assertIn(f"{panel_base}_svg", paths)
            self.assertIn(f"{panel_base}_pdf", paths)
            self.assertIn("variable_uncertainty_distribution_all_metrics_source_data", paths)
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
