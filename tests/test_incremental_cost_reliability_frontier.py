from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.figures.core import matplotlib_context, save_publication_figure
from src.figures.incremental_cost_reliability_frontier import (
    AXES_ASPECT_RATIO,
    FIGURE_SIZE_INCHES,
    POWER_LAW_FIT_SAMPLES,
    X_LIMITS,
    X_TICKS,
    Y_LIMITS,
    Y_TICKS,
    build_frontier_figure,
    build_power_law_fit_figure,
    fit_power_law_through_origin,
    power_law_fit_summary,
    select_incremental_frontier,
)


def candidate_fixture() -> pd.DataFrame:
    rows = [
        # Baseline: global minimum cost and 24 strict-LOLP passes.
        (3.25, 11.50, 2002.609233, 24, "baseline"),
        # A dominated 26% design; the 27% design costs less and must replace it.
        (3.25, 11.75, 2003.600000, 26, "adaptive"),
        (3.50, 11.50, 2002.844315, 27, "adaptive"),
        (3.50, 11.75, 2004.130228, 29, "adaptive"),
        (3.25, 12.00, 2006.216072, 32, "adaptive"),
        # This point is dominated by the preceding 32% point.
        (3.75, 12.00, 2008.000000, 31, "adaptive"),
        (4.00, 15.00, 2177.996109, 95, "adaptive"),
        # Outside the requested absolute CCP window.
        (4.25, 15.25, 2200.000000, 96, "adaptive"),
    ]
    records = []
    for pv, bess, cost, passing, origin in rows:
        records.append(
            {
                "pv_capacity_kw": pv,
                "battery_energy_kwh": bess,
                "battery_power_kw": bess / 2.0,
                "scenario_count": 100,
                "mean_annual_cost_sgd_per_year": cost,
                "candidate_origin": origin,
                "LOLP_lt_20pct_passing_scenarios": passing,
                "CCP_LOLP_lt_20pct": passing / 100.0,
            }
        )
    return pd.DataFrame(records)


class IncrementalCostReliabilityFrontierTests(unittest.TestCase):
    def test_select_frontier_uses_integer_strict_ccp_and_removes_dominated_designs(self):
        candidates = candidate_fixture()
        source, summary = select_incremental_frontier(candidates)
        self.assertEqual(summary["reference_annual_cost_sgd_per_year"], 2002.609233)
        self.assertEqual(summary["reference_passing_scenarios"], 24)
        self.assertEqual(summary["reference_ccp_percent"], 24.0)
        self.assertEqual(len(source), 5)
        self.assertEqual(source.point_type.tolist(), ["baseline", "frontier", "frontier", "frontier", "frontier"])
        self.assertEqual(source.passing_scenarios.tolist(), [24, 27, 29, 32, 95])
        self.assertTrue(np.allclose(source.actual_ccp_percent, source.passing_scenarios))
        self.assertAlmostEqual(source.iloc[0].incremental_annual_cost_sgd_per_year, 0.0)
        self.assertAlmostEqual(source.iloc[0].incremental_ccp_percentage_points, 0.0)
        self.assertTrue(source.mean_annual_cost_sgd_per_year.iloc[1:].is_monotonic_increasing)
        self.assertTrue(source.passing_scenarios.iloc[1:].is_monotonic_increasing)
        self.assertNotIn(26, source.passing_scenarios.tolist())
        self.assertNotIn(31, source.passing_scenarios.tolist())
        self.assertNotIn(96, source.passing_scenarios.tolist())

    def test_frontier_figure_is_scatter_only_with_fixed_16_by_9_axes(self):
        source, _ = select_incremental_frontier(candidate_fixture())
        figure, metadata = build_frontier_figure(source)
        try:
            self.assertEqual(tuple(figure.get_size_inches()), FIGURE_SIZE_INCHES)
            self.assertEqual(len(figure.axes), 1)
            axis = figure.axes[0]
            self.assertEqual(len(axis.lines), 0)
            self.assertEqual(len(axis.collections), 2)
            self.assertTrue(np.allclose(axis.get_xlim(), X_LIMITS))
            self.assertTrue(np.allclose(axis.get_ylim(), Y_LIMITS))
            self.assertTrue(np.allclose(axis.get_xticks(), X_TICKS))
            self.assertTrue(np.allclose(axis.get_yticks(), Y_TICKS))
            self.assertAlmostEqual(metadata["axes_aspect_ratio"], AXES_ASPECT_RATIO, places=10)
            self.assertEqual(metadata["plot_type"], "scatter")
            self.assertFalse(metadata["frontier_points_connected"])
            self.assertFalse(metadata["smoothing"])
            self.assertFalse(metadata["interpolation"])
            self.assertFalse(metadata["trend_fit"])
            self.assertEqual(metadata["frontier_point_count_excluding_baseline"], 4)
        finally:
            _, plt = matplotlib_context()
            plt.close(figure)

    def test_power_law_fit_is_positive_monotonic_and_passes_through_origin(self):
        source, _ = select_incremental_frontier(candidate_fixture())
        fit = fit_power_law_through_origin(source)
        self.assertEqual(fit["model"], "power_law_through_origin")
        self.assertTrue(fit["passes_through_origin"])
        self.assertTrue(fit["monotonic_over_observed_range"])
        self.assertFalse(fit["extrapolation"])
        self.assertGreater(fit["coefficient_a"], 0.0)
        self.assertGreater(fit["exponent_b"], 0.0)
        self.assertEqual(fit["sample_count_including_baseline"], len(source))
        self.assertEqual(fit["sample_count_excluding_baseline"], len(source) - 1)
        self.assertEqual(len(fit["x_curve"]), POWER_LAW_FIT_SAMPLES)
        self.assertEqual(len(fit["y_curve"]), POWER_LAW_FIT_SAMPLES)
        self.assertAlmostEqual(fit["x_curve"][0], 0.0)
        self.assertAlmostEqual(fit["y_curve"][0], 0.0)
        self.assertAlmostEqual(fit["x_curve"][-1], source.incremental_annual_cost_sgd_per_year.max())
        self.assertTrue(np.all(np.diff(fit["y_curve"]) >= 0.0))
        summary = power_law_fit_summary(fit)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(float(summary.loc[0, "coefficient_a"]), fit["coefficient_a"])
        self.assertAlmostEqual(float(summary.loc[0, "exponent_b"]), fit["exponent_b"])
        self.assertTrue(np.isfinite(summary[["r_squared", "rmse_percentage_points"]].to_numpy(dtype=float)).all())

    def test_power_law_figure_adds_one_fit_line_and_preserves_scatter_geometry(self):
        source, _ = select_incremental_frontier(candidate_fixture())
        figure, metadata = build_power_law_fit_figure(source)
        try:
            axis = figure.axes[0]
            self.assertEqual(len(axis.collections), 2)
            self.assertEqual(len(axis.lines), 1)
            self.assertTrue(np.allclose(axis.get_xlim(), X_LIMITS))
            self.assertTrue(np.allclose(axis.get_ylim(), Y_LIMITS))
            self.assertTrue(np.allclose(axis.get_xticks(), X_TICKS))
            self.assertTrue(np.allclose(axis.get_yticks(), Y_TICKS))
            self.assertAlmostEqual(metadata["axes_aspect_ratio"], AXES_ASPECT_RATIO, places=10)
            self.assertTrue(metadata["trend_fit"])
            self.assertEqual(metadata["fit_type"], "power_law_through_origin")
            self.assertFalse(metadata["frontier_points_connected"])
            self.assertFalse(metadata["smoothing"])
            self.assertFalse(metadata["interpolation"])
            fit = metadata["power_law_fit"]
            self.assertEqual(len(axis.lines[0].get_xdata()), POWER_LAW_FIT_SAMPLES)
            self.assertTrue(np.allclose(axis.lines[0].get_xdata(), fit["x_curve"]))
            self.assertTrue(np.allclose(axis.lines[0].get_ydata(), fit["y_curve"]))
            self.assertIn("power-law fit", metadata["title"])
        finally:
            _, plt = matplotlib_context()
            plt.close(figure)

    def test_frontier_exports_png_svg_and_pdf(self):
        source, _ = select_incremental_frontier(candidate_fixture())
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "incremental_cost_reliability_frontier"
            figure, _ = build_frontier_figure(source, out=base)
            _, plt = matplotlib_context(base)
            try:
                paths = save_publication_figure(figure, base)
            finally:
                plt.close(figure)
            self.assertEqual(set(paths), {
                "incremental_cost_reliability_frontier_png",
                "incremental_cost_reliability_frontier_svg",
                "incremental_cost_reliability_frontier_pdf",
            })
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths.values()))

    def test_power_law_figure_exports_png_svg_and_pdf(self):
        source, _ = select_incremental_frontier(candidate_fixture())
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "incremental_cost_reliability_frontier_power_law_fit"
            figure, _ = build_power_law_fit_figure(source, out=base)
            _, plt = matplotlib_context(base)
            try:
                paths = save_publication_figure(figure, base)
            finally:
                plt.close(figure)
            self.assertEqual(set(paths), {
                "incremental_cost_reliability_frontier_power_law_fit_png",
                "incremental_cost_reliability_frontier_power_law_fit_svg",
                "incremental_cost_reliability_frontier_power_law_fit_pdf",
            })
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
