from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.figures.ccp_lolp_target_scatter_plots import (
    AXES_ASPECT_RATIO,
    DEFAULT_CCP_TARGETS,
    DISPLAY_SPECS,
    FIT_CURVE_SAMPLES,
    FIGURE_SIZE_INCHES,
    PANEL_SPECS,
    QUADRATIC_FIT_METRICS,
    X_MAJOR_TICKS,
    build_metric_figure,
    quadratic_fit_summary,
)
from src.figures.core import matplotlib_context, save_publication_figure


def scatter_curve_fixture() -> pd.DataFrame:
    rows = []
    for threshold_index, threshold in enumerate((0.10, 0.20, 0.30), start=1):
        for target_index, target in enumerate(DEFAULT_CCP_TARGETS):
            actual_ccp = min(1.0, target + 0.01)
            rows.append(
                {
                    "lolp_constraint": threshold,
                    "ccp_target": target,
                    "ccp_target_percent": target * 100.0,
                    "comparison": "<",
                    "required_passing_scenarios": int(round(target * 100.0)),
                    "actual_ccp": actual_ccp,
                    "actual_ccp_percent": actual_ccp * 100.0,
                    "passing_scenarios": int(round(actual_ccp * 100.0)),
                    "feasible": True,
                    "optimal_pv_kw": 3.0 + threshold_index * 0.25 + (target_index % 4) * 0.25,
                    "optimal_battery_kwh": 10.0 + threshold_index + target_index * 0.2,
                    "mean_annual_cost_sgd_per_year": 2000.0 + threshold_index * 40.0 + target_index * 9.0,
                    "mean_annual_cost_ksgd_per_year": (2000.0 + threshold_index * 40.0 + target_index * 9.0) / 1000.0,
                }
            )
    return pd.DataFrame(rows)


class CCPLOLPScatterFigureTests(unittest.TestCase):
    def test_default_target_grid_contains_2pct_points_and_95pct_endpoint(self):
        self.assertEqual(len(DEFAULT_CCP_TARGETS), 24)
        self.assertEqual(DEFAULT_CCP_TARGETS[0], 0.50)
        self.assertEqual(DEFAULT_CCP_TARGETS[-1], 0.95)
        self.assertTrue(np.allclose(np.diff(DEFAULT_CCP_TARGETS[:-1]), 0.02))
        self.assertAlmostEqual(DEFAULT_CCP_TARGETS[-1] - DEFAULT_CCP_TARGETS[-2], 0.01)
        self.assertEqual(len(scatter_curve_fixture()), 72)

    def test_simplified_ticks_preserve_points_and_use_a_16_by_9_plotting_area(self):
        curves = scatter_curve_fixture()
        _, plt = matplotlib_context()
        figure_metadata = {}
        for panel in PANEL_SPECS:
            figure, metadata = build_metric_figure(curves, panel_key=str(panel["key"]))
            try:
                self.assertEqual(tuple(figure.get_size_inches()), FIGURE_SIZE_INCHES)
                self.assertEqual(len(figure.axes), 1)
                axis = figure.axes[0]
                self.assertEqual(len(axis.collections), 3)
                self.assertTrue(np.allclose(axis.get_xticks(), X_MAJOR_TICKS))
                self.assertGreater(axis.get_xlim()[1], DEFAULT_CCP_TARGETS[-1])
                self.assertEqual(axis.get_xticklabels()[-1].get_horizontalalignment(), "left")
                self.assertEqual(metadata["plot_type"], "scatter")
                self.assertFalse(metadata["source_points_connected"])
                self.assertFalse(metadata["smoothing"])
                self.assertFalse(metadata["interpolation"])
                self.assertFalse(metadata["extrapolation"])
                self.assertEqual(metadata["legend_labels"], ["LOLP < 10%", "LOLP < 20%", "LOLP < 30%"])
                display = DISPLAY_SPECS[panel["key"]]
                self.assertTrue(np.allclose(axis.get_yticks(), display["y_ticks"]))
                self.assertTrue(np.allclose(axis.get_ylim(), display["y_limits"]))
                self.assertEqual(metadata["panel"]["y_tick_step"], display["y_tick_step"])
                self.assertAlmostEqual(metadata["axes_aspect_ratio"], AXES_ASPECT_RATIO, places=10)
                self.assertEqual(metadata["panel"]["series"][0]["point_count"], 24)
                for collection, series in zip(axis.collections, metadata["panel"]["series"], strict=True):
                    offsets = collection.get_offsets()
                    self.assertEqual(len(offsets), 24)
                    self.assertTrue(np.allclose(offsets[:, 0], series["x_values"]))
                    self.assertTrue(np.allclose(offsets[:, 1], series["y_values"]))
                if panel["key"] == "pv":
                    self.assertEqual(len(axis.lines), 0)
                    self.assertFalse(metadata["quadratic_trend_lines"])
                    self.assertTrue(all(series["quadratic_fit"] is None for series in metadata["panel"]["series"]))
                else:
                    self.assertEqual(len(axis.lines), 3)
                    self.assertTrue(metadata["quadratic_trend_lines"])
                    for line, series in zip(axis.lines, metadata["panel"]["series"], strict=True):
                        fit = series["quadratic_fit"]
                        self.assertIsNotNone(fit)
                        self.assertEqual(len(line.get_xdata()), FIT_CURVE_SAMPLES)
                        self.assertAlmostEqual(line.get_xdata()[0], 0.50)
                        self.assertAlmostEqual(line.get_xdata()[-1], 0.95)
                        self.assertEqual(line.get_color(), series["color"])
                        expected = np.polyval([fit["a"], fit["b"], fit["c"]], line.get_xdata() * 100.0)
                        self.assertTrue(np.allclose(line.get_ydata(), expected))
                figure_metadata[panel["key"]] = metadata
            finally:
                plt.close(figure)

        fit_summary = quadratic_fit_summary(figure_metadata)
        self.assertEqual(len(fit_summary), 6)
        self.assertEqual(set(fit_summary.metric), QUADRATIC_FIT_METRICS)
        self.assertTrue((fit_summary.sample_count == 24).all())
        self.assertTrue(np.allclose(fit_summary.x_min_percent, 50.0))
        self.assertTrue(np.allclose(fit_summary.x_max_percent, 95.0))
        self.assertTrue(np.isfinite(fit_summary[["a", "b", "c", "r_squared"]].to_numpy(dtype=float)).all())

    def test_scatter_figures_export_all_publication_formats(self):
        curves = scatter_curve_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            _, plt = matplotlib_context()
            for panel in PANEL_SPECS:
                base = Path(temporary) / f"ccp2pct_scatter_{panel['key']}"
                figure, _ = build_metric_figure(curves, panel_key=str(panel["key"]), out=base)
                try:
                    paths = save_publication_figure(figure, base)
                finally:
                    plt.close(figure)
                self.assertEqual(set(paths), {
                    f"ccp2pct_scatter_{panel['key']}_png",
                    f"ccp2pct_scatter_{panel['key']}_svg",
                    f"ccp2pct_scatter_{panel['key']}_pdf",
                })
                self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
