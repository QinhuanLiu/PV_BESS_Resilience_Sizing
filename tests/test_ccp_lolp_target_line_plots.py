from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.figures.ccp_lolp_target_line_plots import (
    FIGURE_SIZE_INCHES,
    PANEL_SPECS,
    SMOOTH_SAMPLES,
    build_metric_figure,
    load_completed_fine_curves,
    select_target_designs,
)
from src.figures.capacity_sweep_contours import file_sha256
from src.figures.core import matplotlib_context, save_publication_figure


def selection_fixture() -> pd.DataFrame:
    """Four shared-scenario capacity points for strict-count and tie tests."""
    profiles = {
        (0.0, 10.0): (0.10, 0.20, 0.20, 0.40),
        (0.0, 20.0): (0.10, 0.10, 0.20, 0.50),
        (1.0, 10.0): (0.10, 0.10, 0.10, 0.50),
        (1.0, 20.0): (0.10, 0.10, 0.10, 0.10),
    }
    return pd.DataFrame(
        {
            "pv_capacity_kw": pv,
            "battery_energy_kwh": battery,
            "scenario_id": scenario_id,
            "LOLP": lolp,
            "annual_cost_sgd_per_year": 100.0,
        }
        for (pv, battery), lolps in profiles.items()
        for scenario_id, lolp in enumerate(lolps, start=1)
    )


def complete_curve_fixture() -> pd.DataFrame:
    rows = []
    for threshold in (0.10, 0.20, 0.30):
        for target in (0.50, 0.70, 0.90):
            rows.append(
                {
                    "lolp_constraint": threshold,
                    "ccp_target": target,
                    "ccp_target_percent": target * 100.0,
                    "comparison": "<",
                    "required_passing_scenarios": int(target * 100),
                    "actual_ccp": min(1.0, target + 0.03),
                    "actual_ccp_percent": min(1.0, target + 0.03) * 100.0,
                    "passing_scenarios": int(min(1.0, target + 0.03) * 100),
                    "feasible": True,
                    "optimal_pv_kw": 2.0 + threshold * 10.0 + target,
                    "optimal_battery_kwh": 10.0 + threshold * 10.0 + target * 5.0,
                    "mean_annual_cost_sgd_per_year": 2000.0 + threshold * 100.0 + target * 200.0,
                    "mean_annual_cost_ksgd_per_year": 2.0 + threshold * 0.1 + target * 0.2,
                }
            )
    return pd.DataFrame(rows)


class CCPLOLPSelectionTests(unittest.TestCase):
    def test_strict_lolp_boundary_excludes_equal_values_and_resolves_ties(self):
        curves = select_target_designs(
            selection_fixture(), lolp_thresholds=(0.20,), ccp_targets=(0.50, 0.75)
        )
        half, three_quarters = curves.itertuples(index=False)
        self.assertEqual(half.required_passing_scenarios, 2)
        self.assertEqual(half.passing_scenarios, 2)
        self.assertAlmostEqual(half.actual_ccp, 0.50)
        self.assertEqual((half.optimal_pv_kw, half.optimal_battery_kwh), (0.0, 20.0))
        self.assertEqual(three_quarters.required_passing_scenarios, 3)
        self.assertEqual(three_quarters.passing_scenarios, 3)
        self.assertEqual((three_quarters.optimal_pv_kw, three_quarters.optimal_battery_kwh), (1.0, 10.0))

    def test_default_target_matrix_is_complete_and_stably_sorted(self):
        records = pd.DataFrame(
            {
                "pv_capacity_kw": pv,
                "battery_energy_kwh": battery,
                "scenario_id": scenario_id,
                "LOLP": 0.05 if pv == 0.0 else 0.0,
                "annual_cost_sgd_per_year": 100.0 + pv + battery,
            }
            for pv, battery in ((0.0, 10.0), (1.0, 10.0))
            for scenario_id in range(1, 21)
        )
        curves = select_target_designs(records)
        self.assertEqual(len(curves), 30)
        self.assertTrue(curves.feasible.all())
        self.assertEqual(curves.groupby("lolp_constraint").size().tolist(), [10, 10, 10])
        self.assertTrue(curves.equals(curves.sort_values(["lolp_constraint", "ccp_target"], kind="stable").reset_index(drop=True)))
        self.assertTrue(np.allclose(curves.actual_ccp, 1.0))


class FineSizingInputTests(unittest.TestCase):
    def test_completed_checksum_validated_fine_curve_loads_with_display_columns(self):
        curves = complete_curve_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            curve_path = root / "ccp_target_curve_source.csv"
            curves.to_csv(curve_path, index=False)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "run_status.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
            (root / "validation.json").write_text(
                json.dumps({
                    "passed": True,
                    "file_sha256": {curve_path.name: file_sha256(curve_path)},
                    "fine_precision": {"pv_kw": 0.25, "battery_kwh": 0.25},
                }),
                encoding="utf-8",
            )
            loaded, _manifest, validation = load_completed_fine_curves(
                root,
                lolp_thresholds=(0.10, 0.20, 0.30),
                ccp_targets=(0.50, 0.70, 0.90),
            )
            self.assertEqual(len(loaded), 9)
            self.assertTrue(np.allclose(loaded.ccp_target_percent, loaded.ccp_target * 100.0))
            self.assertTrue(np.allclose(
                loaded.mean_annual_cost_ksgd_per_year,
                loaded.mean_annual_cost_sgd_per_year / 1000.0,
            ))
            self.assertEqual(validation["fine_precision"], {"pv_kw": 0.25, "battery_kwh": 0.25})
            subset, _manifest, _validation = load_completed_fine_curves(
                root,
                lolp_thresholds=(0.10, 0.20, 0.30),
                ccp_targets=(0.50, 0.90),
            )
            self.assertEqual(len(subset), 6)
            self.assertEqual(subset.ccp_target.unique().tolist(), [0.50, 0.90])


class CCPLOLPFigureTests(unittest.TestCase):
    def test_individual_figures_use_marker_free_pchip_lines_and_tight_scales(self):
        curves = complete_curve_fixture()
        _, plt = matplotlib_context()
        for panel in PANEL_SPECS:
            figure, metadata = build_metric_figure(curves, panel_key=str(panel["key"]))
            try:
                self.assertEqual(tuple(figure.get_size_inches()), FIGURE_SIZE_INCHES)
                self.assertEqual(len(figure.axes), 1)
                self.assertEqual(metadata["metric"], panel["key"])
                self.assertEqual(metadata["x_axis"], "CCP target (%)")
                self.assertEqual(metadata["smoothing"], {
                    "method": "shape-preserving PCHIP",
                    "samples_per_series": SMOOTH_SAMPLES,
                    "extrapolation": False,
                })
                self.assertTrue(metadata["interpolation"])
                self.assertFalse(metadata["show_data_points"])
                self.assertEqual(metadata["legend_labels"], ["LOLP < 10%", "LOLP < 20%", "LOLP < 30%"])
                axis = figure.axes[0]
                self.assertEqual(len(axis.lines), 3)
                for line in axis.lines:
                    self.assertIn(line.get_marker(), (None, "None", ""))
                    self.assertEqual(len(line.get_xdata()), SMOOTH_SAMPLES)
                    self.assertAlmostEqual(line.get_xdata()[0], 0.50)
                    self.assertAlmostEqual(line.get_xdata()[-1], 0.90)
                panel_metadata = metadata["panel"]
                all_values = np.concatenate([
                    np.asarray(series["y_values"], dtype=float) for series in panel_metadata["series"]
                ])
                self.assertLess(panel_metadata["y_limits"][0], float(all_values.min()))
                self.assertGreater(panel_metadata["y_limits"][1], float(all_values.max()))
                self.assertEqual(panel_metadata["y_tick_step"], panel["tick_step"])
            finally:
                plt.close(figure)

    def test_publication_exports_pass_layout_validation(self):
        curves = complete_curve_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            # Reuse the shared test cache; a temporary output-specific cache
            # would force Matplotlib to rebuild its font catalogue per test.
            _, plt = matplotlib_context()
            for panel in PANEL_SPECS:
                base = Path(temporary) / f"ccp_target_{panel['key']}"
                figure, _ = build_metric_figure(curves, panel_key=str(panel["key"]), out=base)
                try:
                    paths = save_publication_figure(figure, base)
                finally:
                    plt.close(figure)
                self.assertEqual(set(paths), {
                    f"ccp_target_{panel['key']}_png",
                    f"ccp_target_{panel['key']}_svg",
                    f"ccp_target_{panel['key']}_pdf",
                })
                self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
