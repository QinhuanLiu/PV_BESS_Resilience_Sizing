from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.figures.core import matplotlib_context
from src.figures.sensitivity_grid_contours import METRICS, load_completed_grids
from src.figures.sensitivity_grid_line_plots import LINE_SPECS, SMOOTH_SAMPLES, extract_line, metric_line_color
from src.figures.sensitivity_grid_split_line_plots import (
    AXES_ASPECT_RATIO,
    FIGURE_SIZE_INCHES,
    build_metric_figure,
    run,
)
from tests.test_sensitivity_grid_line_plots import _refresh_validation, _write_completed_grid


class SplitSensitivityLinePlotTests(unittest.TestCase):
    def test_complete_grid_exports_twenty_eight_real_data_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grid_dir = root / "grid"
            output_dir = root / "figures"
            _write_completed_grid(grid_dir)

            paths = run(grid_dir, output_dir)

            self.assertEqual(len(list(output_dir.rglob("*.png"))), 28)
            self.assertEqual(len(list(output_dir.rglob("*.svg"))), 28)
            self.assertEqual(len(list(output_dir.rglob("*.pdf"))), 28)
            self.assertEqual(len(list(output_dir.rglob("*_Source_Data.csv"))), 4)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths.values()))

            report = json.loads((output_dir / "sensitivity_split_line_validation.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["figure_count"], 28)
            self.assertIn("no proxy", report["data_policy"])
            self.assertEqual(set(report["figures"]), {str(spec["key"]) for spec in LINE_SPECS})
            self.assertEqual(report["metrics"], [str(metric["key"]) for metric in METRICS])
            self.assertIn("没有引入虚构", (output_dir / "README.md").read_text())

            expected_x_values = {
                "pv_capex": [600.0, 1200.0, 1800.0],
                "battery_capex": [200.0, 400.0, 600.0],
                "saidi": [0.0, 90.0, 180.0],
                "voll": [0.0, 30.0, 50.0],
            }
            for spec in LINE_SPECS:
                parameter = report["figures"][str(spec["key"])]
                self.assertEqual(parameter["sample_count"], len(expected_x_values[str(spec["key"])]))
                self.assertEqual(set(parameter["metrics"]), {str(metric["key"]) for metric in METRICS})
                for metric in METRICS:
                    figure = parameter["metrics"][str(metric["key"])] ["figure"]
                    self.assertEqual(figure["layout"], "single-panel 16:9 sensitivity line plot")
                    self.assertEqual(figure["figure_size_inches"], list(FIGURE_SIZE_INCHES))
                    self.assertEqual(figure["x_values"], expected_x_values[str(spec["key"])])
                    self.assertAlmostEqual(figure["axes_aspect_ratio"], AXES_ASPECT_RATIO, places=10)
                    self.assertEqual(figure["y_limits"][0], 0.0)
                    self.assertEqual(figure["rendered_sample_count"], SMOOTH_SAMPLES)
                    self.assertIn(figure["marker"], {"None", "", "NoneType"})
                    if metric["key"] in {"lolp", "ssr", "scr"}:
                        self.assertLessEqual(figure["y_limits"][1], 1.0)
                        self.assertTrue(any("%" in label for label in figure["y_tick_labels"]))

    def test_single_metric_figures_keep_style_and_true_axis_aspect(self):
        with tempfile.TemporaryDirectory() as temporary:
            grid_dir = Path(temporary) / "grid"
            _write_completed_grid(grid_dir)
            frames, provenance = load_completed_grids(grid_dir)
            mpl, plt = matplotlib_context()
            try:
                for spec in LINE_SPECS:
                    line = extract_line(
                        frames[str(spec["source_grid"])],
                        spec,
                        manifest=provenance["manifest"],
                        case_key="Brisbane_SingleFamily",
                    )
                    for metric in METRICS:
                        figure, metadata = build_metric_figure(line, spec, metric)
                        try:
                            self.assertEqual(len(figure.axes), 1)
                            self.assertEqual(len(figure.axes[0].lines), 1)
                            rendered = figure.axes[0].lines[0]
                            self.assertEqual(len(rendered.get_xdata()), SMOOTH_SAMPLES)
                            self.assertIn(rendered.get_marker(), {None, "None", ""})
                            self.assertEqual(metadata["color"], metric_line_color(mpl, metric))
                            self.assertAlmostEqual(metadata["axes_aspect_ratio"], AXES_ASPECT_RATIO, places=10)
                            self.assertEqual(metadata["x_axis"], {
                                "pv_capex": "PV CAPEX (SGD kW$^{-1}$)",
                                "battery_capex": "Battery energy CAPEX (SGD kWh$^{-1}$)",
                                "saidi": "SAIDI (min yr$^{-1}$)",
                                "voll": "VoLL (SGD kWh$^{-1}$)",
                            }[str(spec["key"])])
                        finally:
                            plt.close(figure)
            finally:
                plt.close("all")

    def test_rejects_invalid_input_and_preserves_boundary_only_scr_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grid_dir = root / "grid"
            _write_completed_grid(grid_dir)
            (grid_dir / "run_status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed"):
                run(grid_dir, root / "unfinished")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grid_dir = root / "grid"
            _write_completed_grid(grid_dir)
            summary = grid_dir / "SAIDI_VoLL_Grid_Summary.csv"
            frame = pd.read_csv(summary)
            frame.loc[0, "annual_cost_sgd_per_year"] = 9999.0
            frame.to_csv(summary, index=False)
            with self.assertRaisesRegex(ValueError, "checksum"):
                run(grid_dir, root / "checksum")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grid_dir = root / "grid"
            _write_completed_grid(grid_dir)
            summary = grid_dir / "Cost_CAPEX_Grid_Summary.csv"
            frame = pd.read_csv(summary)
            boundary = np.isclose(frame["column_value"], 400.0) & np.isclose(frame["row_value"], 600.0)
            frame.loc[boundary, "SCR"] = np.nan
            frame.loc[boundary, "SCR_defined"] = False
            frame.loc[boundary, "SCR_status"] = "not_applicable_no_pv"
            frame.to_csv(summary, index=False)
            _refresh_validation(grid_dir)
            frames, provenance = load_completed_grids(grid_dir)
            line = extract_line(
                frames["cost_capex"], LINE_SPECS[0], manifest=provenance["manifest"], case_key="Brisbane_SingleFamily"
            )
            _, plt = matplotlib_context()
            figure, metadata = build_metric_figure(line, LINE_SPECS[0], METRICS[-1])
            try:
                self.assertEqual(metadata["undefined_value_count"], 1)
                self.assertIn("No zero-fill or extrapolation", metadata["note"])
                self.assertEqual(metadata["curve_method"], "linear two-point fallback")
            finally:
                plt.close(figure)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grid_dir = root / "grid"
            _write_completed_grid(grid_dir)
            summary = grid_dir / "Cost_CAPEX_Grid_Summary.csv"
            frame = pd.read_csv(summary)
            gap = np.isclose(frame["column_value"], 400.0) & np.isclose(frame["row_value"], 1200.0)
            frame.loc[gap, "SCR"] = np.nan
            frame.loc[gap, "SCR_defined"] = False
            frame.loc[gap, "SCR_status"] = "not_applicable_no_pv"
            frame.to_csv(summary, index=False)
            _refresh_validation(grid_dir)
            with self.assertRaisesRegex(ValueError, "contiguous outer"):
                run(grid_dir, root / "internal_scr_gap")


if __name__ == "__main__":
    unittest.main()
