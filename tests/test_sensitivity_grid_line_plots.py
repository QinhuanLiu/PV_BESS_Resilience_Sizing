from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.figures.core import matplotlib_context
from src.figures.sensitivity_grid_contours import METRICS, load_completed_grids
from src.figures.sensitivity_grid_line_plots import (
    LINE_SPECS,
    PARAMETER_LABELS,
    SMOOTH_SAMPLES,
    build_figure,
    extract_line,
    metric_line_color,
    run,
)


class SensitivityGridLinePlotTests(unittest.TestCase):
    def test_complete_grid_exports_four_seven_panel_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            grid_dir = Path(temporary) / "grid"
            output_dir = Path(temporary) / "figures"
            _write_completed_grid(grid_dir)

            paths = run(grid_dir, output_dir)
            self.assertEqual(len(list(output_dir.rglob("*.png"))), 4)
            self.assertEqual(len(list(output_dir.rglob("*.svg"))), 4)
            self.assertEqual(len(list(output_dir.rglob("*.pdf"))), 4)
            self.assertEqual(len(list(output_dir.rglob("*_Source_Data.csv"))), 4)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths.values()))

            report = json.loads((output_dir / "sensitivity_line_validation.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(set(report["figures"]), {str(spec["key"]) for spec in LINE_SPECS})
            self.assertEqual(report["figures"]["saidi"]["fixed_conditions"]["voll_sgd_per_kwh"], 30.0)
            self.assertIn("VoLL=30", (output_dir / "README.md").read_text())
            for spec in LINE_SPECS:
                key = str(spec["key"])
                figure = report["figures"][key]["figure"]
                self.assertEqual(figure["panel_count"], 7)
                self.assertEqual(figure["x_axis"], PARAMETER_LABELS[str(spec["x_parameter"])])
                self.assertIn("PCHIP", figure["smoothing"])
                self.assertTrue(figure["interpolation"])
                self.assertFalse(figure["show_data_points"])
                self.assertEqual(figure["layout"], "seven-row single-column shared x-axis")
                self.assertEqual(set(figure["metrics"]), {str(metric["key"]) for metric in METRICS})
                for metric in METRICS:
                    panel = figure["metrics"][str(metric["key"])]
                    values = np.asarray([value for value in panel["y_values"] if value is not None], dtype=float)
                    self.assertEqual(panel["y_limits"][0], 0.0)
                    self.assertGreaterEqual(panel["y_limits"][1], values.max())
                    self.assertEqual(panel["curve_method"], "PCHIP")
                    self.assertEqual(panel["rendered_sample_count"], SMOOTH_SAMPLES)
                    self.assertIn(panel["marker"], {"None", "", "NoneType"})
                for probability_key in ("lolp", "ssr", "scr"):
                    self.assertLessEqual(figure["metrics"][probability_key]["y_limits"][1], 1.0)
                    labels = figure["metrics"][probability_key]["y_tick_labels"]
                    self.assertTrue(any("%" in label for label in labels))

    def test_cross_sections_use_exact_samples_and_metric_colours(self):
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
                    expected = {
                        "pv_capex": [600.0, 1200.0, 1800.0],
                        "battery_capex": [200.0, 400.0, 600.0],
                        "saidi": [0.0, 90.0, 180.0],
                        "voll": [0.0, 30.0, 50.0],
                    }[str(spec["key"])]
                    self.assertEqual(line["sweep_value"].tolist(), expected)
                    for parameter, fixed_value in dict(spec["fixed_conditions"]).items():
                        self.assertTrue(np.isclose(line[parameter].to_numpy(dtype=float), fixed_value).all())

                    figure, metadata = build_figure(line, spec)
                    self.assertEqual(metadata["panel_count"], 7)
                    self.assertEqual(len(figure.axes), 7)
                    self.assertTrue(
                        all(np.allclose(axis.get_xlim(), metadata["shared_x_limits"]) for axis in figure.axes)
                    )
                    for axis in figure.axes:
                        self.assertEqual(len(axis.lines), 1)
                        self.assertEqual(len(axis.lines[0].get_xdata()), SMOOTH_SAMPLES)
                        self.assertIn(axis.lines[0].get_marker(), {None, "None", ""})
                    for metric in METRICS:
                        self.assertEqual(
                            metadata["metrics"][str(metric["key"])]["color"],
                            metric_line_color(mpl, metric),
                        )
                    plt.close(figure)
            finally:
                plt.close("all")

    def test_rejects_missing_slice_duplicate_x_nonfinite_and_internal_scr_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            grid_dir = Path(temporary) / "grid"
            _write_completed_grid(grid_dir)

            path = grid_dir / "Cost_CAPEX_Grid_Summary.csv"
            frame = pd.read_csv(path)
            frame.loc[np.isclose(frame["column_value"], 400.0), "column_value"] = 425.0
            frame.to_csv(path, index=False)
            _refresh_validation(grid_dir)
            with self.assertRaisesRegex(ValueError, "Missing exact sensitivity cross-section"):
                run(grid_dir, Path(temporary) / "missing-slice")

        with tempfile.TemporaryDirectory() as temporary:
            grid_dir = Path(temporary) / "grid"
            _write_completed_grid(grid_dir)
            frames, provenance = load_completed_grids(grid_dir)
            spec = LINE_SPECS[0]
            source = frames[str(spec["source_grid"])]
            duplicate_row = source.loc[np.isclose(source["column_value"], 400.0)].iloc[[0]]
            duplicate = pd.concat([source, duplicate_row], ignore_index=True)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                extract_line(duplicate, spec, manifest=provenance["manifest"], case_key="Brisbane_SingleFamily")

        for mutation, message in (("nonfinite", "non-finite"), ("internal_scr_gap", "contiguous outer")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                grid_dir = Path(temporary) / "grid"
                _write_completed_grid(grid_dir)
                path = grid_dir / "Cost_CAPEX_Grid_Summary.csv"
                frame = pd.read_csv(path)
                if mutation == "nonfinite":
                    frame.loc[0, "EUE_kwh_per_year"] = np.nan
                else:
                    gap = np.isclose(frame["column_value"], 400.0) & np.isclose(frame["row_value"], 1200.0)
                    frame.loc[gap, "SCR"] = np.nan
                    frame.loc[gap, "SCR_defined"] = False
                    frame.loc[gap, "SCR_status"] = "not_applicable_no_pv"
                frame.to_csv(path, index=False)
                _refresh_validation(grid_dir)
                with self.assertRaisesRegex(ValueError, message):
                    run(grid_dir, Path(temporary) / mutation)

    def test_boundary_only_undefined_scr_is_preserved_without_fill(self):
        with tempfile.TemporaryDirectory() as temporary:
            grid_dir = Path(temporary) / "grid"
            output_dir = Path(temporary) / "figures"
            _write_completed_grid(grid_dir)
            path = grid_dir / "Cost_CAPEX_Grid_Summary.csv"
            frame = pd.read_csv(path)
            boundary = np.isclose(frame["column_value"], 400.0) & np.isclose(frame["row_value"], 600.0)
            frame.loc[boundary, "SCR"] = np.nan
            frame.loc[boundary, "SCR_defined"] = False
            frame.loc[boundary, "SCR_status"] = "not_applicable_no_pv"
            frame.to_csv(path, index=False)
            _refresh_validation(grid_dir)

            run(grid_dir, output_dir)
            report = json.loads((output_dir / "sensitivity_line_validation.json").read_text())
            panel = report["figures"]["pv_capex"]["figure"]["metrics"]["scr"]
            self.assertEqual(panel["undefined_value_count"], 1)
            self.assertIn("No zero-fill or extrapolation", panel["note"])
            self.assertEqual(panel["curve_method"], "linear two-point fallback")
            self.assertEqual(panel["rendered_sample_count"], SMOOTH_SAMPLES)

    def test_incomplete_run_or_checksum_change_is_rejected_before_plotting(self):
        with tempfile.TemporaryDirectory() as temporary:
            grid_dir = Path(temporary) / "grid"
            _write_completed_grid(grid_dir)
            (grid_dir / "run_status.json").write_text(json.dumps({"state": "running"}))
            with self.assertRaisesRegex(ValueError, "completed"):
                run(grid_dir, Path(temporary) / "unfinished")

        with tempfile.TemporaryDirectory() as temporary:
            grid_dir = Path(temporary) / "grid"
            _write_completed_grid(grid_dir)
            path = grid_dir / "SAIDI_VoLL_Grid_Summary.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "annual_cost_sgd_per_year"] = 9999.0
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "checksum"):
                run(grid_dir, Path(temporary) / "checksum")


def _write_completed_grid(grid_dir: Path) -> None:
    grid_dir.mkdir(parents=True)
    for grid_key, row_values, column_values in (
        ("cost_capex", [600.0, 1200.0, 1800.0], [200.0, 400.0, 600.0]),
        ("saidi_voll", [0.0, 90.0, 180.0], [0.0, 30.0, 50.0]),
    ):
        rows = []
        row_parameter = "pv_capex_sgd_per_kw" if grid_key == "cost_capex" else "saidi_min_per_year"
        column_parameter = "battery_capex_sgd_per_kwh" if grid_key == "cost_capex" else "voll_sgd_per_kwh"
        for row_index, row_value in enumerate(row_values):
            for column_index, column_value in enumerate(column_values):
                factor = row_index * len(column_values) + column_index
                rows.append(
                    {
                        "grid_type": grid_key,
                        "row_parameter": row_parameter,
                        "row_value": row_value,
                        "row_unit": "SGD/kW" if grid_key == "cost_capex" else "min/year",
                        "column_parameter": column_parameter,
                        "column_value": column_value,
                        "column_unit": "SGD/kWh",
                        "case_key": "Brisbane_SingleFamily",
                        "city_key": "Brisbane",
                        "building_key": "SingleFamily",
                        "analysis_years": 15,
                        "annual_cost_sgd_per_year": 1500.0 + 75.0 * factor + (200.0 if grid_key == "saidi_voll" else 0.0),
                        "pv_capacity_kw": 2.5 + 0.15 * factor,
                        "battery_energy_kwh": 9.0 + 0.25 * factor,
                        "EUE_kwh_per_year": 0.1 * factor,
                        "LOLP": 0.02 + 0.02 * factor,
                        "SSR": 0.30 + 0.03 * factor,
                        "SCR": 0.60 + 0.02 * factor,
                        "SCR_defined": True,
                        "SCR_status": "defined",
                    }
                )
        filename = "Cost_CAPEX_Grid_Summary.csv" if grid_key == "cost_capex" else "SAIDI_VoLL_Grid_Summary.csv"
        pd.DataFrame(rows).to_csv(grid_dir / filename, index=False)
    manifest = {
        "spec": {"case_key": "Brisbane_SingleFamily", "analysis_years": 15},
        "effective_config": {
            "pv": {"capex_sgd_per_kw": 1200.0},
            "economics": {"battery_capex_sgd_per_kwh": 400.0},
            "scenarios": {"Brisbane_SingleFamily": {"saidi_min_per_year": 90.0, "voll_sgd_per_kwh": 35.0}},
        },
    }
    (grid_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (grid_dir / "run_status.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    _refresh_validation(grid_dir)


def _refresh_validation(grid_dir: Path) -> None:
    grids = {}
    for grid_key, filename in (
        ("cost_capex", "Cost_CAPEX_Grid_Summary.csv"),
        ("saidi_voll", "SAIDI_VoLL_Grid_Summary.csv"),
    ):
        path = grid_dir / filename
        grids[grid_key] = {
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_count": len(pd.read_csv(path)),
        }
    (grid_dir / "validation.json").write_text(json.dumps({"passed": True, "grids": grids}), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
