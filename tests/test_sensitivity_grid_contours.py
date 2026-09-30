from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.figures.core import matplotlib_context
from src.figures.sensitivity_grid_contours import (
    GRID_GROUPS,
    METRICS,
    build_figure,
    display_values,
    load_completed_grids,
    run,
    group_metric_levels,
)


class SensitivityGridContourTests(unittest.TestCase):
    def test_complete_run_exports_all_groups_with_independent_metric_scales(self):
        with tempfile.TemporaryDirectory() as temporary:
            grid_dir = Path(temporary) / "grid"
            output_dir = Path(temporary) / "figures"
            _write_completed_grid(grid_dir)

            paths = run(grid_dir, output_dir)
            pngs = sorted(output_dir.rglob("*.png"))
            svgs = sorted(output_dir.rglob("*.svg"))
            pdfs = sorted(output_dir.rglob("*.pdf"))
            self.assertEqual(len(pngs), 14)
            self.assertEqual(len(svgs), 14)
            self.assertEqual(len(pdfs), 14)
            self.assertTrue(all(path.stat().st_size > 0 for path in [*pngs, *svgs, *pdfs]))
            self.assertIn("cost_capex_source_data", paths)
            self.assertIn("saidi_voll_source_data", paths)

            report = json.loads((output_dir / "sensitivity_contour_validation.json").read_text())
            self.assertEqual(set(report["groups"]), set(GRID_GROUPS))
            self.assertEqual(set(report["group_levels"]), set(GRID_GROUPS))
            for metric in METRICS:
                key = metric["key"]
                cost_levels = report["groups"]["cost_capex"]["figures"][key]["levels"]
                outage_levels = report["groups"]["saidi_voll"]["figures"][key]["levels"]
                self.assertEqual(cost_levels, report["group_levels"]["cost_capex"][key])
                self.assertEqual(outage_levels, report["group_levels"]["saidi_voll"][key])
            self.assertNotEqual(
                report["group_levels"]["cost_capex"]["annual_system_cost"],
                report["group_levels"]["saidi_voll"]["annual_system_cost"],
            )
            frames, _ = load_completed_grids(grid_dir)
            for grid_key, frame in frames.items():
                for metric in METRICS:
                    values = display_values(frame, metric).dropna().to_numpy(dtype=float)
                    levels = np.asarray(report["group_levels"][grid_key][metric["key"]], dtype=float)
                    self.assertGreaterEqual(values.min(), levels.min())
                    self.assertLessEqual(values.max(), levels.max())
            self.assertIn("No optimisation or dispatch was rerun", (output_dir / "README.md").read_text())

    def test_axes_metric_definitions_and_group_specific_levels(self):
        with tempfile.TemporaryDirectory() as temporary:
            grid_dir = Path(temporary) / "grid"
            _write_completed_grid(grid_dir)
            frames, _ = load_completed_grids(grid_dir)
            levels = group_metric_levels(frames)
            pv_metric = next(metric for metric in METRICS if metric["key"] == "optimal_pv_capacity")
            cost_figure, cost_metadata = build_figure(
                frames["cost_capex"], GRID_GROUPS["cost_capex"], pv_metric, levels["cost_capex"][pv_metric["key"]]
            )
            outage_figure, outage_metadata = build_figure(
                frames["saidi_voll"], GRID_GROUPS["saidi_voll"], pv_metric, levels["saidi_voll"][pv_metric["key"]]
            )
            _, plt = matplotlib_context()
            try:
                self.assertEqual(cost_metadata["x_axis"], "Battery energy CAPEX (SGD kWh$^{-1}$)")
                self.assertEqual(cost_metadata["y_axis"], "PV CAPEX (SGD kW$^{-1}$)")
                self.assertEqual(outage_metadata["x_axis"], "VoLL (SGD kWh$^{-1}$)")
                self.assertEqual(outage_metadata["y_axis"], "SAIDI (min yr$^{-1}$)")
                self.assertNotEqual(
                    levels["cost_capex"]["annual_system_cost"].tolist(),
                    levels["saidi_voll"]["annual_system_cost"].tolist(),
                )
                self.assertEqual(cost_metadata["grid_shape"], [3, 3])
                self.assertEqual(outage_metadata["grid_shape"], [3, 3])
                self.assertIn("Optimal PV capacity", cost_metadata["title"])
            finally:
                plt.close(cost_figure)
                plt.close(outage_figure)

    def test_boundary_only_undefined_scr_is_left_unfilled(self):
        with tempfile.TemporaryDirectory() as temporary:
            grid_dir = Path(temporary) / "grid"
            _write_completed_grid(grid_dir)
            path = grid_dir / GRID_GROUPS["cost_capex"]["summary_filename"]
            frame = pd.read_csv(path)
            edge = frame["row_value"].eq(frame["row_value"].min())
            frame.loc[edge, "SCR"] = np.nan
            frame.loc[edge, "SCR_defined"] = False
            frame.loc[edge, "SCR_status"] = "not_applicable_no_pv"
            frame.to_csv(path, index=False)
            _refresh_validation(grid_dir)

            frames, _ = load_completed_grids(grid_dir)
            metric = next(metric for metric in METRICS if metric["key"] == "scr")
            figure, metadata = build_figure(
                frames["cost_capex"], GRID_GROUPS["cost_capex"], metric, group_metric_levels(frames)["cost_capex"]["scr"]
            )
            _, plt = matplotlib_context()
            try:
                self.assertEqual(metadata["grid_shape"], [2, 3])
                self.assertIn("No zero-fill or extrapolation", metadata["note"])
            finally:
                plt.close(figure)

    def test_validation_rejects_incomplete_or_invalid_inputs(self):
        mutations = {
            "unfinished_run": _make_unfinished,
            "checksum_mismatch": _make_checksum_mismatch,
            "axis_parameter": _make_axis_parameter_mismatch,
            "incomplete_grid": _make_incomplete_grid,
            "nonfinite_metric": _make_nonfinite_metric,
            "invalid_scr_status": _make_invalid_scr_status,
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                grid_dir = Path(temporary) / "grid"
                _write_completed_grid(grid_dir)
                mutation(grid_dir)
                with self.assertRaises(ValueError):
                    load_completed_grids(grid_dir)


def _write_completed_grid(grid_dir: Path) -> None:
    grid_dir.mkdir(parents=True)
    for grid_key, group in GRID_GROUPS.items():
        rows = []
        row_values = [600.0, 800.0, 1000.0] if grid_key == "cost_capex" else [0.0, 45.0, 90.0]
        column_values = [200.0, 300.0, 400.0] if grid_key == "cost_capex" else [0.0, 10.0, 20.0]
        for row_index, row_value in enumerate(row_values):
            for column_index, column_value in enumerate(column_values):
                factor = row_index * len(column_values) + column_index
                rows.append(
                    {
                        "grid_type": grid_key,
                        "row_parameter": group["row_parameter"],
                        "row_value": row_value,
                        "row_unit": "SGD/kW" if grid_key == "cost_capex" else "min/year",
                        "column_parameter": group["column_parameter"],
                        "column_value": column_value,
                        "column_unit": "SGD/kWh",
                        "case_key": "Brisbane_SingleFamily",
                        "city_key": "Brisbane",
                        "building_key": "SingleFamily",
                        "analysis_years": 15,
                        "annual_cost_sgd_per_year": 1500.0 + 100.0 * factor + (300.0 if grid_key == "saidi_voll" else 0.0),
                        "pv_capacity_kw": 3.0 + 0.1 * factor,
                        "battery_energy_kwh": 10.0 + 0.2 * factor,
                        "EUE_kwh_per_year": 0.1 * factor,
                        "LOLP": 0.02 * factor,
                        "SSR": 0.35 + 0.03 * factor,
                        "SCR": 0.70 + 0.02 * factor,
                        "SCR_defined": True,
                        "SCR_status": "defined",
                    }
                )
        pd.DataFrame(rows).to_csv(grid_dir / group["summary_filename"], index=False)
    (grid_dir / "manifest.json").write_text(
        json.dumps({"spec": {"case_key": "Brisbane_SingleFamily", "analysis_years": 15}}), encoding="utf-8"
    )
    (grid_dir / "run_status.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    _refresh_validation(grid_dir)


def _refresh_validation(grid_dir: Path) -> None:
    grids = {}
    for grid_key, group in GRID_GROUPS.items():
        path = grid_dir / group["summary_filename"]
        grids[grid_key] = {
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_count": len(pd.read_csv(path)),
        }
    (grid_dir / "validation.json").write_text(json.dumps({"passed": True, "grids": grids}), encoding="utf-8")


def _make_unfinished(grid_dir: Path) -> None:
    (grid_dir / "run_status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")


def _make_checksum_mismatch(grid_dir: Path) -> None:
    path = grid_dir / GRID_GROUPS["cost_capex"]["summary_filename"]
    frame = pd.read_csv(path)
    frame.loc[0, "EUE_kwh_per_year"] = 999.0
    frame.to_csv(path, index=False)


def _make_axis_parameter_mismatch(grid_dir: Path) -> None:
    path = grid_dir / GRID_GROUPS["cost_capex"]["summary_filename"]
    frame = pd.read_csv(path)
    frame.loc[0, "row_parameter"] = "wrong_parameter"
    frame.to_csv(path, index=False)
    _refresh_validation(grid_dir)


def _make_incomplete_grid(grid_dir: Path) -> None:
    path = grid_dir / GRID_GROUPS["cost_capex"]["summary_filename"]
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    _refresh_validation(grid_dir)


def _make_nonfinite_metric(grid_dir: Path) -> None:
    path = grid_dir / GRID_GROUPS["saidi_voll"]["summary_filename"]
    frame = pd.read_csv(path)
    frame.loc[0, "EUE_kwh_per_year"] = np.nan
    frame.to_csv(path, index=False)
    _refresh_validation(grid_dir)


def _make_invalid_scr_status(grid_dir: Path) -> None:
    path = grid_dir / GRID_GROUPS["saidi_voll"]["summary_filename"]
    frame = pd.read_csv(path)
    frame.loc[0, "SCR_status"] = "not_applicable_no_pv"
    frame.to_csv(path, index=False)
    _refresh_validation(grid_dir)


if __name__ == "__main__":
    unittest.main()
