from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.capacity_sweep import ROOT, SweepSpec, run_sweep
from src.ccp_cost_sensitivity import (
    BATTERY_PARAMETER,
    DEFAULT_PARAMETER_CSV,
    PV_PARAMETER,
    CostSensitivitySpec,
    PricePoint,
    PriceSizingRun,
    SharedPhysicalCache,
    _price_config,
    _reprice_rows,
    _search_spec,
    _validate_results,
    build_price_points,
)
from src.ccp_lolp_sizing import CapacityGrid, _validate_source
from src.figures.ccp_cost_sensitivity_line_plots import LINE_SPECS, SMOOTH_SAMPLES, load_completed_results, run as run_plots


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_result_frame() -> tuple[pd.DataFrame, tuple[PricePoint, ...]]:
    points = build_price_points(DEFAULT_PARAMETER_CSV)
    rows: list[dict] = []
    for index, point in enumerate(points):
        rows.append({
            "sweep_key": point.sweep_key,
            "sweep_parameter": point.sweep_parameter,
            "sweep_value": point.sweep_value,
            "sweep_unit": point.sweep_unit,
            "pv_capex_sgd_per_kw": point.pv_capex_sgd_per_kw,
            "battery_capex_sgd_per_kwh": point.battery_capex_sgd_per_kwh,
            "battery_power_capex_sgd_per_kw": 150.0,
            "case_key": "Brisbane_SingleFamily",
            "analysis_years": 15,
            "lolp_threshold": 0.20,
            "ccp_target": 0.90,
            "comparison": "LOLP < threshold",
            "scenario_count": 100,
            "required_passing_scenarios": 90,
            "feasible": True,
            "passing_scenarios": 90,
            "actual_ccp": 0.90,
            "domain_limited": False,
            "boundary_status": "interior_or_screened_boundary",
            "candidate_origin": "shared_physical",
            "annual_cost_sgd_per_year": 2000.0 + index,
            "annual_system_cost_ksgd_per_year": 2.0 + index / 1000.0,
            "pv_capacity_kw": 3.25,
            "battery_energy_kwh": 14.50,
            "battery_power_kw": 7.25,
            "EUE_kwh_per_year": 4.0 + index / 10.0,
            "LOLP": 0.10,
            "SSR": 0.70,
            "SCR": 0.80,
            "SCR_defined": True,
            "SCR_status": "defined",
        })
    return pd.DataFrame(rows), points


def _write_completed_results(directory: Path, frame: pd.DataFrame, points: tuple[PricePoint, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    results = directory / "ccp_cost_sensitivity_results.csv"
    frame.to_csv(results, index=False)
    fingerprint = "ccp-cost-sensitivity-test"
    (directory / "manifest.json").write_text(json.dumps({
        "fingerprint": fingerprint,
        "points": [asdict(point) for point in points],
    }, indent=2), encoding="utf-8")
    (directory / "run_status.json").write_text(json.dumps({"state": "complete", "fingerprint": fingerprint}), encoding="utf-8")
    (directory / "validation.json").write_text(json.dumps({
        "passed": True,
        "fingerprint": fingerprint,
        "result_sha256": _sha256(results),
        "result_row_count": len(frame),
    }, indent=2), encoding="utf-8")


class CCPCostSensitivityTests(unittest.TestCase):
    def test_brisbane_price_axes_and_shared_baseline_match_the_plan(self):
        points = build_price_points(DEFAULT_PARAMETER_CSV)
        pv = [point for point in points if point.sweep_key == "pv_capex"]
        battery = [point for point in points if point.sweep_key == "battery_capex"]
        self.assertEqual([point.sweep_value for point in pv], [600, 800, 1000, 1200, 1400, 1600, 1800])
        self.assertEqual([point.sweep_value for point in battery], [200, 250, 300, 350, 400, 450, 500, 550, 600])
        self.assertTrue(all(point.battery_capex_sgd_per_kwh == 400.0 for point in pv))
        self.assertTrue(all(point.pv_capex_sgd_per_kw == 1200.0 for point in battery))
        self.assertEqual(len(points), 16)
        self.assertEqual(len({point.price_key for point in points}), 15)
        self.assertEqual(sum(point.price_key == (1200.0, 400.0) for point in points), 2)
        self.assertEqual(PV_PARAMETER, "pv_capex_sgd_per_kw")
        self.assertEqual(BATTERY_PARAMETER, "battery_capex_sgd_per_kwh")

    def test_exactly_ninety_passing_scenarios_is_feasible_and_grid_aligned(self):
        frame, points = _valid_result_frame()
        report = _validate_results(frame, points, CostSensitivitySpec())
        self.assertTrue(report["passed"])
        self.assertEqual(report["result_row_count"], 16)
        self.assertEqual(report["pv_capex_point_count"], 7)
        self.assertEqual(report["battery_capex_point_count"], 9)

        failing = frame.copy()
        failing.loc[0, ["passing_scenarios", "actual_ccp"]] = [89, 0.89]
        with self.assertRaisesRegex(ValueError, "passing-scenario"):
            _validate_results(failing, points, CostSensitivitySpec())
        boundary = frame.copy()
        boundary.loc[0, "domain_limited"] = True
        with self.assertRaisesRegex(ValueError, "upper boundary"):
            _validate_results(boundary, points, CostSensitivitySpec())
        misaligned = frame.copy()
        misaligned.loc[0, "pv_capacity_kw"] = 3.3
        with self.assertRaisesRegex(ValueError, "capacity grid"):
            _validate_results(misaligned, points, CostSensitivitySpec())

    def test_repricing_keeps_physical_metrics_and_shared_cache_avoids_dispatch(self):
        with tempfile.TemporaryDirectory(prefix="ccp_cost_sensitivity_test_") as temporary:
            root = Path(temporary)
            inputs = root / "inputs" / "Brisbane_SingleFamily"
            inputs.mkdir(parents=True)
            for name in (
                "Brisbane_SingleFamily_Load.csv", "Brisbane_SingleFamily_PV.csv", "Brisbane_SingleFamily_TOU.csv",
            ):
                shutil.copy2(ROOT / "Input" / "Brisbane_SingleFamily" / name, inputs / name)
            shutil.copy2(
                ROOT / "tests/fixtures/Brisbane_SingleFamily_Parameter.csv",
                inputs / "Brisbane_SingleFamily_Parameter.csv",
            )
            source_dir = root / "source"
            run_sweep(
                SweepSpec(
                    case_key="Brisbane_SingleFamily", analysis_years=1, scenario_count=2,
                    pv_capacities=(0.0, 1.0), battery_capacities=(0.0, 1.0),
                ),
                source_dir, workers=1, input_dir=root / "inputs", include_economic_metrics=True,
            )
            source = _validate_source(source_dir)
            point = PricePoint("pv_capex", "pv_capex_sgd_per_kw", 1800.0, "SGD/kW", 1800.0, 600.0)
            group = source["results"].loc[
                (source["results"].pv_capacity_kw == 1.0) & (source["results"].battery_energy_kwh == 1.0)
            ].copy()
            repriced = _reprice_rows(group, _price_config(source["config"], point))
            for column in ("LOLP", "EUE_kwh_total", "outage_load_kwh_total"):
                self.assertTrue(np.allclose(group[column], repriced[column], rtol=0.0, atol=1e-12), column)
            self.assertTrue((repriced.annual_cost_sgd_per_year > group.annual_cost_sgd_per_year).all())
            recomposed = (
                repriced.annualized_pv_cost_sgd + repriced.pv_om_cost_sgd
                + repriced.annualized_battery_cost_sgd + repriced.battery_om_cost_sgd
                + repriced.pv_bess_net_grid_cost_sgd + repriced.unserved_energy_cost_sgd_per_year
            )
            self.assertTrue(np.allclose(repriced.annual_cost_sgd_per_year, recomposed, rtol=0.0, atol=1e-9))

            search = _search_spec(CostSensitivitySpec())
            shared = SharedPhysicalCache(
                source,
                pv_grid=CapacityGrid(0.25, "PV"),
                battery_grid=CapacityGrid(0.25, "BESS"),
                output=root / "shared",
                fingerprint="shared-cache-test",
            )
            fine = group.copy()
            fine["pv_capacity_kw"] = 0.5
            fine["battery_energy_kwh"] = 0.5
            shared.add(fine)
            run = PriceSizingRun(
                _price_source_for_test(source, point), root / "price", search, 1,
                "price-test", [], shared,
            )
            try:
                run.evaluate([(2, 2)], reason="shared cache test")
                self.assertEqual(run.cache.origin((2, 2)), "shared_physical")
                self.assertEqual(run.worker_pids, set())
            finally:
                run.close_workers()

    def test_plotting_exports_two_validated_seven_panel_figures_and_preserves_blank_scr(self):
        with tempfile.TemporaryDirectory(prefix="ccp_cost_sensitivity_plot_test_") as temporary:
            root = Path(temporary)
            frame, points = _valid_result_frame()
            # A boundary SCR gap is permitted and must stay blank rather than becoming zero.
            first_pv = frame.sweep_key.eq("pv_capex") & frame.sweep_value.eq(600.0)
            frame.loc[first_pv, ["SCR", "SCR_defined", "SCR_status"]] = [np.nan, False, "not_applicable_no_pv"]
            results_dir, figures_dir = root / "results", root / "figures"
            _write_completed_results(results_dir, frame, points)
            paths = run_plots(results_dir, figures_dir)
            self.assertEqual(len(list(figures_dir.rglob("*.png"))), 2)
            self.assertEqual(len(list(figures_dir.rglob("*.svg"))), 2)
            self.assertEqual(len(list(figures_dir.rglob("*.pdf"))), 2)
            self.assertEqual(len(list(figures_dir.rglob("*_Source_Data.csv"))), 2)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths.values()))
            report = json.loads((figures_dir / "ccp_cost_sensitivity_line_validation.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(set(report["figures"]), {"pv_capex", "battery_capex"})
            for spec in LINE_SPECS:
                figure = report["figures"][spec["key"]]["figure"]
                self.assertEqual(figure["panel_count"], 7)
                self.assertEqual(figure["constraint"], "P(LOLP < 20%) >= 90%")
                self.assertEqual(figure["shared_x_limits"], [
                    min(report["figures"][spec["key"]]["figure"]["metrics"]["lolp"]["x_values"]),
                    max(report["figures"][spec["key"]]["figure"]["metrics"]["lolp"]["x_values"]),
                ])
                for metric in figure["metrics"].values():
                    self.assertEqual(metric["y_limits"][0], 0.0)
                    self.assertEqual(metric["curve_method"], "PCHIP")
                    self.assertEqual(metric["rendered_sample_count"], SMOOTH_SAMPLES)
                self.assertEqual(figure["metrics"]["scr"]["undefined_value_count"], 1 if spec["key"] == "pv_capex" else 0)
            source = pd.read_csv(figures_dir / "PV_CAPEX" / "Brisbane_SingleFamily_PV_CAPEX_CCP_LOLP_Cost_Sensitivity_Source_Data.csv")
            self.assertTrue(source.loc[source.sweep_value.eq(600.0), "SCR"].isna().all())
            with self.assertRaisesRegex(ValueError, "checksum"):
                (results_dir / "ccp_cost_sensitivity_results.csv").write_text("tampered\n", encoding="utf-8")
                load_completed_results(results_dir)


def _price_source_for_test(source: dict, point: PricePoint) -> dict:
    """Avoid exposing the runner's private source-copy helper in the test API."""
    adjusted = dict(source)
    adjusted["config"] = _price_config(source["config"], point)
    adjusted["results"] = _reprice_rows(source["results"], adjusted["config"])
    return adjusted


if __name__ == "__main__":
    unittest.main()
