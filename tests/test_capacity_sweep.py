from __future__ import annotations

from dataclasses import replace
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.capacity_sweep import (
    ROOT, SweepSpec, _availability_hash, _init_worker, _evaluate_block,
    _make_scenarios, capacity_summary, run_sweep, summarize_dispatch,
)
from src.dispatch import solve_dispatch, solve_dispatch_greedy, _reconcile_battery_origin_soc
from src.costing import pv_bess_annual_cost
from src.sweep_economics import economic_metrics, validate_economic_results
from src.io.config import load_config
from src.io.data_loader import discover_input_cases, load_input_case
from src.optimizer import battery_sizing, scenario_config
from src.resilience_metrics import electric_equivalent_building_load, ue


def brisbane_config(years=1):
    case = next(c for c in discover_input_cases(ROOT / "Input") if c.case_key == "Brisbane_SingleFamily")
    case = replace(case, parameter_csv=ROOT / "tests/fixtures/Brisbane_SingleFamily_Parameter.csv")
    _, _, config, _ = load_input_case(case, load_config(ROOT / "config.yaml"), {"analysis_years": years})
    config = scenario_config(config, case.case_key)
    config["dispatch"]["backend"] = "greedy"
    return case, config


def test_inputs(root):
    """Use a fixed parameter fixture, independent of live user parameter edits."""
    case, _ = brisbane_config()
    destination = root / case.case_key
    destination.mkdir(parents=True)
    for role in ("load_csv", "pv_csv", "tou_csv", "parameter_csv"):
        source = getattr(case, role)
        shutil.copy2(source, destination / source.name)
    return root


test_inputs.__test__ = False


class MetricTests(unittest.TestCase):
    def setUp(self):
        _, self.config = brisbane_config(2)
        self.frame = pd.DataFrame({
            "hour_index": [8758, 8759, 8760, 8761], "analysis_year": [1, 1, 2, 2],
            "electric_load_kwh": [2.0] * 4, "cooling_load_kwh_th": [0.0] * 4,
            "heating_load_kwh_th": [0.0] * 4, "pv_kwh": [0.0] * 4,
            "grid_availability": [1.0, 0.5, 0.0, 1.0], "outage_fraction": [0.0, 0.5, 1.0, 0.0],
            "tou_sgd_per_kwh": [0.3] * 4,
        })

    def evaluate(self, frame, battery):
        dispatch = solve_dispatch(frame, battery_sizing(battery, self.config), self.config)
        total_load = electric_equivalent_building_load(frame, 5.0, 3.5).to_numpy()
        row, yearly = summarize_dispatch(frame, dispatch, total_load, 2, {})
        return row, yearly, dispatch

    def test_zero_capacity_fractional_outage_and_multiyear_units(self):
        row, yearly, dispatch = self.evaluate(self.frame, 0.0)
        self.assertEqual(row["EUE_kwh_total"], 3.0)
        self.assertEqual(row["EUE_kwh_per_year"], 1.5)
        self.assertEqual(row["LOLP"], 1.0)
        self.assertEqual(row["outage_hours_total"], 1.5)
        self.assertEqual([r["EUE_kwh"] for r in yearly], [1.0, 2.0])
        self.assertEqual(row["EUE_kwh_total"], ue(dispatch))

    def test_no_outage(self):
        frame = self.frame.assign(grid_availability=1.0, outage_fraction=0.0)
        row, yearly, _ = self.evaluate(frame, 0.0)
        self.assertEqual(row["LOLP"], 0.0)
        self.assertEqual(row["EUE_kwh_total"], 0.0)
        self.assertTrue(row["no_outage_load"])
        self.assertTrue(all(r["no_outage_load"] for r in yearly))

    def test_pv_only(self):
        row, _, _ = self.evaluate(self.frame.assign(pv_kwh=1.0), 0.0)
        self.assertEqual(row["EUE_kwh_total"], 1.0)
        self.assertAlmostEqual(row["LOLP"], 1.0 / 3.0)

    def test_battery_only_and_soc_continuity(self):
        row, yearly, dispatch = self.evaluate(self.frame, 2.5)
        self.assertLess(row["EUE_kwh_total"], 3.0)
        self.assertAlmostEqual(sum(r["EUE_kwh"] for r in yearly), row["EUE_kwh_total"])
        eta = self.config["battery"]["roundtrip_efficiency"] ** 0.5
        expected = dispatch.battery_soc_kwh.iloc[1] - dispatch.battery_discharge_kwh.iloc[2] / eta
        self.assertAlmostEqual(dispatch.battery_soc_kwh.iloc[2], expected)
        # Restarting the second year would incorrectly replenish the battery.
        fresh = solve_dispatch(self.frame.iloc[2:], battery_sizing(2.5, self.config), self.config)
        self.assertGreater(dispatch.unmet_load_kwh_equiv.iloc[2], fresh.unmet_load_kwh_equiv.iloc[0])

    def test_lolp_summary_is_mean_of_ratios(self):
        rows = pd.DataFrame({
            "pv_capacity_kw": [0, 0], "battery_energy_kwh": [0, 0],
            "EUE_kwh_total": [1.0, 1.0], "EUE_kwh_per_year": [0.5, 0.5], "LOLP": [1.0, 0.1],
        })
        summary = capacity_summary(rows).iloc[0]
        self.assertAlmostEqual(summary.LOLP_mean, 0.55)
        self.assertAlmostEqual(summary.LOLP_std, np.std([1.0, 0.1], ddof=1))

    def test_lightweight_output_equals_full_hourly_output(self):
        for pv in (0.0, 1.0, 5.0):
            for battery in (0.0, 2.5, 20.0):
                frame = self.frame.assign(pv_kwh=pv)
                sizing = battery_sizing(battery, self.config)
                full = solve_dispatch_greedy(frame, sizing, self.config)
                light = solve_dispatch_greedy(frame, sizing, self.config, metrics_only=True)
                pd.testing.assert_frame_equal(light, full[light.columns])

    def test_scalar_soc_tolerance_matches_original_numpy_condition(self):
        rng = np.random.default_rng(123)
        for _ in range(2000):
            pv, grid, initial = rng.uniform(0, 20, 3)
            total = pv + grid + initial
            target = total + rng.choice([-1, 1]) * rng.uniform(0, 2) * (1e-9 + 1e-5 * total)
            if np.isclose(total, target, atol=1e-9):
                expected = (pv, grid, initial)
            elif total > target:
                scale = target / total
                expected = (pv * scale, grid * scale, initial * scale)
            else:
                expected = (pv, grid, initial + (target - total))
            self.assertEqual(_reconcile_battery_origin_soc(target, pv, grid, initial), expected)

    def test_bad_specs(self):
        for spec in (SweepSpec(analysis_years=0), SweepSpec(pv_capacities=(0.0, 0.0)),
                     SweepSpec(battery_capacities=(-1.0,)), SweepSpec(scenario_count=0)):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                spec.validate()


class ScenarioTests(unittest.TestCase):
    def test_full_horizon_and_exact_sparse_reconstruction(self):
        _, config = brisbane_config(15)
        payloads, table, events = _make_scenarios(config, SweepSpec(scenario_count=2))
        self.assertEqual(len(table), 2)
        for payload in payloads:
            availability = np.ones(131400)
            availability[payload["hour_indices"]] = payload["availability"]
            self.assertEqual(_availability_hash(availability), payload["outage_profile_sha256"])
            self.assertAlmostEqual(float((1 - availability).sum()) * 60, 15 * 90, places=6)
        self.assertTrue(events.start_analysis_year.max() > 1)
        self.assertNotEqual(payloads[0]["outage_profile_sha256"], payloads[1]["outage_profile_sha256"])

    def test_zero_event_scenarios(self):
        case, config = brisbane_config()
        config["active_scenario"]["saifi_per_year"] = 0.0
        payloads, table, events = _make_scenarios(config, SweepSpec(analysis_years=1, scenario_count=1))
        self.assertTrue(events.empty)
        self.assertEqual(payloads[0]["hour_indices"], [])
        _init_worker(case, config, 1.0)
        block = _evaluate_block(payloads[0], 0.0, (0.0, 2.5))
        self.assertTrue(all(row["LOLP"] == 0 and row["EUE_kwh_total"] == 0 for row in block["scenario_rows"]))


class EconomicMetricTests(unittest.TestCase):
    def setUp(self):
        _, self.config = brisbane_config(1)
        self.frame = pd.DataFrame({
            "hour_index": range(8), "analysis_year": [1] * 8,
            "electric_load_kwh": [2.0] * 8, "cooling_load_kwh_th": [0.0] * 8,
            "heating_load_kwh_th": [0.0] * 8, "pv_kwh": [0, 0, 4, 8, 0, 0, 2, 0],
            "grid_availability": [1, 1, 1, 0.75, 0, 1, 1, 1],
            "outage_fraction": [0, 0, 0, 0.25, 1, 0, 0, 0],
            "tou_sgd_per_kwh": [0.1, 0.2, 0.3, 0.3, 0.6, 0.4, 0.3, 0.6],
            "pv_export_sgd_per_kwh": [0.05] * 8,
        })

    def metrics(self, frame, battery, pv_capacity):
        sizing = battery_sizing(battery, self.config)
        light = solve_dispatch_greedy(frame, sizing, self.config, metrics_only=True, collect_totals=True)
        total = economic_metrics(light.attrs["energy_totals"], sizing, pv_capacity, self.config)
        reliability, _ = summarize_dispatch(frame, light, frame.electric_load_kwh.to_numpy(), 1, {})
        total.update(reliability)
        total["EUE_threshold_difference_kwh_total"] = total["unserved_energy_kwh_total"] - total["EUE_kwh_total"]
        validate_economic_results(pd.DataFrame([total]), self.config)
        return total, light

    def test_full_and_light_costs_energy_and_export(self):
        for export in (False, True):
            self.config["pv"]["export_enabled"] = export
            for pv in (0, 1, 5):
                for battery in (0, 2.5, 20):
                    with self.subTest(export=export, pv=pv, battery=battery):
                        frame = self.frame.assign(pv_kwh=self.frame.pv_kwh * pv)
                        values, light = self.metrics(frame, battery, pv)
                        sizing = battery_sizing(battery, self.config)
                        full = solve_dispatch_greedy(frame, sizing, self.config)
                        pd.testing.assert_frame_equal(light, full[light.columns])
                        costs = pv_bess_annual_cost(frame, full, sizing, {"pv_capacity_kw": pv}, self.config)
                        for key, expected in costs.items():
                            self.assertAlmostEqual(values[key], expected, places=9, msg=key)
                        for scalar, hourly in {
                            "pv_self_served_load_kwh_total": "pv_self_served_load_kwh",
                            "pv_to_battery_kwh_total": "pv_to_battery_kwh",
                            "pv_curtailed_kwh_total": "pv_curtailed_kwh",
                            "battery_soc_kwh_sum": "battery_soc_kwh",
                        }.items():
                            self.assertAlmostEqual(values[scalar], full[hourly].sum(), places=9)

    def test_zero_pv_excludes_grid_and_initial_energy(self):
        values, _ = self.metrics(self.frame.assign(pv_kwh=0), 10, 0)
        self.assertEqual(values["SSR"], 0)
        self.assertIsNone(values["SCR"])
        self.assertFalse(values["SCR_defined"])
        self.assertGreater(values["grid_to_battery_kwh_total"], 0)
        self.assertGreater(values["battery_discharge_from_initial_kwh_total"], 0)
        self.assertGreater(values["battery_discharge_from_grid_kwh_total"], 0)

    def test_scr_includes_stored_energy_and_excludes_curtailment(self):
        self.config["battery"]["initial_soc_fraction"] = 0
        self.config["battery"]["soc_min_fraction"] = 0
        frame = self.frame.iloc[:1].assign(pv_kwh=10, grid_availability=1, outage_fraction=0)
        values, _ = self.metrics(frame, 10, 10)
        self.assertEqual(values["SSR"], 1)
        self.assertAlmostEqual(values["SCR"], (2 + 5) / 10)
        self.assertGreater(values["battery_soc_pv_final_kwh"], 0)
        self.assertEqual(values["pv_curtailed_kwh_total"], 3)

    def test_raw_cost_retains_sub_tolerance_unmet(self):
        frame = self.frame.iloc[:1].assign(pv_kwh=0, electric_load_kwh=5e-7, grid_availability=0, outage_fraction=1)
        values, _ = self.metrics(frame, 0, 0)
        self.assertEqual(values["EUE_kwh_total"], 0)
        self.assertEqual(values["EUE_threshold_difference_kwh_total"], 5e-7)
        self.assertAlmostEqual(values["annual_cost_sgd_per_year"], 5e-7 * self.config["active_scenario"]["voll_sgd_per_kwh"])

    def test_zero_system_cost_has_only_energy_and_outage_cost(self):
        frame = self.frame.assign(pv_kwh=0)
        values, _ = self.metrics(frame, 0, 0)
        grid_bill = (frame.electric_load_kwh * frame.grid_availability * frame.tou_sgd_per_kwh).sum()
        outage_energy = (frame.electric_load_kwh * frame.outage_fraction).sum()
        self.assertAlmostEqual(values["annual_cost_sgd_per_year"], grid_bill + outage_energy * self.config["active_scenario"]["voll_sgd_per_kwh"])


class SweepIntegrationTests(unittest.TestCase):
    def test_enrichment_replays_fifteen_year_source_and_resumes(self):
        spec = SweepSpec(analysis_years=15, scenario_count=2, pv_capacities=(0.0, 5.0), battery_capacities=(0.0, 10.0))
        with tempfile.TemporaryDirectory(prefix="enrichment_test_") as directory:
            root = Path(directory)
            inputs = test_inputs(root / "inputs")
            source, serial, parallel = (root / name for name in ("source", "serial", "parallel"))
            run_sweep(spec, source, workers=2, input_dir=inputs)
            original = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
            # Neither new random sampling nor the live parameter/config loader is allowed.
            with patch("src.capacity_sweep._make_scenarios", side_effect=AssertionError("resampled")), patch("src.capacity_sweep.load_input_case", side_effect=AssertionError("live inputs")):
                run_sweep(SweepSpec(), serial, workers=1, enrich_from=source)
                run_sweep(SweepSpec(), parallel, workers=2, enrich_from=source)
                for name in ("scenario_results.csv", "yearly_results.csv", "capacity_summary.csv"):
                    self.assertEqual((serial / name).read_bytes(), (parallel / name).read_bytes())
                before = (parallel / "scenario_results.csv").read_bytes()
                next((parallel / "checkpoints").glob("*.json")).unlink()
                status = run_sweep(SweepSpec(), parallel, workers=2, resume=True, enrich_from=source)
                self.assertEqual(status["tasks_resumed"], 3)
                self.assertEqual(before, (parallel / "scenario_results.csv").read_bytes())
            for name, content in original.items():
                self.assertEqual(content, (source / name).read_bytes())
            read = lambda path: pd.read_csv(path, float_precision="round_trip")
            old, new = read(source / "scenario_results.csv"), read(parallel / "scenario_results.csv")
            pd.testing.assert_frame_equal(old, new[old.columns], check_exact=True)
            self.assertEqual(new.SCR.isna().sum(), 4)
            self.assertEqual(len(new), 8)
            self.assertFalse(new[["annual_cost_sgd_per_year", "EUE_kwh_per_year", "LOLP", "SSR"]].isna().any().any())
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                run_sweep(spec, source, enrich_from=source)
            payload_path = source / "outage_profiles_sparse.json"
            payloads = json.loads(payload_path.read_text())
            payloads[0]["availability"][0] *= 0.5
            payload_path.write_text(json.dumps(payloads))
            with self.assertRaisesRegex(ValueError, "profile checksum mismatch"):
                run_sweep(spec, root / "invalid", enrich_from=source)

    def test_direct_economic_metrics_are_written_in_the_initial_run(self):
        spec = SweepSpec(
            analysis_years=1, scenario_count=2, pv_capacities=(0.0, 1.0), battery_capacities=(0.0, 2.5),
        )
        with tempfile.TemporaryDirectory(prefix="direct_economic_sweep_test_") as directory:
            root = Path(directory)
            output = root / "direct"
            status = run_sweep(
                spec, output, workers=2, input_dir=test_inputs(root / "inputs"), include_economic_metrics=True,
            )
            self.assertEqual(status["metrics"], "cost_EUE_LOLP_SSR_SCR")
            results = pd.read_csv(output / "scenario_results.csv")
            self.assertEqual(len(results), 8)
            self.assertFalse(results[["annual_cost_sgd_per_year", "EUE_kwh_per_year", "LOLP", "SSR"]].isna().any().any())
            self.assertEqual(int(results.SCR.isna().sum()), 4)
            self.assertTrue((output / "数据说明.md").exists())
            validation = json.loads((output / "validation.json").read_text())
            self.assertTrue(validation["economic_metrics_passed"])

    def test_serial_parallel_resume_and_fingerprint(self):
        spec = SweepSpec(analysis_years=1, scenario_count=2, pv_capacities=(0.0, 1.0), battery_capacities=(0.0, 2.5))
        with tempfile.TemporaryDirectory(prefix="capacity_sweep_test_") as directory:
            inputs = test_inputs(Path(directory) / "inputs")
            serial, parallel = Path(directory) / "serial", Path(directory) / "parallel"
            run_sweep(spec, serial, workers=1, input_dir=inputs)
            status = run_sweep(spec, parallel, workers=2, input_dir=inputs)
            self.assertEqual(status["state"], "complete")
            self.assertEqual(len(status["worker_pids"]), 2)
            for name in ("scenario_results.csv", "yearly_results.csv", "capacity_summary.csv"):
                pd.testing.assert_frame_equal(pd.read_csv(serial / name), pd.read_csv(parallel / name))
            before = (parallel / "scenario_results.csv").read_bytes()
            # Delete one completed block to model a run interrupted before that checkpoint.
            next((parallel / "checkpoints").glob("*.json")).unlink()
            status = run_sweep(spec, parallel, workers=2, resume=True, input_dir=inputs)
            self.assertEqual(status["tasks_resumed"], 3)
            self.assertEqual(before, (parallel / "scenario_results.csv").read_bytes())
            status = run_sweep(spec, parallel, workers=1, resume=True, input_dir=inputs)
            self.assertEqual(status["tasks_resumed"], 4)
            with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                run_sweep(replace(spec, seed_start=43), parallel, resume=True, input_dir=inputs)
            checkpoint = next((parallel / "checkpoints").glob("*.json"))
            data = json.loads(checkpoint.read_text())
            data["payload"]["scenario_rows"][0]["EUE_kwh_total"] += 1
            checkpoint.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                run_sweep(spec, parallel, resume=True, input_dir=inputs)


if __name__ == "__main__":
    unittest.main()
