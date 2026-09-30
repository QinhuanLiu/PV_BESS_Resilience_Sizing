from __future__ import annotations

from dataclasses import replace
import shutil
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.capacity_sweep import ROOT, SweepSpec, run_sweep, summarize_dispatch
from src.ccp_lolp_sizing import (
    DEFAULT_CCP_TARGETS,
    DEFAULT_LOLP_THRESHOLDS,
    SearchSpec,
    SizingRun,
    _validate_source,
    run_sizing,
    strict_ccp_count,
)
from src.dispatch import solve_dispatch_greedy
from src.io.config import load_config
from src.io.data_loader import build_case_frame, discover_input_cases, load_input_case
from src.optimizer import battery_sizing, scenario_config
from src.resilience_metrics import electric_equivalent_building_load


class CCPLOLPSizingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory(prefix="ccp_lolp_sizing_test_")
        cls.root = Path(cls._temporary.name)
        inputs = cls.root / "inputs" / "Brisbane_SingleFamily"
        inputs.mkdir(parents=True)
        for name in (
            "Brisbane_SingleFamily_Load.csv", "Brisbane_SingleFamily_PV.csv", "Brisbane_SingleFamily_TOU.csv",
        ):
            shutil.copy2(ROOT / "Input" / "Brisbane_SingleFamily" / name, inputs / name)
        shutil.copy2(
            ROOT / "tests/fixtures/Brisbane_SingleFamily_Parameter.csv",
            inputs / "Brisbane_SingleFamily_Parameter.csv",
        )
        cls.source = cls.root / "source"
        run_sweep(
            SweepSpec(
                case_key="Brisbane_SingleFamily", analysis_years=1, scenario_count=2,
                pv_capacities=(0.0, 1.0), battery_capacities=(0.0, 1.0),
            ),
            cls.source, workers=1, input_dir=cls.root / "inputs", include_economic_metrics=True,
        )
        cls.spec = SearchSpec(
            lolp_thresholds=(0.90,), ccp_targets=(0.50,), pv_step_kw=0.5,
            battery_step_kwh=0.5, bess_batch_size=2,
        )

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def test_strict_threshold_excludes_equal_lolp(self):
        self.assertEqual(strict_ccp_count([0.099, 0.100, 0.101], 0.10), 1)

    def test_streaming_dispatch_matches_metrics_only_reliability_and_cost_totals(self):
        case = next(case for case in discover_input_cases(ROOT / "Input") if case.case_key == "Brisbane_SingleFamily")
        case = replace(case, parameter_csv=ROOT / "tests/fixtures/Brisbane_SingleFamily_Parameter.csv")
        _, _, config, _ = load_input_case(case, load_config(ROOT / "config.yaml"), {"analysis_years": 1}, 1.0)
        config = scenario_config(config, case.case_key)
        config["dispatch"]["backend"] = "greedy"
        frame, _ = build_case_frame(case, config, 1.0)
        frame = frame.iloc[:48].copy()
        frame["grid_availability"] = 1.0
        frame.loc[frame.index[12], "grid_availability"] = 0.0
        frame["outage_fraction"] = 1.0 - frame["grid_availability"]
        sizing = battery_sizing(2.5, config)
        metrics = solve_dispatch_greedy(frame, sizing, config, metrics_only=True, collect_totals=True)
        streaming = solve_dispatch_greedy(
            frame, sizing, config, metrics_only=True, collect_totals=True, streaming_only=True,
        )
        total_load = electric_equivalent_building_load(
            frame, config["technical"]["cooling_cop"], config["technical"]["heating_cop"],
        ).to_numpy()
        ordinary, _ = summarize_dispatch(frame, metrics, total_load, 1, {})
        self.assertEqual(len(streaming), 0)
        self.assertAlmostEqual(ordinary["EUE_kwh_total"], streaming.attrs["reliability_totals"]["EUE_kwh_total"])
        self.assertAlmostEqual(ordinary["LOLP"], streaming.attrs["reliability_totals"]["LOLP"])
        self.assertEqual(metrics.attrs["energy_totals"], streaming.attrs["energy_totals"])

    def test_serial_parallel_and_resume_preserve_candidate_outputs(self):
        serial, parallel = self.root / "serial", self.root / "parallel"
        run_sizing(self.source, serial, workers=1, spec=self.spec)
        run_sizing(self.source, parallel, workers=2, spec=self.spec)
        for name in (
            "ccp_target_curve_source.csv", "candidate_summary.csv", "fine_scenario_results.csv", "refinement_trace.csv",
        ):
            self.assertEqual((serial / name).read_bytes(), (parallel / name).read_bytes(), name)
        before = (serial / "fine_scenario_results.csv").read_bytes()
        checkpoint = next((serial / "checkpoints").glob("*.json"))
        checkpoint.unlink()
        run_sizing(self.source, serial, workers=2, resume=True, spec=self.spec)
        self.assertEqual(before, (serial / "fine_scenario_results.csv").read_bytes())
        candidates = pd.read_csv(serial / "candidate_summary.csv")
        self.assertFalse(candidates.duplicated(["pv_capacity_kw", "battery_energy_kwh"]).any())
        curves = pd.read_csv(serial / "ccp_target_curve_source.csv")
        self.assertEqual(len(curves), 1)
        self.assertTrue(bool(curves.feasible.iloc[0]))

    def test_invalid_source_checksum_is_rejected(self):
        invalid = self.root / "invalid"
        shutil.copytree(self.source, invalid)
        path = invalid / "scenario_results.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            run_sizing(invalid, self.root / "bad-output", workers=1, spec=self.spec)

    def test_default_target_matrix_emits_all_seventy_two_curve_rows(self):
        result = run_sizing(
            self.source, self.root / "default-targets", workers=1,
            spec=SearchSpec(pv_step_kw=0.5, battery_step_kwh=0.5, bess_batch_size=2),
        )
        curves = pd.read_csv(self.root / "default-targets" / "ccp_target_curve_source.csv")
        self.assertEqual(len(curves), len(DEFAULT_LOLP_THRESHOLDS) * len(DEFAULT_CCP_TARGETS))
        self.assertEqual(result["curve_rows"], 72)
        self.assertEqual(len(DEFAULT_CCP_TARGETS), 24)
        self.assertEqual(DEFAULT_CCP_TARGETS[-2:], (0.94, 0.95))
        self.assertEqual(DEFAULT_CCP_TARGETS[:-1], tuple(round(value / 100.0, 10) for value in range(50, 95, 2)))

    def test_nonmonotone_source_ccp_forces_full_local_bess_fallback(self):
        source = _validate_source(self.source)
        fallback_spec = SearchSpec(
            lolp_thresholds=(0.90,), ccp_targets=(0.50,), pv_step_kw=1.0,
            battery_step_kwh=0.5, bess_batch_size=2,
        )
        trace: list[dict] = []
        run = SizingRun(source, self.root / "fallback", fallback_spec, 1, "fallback", trace)
        # Create a deliberately non-monotone source CCP row: more BESS has
        # fewer passing scenarios.  The fallback must evaluate the complete
        # 0.5 kWh axis rather than trusting a binary-search result.
        run.cache.rows((0, 0)).loc[:, "LOLP"] = 0.0
        run.cache.rows((0, 2)).loc[:, "LOLP"] = 1.0
        try:
            run.reliability_frontiers(set(run.all_cells), reason="nonmonotone unit test")
        finally:
            run.close_workers()
        self.assertTrue(any(item["event"] == "bisection_fallback" for item in trace))
        self.assertTrue(run.cache.has((0, 1)))  # 0.5 kWh, absent from the source grid

    def test_active_mesh_refines_only_classified_cells_on_nested_grid(self):
        """The 0.5 -> 0.25 loop writes an auditable, aligned mesh result."""
        spec = SearchSpec(
            lolp_thresholds=(0.90,), ccp_targets=(0.50,),
            pv_step_kw=0.25, battery_step_kwh=0.25,
            intermediate_pv_step_kw=0.5, intermediate_battery_step_kwh=0.5,
            bess_batch_size=2,
        )
        output = self.root / "active-mesh"
        result = run_sizing(self.source, output, workers=1, spec=spec)
        self.assertEqual(result["curve_rows"], 1)
        cells = pd.read_csv(output / "mesh_cells.csv")
        rounds = pd.read_csv(output / "mesh_rounds.csv")
        curves = pd.read_csv(output / "ccp_target_curve_source.csv")
        candidates = pd.read_csv(output / "candidate_summary.csv")
        self.assertTrue((cells.level == "intermediate_0p5").any())
        self.assertTrue((cells.level == "final_0p25").any())
        self.assertIn("intermediate_0p5", set(rounds["round"]))
        self.assertIn("final_0p25", set(rounds["round"]))
        self.assertEqual(len(curves), 1)
        self.assertTrue(bool(curves.feasible.iloc[0]))
        self.assertTrue(((candidates.pv_capacity_kw / 0.25).round() == candidates.pv_capacity_kw / 0.25).all())
        self.assertTrue(((candidates.battery_energy_kwh / 0.25).round() == candidates.battery_energy_kwh / 0.25).all())

    def test_checkpoint_import_reuses_compatible_final_grid_candidates(self):
        donor = self.root / "donor"
        run_sizing(self.source, donor, workers=1, spec=self.spec)
        self.assertTrue(list((donor / "checkpoints").glob("*.json")))
        spec = SearchSpec(
            lolp_thresholds=(0.90,), ccp_targets=(0.50,),
            pv_step_kw=0.25, battery_step_kwh=0.25,
            intermediate_pv_step_kw=0.5, intermediate_battery_step_kwh=0.5,
            import_checkpoints_from=donor, bess_batch_size=2,
        )
        output = self.root / "imported-mesh"
        run_sizing(self.source, output, workers=1, spec=spec)
        candidates = pd.read_csv(output / "candidate_summary.csv")
        self.assertIn("imported_checkpoint", set(candidates.candidate_origin))
        trace = pd.read_csv(output / "refinement_trace.csv")
        imported = trace.loc[trace.event == "checkpoint_import", "imported_candidate_count"]
        self.assertEqual(len(imported), 1)
        self.assertGreater(int(imported.iloc[0]), 0)


if __name__ == "__main__":
    unittest.main()
