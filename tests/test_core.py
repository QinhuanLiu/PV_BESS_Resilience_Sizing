from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import pandas as pd

from src.io.config import DEFAULT_CONFIG
from src.costing import pv_annual_cost
from src.io.data_loader import (
    SENSITIVITY_PARAMETERS,
    build_case_frame,
    discover_input_cases,
    load_case_parameters,
    load_input_case,
    sensitivity_runs,
)
from src.degradation import annualized_battery_cost
from src.dispatch import outage_free_frame, solve_dispatch, solve_dispatch_greedy, solve_dispatch_milp
from src.optimizer import CandidateEvaluator, CandidateStore, evaluate_candidate, resilience_design, scenario_config
from src.outage import simulate_outages, summarize_outages
from src.uncertainty import run_uncertainty_analysis, summarize_uncertainty_results
from src.app.outputs import write_candidate_dispatch_workbooks
from pv_bess_sizing import (
    DISPATCH_BACKEND_CHOICES,
    run_all,
    run_selected,
)


PARAMETER_COLUMNS = [
    "parameter",
    "base_value",
    "unit",
    "sensitivity_start",
    "sensitivity_end",
    "sensitivity_step",
    "description",
    "中文说明",
]


class CoreTests(unittest.TestCase):
    def _fast_greedy_config(self) -> dict[str, object]:
        config = deepcopy(DEFAULT_CONFIG)
        config.setdefault("dispatch", {})["backend"] = "greedy"
        return config

    def _case_dir(self, root: Path) -> Path:
        case_dir = root / "Input" / "SG"
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def _parameter_rows(self) -> list[dict[str, object]]:
        rows = [
            ("pv_capex_sgd_per_kw", 1200, "SGD/kW", 1000, 1200, 200, "PV unit CAPEX.", "光伏单位装机投资成本。"),
            ("battery_capex_sgd_per_kwh", 450, "SGD/kWh", 350, 550, 100, "Battery energy CAPEX.", "电池储能单位能量容量投资成本。"),
            ("battery_power_capex_sgd_per_kw", 180, "SGD/kW", 120, 200, 40, "Battery power CAPEX.", "电池储能单位功率容量投资成本。"),
            ("saidi_min_per_year", 0.26, "min/year", 0, 5, 5, "Annual outage duration.", "年平均停电持续时间。"),
            ("saifi_per_year", 0.006, "events/year", 0, 1, 1, "Annual outage count.", "年平均停电次数。"),
            ("voll_sgd_per_kwh", 35, "SGD/kWh", 10, 40, 15, "Value of lost load.", "单位未供电电量对应的停电损失价值。"),
            ("city_name", "SG", "-", "", "", "", "City key.", "城市或地区名称。"),
            ("building_name", "UTown", "-", "", "", "", "Building key.", "建筑或建筑群名称。"),
            ("local_refine_window_kw", 0, "kW", "", "", "", "PV local refinement window.", "光伏容量局部细化窗口。"),
            ("local_refine_step_kw", 500, "kW", "", "", "", "PV local refinement step.", "光伏容量局部细化步长。"),
            ("rooftop_area_m2", 1000, "m2", "", "", "", "Effective rooftop area available for PV.", "可安装光伏的有效屋顶面积。"),
            ("pv_module_power_density_kw_per_m2", 0.2, "kW/m2", "", "", "", "PV module power density.", "单位面积光伏组件装机功率。"),
            ("pv_om_fraction_per_year", 0.015, "fraction/year", "", "", "", "PV annual O&M.", "光伏年度运维成本占 CAPEX 比例。"),
            ("pv_project_years", 25, "years", "", "", "", "PV project life.", "光伏项目寿命。"),
            ("cooling_cop", 5.0, "-", "", "", "", "Cooling COP.", "制冷系统 COP。"),
            ("heating_cop", 3.5, "-", "", "", "", "Heating COP.", "供热系统 COP。"),
            ("battery_om_fraction_per_year", 0.02, "fraction/year", "", "", "", "Battery O&M.", "电池年度运维成本占 CAPEX 比例。"),
            ("battery_max_c_rate", 0.5, "1/h", "", "", "", "Battery max C-rate.", "电池最大充放电倍率。"),
            ("battery_roundtrip_efficiency", 0.92, "fraction", "", "", "", "Battery round-trip efficiency.", "电池往返效率。"),
            ("battery_soc_min_fraction", 0.10, "fraction", "", "", "", "Minimum SOC.", "电池最小 SOC。"),
            ("battery_soc_max_fraction", 0.95, "fraction", "", "", "", "Maximum SOC.", "电池最大 SOC。"),
            ("battery_initial_soc_fraction", 0.50, "fraction", "", "", "", "Initial SOC.", "初始 SOC。"),
            ("battery_eol_soh_fraction", 0.80, "fraction", "", "", "", "End-of-life SOH.", "电池寿命终止 SOH。"),
            ("battery_rated_cycle_life_to_eol", 9000, "cycles", "", "", "", "Rated cycle life.", "到寿命终止的额定循环次数。"),
            ("battery_calendar_loss_per_year", 0.008, "fraction/year", "", "", "", "Calendar degradation.", "电池日历衰减率。"),
            ("battery_max_replacement_years", 30, "years", "", "", "", "Maximum replacement interval.", "最长更换周期。"),
            ("local_refine_window_kwh", 0, "kWh", "", "", "", "Local refinement window.", "在粗筛最优容量附近继续细化的窗口。"),
            ("local_refine_step_kwh", 500, "kWh", "", "", "", "Local refinement step.", "细化搜索步长。"),
            ("outage_seed", 42, "-", "", "", "", "Outage random seed.", "停电随机生成种子。"),
            ("forced_event_mode", "match_saidi", "-", "", "", "", "Outage event generation mode.", "停电事件生成方式。"),
            ("uncertainty_baseline_scenario_count", 2, "scenarios", "", "", "", "Uncertainty baseline scenario count.", "不确定性基准分析场景数量。"),
            ("uncertainty_sensitivity_scenario_count", 2, "scenarios", "", "", "", "Uncertainty sensitivity scenario count.", "不确定性敏感性分析场景数量。"),
            ("uncertainty_seed_start", 2000, "-", "", "", "", "Uncertainty seed start.", "不确定性停电场景起始随机种子。"),
            ("parallel_workers", 1, "workers", "", "", "", "Parallel workers for uncertainty analysis.", "不确定性分析并行进程数。"),
            ("pv_during_outage_available", "true", "boolean", "", "", "", "PV availability during outage.", "停电期间 PV 是否可用。"),
        ]
        return [dict(zip(PARAMETER_COLUMNS, row, strict=True)) for row in rows]

    def _write_valid_case(self, root: Path) -> Path:
        case_dir = self._case_dir(root)
        load = pd.DataFrame(
            {
                "Cooling": [25.0] * 8760,
                "Heating": [7.0] * 8760,
                "Lighting": [6.0] * 8760,
                "Equipment": [4.0] * 8760,
            }
        )
        pv = pd.DataFrame({"PV_output_kWh_per_m2": [1.0] * 8760})
        tou = pd.DataFrame({"hour": range(24), "tou_sgd_per_kwh": [0.239] * 9 + [0.329] * 12 + [0.239] * 3})
        params = pd.DataFrame(self._parameter_rows(), columns=PARAMETER_COLUMNS)
        load.to_csv(case_dir / "SG_UTown_Load.csv", index=False)
        pv.to_csv(case_dir / "SG_UTown_PV.csv", index=False)
        tou.to_csv(case_dir / "SG_UTown_TOU.csv", index=False)
        params.to_csv(case_dir / "SG_UTown_Parameter.csv", index=False)
        return root / "Input"

    def _fast_config(self) -> dict[str, object]:
        config = deepcopy(DEFAULT_CONFIG)
        config["capacity_search"]["coarse_intervals"] = 2
        config["capacity_search"]["local_refine_divisions"] = 1
        config["capacity_search"]["max_upper_expansions"] = 0
        return config

    def test_input_discovery_finds_four_file_case(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            cases = discover_input_cases(input_dir)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_key, "SG_UTown")

    def test_input_discovery_reports_missing_file(self) -> None:
        with TemporaryDirectory() as tmp:
            case_dir = self._case_dir(Path(tmp))
            pd.DataFrame({"Cooling": [0.0]}).to_csv(case_dir / "SG_UTown_Load.csv", index=False)
            with self.assertRaisesRegex(ValueError, "missing files"):
                discover_input_cases(Path(tmp) / "Input")

    def test_parameter_file_requires_first_six_sensitivity_parameters(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            raw.loc[0, "parameter"] = "voll_sgd_per_kwh"
            raw.to_csv(case.parameter_csv, index=False)
            with self.assertRaisesRegex(ValueError, "first six parameters"):
                load_case_parameters(case.parameter_csv)

    def test_old_fixed_pv_coverage_sensitivity_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            raw.loc[0, "parameter"] = "fixed_pv_coverage_fraction"
            raw.to_csv(case.parameter_csv, index=False)
            with self.assertRaisesRegex(ValueError, "no longer a sensitivity parameter"):
                load_case_parameters(case.parameter_csv)

    def test_removed_capacity_candidate_parameters_are_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            extra = dict(
                zip(
                    PARAMETER_COLUMNS,
                    ("pv_capacity_kw_candidates", "[0,1000]", "kW", "", "", "", "Old PV candidates.", "旧光伏候选列表。"),
                    strict=True,
                )
            )
            raw = pd.concat([raw, pd.DataFrame([extra])], ignore_index=True)
            raw.to_csv(case.parameter_csv, index=False)
            with self.assertRaisesRegex(ValueError, "Removed PV parameter"):
                load_case_parameters(case.parameter_csv)

    def test_adaptive_capacity_search_generates_positive_candidate_axes(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            _, _, cfg, _ = load_input_case(case, DEFAULT_CONFIG)
            self.assertEqual(cfg["pv"]["capacity_kw_candidates"][0], 0.0)
            self.assertEqual(cfg["battery"]["energy_kwh_candidates"][0], 0.0)
            self.assertGreater(max(cfg["pv"]["capacity_kw_candidates"]), 0.0)
            self.assertGreater(max(cfg["battery"]["energy_kwh_candidates"]), 0.0)
            self.assertEqual(len(cfg["pv"]["capacity_kw_candidates"]), 9)
            self.assertEqual(len(cfg["battery"]["energy_kwh_candidates"]), 9)
            self.assertIn("annual_load_e_equiv_kwh", cfg["capacity_search"])

    def test_zero_pv_generation_is_rejected_for_adaptive_search(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            pd.DataFrame({"PV_output_kWh_per_m2": [0.0] * 8760}).to_csv(case.pv_csv, index=False)
            with self.assertRaisesRegex(ValueError, "positive PV_output_kWh_per_m2"):
                load_input_case(case, DEFAULT_CONFIG)

    def test_parameter_parsing_generates_inclusive_sensitivity_ranges(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            params = load_case_parameters(case.parameter_csv)
            self.assertNotIn("battery_energy_kwh_candidates", params.base)
            self.assertNotIn("pv_capacity_kw_candidates", params.base)
            self.assertIs(params.base["pv_during_outage_available"], True)
            self.assertEqual(params.sensitivity["pv_capex_sgd_per_kw"], [1000.0, 1200.0])
            self.assertEqual(params.sensitivity["saifi_per_year"], [0.0, 1.0])
            self.assertEqual(len(sensitivity_runs(params)), 15)

    def test_removed_pv_parameters_are_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            extra = dict(zip(PARAMETER_COLUMNS, ("base_pv_coverage_fraction", 1.0, "fraction", "", "", "", "Old PV base coverage.", "旧基准覆盖率。"), strict=True))
            raw = pd.concat([raw, pd.DataFrame([extra])], ignore_index=True)
            raw.to_csv(case.parameter_csv, index=False)
            with self.assertRaisesRegex(ValueError, "Removed PV parameter"):
                load_case_parameters(case.parameter_csv)

    def test_required_pv_area_parameters_are_validated(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            raw.loc[raw["parameter"] == "rooftop_area_m2", "base_value"] = ""
            raw.to_csv(case.parameter_csv, index=False)
            with self.assertRaisesRegex(ValueError, "rooftop_area_m2 is required"):
                load_input_case(case, DEFAULT_CONFIG)

    def test_uncertainty_parameters_are_required_and_validated(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            raw.loc[raw["parameter"] == "uncertainty_baseline_scenario_count", "base_value"] = ""
            raw.to_csv(case.parameter_csv, index=False)
            with self.assertRaisesRegex(ValueError, "uncertainty_baseline_scenario_count is required"):
                load_input_case(case, DEFAULT_CONFIG)

        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            raw.loc[raw["parameter"] == "uncertainty_sensitivity_scenario_count", "base_value"] = "1.5"
            raw.to_csv(case.parameter_csv, index=False)
            with self.assertRaisesRegex(ValueError, "uncertainty_sensitivity_scenario_count must be a positive integer"):
                load_input_case(case, DEFAULT_CONFIG)

        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            raw = raw[raw["parameter"] != "parallel_workers"].copy()
            raw.to_csv(case.parameter_csv, index=False)
            _, _, cfg, _ = load_input_case(case, DEFAULT_CONFIG)
            self.assertEqual(cfg["uncertainty"]["parallel_workers"], 1)

        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            raw.loc[raw["parameter"] == "parallel_workers", "base_value"] = "0"
            raw.to_csv(case.parameter_csv, index=False)
            with self.assertRaisesRegex(ValueError, "parallel_workers must be a positive integer"):
                load_input_case(case, DEFAULT_CONFIG)

        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            raw.loc[raw["parameter"] == "parallel_workers", "base_value"] = "1.5"
            raw.to_csv(case.parameter_csv, index=False)
            with self.assertRaisesRegex(ValueError, "parallel_workers must be a positive integer"):
                load_input_case(case, DEFAULT_CONFIG)

    def test_load_input_case_builds_8760_frame_and_scales_pv_from_per_m2_input(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            frame, meta, cfg, _ = load_input_case(case, DEFAULT_CONFIG, pv_capacity_kw=200.0)
            self.assertEqual(len(frame), 8760)
            self.assertFalse(frame.isna().any().any())
            self.assertAlmostEqual(float(frame["electric_load_kwh"].iloc[0]), 10.0)
            self.assertAlmostEqual(float(frame["pv_kwh"].iloc[0]), 1000.0)
            self.assertAlmostEqual(float(meta["pv_capacity_kw"]), 200.0)
            self.assertAlmostEqual(float(meta["pv_capacity_mw"]), 0.2)
            self.assertAlmostEqual(float(meta["pv_installed_area_m2"]), 1000.0)
            self.assertAlmostEqual(float(meta["pv_rooftop_coverage_fraction"]), 1.0)
            self.assertAlmostEqual(float(meta["pv_rooftop_coverage_percent"]), 100.0)
            self.assertAlmostEqual(float(meta["pv_rooftop_utilization_fraction"]), 1.0)
            self.assertAlmostEqual(float(meta["pv_module_power_density_kw_per_m2"]), 0.2)
            self.assertAlmostEqual(float(frame["pv_rooftop_coverage_fraction"].iloc[0]), 1.0)
            self.assertAlmostEqual(float(frame["pv_capacity_kw"].iloc[0]), 200.0)
            self.assertEqual(cfg["scenario_order"], ["SG_UTown"])
            self.assertEqual(sorted(frame["tou_sgd_per_kwh"].unique().tolist()), [0.239, 0.329])
            self.assertEqual(len(cfg["pv"]["capacity_kw_candidates"]), 9)
            self.assertEqual(cfg["uncertainty"]["baseline_scenario_count"], 2)
            self.assertEqual(cfg["uncertainty"]["sensitivity_scenario_count"], 2)
            self.assertEqual(cfg["uncertainty"]["seed_start"], 2000)
            self.assertEqual(cfg["uncertainty"]["parallel_workers"], 1)

    def test_pv_capacity_can_exceed_rooftop_coverage_without_error(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            _, meta, _, _ = load_input_case(case, DEFAULT_CONFIG, pv_capacity_kw=1000.0)
            self.assertGreater(float(meta["pv_rooftop_coverage_fraction"]), 1.0)
            self.assertGreater(float(meta["pv_rooftop_utilization_fraction"]), 1.0)

    def test_joint_pv_battery_optimization_returns_pv_capacity_and_battery(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            _, _, cfg, _ = load_input_case(case, self._fast_greedy_config())

            def pv_capacity_frame_builder(pv_capacity_kw: float):
                frame, meta = build_case_frame(case, cfg, pv_capacity_kw)
                frame["grid_availability"] = 1.0
                frame["outage_fraction"] = 0.0
                return frame, meta

            rows, dispatch, summary, candidate_dispatches = resilience_design(pv_capacity_frame_builder, cfg, "SG_UTown")
            self.assertEqual(len(rows), 1)
            self.assertIn("pv_capacity_kw", rows.columns)
            self.assertNotIn("pv_coverage_fraction", rows.columns)
            self.assertIn("battery_energy_kwh", rows.columns)
            self.assertIn("pv_bess_system_annual_cost_sgd", rows.columns)
            self.assertFalse(any(column.startswith("hess_") for column in rows.columns))
            self.assertIn("hour_index", dispatch.columns)
            self.assertIn("pv_capacity_kw", summary.columns)
            self.assertEqual(len(candidate_dispatches), len(summary))
            one_dispatch = next(iter(candidate_dispatches.values()))
            for column in ("search_stage", "search_iteration", "pv_capacity_kw", "battery_energy_kwh"):
                self.assertIn(column, one_dispatch.columns)

    def test_pv_annual_cost(self) -> None:
        meta = {"pv_capacity_kw": 1000.0}
        cost = pv_annual_cost(meta, DEFAULT_CONFIG)
        self.assertAlmostEqual(cost["pv_capex_sgd"], 1_200_000.0)
        self.assertAlmostEqual(cost["annualized_pv_cost_sgd"], 48_000.0)
        self.assertAlmostEqual(cost["pv_om_cost_sgd"], 18_000.0)

    def test_outage_free_frame_removes_outage(self) -> None:
        frame = pd.DataFrame({"grid_availability": [0.0, 0.5], "outage_fraction": [1.0, 0.5]})
        no_outage = outage_free_frame(frame)
        self.assertEqual(no_outage["grid_availability"].tolist(), [1.0, 1.0])
        self.assertEqual(no_outage["outage_fraction"].tolist(), [0.0, 0.0])

    def test_uncertainty_outage_seed_is_reproducible_and_saidi_matches(self) -> None:
        profile_a, events_a = simulate_outages(2.0, 120.0, seed=123)
        profile_b, events_b = simulate_outages(2.0, 120.0, seed=123)
        profile_c, events_c = simulate_outages(2.0, 120.0, seed=124)
        pd.testing.assert_frame_equal(profile_a, profile_b)
        pd.testing.assert_frame_equal(events_a, events_b)
        self.assertFalse(events_a.equals(events_c))
        summary = summarize_outages(profile_a, events_a)
        self.assertAlmostEqual(summary["simulated_saidi_min_per_year"], 120.0, places=6)
        self.assertEqual(summary["simulated_saifi_per_year"], 2.0)

    def test_dispatch_and_no_bess_reference_include_heating_load(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            frame, meta, cfg, _ = load_input_case(case, DEFAULT_CONFIG, pv_capacity_kw=0.0)
            resolved = scenario_config(cfg, "SG_UTown")
            two_hours = frame.head(2).copy()
            two_hours["grid_availability"] = [1.0, 1.0]
            two_hours["outage_fraction"] = [0.0, 0.0]
            sizing = {"battery_energy_kwh": 0.0, "battery_power_kw": 0.0}
            dispatch = solve_dispatch_greedy(two_hours, sizing, resolved)
            expected = 10.0 + 25.0 / 5.0 + 7.0 / 3.5
            self.assertAlmostEqual(float(dispatch["total_load_e_equiv_kwh"].iloc[0]), expected)
            result, _ = evaluate_candidate(two_hours, 0.0, meta, resolved)
            self.assertGreater(result["no_bess_reference_grid_energy_kwh"], 0.0)
            self.assertIn("no_bess_reference_private_annual_cost_sgd", result)
            self.assertIn("no_bess_reference_system_annual_cost_sgd", result)
            self.assertAlmostEqual(result["no_bess_reference_grid_energy_kwh"], result["pv_bess_grid_energy_kwh"])
            self.assertAlmostEqual(result["no_bess_reference_grid_cost_sgd"], result["pv_bess_grid_cost_sgd"])
            self.assertAlmostEqual(
                result["no_bess_reference_private_annual_cost_sgd"],
                result["pv_bess_private_annual_cost_sgd"],
            )
            self.assertAlmostEqual(
                result["no_bess_reference_unserved_energy_kwh_per_year"],
                result["unserved_energy_kwh_per_year"],
            )
            self.assertAlmostEqual(
                result["no_bess_reference_system_annual_cost_sgd"],
                result["pv_bess_system_annual_cost_sgd"],
            )

    def _small_dispatch_config(self, backend: str = "milp", include_degradation_penalty: bool = True) -> dict[str, object]:
        return {
            "technical": {"cooling_cop": 5.0, "heating_cop": 3.5, "pv_during_outage_available": True},
            "battery": {
                "roundtrip_efficiency": 1.0,
                "max_c_rate": 0.5,
                "initial_soc_fraction": 0.5,
                "soc_min_fraction": 0.1,
                "soc_max_fraction": 0.95,
                "eol_soh_fraction": 0.8,
                "rated_cycle_life_to_eol": 9000.0,
                "calendar_loss_per_year": 0.008,
                "max_replacement_years": 30.0,
            },
            "economics": {
                "battery_capex_sgd_per_kwh": 100.0,
                "battery_power_capex_sgd_per_kw": 10.0,
                "battery_om_fraction_per_year": 0.02,
            },
            "active_scenario": {"voll_sgd_per_kwh": 100.0},
            "dispatch": {
                "backend": backend,
                "solver": "HiGHS",
                "include_degradation_penalty": include_degradation_penalty,
            },
        }

    def test_milp_dispatch_blocks_grid_import_during_full_outage(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0],
                "grid_availability": [0.0],
                "tou_sgd_per_kwh": [1.0],
                "pv_kwh": [0.0],
                "electric_load_kwh": [10.0],
                "cooling_load_kwh_th": [0.0],
                "heating_load_kwh_th": [0.0],
            }
        )
        sizing = {"battery_energy_kwh": 0.0, "battery_power_kw": 0.0}
        dispatch = solve_dispatch_milp(frame, sizing, self._small_dispatch_config())
        self.assertAlmostEqual(float(dispatch["grid_import_kwh"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(dispatch["unmet_load_kwh_equiv"].iloc[0]), 10.0)
        self.assertEqual(dispatch["dispatch_backend"].unique().tolist(), ["milp"])
        self.assertAlmostEqual(float(dispatch["simultaneous_charge_discharge_kwh"].sum()), 0.0)

    def test_auto_dispatch_uses_milp_when_available(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0, 1],
                "grid_availability": [1.0, 1.0],
                "tou_sgd_per_kwh": [0.10, 0.20],
                "pv_kwh": [0.0, 0.0],
                "electric_load_kwh": [0.0, 10.0],
                "cooling_load_kwh_th": [0.0, 0.0],
                "heating_load_kwh_th": [0.0, 0.0],
            }
        )
        sizing = {"battery_energy_kwh": 100.0, "battery_power_kw": 100.0}
        dispatch = solve_dispatch(frame, sizing, self._small_dispatch_config(backend="auto"))
        self.assertEqual(dispatch["dispatch_backend"].unique().tolist(), ["milp"])
        self.assertAlmostEqual(float(dispatch["simultaneous_charge_discharge_kwh"].sum()), 0.0)

    def test_auto_dispatch_falls_back_to_greedy_when_milp_fails(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0],
                "grid_availability": [1.0],
                "tou_sgd_per_kwh": [0.10],
                "pv_kwh": [0.0],
                "electric_load_kwh": [1.0],
                "cooling_load_kwh_th": [0.0],
                "heating_load_kwh_th": [0.0],
            }
        )
        sizing = {"battery_energy_kwh": 0.0, "battery_power_kw": 0.0}
        with patch("src.dispatch.solve_dispatch_milp", side_effect=RuntimeError("solver unavailable")):
            dispatch = solve_dispatch(frame, sizing, self._small_dispatch_config(backend="auto"))
        self.assertEqual(dispatch["dispatch_backend"].unique().tolist(), ["greedy"])
        self.assertEqual(dispatch["solver_status"].unique().tolist(), ["heuristic"])

    def test_explicit_milp_dispatch_does_not_fall_back_to_greedy(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0],
                "grid_availability": [1.0],
                "tou_sgd_per_kwh": [0.10],
                "pv_kwh": [0.0],
                "electric_load_kwh": [1.0],
                "cooling_load_kwh_th": [0.0],
                "heating_load_kwh_th": [0.0],
            }
        )
        sizing = {"battery_energy_kwh": 0.0, "battery_power_kw": 0.0}
        with patch("src.dispatch.solve_dispatch_milp", side_effect=RuntimeError("solver unavailable")):
            with self.assertRaisesRegex(RuntimeError, "solver unavailable"):
                solve_dispatch(frame, sizing, self._small_dispatch_config(backend="milp"))

    def test_cli_dispatch_backend_choices_exclude_benchmark(self) -> None:
        self.assertEqual(DISPATCH_BACKEND_CHOICES, ["auto", "greedy", "milp"])

    def test_milp_degradation_penalty_reduces_unnecessary_cycling(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0, 1],
                "grid_availability": [1.0, 1.0],
                "tou_sgd_per_kwh": [0.10, 0.20],
                "pv_kwh": [0.0, 0.0],
                "electric_load_kwh": [0.0, 10.0],
                "cooling_load_kwh_th": [0.0, 0.0],
                "heating_load_kwh_th": [0.0, 0.0],
            }
        )
        sizing = {"battery_energy_kwh": 100.0, "battery_power_kw": 100.0}
        no_penalty = solve_dispatch_milp(frame, sizing, self._small_dispatch_config(include_degradation_penalty=False))
        high_penalty_cfg = self._small_dispatch_config(include_degradation_penalty=True)
        high_penalty_cfg["economics"]["battery_capex_sgd_per_kwh"] = 10000.0
        with_penalty = solve_dispatch_milp(frame, sizing, high_penalty_cfg)
        self.assertGreater(float(no_penalty["battery_discharge_kwh"].sum()), 0.0)
        self.assertAlmostEqual(float(with_penalty["battery_discharge_kwh"].sum()), 0.0)

    def test_candidate_evaluator_reuses_cached_candidate_result(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0],
                "grid_availability": [1.0],
                "tou_sgd_per_kwh": [0.10],
                "pv_kwh": [0.0],
                "electric_load_kwh": [1.0],
                "cooling_load_kwh_th": [0.0],
                "heating_load_kwh_th": [0.0],
            }
        )
        config = self._small_dispatch_config(backend="greedy")
        config["pv"] = {"capex_sgd_per_kw": 1200.0, "om_fraction_per_year": 0.015, "project_years": 25.0}
        evaluator = CandidateEvaluator(config, CandidateStore(retain_dispatch=True))
        pv_meta = {"pv_capacity_kw": 0.0, "pv_capacity_mw": 0.0}
        with patch("src.optimizer.solve_dispatch", wraps=solve_dispatch) as solve:
            first, first_dispatch = evaluator.evaluate(frame, 0.0, pv_meta)
            second, second_dispatch = evaluator.evaluate(frame, 0.0, pv_meta)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(first["pv_bess_system_annual_cost_sgd"], second["pv_bess_system_annual_cost_sgd"])
        self.assertIsNotNone(first_dispatch)
        self.assertIsNotNone(second_dispatch)
        self.assertIsNot(first_dispatch, second_dispatch)

    def test_candidate_evaluator_invalidates_cache_when_context_changes(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0],
                "grid_availability": [0.0],
                "tou_sgd_per_kwh": [0.10],
                "pv_kwh": [0.0],
                "electric_load_kwh": [1.0],
                "cooling_load_kwh_th": [0.0],
                "heating_load_kwh_th": [0.0],
            }
        )
        base_config = self._small_dispatch_config(backend="greedy")
        base_config["pv"] = {"capex_sgd_per_kw": 1200.0, "om_fraction_per_year": 0.015, "project_years": 25.0}
        base_config["active_scenario"] = {"voll_sgd_per_kwh": 100.0}
        store = CandidateStore(retain_dispatch=True)
        pv_meta = {"pv_capacity_kw": 0.0, "pv_capacity_mw": 0.0}
        with patch("src.optimizer.solve_dispatch", wraps=solve_dispatch) as solve:
            CandidateEvaluator(base_config, store).evaluate(frame, 0.0, pv_meta)
            voll_config = deepcopy(base_config)
            voll_config["active_scenario"] = {"voll_sgd_per_kwh": 200.0}
            CandidateEvaluator(voll_config, store).evaluate(frame, 0.0, pv_meta)
            outage_frame = frame.copy()
            outage_frame["grid_availability"] = [1.0]
            CandidateEvaluator(base_config, store).evaluate(outage_frame, 0.0, pv_meta)
            reserve_config = deepcopy(base_config)
            reserve_config["battery"]["normal_operation_reserve_fraction"] = 0.5
            CandidateEvaluator(reserve_config, store).evaluate(frame, 0.0, pv_meta)
            backend_config = deepcopy(base_config)
            backend_config["dispatch"]["backend"] = "auto"
            CandidateEvaluator(backend_config, store).evaluate(frame, 0.0, pv_meta)
        self.assertEqual(solve.call_count, 5)

    def test_candidate_evaluator_reuses_no_bess_reference_for_same_pv_context(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0],
                "grid_availability": [0.0],
                "tou_sgd_per_kwh": [0.10],
                "pv_kwh": [0.0],
                "electric_load_kwh": [1.0],
                "cooling_load_kwh_th": [0.0],
                "heating_load_kwh_th": [0.0],
            }
        )
        config = self._small_dispatch_config(backend="greedy")
        config["pv"] = {"capex_sgd_per_kw": 1200.0, "om_fraction_per_year": 0.015, "project_years": 25.0}
        evaluator = CandidateEvaluator(config, CandidateStore(retain_dispatch=False))
        pv_meta = {"pv_capacity_kw": 0.0, "pv_capacity_mw": 0.0}
        with patch("src.optimizer.solve_dispatch", wraps=solve_dispatch) as solve:
            first, _ = evaluator.evaluate(frame, 100.0, pv_meta, require_dispatch=False)
            second, _ = evaluator.evaluate(frame, 200.0, pv_meta, require_dispatch=False)
        self.assertEqual(solve.call_count, 3)
        self.assertAlmostEqual(first["no_bess_reference_unserved_energy_kwh_per_year"], 1.0)
        self.assertAlmostEqual(second["no_bess_reference_unserved_energy_kwh_per_year"], 1.0)
        self.assertAlmostEqual(first["avoided_unserved_energy_kwh_per_year"], 1.0 - first["unserved_energy_kwh_per_year"])
        self.assertAlmostEqual(
            first["system_annual_cost_delta_vs_no_bess_reference_sgd"],
            first["pv_bess_system_annual_cost_sgd"] - first["no_bess_reference_system_annual_cost_sgd"],
        )

    def test_resilience_design_uses_unserved_energy_cost(self) -> None:
        config = {
            "scenario_order": ["SG_UTown"],
            "scenarios": {
                "SG_UTown": {
                    "name": "SG UTown",
                    "city_or_service_area": "SG",
                    "saidi_min_per_year": 60.0,
                    "saifi_per_year": 1.0,
                    "voll_sgd_per_kwh": 100.0,
                }
            },
            "economics": {
                "battery_capex_sgd_per_kwh": 450.0,
                "battery_power_capex_sgd_per_kw": 180.0,
                "battery_om_fraction_per_year": 0.02,
            },
            "battery": {
                "energy_kwh_candidates": [0.0],
                "local_refine_window_kwh": 0.0,
                "local_refine_step_kwh": 500.0,
                "max_c_rate": 0.5,
                "roundtrip_efficiency": 0.92,
                "soc_min_fraction": 0.1,
                "soc_max_fraction": 0.95,
                "initial_soc_fraction": 0.5,
                "eol_soh_fraction": 0.8,
                "rated_cycle_life_to_eol": 9000.0,
                "calendar_loss_per_year": 0.008,
                "max_replacement_years": 30.0,
            },
            "pv": {
                "capacity_kw_candidates": [0.0],
                "local_refine_window_kw": 0.0,
                "local_refine_step_kw": 500.0,
                "capex_sgd_per_kw": 1200.0,
                "om_fraction_per_year": 0.015,
                "project_years": 25.0,
            },
            "technical": {"cooling_cop": 5.0, "heating_cop": 3.5},
        }

        def outage_builder(pv_capacity_kw: float):
            frame = pd.DataFrame(
                {
                    "hour_index": [0],
                    "electric_load_kwh": [10.0],
                    "cooling_load_kwh_th": [0.0],
                    "heating_load_kwh_th": [0.0],
                    "pv_kwh": [0.0],
                    "tou_sgd_per_kwh": [1.0],
                    "grid_availability": [0.0],
                    "outage_fraction": [1.0],
                }
            )
            return frame, {"pv_capacity_kw": pv_capacity_kw, "pv_capacity_mw": pv_capacity_kw / 1000.0}

        rows, _, _, _ = resilience_design(outage_builder, config, "SG_UTown")
        design = rows.iloc[0]
        self.assertEqual(float(design["optimization_voll_sgd_per_kwh"]), 100.0)
        self.assertEqual(float(design["evaluation_voll_sgd_per_kwh"]), 100.0)
        self.assertAlmostEqual(float(design["unserved_energy_kwh_per_year"]), 10.0)
        self.assertAlmostEqual(float(design["unserved_energy_cost_sgd_per_year"]), 1000.0)
        self.assertAlmostEqual(
            float(design["pv_bess_system_annual_cost_sgd"]),
            float(design["pv_bess_private_annual_cost_sgd"]) + 1000.0,
        )

    def test_battery_cost_uses_input_parameter_economics(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            _, _, cfg, _ = load_input_case(case, DEFAULT_CONFIG)
            resolved = scenario_config(cfg, "SG_UTown")
            cost = annualized_battery_cost(1000.0, 500.0, 0.0, resolved)
            self.assertAlmostEqual(cost["battery_capex_sgd"], 540_000.0)
            self.assertAlmostEqual(cost["battery_capex_sgd_per_kwh"], 450.0)
            self.assertAlmostEqual(cost["battery_power_capex_sgd_per_kw"], 180.0)

    def test_uncertainty_summary_exports_distribution_statistics_without_regret(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "pv_capacity_kw": 1000.0,
                    "battery_energy_kwh": 0.0,
                    "pv_bess_private_annual_cost_sgd": 10.0,
                    "unserved_energy_kwh_per_year": 3.0,
                    "unserved_energy_cost_sgd_per_year": 6.0,
                    "pv_bess_system_annual_cost_sgd": 16.0,
                },
                {
                    "pv_capacity_kw": 1000.0,
                    "battery_energy_kwh": 0.0,
                    "pv_bess_private_annual_cost_sgd": 11.0,
                    "unserved_energy_kwh_per_year": 4.0,
                    "unserved_energy_cost_sgd_per_year": 8.0,
                    "pv_bess_system_annual_cost_sgd": 19.0,
                },
            ]
        )
        summary = summarize_uncertainty_results(rows)
        self.assertIn("mean", summary.columns)
        self.assertIn("median", summary.columns)
        self.assertIn("p10", summary.columns)
        self.assertIn("p25", summary.columns)
        self.assertIn("p75", summary.columns)
        self.assertIn("p90", summary.columns)
        self.assertFalse(any("regret" in column.lower() for column in summary.columns))

    def test_uncertainty_parallel_workers_match_serial_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]

            raw = pd.read_csv(case.parameter_csv)
            raw.loc[raw["parameter"] == "parallel_workers", "base_value"] = "1"
            raw.to_csv(case.parameter_csv, index=False)
            _, _, serial_cfg, _ = load_input_case(case, self._fast_config())
            serial_results, serial_scenarios, _ = run_uncertainty_analysis(
                case,
                serial_cfg,
                "SG_UTown",
                scenario_count=2,
                seed_start=2000,
            )

            raw = pd.read_csv(case.parameter_csv)
            raw.loc[raw["parameter"] == "parallel_workers", "base_value"] = "2"
            raw.to_csv(case.parameter_csv, index=False)
            _, _, parallel_cfg, _ = load_input_case(case, self._fast_config())
            parallel_results, parallel_scenarios, _ = run_uncertainty_analysis(
                case,
                parallel_cfg,
                "SG_UTown",
                scenario_count=2,
                seed_start=2000,
            )

            self.assertEqual(parallel_cfg["uncertainty"]["parallel_workers"], 2)
            pd.testing.assert_frame_equal(serial_scenarios, parallel_scenarios)
            self.assertEqual(serial_results["uncertainty_scenario_id"].tolist(), parallel_results["uncertainty_scenario_id"].tolist())
            pd.testing.assert_series_equal(
                serial_results["pv_bess_system_annual_cost_sgd"],
                parallel_results["pv_bess_system_annual_cost_sgd"],
                check_names=False,
            )

    def test_candidate_dispatch_workbook_has_dispatch_and_parameter_sheets(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deepcopy(DEFAULT_CONFIG)
            config["scenarios"] = {
                "SG_UTown": {
                    "name": "SG UTown",
                    "city_or_service_area": "SG",
                    "saidi_min_per_year": 60.0,
                    "saifi_per_year": 1.0,
                    "voll_sgd_per_kwh": 50.0,
                }
            }
            summary = pd.DataFrame(
                [
                    {
                        "search_stage": "coarse",
                        "search_iteration": 0,
                        "pv_capacity_kw": 10.0,
                        "battery_energy_kwh": 20.0,
                        "battery_power_kw": 10.0,
                        "battery_c_rate": 0.5,
                        "pv_bess_private_annual_cost_sgd": 100.0,
                        "unserved_energy_kwh_per_year": 2.0,
                        "unserved_energy_cost_sgd_per_year": 100.0,
                        "pv_bess_system_annual_cost_sgd": 200.0,
                    }
                ]
            )
            dispatch = pd.DataFrame(
                {
                    "hour_index": [0],
                    "grid_availability": [1.0],
                    "outage_fraction": [0.0],
                    "tou_sgd_per_kwh": [0.25],
                    "total_load_e_equiv_kwh": [1.0],
                    "pv_available_kwh": [0.5],
                    "pv_used_kwh": [0.5],
                    "pv_curtailed_kwh": [0.0],
                    "grid_import_kwh": [0.5],
                    "battery_charge_kwh": [0.0],
                    "battery_discharge_kwh": [0.0],
                    "battery_soc_kwh": [10.0],
                    "battery_capacity_kwh": [20.0],
                    "unmet_load_kwh_equiv": [0.0],
                }
            )
            paths = write_candidate_dispatch_workbooks(
                root,
                "SG_UTown",
                config,
                {("coarse", 0, 10.0, 20.0): dispatch},
                summary,
                sensitivity_parameter="voll_sgd_per_kwh",
                sensitivity_value=50.0,
            )
            workbook = next(iter(paths.values()))
            self.assertEqual(workbook.suffix, ".xlsx")
            dispatch_sheet = pd.read_excel(workbook, sheet_name="Dispatch_8760")
            parameter_sheet = pd.read_excel(workbook, sheet_name="Parameters")
            self.assertIn("grid_import_kwh", dispatch_sheet.columns)
            self.assertIn("sensitivity_parameter", dispatch_sheet.columns)
            self.assertEqual(parameter_sheet.columns.tolist(), ["category", "parameter", "value", "unit", "description"])
            self.assertIn("pv_bess_system_annual_cost_sgd", parameter_sheet["parameter"].tolist())

    def test_run_all_writes_case_output_structure_and_key_figures(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = self._write_valid_case(root)
            output_dir = root / "outputs"
            config = self._fast_config()
            config.setdefault("dispatch", {})["backend"] = "greedy"
            lolp_slice_fixture = pd.DataFrame(
                [
                    {
                        "pv_slice_label": label,
                        "pv_slice_rank": rank,
                        "pv_slice_target_mw": pv,
                        "pv_slice_selected_mw": pv,
                        "pv_slice_source": "fixed_8_20_32_mw",
                        "pv_capacity_kw": pv * 1000.0,
                        "pv_capacity_mw": pv,
                        "battery_energy_kwh": bess * 1000.0,
                        "battery_energy_mwh": bess,
                        "energy_based_LOLP": max(0.0, 0.4 - 0.005 * pv - 0.01 * bess),
                        "UE_kwh_per_year": 1000.0,
                        "UE_mwh_per_year": 1.0,
                        "system_total_cost_sgd_per_year": 1_000_000.0,
                        "system_total_cost_million_sgd_per_year": 1.0,
                    }
                    for label, rank, pv in (("Low PV", 1, 8.0), ("Medium PV", 2, 20.0), ("High PV", 3, 32.0))
                    for bess in (0.0, 5.0, 10.0)
                ]
            )
            with (
                patch("src.app.workflow.make_baseline_design_space_figures", return_value={}),
                patch("src.app.workflow.build_baseline_lolp_bess_slice_source", return_value=lolp_slice_fixture),
            ):
                paths = run_selected(
                    config,
                    input_dir,
                    output_dir,
                    run_baseline=True,
                    run_sensitivity=True,
                    sensitivity_parameters=["battery_capex_sgd_per_kwh", "saidi_min_per_year"],
                )
            baseline = output_dir / "SG" / "SG_UTown_Baseline"
            sensitivity = output_dir / "SG" / "SG_UTown_Sensitive_Analysis"
            self.assertTrue(baseline.is_dir())
            self.assertTrue(sensitivity.is_dir())
            self.assertTrue((baseline / "SG_UTown_Baseline_Optimal_Design.csv").exists())
            old_design_path = baseline / ("SG_UTown_Baseline_Design" + "_Comparison.csv")
            self.assertFalse(old_design_path.exists())
            lolp_slice_source = (
                baseline
                / "SG_UTown_Baseline_Result_Figures"
                / "SG_UTown_Figure_X_Baseline_LOLP_BESS_Slices_Source_Data.csv"
            )
            self.assertTrue(lolp_slice_source.exists())
            lolp_slice = pd.read_csv(lolp_slice_source)
            self.assertEqual(lolp_slice["pv_capacity_mw"].drop_duplicates().tolist(), [8.0, 20.0, 32.0])
            self.assertEqual(set(lolp_slice["pv_slice_label"]), {"Low PV", "Medium PV", "High PV"})
            self.assertNotIn("chance_constraint_probability", lolp_slice.columns)
            self.assertIn("SG_UTown_baseline_lolp_bess_slices_source_data", paths)
            design = pd.read_csv(baseline / "SG_UTown_Baseline_Optimal_Design.csv")
            self.assertIn("pv_bess_system_annual_cost_sgd", design.columns)
            self.assertIn("pv_capacity_kw", design.columns)
            self.assertNotIn("design_case", design.columns)
            self.assertNotIn("pv_coverage_fraction", design.columns)
            self.assertFalse(any(column.startswith("hess_") for column in design.columns))
            candidate = pd.read_csv(baseline / "SG_UTown_Baseline_Candidate_Summary.csv")
            self.assertIn("search_stage", candidate.columns)
            self.assertIn("coarse", candidate["search_stage"].unique().tolist())
            self.assertIn("optimal_region", candidate["search_stage"].unique().tolist())
            candidate_dispatch_dir = baseline / "Candidate_Dispatch"
            self.assertFalse(candidate_dispatch_dir.exists())
            self.assertFalse(any(path.suffix == ".png" for path in baseline.iterdir()))
            sensitivity_all = sensitivity / "SG_UTown_Sensitive_Analysis_All_Results.csv"
            self.assertTrue(sensitivity_all.exists())
            sensitivity_results = pd.read_csv(sensitivity_all)
            self.assertEqual(
                sorted(sensitivity_results["sensitivity_parameter"].unique().tolist()),
                ["battery_capex_sgd_per_kwh", "saidi_min_per_year"],
            )
            self.assertNotIn("design_case", sensitivity_results.columns)
            self.assertIn("UE_mwh_per_year", sensitivity_results.columns)
            self.assertIn("system_total_cost_million_sgd_per_year", sensitivity_results.columns)
            self.assertIn("energy_based_LOLP", sensitivity_results.columns)
            self.assertFalse(any(column.startswith("hess_") for column in sensitivity_results.columns))
            sensitivity_dispatch_dir = (
                sensitivity / "SG_UTown_Sensitive_Analysis_battery_capex_sgd_per_kwh" / "Candidate_Dispatch"
            )
            self.assertFalse(sensitivity_dispatch_dir.exists())
            versions = output_dir / "SG" / "SG_UTown_Figure_Versions"
            self.assertTrue((versions / "Figure_Version_Manifest.csv").exists())
            self.assertTrue((versions / "Figure_Version_2" / "SG_UTown_Figure_4A_Baseline_Cost_Comparison.png").exists())
            self.assertTrue((versions / "Figure_Version_2" / "SG_UTown_Figure_4B_Baseline_Cost_Comparison.png").exists())
            self.assertTrue(
                (
                    versions
                    / "Figure_Version_2"
                    / "SG_UTown_Figure_5_Baseline_Outage_Energy_Balance_Comparison.png"
                ).exists()
            )
            self.assertTrue(
                (
                    versions
                    / "Figure_Version_1"
                    / "SG_UTown_Figure_X_Baseline_LOLP_BESS_Slices_Source_Data.csv"
                ).exists()
            )

    def test_run_selected_writes_candidate_dispatch_workbooks_when_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = self._write_valid_case(root)
            output_dir = root / "outputs"
            config = self._fast_config()
            config.setdefault("dispatch", {})["backend"] = "greedy"
            config.setdefault("outputs", {})["save_candidate_dispatch_workbooks"] = True
            with (
                patch("src.app.workflow.make_baseline_design_space_figures", return_value={}),
                patch("src.app.workflow.build_baseline_lolp_bess_slice_source", return_value=pd.DataFrame()),
                patch("src.app.workflow.make_baseline_lolp_bess_slice_figure", return_value={}),
                patch(
                    "src.app.outputs.write_candidate_dispatch_workbooks",
                    return_value={"candidate_dispatch_probe": output_dir / "probe.xlsx"},
                ) as write_workbooks,
            ):
                paths = run_selected(
                    config,
                    input_dir,
                    output_dir,
                    run_baseline=True,
                    run_sensitivity=True,
                    sensitivity_parameters=["battery_capex_sgd_per_kwh"],
                )
            self.assertGreaterEqual(write_workbooks.call_count, 2)
            self.assertTrue(any(key.endswith("candidate_dispatch_probe") for key in paths))

    def test_run_selected_writes_uncertainty_analysis_outputs_without_regret(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = self._write_valid_case(root)
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            raw.loc[raw["parameter"] == "parallel_workers", "base_value"] = "2"
            raw.to_csv(case.parameter_csv, index=False)
            output_dir = root / "outputs"
            paths = run_selected(
                self._fast_config(),
                input_dir,
                output_dir,
                run_baseline=False,
                run_sensitivity=False,
                run_uncertainty_baseline=True,
                case_filters=["SG_UTown"],
            )
            uncertainty_root = output_dir / "SG" / "SG_UTown_Uncertainty_Analysis"
            uncertainty = uncertainty_root / "SG_UTown_Uncertainty_Baseline"
            self.assertTrue(uncertainty_root.is_dir())
            self.assertTrue(uncertainty.is_dir())
            self.assertFalse((uncertainty_root / "SG_UTown_Uncertainty_Sensitive_Analysis").exists())
            all_results_path = uncertainty / "SG_UTown_Uncertainty_Baseline_All_Results.csv"
            scenarios_path = uncertainty / "SG_UTown_Uncertainty_Baseline_Outage_Scenarios.csv"
            summary_path = uncertainty / "SG_UTown_Uncertainty_Baseline_Design_Summary.csv"
            for path in (all_results_path, scenarios_path, summary_path):
                self.assertTrue(path.exists())
            all_results = pd.read_csv(all_results_path)
            outage_scenarios = pd.read_csv(scenarios_path)
            summary = pd.read_csv(summary_path)
            self.assertEqual(len(all_results), 2)
            self.assertEqual(len(outage_scenarios), 2)
            self.assertNotIn("design_case", all_results.columns)
            self.assertFalse(
                all_results["optimization_reused_across_uncertainty_scenarios"].astype(bool).any()
            )
            self.assertIn("p10", summary.columns)
            self.assertIn("p90", summary.columns)
            self.assertFalse(any("regret" in column.lower() for column in all_results.columns))
            self.assertFalse(any("regret" in path.name.lower() for path in uncertainty_root.rglob("*")))
            self.assertFalse(any("Outage_Weighted" in path.name for path in uncertainty_root.rglob("*")))
            expected_figures = [
                "SG_UTown_Uncertainty_Baseline_Distribution_Violins",
                "SG_UTown_Uncertainty_Baseline_Capacity_Cost_Bubble",
            ]
            for stem in expected_figures:
                self.assertTrue((uncertainty / f"{stem}.png").exists())
                self.assertTrue((uncertainty / f"{stem}.svg").exists())
                self.assertTrue((uncertainty / f"{stem}.pdf").exists())
                self.assertTrue((uncertainty / f"{stem}_Source_Data.csv").exists())
            self.assertFalse(any(path.suffix == ".tiff" for path in uncertainty_root.rglob("*")))
            self.assertGreater(len(paths), 0)

    def test_run_selected_writes_uncertainty_sensitivity_band_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = self._write_valid_case(root)
            output_dir = root / "outputs"
            paths = run_selected(
                self._fast_config(),
                input_dir,
                output_dir,
                run_baseline=False,
                run_sensitivity=False,
                run_uncertainty_sensitivity=True,
                case_filters=["SG_UTown"],
                uncertainty_sensitivity_parameters=["battery_capex_sgd_per_kwh"],
            )
            uncertainty_root = output_dir / "SG" / "SG_UTown_Uncertainty_Analysis"
            sensitivity = uncertainty_root / "SG_UTown_Uncertainty_Sensitive_Analysis"
            param_dir = sensitivity / "SG_UTown_Uncertainty_Sensitive_Analysis_battery_capex_sgd_per_kwh"
            self.assertTrue(sensitivity.is_dir())
            self.assertTrue(param_dir.is_dir())
            self.assertFalse((uncertainty_root / "SG_UTown_Uncertainty_Baseline").exists())
            all_results = pd.read_csv(sensitivity / "SG_UTown_Uncertainty_Sensitive_Analysis_All_Results.csv")
            self.assertEqual(sorted(all_results["sensitivity_parameter"].unique().tolist()), ["battery_capex_sgd_per_kwh"])
            self.assertNotIn("design_case", all_results.columns)
            self.assertEqual(sorted(all_results["sensitivity_value"].unique().tolist()), [350.0, 450.0, 550.0])
            self.assertFalse(any("Outage_Weighted" in path.name for path in uncertainty_root.rglob("*")))
            self.assertFalse(any("regret" in path.name.lower() for path in uncertainty_root.rglob("*")))
            self.assertFalse(any(path.suffix == ".png" for path in uncertainty_root.rglob("*")))
            self.assertFalse(any(path.suffix in {".svg", ".pdf", ".tiff"} for path in uncertainty_root.rglob("*")))
            self.assertGreater(len(paths), 0)

    def test_run_selected_can_run_only_one_sensitivity_parameter(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = self._write_valid_case(root)
            output_dir = root / "outputs"
            run_selected(
                self._fast_greedy_config(),
                input_dir,
                output_dir,
                run_baseline=False,
                run_sensitivity=True,
                case_filters=["SG_UTown"],
                sensitivity_parameters=["voll_sgd_per_kwh"],
            )
            sensitivity = output_dir / "SG" / "SG_UTown_Sensitive_Analysis"
            all_results = pd.read_csv(sensitivity / "SG_UTown_Sensitive_Analysis_All_Results.csv")
            self.assertEqual(sorted(all_results["sensitivity_parameter"].unique().tolist()), ["voll_sgd_per_kwh"])
            self.assertNotIn("design_case", all_results.columns)
            self.assertTrue((sensitivity / "SG_UTown_Sensitive_Analysis_voll_sgd_per_kwh").is_dir())
            self.assertFalse((sensitivity / "SG_UTown_Sensitive_Analysis_battery_capex_sgd_per_kwh").exists())

    def test_sensitivity_forces_greedy_backend_even_when_global_backend_is_milp(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = self._write_valid_case(root)
            output_dir = root / "outputs"
            config = self._fast_config()
            config.setdefault("dispatch", {})["backend"] = "milp"
            run_selected(
                config,
                input_dir,
                output_dir,
                run_baseline=False,
                run_sensitivity=True,
                case_filters=["SG_UTown"],
                sensitivity_parameters=["battery_capex_sgd_per_kwh"],
            )
            sensitivity = output_dir / "SG" / "SG_UTown_Sensitive_Analysis"
            all_results = pd.read_csv(sensitivity / "SG_UTown_Sensitive_Analysis_All_Results.csv")
            self.assertEqual(sorted(all_results["dispatch_backend"].unique().tolist()), ["greedy"])
            self.assertNotIn("design_case", all_results.columns)

    def test_run_selected_reports_unknown_case_and_sensitivity_parameter(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            with self.assertRaisesRegex(ValueError, "Unknown case"):
                run_selected(DEFAULT_CONFIG, input_dir, Path(tmp) / "outputs", case_filters=["Missing_Case"])
            with self.assertRaisesRegex(ValueError, "Unknown sensitivity parameter"):
                run_selected(
                    DEFAULT_CONFIG,
                    input_dir,
                    Path(tmp) / "outputs",
                    run_baseline=False,
                    sensitivity_parameters=["bad_parameter"],
                )

    def test_sensitivity_parameter_order_constant(self) -> None:
        self.assertEqual(
            SENSITIVITY_PARAMETERS,
            [
                "pv_capex_sgd_per_kw",
                "battery_capex_sgd_per_kwh",
                "battery_power_capex_sgd_per_kw",
                "saidi_min_per_year",
                "saifi_per_year",
                "voll_sgd_per_kwh",
            ],
        )


if __name__ == "__main__":
    unittest.main()
