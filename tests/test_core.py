from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.config import DEFAULT_CONFIG
from src.costing import baseline_no_battery_cost, pv_annual_cost
from src.data_loader import (
    SENSITIVITY_PARAMETERS,
    build_case_frame,
    discover_input_cases,
    load_case_parameters,
    load_input_case,
    sensitivity_runs,
)
from src.degradation import annualized_battery_cost
from src.dispatch import outage_free_frame, solve_dispatch_greedy
from src.optimizer import resilience_design_pair, scenario_config
from src.outage import simulate_outages, summarize_outages
from src.uncertainty import run_uncertainty_analysis, summarize_uncertainty_results
from src.visualization import DESIGN_CASE_ORDER, cost_breakdown_table, optimal_sizing_table
from pv_bess_sizing import run_all, run_selected


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
    def _case_dir(self, root: Path) -> Path:
        case_dir = root / "Input" / "SG"
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def _parameter_rows(self) -> list[dict[str, object]]:
        rows = [
            ("pv_capex_sgd_per_kwp", 1200, "SGD/kWp", 1000, 1200, 200, "PV unit CAPEX.", "光伏单位装机投资成本。"),
            ("battery_capex_sgd_per_kwh", 450, "SGD/kWh", 350, 550, 100, "Battery energy CAPEX.", "电池储能单位能量容量投资成本。"),
            ("battery_power_capex_sgd_per_kw", 180, "SGD/kW", 120, 200, 40, "Battery power CAPEX.", "电池储能单位功率容量投资成本。"),
            ("saidi_min_per_year", 0.26, "min/year", 0, 5, 5, "Annual outage duration.", "年平均停电持续时间。"),
            ("voll_sgd_per_kwh", 35, "SGD/kWh", 10, 40, 15, "Value of lost load.", "单位未供电电量对应的停电损失价值。"),
            ("city_name", "SG", "-", "", "", "", "City key.", "城市或地区名称。"),
            ("building_name", "UTown", "-", "", "", "", "Building key.", "建筑或建筑群名称。"),
            ("local_refine_window_kwp", 0, "kWp", "", "", "", "PV local refinement window.", "光伏容量局部细化窗口。"),
            ("local_refine_step_kwp", 500, "kWp", "", "", "", "PV local refinement step.", "光伏容量局部细化步长。"),
            ("rooftop_area_m2", 1000, "m2", "", "", "", "Effective rooftop area available for PV.", "可安装光伏的有效屋顶面积。"),
            ("pv_module_power_density_kwp_per_m2", 0.2, "kWp/m2", "", "", "", "PV module power density.", "单位面积光伏组件装机功率。"),
            ("pv_om_fraction_per_year", 0.015, "fraction/year", "", "", "", "PV annual O&M.", "光伏年度运维成本占 CAPEX 比例。"),
            ("pv_project_years", 25, "years", "", "", "", "PV project life.", "光伏项目寿命。"),
            ("cooling_cop", 5.0, "-", "", "", "", "Cooling COP.", "制冷系统 COP。"),
            ("heating_cop", 3.5, "-", "", "", "", "Heating COP.", "供热系统 COP。"),
            ("saifi_per_year", 0.006, "events/year", "", "", "", "Annual outage count.", "年平均停电次数。"),
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

    def test_parameter_file_requires_first_five_sensitivity_parameters(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            raw.loc[0, "parameter"] = "voll_sgd_per_kwh"
            raw.to_csv(case.parameter_csv, index=False)
            with self.assertRaisesRegex(ValueError, "first five parameters"):
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
                    ("pv_capacity_kwp_candidates", "[0,1000]", "kWp", "", "", "", "Old PV candidates.", "旧光伏候选列表。"),
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
            self.assertEqual(cfg["pv"]["capacity_kwp_candidates"][0], 0.0)
            self.assertEqual(cfg["battery"]["energy_kwh_candidates"][0], 0.0)
            self.assertGreater(max(cfg["pv"]["capacity_kwp_candidates"]), 0.0)
            self.assertGreater(max(cfg["battery"]["energy_kwh_candidates"]), 0.0)
            self.assertEqual(len(cfg["pv"]["capacity_kwp_candidates"]), 9)
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
            self.assertNotIn("pv_capacity_kwp_candidates", params.base)
            self.assertIs(params.base["pv_during_outage_available"], True)
            self.assertEqual(params.sensitivity["pv_capex_sgd_per_kwp"], [1000.0, 1200.0])
            self.assertEqual(len(sensitivity_runs(params)), 13)

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
            frame, meta, cfg, _ = load_input_case(case, DEFAULT_CONFIG, pv_capacity_kwp=200.0)
            self.assertEqual(len(frame), 8760)
            self.assertFalse(frame.isna().any().any())
            self.assertAlmostEqual(float(frame["electric_load_kwh"].iloc[0]), 10.0)
            self.assertAlmostEqual(float(frame["pv_kwh"].iloc[0]), 1000.0)
            self.assertAlmostEqual(float(meta["pv_capacity_kwp"]), 200.0)
            self.assertAlmostEqual(float(meta["pv_capacity_mwp"]), 0.2)
            self.assertAlmostEqual(float(meta["pv_installed_area_m2"]), 1000.0)
            self.assertAlmostEqual(float(meta["pv_rooftop_utilization_fraction"]), 1.0)
            self.assertAlmostEqual(float(meta["pv_module_power_density_kwp_per_m2"]), 0.2)
            self.assertAlmostEqual(float(frame["pv_capacity_kwp"].iloc[0]), 200.0)
            self.assertEqual(cfg["scenario_order"], ["SG_UTown"])
            self.assertEqual(sorted(frame["tou_sgd_per_kwh"].unique().tolist()), [0.239, 0.329])
            self.assertEqual(len(cfg["pv"]["capacity_kwp_candidates"]), 9)
            self.assertEqual(cfg["uncertainty"]["baseline_scenario_count"], 2)
            self.assertEqual(cfg["uncertainty"]["sensitivity_scenario_count"], 2)
            self.assertEqual(cfg["uncertainty"]["seed_start"], 2000)
            self.assertEqual(cfg["uncertainty"]["parallel_workers"], 1)

    def test_pv_capacity_can_exceed_rooftop_utilization_without_error(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            _, meta, _, _ = load_input_case(case, DEFAULT_CONFIG, pv_capacity_kwp=1000.0)
            self.assertGreater(float(meta["pv_rooftop_utilization_fraction"]), 1.0)

    def test_joint_pv_battery_optimization_returns_coverage_and_battery(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            _, _, cfg, _ = load_input_case(case, DEFAULT_CONFIG)

            def coverage_frame_builder(coverage: float):
                frame, meta = build_case_frame(case, cfg, coverage)
                frame["grid_availability"] = 1.0
                frame["outage_fraction"] = 0.0
                return frame, meta

            rows, dispatches, summaries = resilience_design_pair(coverage_frame_builder, cfg, "SG_UTown")
            self.assertEqual(sorted(rows["design_case"].tolist()), ["flexibility_only", "resilience_informed"])
            self.assertIn("pv_capacity_kwp", rows.columns)
            self.assertNotIn("pv_coverage_fraction", rows.columns)
            self.assertIn("battery_energy_kwh", rows.columns)
            self.assertIn("pv_bess_system_annual_cost_sgd", rows.columns)
            self.assertFalse(any(column.startswith("hess_") for column in rows.columns))
            self.assertIn("resilience_informed", dispatches)
            self.assertIn("pv_capacity_kwp", summaries["resilience_informed"].columns)

    def test_pv_annual_cost(self) -> None:
        meta = {"pv_capacity_kwp": 1000.0}
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

    def test_dispatch_and_baseline_include_heating_load(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            frame, meta, cfg, _ = load_input_case(case, DEFAULT_CONFIG, pv_capacity_kwp=0.0)
            resolved = scenario_config(cfg, "SG_UTown", resilience_informed=True)
            two_hours = frame.head(2).copy()
            two_hours["grid_availability"] = [1.0, 1.0]
            two_hours["outage_fraction"] = [0.0, 0.0]
            sizing = {"battery_energy_kwh": 0.0, "battery_power_kw": 0.0}
            dispatch = solve_dispatch_greedy(two_hours, sizing, resolved)
            expected = 10.0 + 25.0 / 5.0 + 7.0 / 3.5
            self.assertAlmostEqual(float(dispatch["total_load_e_equiv_kwh"].iloc[0]), expected)
            base_cost = baseline_no_battery_cost(two_hours, meta, resolved)
            self.assertGreater(base_cost["baseline_grid_energy_kwh"], 0.0)
            self.assertIn("baseline_private_annual_cost_sgd", base_cost)
            self.assertIn("baseline_system_annual_cost_sgd", base_cost)

    def test_flexibility_only_is_evaluated_with_unserved_energy_cost(self) -> None:
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
                "capacity_kwp_candidates": [0.0],
                "local_refine_window_kwp": 0.0,
                "local_refine_step_kwp": 500.0,
                "capex_sgd_per_kwp": 1200.0,
                "om_fraction_per_year": 0.015,
                "project_years": 25.0,
            },
            "technical": {"cooling_cop": 5.0, "heating_cop": 3.5},
        }

        def outage_builder(pv_capacity_kwp: float):
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
            return frame, {"pv_capacity_kwp": pv_capacity_kwp, "pv_capacity_mwp": pv_capacity_kwp / 1000.0}

        rows, _, _ = resilience_design_pair(outage_builder, config, "SG_UTown")
        flex = rows[rows["design_case"] == "flexibility_only"].iloc[0]
        self.assertEqual(float(flex["optimization_voll_sgd_per_kwh"]), 0.0)
        self.assertEqual(float(flex["evaluation_voll_sgd_per_kwh"]), 100.0)
        self.assertAlmostEqual(float(flex["unserved_energy_kwh_per_year"]), 10.0)
        self.assertAlmostEqual(float(flex["unserved_energy_cost_sgd_per_year"]), 1000.0)
        self.assertAlmostEqual(
            float(flex["pv_bess_system_annual_cost_sgd"]),
            float(flex["pv_bess_private_annual_cost_sgd"]) + 1000.0,
        )

    def test_battery_cost_uses_input_parameter_economics(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_valid_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            _, _, cfg, _ = load_input_case(case, DEFAULT_CONFIG)
            resolved = scenario_config(cfg, "SG_UTown", resilience_informed=True)
            cost = annualized_battery_cost(1000.0, 500.0, 0.0, resolved)
            self.assertAlmostEqual(cost["battery_capex_sgd"], 540_000.0)
            self.assertAlmostEqual(cost["battery_capex_sgd_per_kwh"], 450.0)
            self.assertAlmostEqual(cost["battery_power_capex_sgd_per_kw"], 180.0)

    def test_cost_breakdown_sums_to_total_and_labels_unserved_energy_cost(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "design_case": "flexibility_only",
                    "annualized_pv_cost_sgd": 10.0,
                    "annualized_battery_cost_sgd": 5.0,
                    "pv_bess_grid_cost_sgd": 20.0,
                    "pv_om_cost_sgd": 2.0,
                    "battery_om_cost_sgd": 1.0,
                    "pv_bess_private_annual_cost_sgd": 38.0,
                    "unserved_energy_kwh_per_year": 4.0,
                    "unserved_energy_cost_sgd_per_year": 12.0,
                    "pv_bess_system_annual_cost_sgd": 50.0,
                    "pv_capacity_kwp": 800.0,
                    "battery_energy_kwh": 1000.0,
                },
                {
                    "design_case": "resilience_informed",
                    "annualized_pv_cost_sgd": 9.0,
                    "annualized_battery_cost_sgd": 6.0,
                    "pv_bess_grid_cost_sgd": 18.0,
                    "pv_om_cost_sgd": 2.0,
                    "battery_om_cost_sgd": 2.0,
                    "pv_bess_private_annual_cost_sgd": 37.0,
                    "unserved_energy_kwh_per_year": 2.0,
                    "unserved_energy_cost_sgd_per_year": 8.0,
                    "pv_bess_system_annual_cost_sgd": 45.0,
                    "pv_capacity_kwp": 700.0,
                    "battery_energy_kwh": 1500.0,
                },
            ]
        )
        table = cost_breakdown_table(rows)
        self.assertIn("annualized_pv_capital_cost_sgd_per_year", table.columns)
        self.assertIn("annualized_battery_capital_cost_sgd_per_year", table.columns)
        self.assertIn("unserved_energy_kwh_per_year", table.columns)
        self.assertIn("unserved_energy_cost_sgd_per_year", table.columns)
        for row in table.itertuples(index=False):
            self.assertAlmostEqual(row.component_sum_sgd_per_year, row.pv_bess_system_annual_cost_sgd_per_year)

    def test_optimal_sizing_table_exports_battery_and_pv_units(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "design_case": "flexibility_only",
                    "battery_energy_kwh": 1000.0,
                    "battery_power_kw": 500.0,
                    "pv_capacity_kwp": 6400.0,
                    "pv_bess_private_annual_cost_sgd": 8_900_000.0,
                    "pv_bess_system_annual_cost_sgd": 9_000_000.0,
                },
                {
                    "design_case": "resilience_informed",
                    "battery_energy_kwh": 2500.0,
                    "battery_power_kw": 1250.0,
                    "pv_capacity_kwp": 5600.0,
                    "pv_bess_private_annual_cost_sgd": 9_050_000.0,
                    "pv_bess_system_annual_cost_sgd": 9_100_000.0,
                },
            ]
        )
        table = optimal_sizing_table(rows)
        self.assertEqual(table["design_case_label"].tolist(), ["Flexibility-only", "Resilience-informed"])
        self.assertEqual(table["battery_energy_mwh"].tolist(), [1.0, 2.5])
        self.assertEqual(table["pv_capacity_mwp"].tolist(), [6.4, 5.6])
        self.assertNotIn("pv_coverage_fraction", table.columns)
        self.assertIn("pv_bess_private_annual_cost_sgd_per_year", table.columns)
        self.assertIn("pv_bess_system_annual_cost_sgd_per_year", table.columns)

    def test_uncertainty_summary_exports_distribution_statistics_without_regret(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "design_case": "flexibility_only",
                    "design_case_label": "Flexibility-only",
                    "pv_capacity_kwp": 1000.0,
                    "battery_energy_kwh": 0.0,
                    "pv_bess_private_annual_cost_sgd": 10.0,
                    "unserved_energy_kwh_per_year": 3.0,
                    "unserved_energy_cost_sgd_per_year": 6.0,
                    "pv_bess_system_annual_cost_sgd": 16.0,
                },
                {
                    "design_case": "flexibility_only",
                    "design_case_label": "Flexibility-only",
                    "pv_capacity_kwp": 1000.0,
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
            self.assertEqual(serial_results["design_case"].tolist(), parallel_results["design_case"].tolist())
            pd.testing.assert_series_equal(
                serial_results["pv_bess_system_annual_cost_sgd"],
                parallel_results["pv_bess_system_annual_cost_sgd"],
                check_names=False,
            )

    def test_run_all_writes_case_output_structure_and_key_figures(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = self._write_valid_case(root)
            output_dir = root / "outputs"
            paths = run_selected(
                DEFAULT_CONFIG,
                input_dir,
                output_dir,
                run_baseline=True,
                run_sensitivity=True,
                sensitivity_parameters=["pv_capex_sgd_per_kwp", "saidi_min_per_year"],
            )
            baseline = output_dir / "SG" / "SG_UTown_Baseline"
            sensitivity = output_dir / "SG" / "SG_UTown_Sensitive_Analysis"
            self.assertTrue(baseline.is_dir())
            self.assertTrue(sensitivity.is_dir())
            self.assertTrue((baseline / "SG_UTown_Baseline_Design_Comparison.csv").exists())
            design = pd.read_csv(baseline / "SG_UTown_Baseline_Design_Comparison.csv")
            self.assertIn("pv_bess_system_annual_cost_sgd", design.columns)
            self.assertIn("pv_capacity_kwp", design.columns)
            self.assertNotIn("pv_coverage_fraction", design.columns)
            self.assertFalse(any(column.startswith("hess_") for column in design.columns))
            candidate = pd.read_csv(baseline / "SG_UTown_Baseline_Candidate_Summary_Resilience_Informed.csv")
            self.assertIn("search_stage", candidate.columns)
            self.assertIn("coarse", candidate["search_stage"].unique().tolist())
            self.assertIn("optimal_region", candidate["search_stage"].unique().tolist())
            self.assertTrue(
                (baseline / "SG_UTown_Baseline_Annualized_Cost_Stacked_Flexibility_vs_Resilience_Informed.png").exists()
            )
            cost_source = pd.read_csv(
                baseline / "SG_UTown_Baseline_Annualized_Cost_Stacked_Flexibility_vs_Resilience_Informed_Source_Data.csv"
            )
            self.assertIn("annualized_pv_capital_cost_sgd_per_year", cost_source.columns)
            self.assertIn("annualized_battery_capital_cost_sgd_per_year", cost_source.columns)
            sizing_base = baseline / "SG_UTown_Baseline_Optimal_Battery_Capacity_PV_Capacity_Flexibility_vs_Resilience_Informed"
            for suffix in (".png", "_Source_Data.csv"):
                self.assertTrue(sizing_base.with_name(sizing_base.name + suffix).exists())
            sizing_source = pd.read_csv(sizing_base.with_name(sizing_base.name + "_Source_Data.csv"))
            self.assertEqual(sorted(sizing_source["design_case"].tolist()), ["flexibility_only", "resilience_informed"])
            self.assertIn("battery_energy_mwh", sizing_source.columns)
            self.assertIn("pv_capacity_mwp", sizing_source.columns)
            self.assertNotIn("pv_coverage_fraction", sizing_source.columns)
            self.assertFalse(any(column.startswith("hess_") for column in sizing_source.columns))
            full_heatmap = (
                baseline
                / "SG_UTown_Baseline_Full_Search_PV_Capacity_Battery_Capacity_Annualized_Cost_Heatmap_Resilience_Informed"
            )
            optimal_heatmap = (
                baseline
                / "SG_UTown_Baseline_Optimal_Region_PV_Capacity_Battery_Capacity_Annualized_Cost_Heatmap_Resilience_Informed"
            )
            for base, scope in ((full_heatmap, "full_search"), (optimal_heatmap, "optimal_region")):
                self.assertTrue(base.with_name(base.name + ".png").exists())
                source = pd.read_csv(base.with_name(base.name + "_Source_Data.csv"))
                self.assertEqual(source["plot_scope"].unique().tolist(), [scope])
                self.assertIn("pv_capacity_mwp", source.columns)
                self.assertIn("battery_energy_mwh", source.columns)
            self.assertTrue(
                (
                    sensitivity
                    / "SG_UTown_Sensitive_Analysis_pv_capex_sgd_per_kwp"
                    / "SG_UTown_Sensitive_Analysis_pv_capex_sgd_per_kwp_Optimal_Annualized_Cost_vs_PV_CAPEX.png"
                ).exists()
            )
            pv_cost_base = (
                sensitivity
                / "SG_UTown_Sensitive_Analysis_pv_capex_sgd_per_kwp"
                / "SG_UTown_Sensitive_Analysis_pv_capex_sgd_per_kwp_Optimal_Annualized_Cost_vs_PV_CAPEX"
            )
            pv_cost_source = pd.read_csv(pv_cost_base.with_name(pv_cost_base.name + "_Source_Data.csv"))
            self.assertEqual(sorted(pv_cost_source["sensitivity_value"].unique().tolist()), [1000.0, 1200.0])
            self.assertEqual(sorted(pv_cost_source["design_case"].unique().tolist()), DESIGN_CASE_ORDER)
            self.assertFalse(any(column.startswith("hess_") for column in pv_cost_source.columns))
            saidi_base_names = [
                "SG_UTown_Sensitive_Analysis_saidi_min_per_year_Optimal_Annualized_Cost_vs_SAIDI",
                "SG_UTown_Sensitive_Analysis_saidi_min_per_year_Optimal_Battery_Capacity_vs_SAIDI",
                "SG_UTown_Sensitive_Analysis_saidi_min_per_year_Optimal_PV_Capacity_vs_SAIDI",
            ]
            saidi_dir = sensitivity / "SG_UTown_Sensitive_Analysis_saidi_min_per_year"
            for name in saidi_base_names:
                base = saidi_dir / name
                for suffix in (".png", "_Source_Data.csv"):
                    self.assertTrue(base.with_name(base.name + suffix).exists())
            saidi_cost_base = saidi_dir / "SG_UTown_Sensitive_Analysis_saidi_min_per_year_Optimal_Annualized_Cost_vs_SAIDI"
            saidi_cost_source = pd.read_csv(saidi_cost_base.with_name(saidi_cost_base.name + "_Source_Data.csv"))
            self.assertEqual(sorted(saidi_cost_source["sensitivity_value"].unique().tolist()), [0.0, 5.0])
            self.assertEqual(sorted(saidi_cost_source["design_case"].unique().tolist()), DESIGN_CASE_ORDER)
            self.assertFalse(any("Annualized_Cost_Reduction_Percentage" in path.name for path in sensitivity.rglob("*")))
            self.assertFalse(any("Representative" in path.name for path in sensitivity.rglob("*")))
            self.assertNotIn("Comparison", [path.name for path in output_dir.iterdir()])
            self.assertGreater(len(paths), 0)

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
            summary_path = uncertainty / "SG_UTown_Uncertainty_Baseline_Design_Comparison_Summary.csv"
            for path in (all_results_path, scenarios_path, summary_path):
                self.assertTrue(path.exists())
            all_results = pd.read_csv(all_results_path)
            outage_scenarios = pd.read_csv(scenarios_path)
            summary = pd.read_csv(summary_path)
            self.assertEqual(len(all_results), 4)
            self.assertEqual(len(outage_scenarios), 2)
            self.assertEqual(sorted(all_results["design_case"].unique().tolist()), DESIGN_CASE_ORDER)
            self.assertTrue(
                all_results[all_results["design_case"] == "flexibility_only"][
                    "optimization_reused_across_uncertainty_scenarios"
                ].astype(bool).all()
            )
            self.assertFalse(
                all_results[all_results["design_case"] == "resilience_informed"][
                    "optimization_reused_across_uncertainty_scenarios"
                ].astype(bool).any()
            )
            self.assertIn("p10", summary.columns)
            self.assertIn("p90", summary.columns)
            self.assertFalse(any("regret" in column.lower() for column in all_results.columns))
            self.assertFalse(any("regret" in path.name.lower() for path in uncertainty_root.rglob("*")))
            self.assertFalse(any("Outage_Weighted" in path.name for path in uncertainty_root.rglob("*")))
            distribution = uncertainty / "SG_UTown_Uncertainty_Baseline_Optimal_Battery_Capacity_Distribution_Box_Jitter_Half_Violin"
            cost_ecdf = uncertainty / "SG_UTown_Uncertainty_Baseline_Annualized_System_Cost_ECDF"
            battery_ecdf = uncertainty / "SG_UTown_Uncertainty_Baseline_Optimal_Battery_Capacity_ECDF"
            for base in (distribution, cost_ecdf, battery_ecdf):
                self.assertTrue(base.with_name(base.name + ".png").exists())
                self.assertTrue(base.with_name(base.name + "_Source_Data.csv").exists())
            self.assertFalse(any(path.suffix in {".svg", ".pdf", ".tiff"} for path in uncertainty_root.rglob("*")))
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
            self.assertEqual(sorted(all_results["design_case"].unique().tolist()), DESIGN_CASE_ORDER)
            self.assertEqual(sorted(all_results["sensitivity_value"].unique().tolist()), [350.0, 450.0, 550.0])
            expected = [
                "SG_UTown_Uncertainty_Sensitive_Analysis_battery_capex_sgd_per_kwh_Optimal_Annualized_Cost_vs_Battery_energy_CAPEX",
                "SG_UTown_Uncertainty_Sensitive_Analysis_battery_capex_sgd_per_kwh_Optimal_Battery_Capacity_vs_Battery_energy_CAPEX",
                "SG_UTown_Uncertainty_Sensitive_Analysis_battery_capex_sgd_per_kwh_Optimal_PV_Capacity_vs_Battery_energy_CAPEX",
            ]
            for name in expected:
                for design_file_label, design_case in (
                    ("Flexibility_Only", "flexibility_only"),
                    ("Resilience_Informed", "resilience_informed"),
                ):
                    base = param_dir / f"{name}_{design_file_label}"
                    self.assertTrue(base.with_name(base.name + ".png").exists())
                    source = pd.read_csv(base.with_name(base.name + "_Source_Data.csv"))
                    self.assertEqual(source["design_case"].unique().tolist(), [design_case])
                    self.assertIn("median", source.columns)
                    self.assertIn("p10", source.columns)
                    self.assertIn("p90", source.columns)
            self.assertFalse(any("Outage_Weighted" in path.name for path in uncertainty_root.rglob("*")))
            self.assertFalse(any("regret" in path.name.lower() for path in uncertainty_root.rglob("*")))
            self.assertFalse(any(path.suffix in {".svg", ".pdf", ".tiff"} for path in uncertainty_root.rglob("*")))
            self.assertGreater(len(paths), 0)

    def test_run_selected_can_run_only_one_sensitivity_parameter(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = self._write_valid_case(root)
            output_dir = root / "outputs"
            run_selected(
                DEFAULT_CONFIG,
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
            self.assertTrue((sensitivity / "SG_UTown_Sensitive_Analysis_voll_sgd_per_kwh").is_dir())
            self.assertFalse((sensitivity / "SG_UTown_Sensitive_Analysis_pv_capex_sgd_per_kwp").exists())

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
                "pv_capex_sgd_per_kwp",
                "battery_capex_sgd_per_kwh",
                "battery_power_capex_sgd_per_kw",
                "saidi_min_per_year",
                "voll_sgd_per_kwh",
            ],
        )


if __name__ == "__main__":
    unittest.main()
