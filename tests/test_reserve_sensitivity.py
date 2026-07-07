from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from pv_bess_sizing import run_selected
from src.config import DEFAULT_CONFIG
from src.data_loader import discover_input_cases, load_input_case
from src.dispatch import solve_dispatch_greedy
from src.optimizer import scenario_config
from src.reserve_sensitivity import run_reserve_sensitivity
from src.resilience_frontier import run_resilience_frontier


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


class ReserveSensitivityTests(unittest.TestCase):
    def _fast_config(self) -> dict[str, object]:
        config = deepcopy(DEFAULT_CONFIG)
        config["capacity_search"]["coarse_intervals"] = 1
        config["capacity_search"]["local_refine_divisions"] = 1
        config["capacity_search"]["max_upper_expansions"] = 0
        return config

    def _write_case(self, root: Path) -> Path:
        case_dir = root / "Input" / "SG"
        case_dir.mkdir(parents=True, exist_ok=True)
        load = pd.DataFrame(
            {
                "Cooling": [25.0] * 8760,
                "Heating": [0.0] * 8760,
                "Lighting": [6.0] * 8760,
                "Equipment": [4.0] * 8760,
            }
        )
        pv_values = [1.0] * 8760
        pv_values[2088] = 0.0
        pv_values[2089] = 0.0
        pv = pd.DataFrame({"PV_output_kWh_per_m2": pv_values})
        tou = pd.DataFrame({"hour": range(24), "tou_sgd_per_kwh": [0.25] * 24})
        params = pd.DataFrame(self._parameter_rows(), columns=PARAMETER_COLUMNS)
        load.to_csv(case_dir / "SG_UTown_Load.csv", index=False)
        pv.to_csv(case_dir / "SG_UTown_PV.csv", index=False)
        tou.to_csv(case_dir / "SG_UTown_TOU.csv", index=False)
        params.to_csv(case_dir / "SG_UTown_Parameter.csv", index=False)
        return root / "Input"

    def _parameter_rows(self) -> list[dict[str, object]]:
        rows = [
            ("pv_capex_sgd_per_kwp", 1200, "SGD/kWp", 1200, 1200, 1, "PV unit CAPEX.", "光伏单位装机投资成本。"),
            ("battery_capex_sgd_per_kwh", 450, "SGD/kWh", 450, 450, 1, "Battery energy CAPEX.", "电池储能单位能量容量投资成本。"),
            ("battery_power_capex_sgd_per_kw", 180, "SGD/kW", 180, 180, 1, "Battery power CAPEX.", "电池功率转换单位投资成本。"),
            ("saidi_min_per_year", 60, "min/year", 60, 60, 1, "Annual outage duration.", "年平均停电持续时间。"),
            ("voll_sgd_per_kwh", 50, "SGD/kWh", 50, 50, 1, "Value of lost load.", "单位未供电电量对应的停电损失价值。"),
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
            ("saifi_per_year", 1, "events/year", "", "", "", "Annual outage count.", "年平均停电次数。"),
            ("battery_om_fraction_per_year", 0.02, "fraction/year", "", "", "", "Battery O&M.", "电池年度运维成本占 CAPEX 比例。"),
            ("battery_max_c_rate", 0.0, "1/h", "", "", "", "Battery max C-rate.", "电池最大充放电倍率。"),
            ("battery_roundtrip_efficiency", 1.0, "fraction", "", "", "", "Battery round-trip efficiency.", "电池往返效率。"),
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
            ("uncertainty_baseline_scenario_count", 1, "scenarios", "", "", "", "Uncertainty baseline scenario count.", "不确定性基准分析场景数量。"),
            ("uncertainty_sensitivity_scenario_count", 1, "scenarios", "", "", "", "Uncertainty sensitivity scenario count.", "不确定性敏感性分析场景数量。"),
            ("uncertainty_seed_start", 2000, "-", "", "", "", "Uncertainty seed start.", "不确定性停电场景起始随机种子。"),
            ("parallel_workers", 1, "workers", "", "", "", "Parallel workers for uncertainty analysis.", "不确定性分析并行进程数。"),
            ("pv_during_outage_available", "true", "boolean", "", "", "", "PV availability during outage.", "停电期间 PV 是否可用。"),
            ("frontier_outage_scenario_count", 1, "scenarios", "", "", "", "Frontier outage scenarios.", "前沿停电场景数量。"),
            ("frontier_seed_start", 1000, "-", "", "", "", "Frontier seed start.", "前沿随机种子起点。"),
            ("lolp_threshold", 0.0, "-", "", "", "", "LOLP threshold.", "LOLP 阈值。"),
            ("reserve_min_fraction", 0.20, "fraction", "", "", "", "Minimum reserve.", "最小预留比例。"),
            ("reserve_max_fraction", 0.80, "fraction", "", "", "", "Maximum reserve.", "最大预留比例。"),
            ("reserve_step_fraction", 0.30, "fraction", "", "", "", "Reserve step.", "预留比例步长。"),
        ]
        return [dict(zip(PARAMETER_COLUMNS, row, strict=True)) for row in rows]

    def _dispatch_config(self, reserve_fraction: float) -> dict[str, object]:
        return {
            "technical": {"cooling_cop": 5.0, "heating_cop": 3.5},
            "battery": {
                "roundtrip_efficiency": 1.0,
                "initial_soc_fraction": 0.5,
                "soc_min_fraction": 0.1,
                "soc_max_fraction": 0.95,
                "normal_operation_reserve_fraction": reserve_fraction,
            },
        }

    def test_grid_connected_discharge_respects_reserve_floor(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0, 1, 2],
                "grid_availability": [1.0, 1.0, 1.0],
                "tou_sgd_per_kwh": [2.0, 1.0, 0.0],
                "pv_kwh": [0.0, 0.0, 0.0],
                "electric_load_kwh": [30.0, 0.0, 0.0],
                "cooling_load_kwh_th": [0.0, 0.0, 0.0],
                "heating_load_kwh_th": [0.0, 0.0, 0.0],
            }
        )
        sizing = {"battery_energy_kwh": 100.0, "battery_power_kw": 100.0}
        dispatch = solve_dispatch_greedy(frame, sizing, self._dispatch_config(0.4))
        self.assertAlmostEqual(float(dispatch["battery_soc_kwh"].iloc[0]), 40.0)
        self.assertAlmostEqual(float(dispatch["battery_discharge_kwh"].iloc[0]), 10.0)

    def test_outage_discharge_can_go_below_reserve_to_technical_soc(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0],
                "grid_availability": [0.0],
                "tou_sgd_per_kwh": [1.0],
                "pv_kwh": [0.0],
                "electric_load_kwh": [30.0],
                "cooling_load_kwh_th": [0.0],
                "heating_load_kwh_th": [0.0],
            }
        )
        sizing = {"battery_energy_kwh": 100.0, "battery_power_kw": 100.0}
        dispatch = solve_dispatch_greedy(frame, sizing, self._dispatch_config(0.4))
        self.assertAlmostEqual(float(dispatch["battery_soc_kwh"].iloc[0]), 20.0)
        self.assertAlmostEqual(float(dispatch["battery_discharge_kwh"].iloc[0]), 30.0)

    def test_reserve_above_soc_max_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "hour_index": [0],
                "grid_availability": [1.0],
                "tou_sgd_per_kwh": [1.0],
                "pv_kwh": [0.0],
                "electric_load_kwh": [1.0],
                "cooling_load_kwh_th": [0.0],
                "heating_load_kwh_th": [0.0],
            }
        )
        sizing = {"battery_energy_kwh": 100.0, "battery_power_kw": 100.0}
        with self.assertRaisesRegex(ValueError, "normal_operation_reserve_fraction"):
            solve_dispatch_greedy(frame, sizing, self._dispatch_config(0.99))

    def test_reserve_sensitivity_returns_required_tables(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            _, _, cfg, parameters = load_input_case(case, self._fast_config())
            frontier = run_resilience_frontier(case, cfg, parameters)
            _, _, optimal_designs, outage_scenarios = frontier
            all_results, cost_components, reliability_metrics, marginal_cost = run_reserve_sensitivity(
                case,
                cfg,
                parameters,
                optimal_designs,
                outage_scenarios,
            )
            self.assertEqual(all_results["reserve_fraction"].tolist(), [0.2, 0.5, 0.8])
            required_all = {
                "reserve_reference_design",
                "reference_pv_capacity_kwp",
                "reference_battery_energy_kwh",
                "normal_operation_bill_cost_sgd_per_year",
                "outage_unserved_energy_cost_sgd_per_year",
                "net_cost_sgd_per_year",
                "EUE_kwh_per_year",
                "achieved_LOLP",
                "chance_constraint_probability",
                "outage_load_served_ratio",
            }
            self.assertTrue(required_all.issubset(all_results.columns))
            self.assertIn("normal_operation_bill_cost_sgd_per_year", cost_components.columns)
            self.assertIn("achieved_LOLP", reliability_metrics.columns)
            self.assertEqual(marginal_cost["reserve_interval"].tolist(), ["20% to 50%", "50% to 80%"])
            self.assertIn("marginal_net_cost_change_sgd_per_year", marginal_cost.columns)

    def test_reliability_constrained_reference_raises_when_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            raw = pd.read_csv(case.parameter_csv)
            extra = dict(
                zip(
                    PARAMETER_COLUMNS,
                    (
                        "reserve_reference_design",
                        "reliability_constrained",
                        "-",
                        "",
                        "",
                        "",
                        "Reserve reference design.",
                        "预留分析参考设计。",
                    ),
                    strict=True,
                )
            )
            raw = pd.concat([raw, pd.DataFrame([extra])], ignore_index=True)
            raw.to_csv(case.parameter_csv, index=False)
            _, _, cfg, parameters = load_input_case(case, self._fast_config())
            frontier = run_resilience_frontier(case, cfg, parameters)
            _, _, optimal_designs, outage_scenarios = frontier
            with self.assertRaisesRegex(ValueError, "unavailable"):
                run_reserve_sensitivity(case, cfg, parameters, optimal_designs, outage_scenarios)

    def test_run_selected_writes_reserve_outputs_and_framework_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = self._write_case(root)
            output_dir = root / "outputs"
            paths = run_selected(
                self._fast_config(),
                input_dir,
                output_dir,
                run_baseline=False,
                run_sensitivity=False,
                run_uncertainty_baseline=False,
                run_uncertainty_sensitivity=False,
                run_frontier=False,
                run_reserve=True,
                case_filters=["SG_UTown"],
            )
            reserve = output_dir / "SG" / "SG_UTown_Reserve_Sensitivity"
            frontier = output_dir / "SG" / "SG_UTown_Resilience_Frontier"
            self.assertTrue((frontier / "SG_UTown_Frontier_Optimal_Designs.csv").exists())
            self.assertTrue((reserve / "SG_UTown_Reserve_Sensitivity_All_Results.csv").exists())
            self.assertTrue((reserve / "SG_UTown_Reserve_Sensitivity_Cost_Components.csv").exists())
            self.assertTrue((reserve / "SG_UTown_Reserve_Sensitivity_Reliability_Metrics.csv").exists())
            self.assertTrue((reserve / "SG_UTown_Reserve_Sensitivity_Marginal_Cost.csv").exists())
            for label in (
                "Figure_13_Reserve_vs_Cost_Components",
                "Figure_14_Reserve_vs_Reliability_Metrics",
                "Figure_15_Reserve_Marginal_Cost",
            ):
                base = reserve / f"SG_UTown_{label}"
                self.assertTrue(base.with_suffix(".png").exists())
                self.assertTrue(base.with_name(base.name + "_Source_Data.csv").exists())
            figure_13_source = pd.read_csv(
                reserve / "SG_UTown_Figure_13_Reserve_vs_Cost_Components_Source_Data.csv"
            )
            self.assertIn("reserve_fraction", figure_13_source.columns)
            self.assertIn("net_cost_sgd_per_year", figure_13_source.columns)
            self.assertIn("SG_UTown_reserve_sensitivity_all_results", paths)

            framework_output = root / "framework_outputs"
            framework_paths = run_selected(
                self._fast_config(),
                input_dir,
                framework_output,
                run_baseline=False,
                run_sensitivity=False,
                run_uncertainty_baseline=False,
                run_uncertainty_sensitivity=False,
                run_frontier=True,
                run_reserve=True,
                case_filters=["SG_UTown"],
            )
            self.assertIn("SG_UTown_frontier_all_candidates", framework_paths)
            self.assertIn("SG_UTown_reserve_sensitivity_all_results", framework_paths)


if __name__ == "__main__":
    unittest.main()
