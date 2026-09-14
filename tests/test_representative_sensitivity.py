from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from pv_bess_sizing import run_selected
from src.io.config import DEFAULT_CONFIG
from src.io.data_loader import discover_input_cases, load_input_case
from src.representative_sensitivity import representative_outage_seed, run_representative_sensitivity
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


class RepresentativeSensitivityTests(unittest.TestCase):
    def _fast_config(self) -> dict[str, object]:
        config = deepcopy(DEFAULT_CONFIG)
        config.setdefault("dispatch", {})["backend"] = "greedy"
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
        pv = pd.DataFrame({"PV_output_kWh_per_m2": [1.0] * 8760})
        tou = pd.DataFrame({"hour": range(24), "tou_sgd_per_kwh": [0.20] * 8 + [0.40] * 8 + [0.25] * 8})
        params = pd.DataFrame(self._parameter_rows(), columns=PARAMETER_COLUMNS)
        load.to_csv(case_dir / "SG_UTown_Load.csv", index=False)
        pv.to_csv(case_dir / "SG_UTown_PV.csv", index=False)
        tou.to_csv(case_dir / "SG_UTown_TOU.csv", index=False)
        params.to_csv(case_dir / "SG_UTown_Parameter.csv", index=False)
        return root / "Input"

    def _parameter_rows(self) -> list[dict[str, object]]:
        rows = [
            ("pv_capex_sgd_per_kw", 1200, "SGD/kW", 1200, 1200, 1, "PV unit CAPEX.", "光伏单位装机投资成本。"),
            ("battery_capex_sgd_per_kwh", 450, "SGD/kWh", 450, 450, 1, "Battery energy CAPEX.", "电池储能单位能量容量投资成本。"),
            ("battery_power_capex_sgd_per_kw", 180, "SGD/kW", 180, 180, 1, "Battery power CAPEX.", "电池功率转换单位投资成本。"),
            ("saidi_min_per_year", 60, "min/year", 60, 60, 1, "Annual outage duration.", "年平均停电持续时间。"),
            ("saifi_per_year", 1, "events/year", 1, 1, 1, "Annual outage count.", "年平均停电次数。"),
            ("voll_sgd_per_kwh", 50, "SGD/kWh", 50, 50, 1, "Value of lost load.", "单位未供电电量对应的停电损失价值。"),
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
            ("uncertainty_baseline_scenario_count", 1, "scenarios", "", "", "", "Uncertainty baseline scenario count.", "不确定性基准分析场景数量。"),
            ("uncertainty_sensitivity_scenario_count", 1, "scenarios", "", "", "", "Uncertainty sensitivity scenario count.", "不确定性敏感性分析场景数量。"),
            ("uncertainty_seed_start", 2000, "-", "", "", "", "Uncertainty seed start.", "不确定性停电场景起始随机种子。"),
            ("parallel_workers", 1, "workers", "", "", "", "Parallel workers for uncertainty analysis.", "不确定性分析并行进程数。"),
            ("pv_during_outage_available", "true", "boolean", "", "", "", "PV availability during outage.", "停电期间 PV 是否可用。"),
            ("frontier_outage_scenario_count", 1, "scenarios", "", "", "", "Frontier outage scenarios.", "前沿停电场景数量。"),
            ("frontier_seed_start", 1000, "-", "", "", "", "Frontier seed start.", "前沿随机种子起点。"),
            ("lolp_threshold", 0.10, "-", "", "", "", "LOLP threshold.", "LOLP 阈值。"),
        ]
        return [dict(zip(PARAMETER_COLUMNS, row, strict=True)) for row in rows]

    def test_representative_sensitivity_is_deterministic_and_fixed_design(self) -> None:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            _, _, cfg, parameters = load_input_case(case, self._fast_config())
            self.assertEqual(representative_outage_seed(parameters, cfg), 1000)
            _, _, optimal_designs, _ = run_resilience_frontier(case, cfg, parameters)
            all_results, cost_reliability = run_representative_sensitivity(
                case,
                cfg,
                parameters,
                optimal_designs,
            )
            self.assertFalse(all_results.empty)
            self.assertFalse(cost_reliability.empty)
            self.assertEqual(all_results["reference_pv_capacity_kw"].nunique(), 1)
            self.assertEqual(all_results["reference_battery_energy_kwh"].nunique(), 1)
            capex = all_results[all_results["sensitivity_parameter"] == "pv_capex_sgd_per_kw"]
            self.assertEqual(sorted(capex["sensitivity_multiplier"].unique().tolist()), [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
            base_pv_cost = float(cfg["pv"]["capex_sgd_per_kw"])
            self.assertIn(base_pv_cost * 2.0, capex["sensitivity_value"].tolist())
            saidi_voll = all_results[all_results["sensitivity_family"] == "saidi_voll"]
            self.assertIn(10.0, saidi_voll["voll_multiplier"].tolist())
            self.assertIn(10.0, saidi_voll["saidi_multiplier"].tolist())

    def test_run_selected_writes_representative_sensitivity_outputs(self) -> None:
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
                run_representative_sensitivity=True,
                case_filters=["SG_UTown"],
            )
            sensitivity = output_dir / "SG" / "SG_UTown_Representative_Sensitivity"
            self.assertTrue((sensitivity / "SG_UTown_Representative_Sensitivity_All_Results.csv").exists())
            self.assertTrue((sensitivity / "SG_UTown_Representative_Sensitivity_Cost_Reliability.csv").exists())
            csv_names = sorted(path.name for path in sensitivity.glob("*.csv"))
            self.assertEqual(
                csv_names,
                [
                    "SG_UTown_Representative_Sensitivity_All_Results.csv",
                    "SG_UTown_Representative_Sensitivity_Cost_Reliability.csv",
                ],
            )
            self.assertFalse(any(path.suffix == ".png" for path in sensitivity.iterdir()))
            self.assertIn("SG_UTown_representative_sensitivity_all_results", paths)


if __name__ == "__main__":
    unittest.main()
