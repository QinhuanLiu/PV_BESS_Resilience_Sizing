from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.io.config import DEFAULT_CONFIG
from src.io.data_loader import discover_input_cases, load_input_case
from src.figures.outage_balance import (
    _conversion_loss,
    _select_max_impact_event,
    _soc_ylim_for_zero_alignment,
    build_outage_energy_balance_sources,
    plot_outage_context_figure,
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


class OutageEnergyBalanceFigureTests(unittest.TestCase):
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
                "Cooling": [20.0] * 8760,
                "Heating": [0.0] * 8760,
                "Lighting": [10.0] * 8760,
                "Equipment": [5.0] * 8760,
            }
        )
        pv_values = [0.8] * 8760
        for hour in range(0, 8760, 24):
            for night_hour in range(hour, min(hour + 7, 8760)):
                pv_values[night_hour] = 0.0
        pd.DataFrame({"PV_output_kWh_per_m2": pv_values}).to_csv(case_dir / "SG_UTown_PV.csv", index=False)
        load.to_csv(case_dir / "SG_UTown_Load.csv", index=False)
        pd.DataFrame({"hour": range(24), "tou_sgd_per_kwh": [0.25] * 24}).to_csv(case_dir / "SG_UTown_TOU.csv", index=False)
        pd.DataFrame(self._parameter_rows(), columns=PARAMETER_COLUMNS).to_csv(case_dir / "SG_UTown_Parameter.csv", index=False)
        return root / "Input"

    def _parameter_rows(self) -> list[dict[str, object]]:
        rows = [
            ("pv_capex_sgd_per_kw", 1200, "SGD/kW", 1200, 1200, 1, "PV unit CAPEX.", "光伏单位装机投资成本。"),
            ("battery_capex_sgd_per_kwh", 450, "SGD/kWh", 450, 450, 1, "Battery energy CAPEX.", "电池储能单位能量容量投资成本。"),
            ("battery_power_capex_sgd_per_kw", 180, "SGD/kW", 180, 180, 1, "Battery power CAPEX.", "电池功率转换单位投资成本。"),
            ("saidi_min_per_year", 180, "min/year", 180, 180, 1, "Annual outage duration.", "年平均停电持续时间。"),
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
            ("battery_roundtrip_efficiency", 0.81, "fraction", "", "", "", "Battery round-trip efficiency.", "电池往返效率。"),
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
        ]
        return [dict(zip(PARAMETER_COLUMNS, row, strict=True)) for row in rows]

    def _sources(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        with TemporaryDirectory() as tmp:
            input_dir = self._write_case(Path(tmp))
            case = discover_input_cases(input_dir)[0]
            _, _, cfg, _ = load_input_case(case, self._fast_config())
            optimal_designs = pd.DataFrame(
                [
                    {
                        "design_type": "reliability_constrained_design",
                        "available": True,
                        "pv_capacity_kw": 100.0,
                        "battery_energy_kwh": 200.0,
                    }
                ]
            )
            outage_scenarios = pd.DataFrame(
                [
                    {"frontier_scenario_id": 1, "outage_seed": 1000},
                    {"frontier_scenario_id": 2, "outage_seed": 1001},
                    {"frontier_scenario_id": 3, "outage_seed": 1002},
                ]
            )
            return build_outage_energy_balance_sources(case, cfg, optimal_designs, outage_scenarios)

    def test_select_max_impact_event_uses_duration_only_as_tie_breaker(self) -> None:
        catalog = pd.DataFrame(
            [
                {"frontier_scenario_id": 1, "event_id": 1, "event_start_hour": 100.0, "event_net_deficit_kwh": 10.0, "event_duration_hours": 10.0},
                {"frontier_scenario_id": 2, "event_id": 1, "event_start_hour": 200.0, "event_net_deficit_kwh": 30.0, "event_duration_hours": 1.0},
                {"frontier_scenario_id": 3, "event_id": 1, "event_start_hour": 300.0, "event_net_deficit_kwh": 30.0, "event_duration_hours": 2.0},
                {"frontier_scenario_id": 4, "event_id": 1, "event_start_hour": 5.0, "event_net_deficit_kwh": 100.0, "event_duration_hours": 5.0},
            ]
        )

        selected = _select_max_impact_event(catalog, total_hours=8760)

        self.assertEqual(selected["frontier_scenario_id"], 3)

    def test_build_sources_returns_complete_72h_window_on_outage_day(self) -> None:
        summary, dispatch = self._sources()

        self.assertEqual(len(dispatch), 72)
        self.assertEqual(dispatch["plot_hour"].astype(int).tolist(), list(range(72)))
        self.assertEqual(summary["selection_rule"].iloc[0], "max_event_net_deficit")
        self.assertAlmostEqual(float(summary["minimum_soc_fraction"].iloc[0]), 0.1)
        self.assertAlmostEqual(float(summary["reserve_fraction"].iloc[0]), 0.1)
        outage_hours = dispatch.loc[dispatch["is_outage_interval"].map(bool), "plot_hour"].astype(float)
        self.assertFalse(outage_hours.empty)
        self.assertGreaterEqual(float(outage_hours.min()), 24.0)
        self.assertLess(float(outage_hours.max()), 48.0)
        self.assertTrue((dispatch["conversion_loss_kwh"].astype(float) >= -1e-12).all())

    def test_conversion_loss_matches_efficiency_formula(self) -> None:
        loss = _conversion_loss(
            pd.Series([10.0, 0.0]),
            pd.Series([0.0, 9.0]),
            eta=0.9,
        )

        self.assertAlmostEqual(float(loss[0]), 1.0)
        self.assertAlmostEqual(float(loss[1]), 1.0)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_soc_zero_alignment_helper_matches_axis_transform(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.0, 2.0))
        ax.set_ylim(-800.0, 400.0)
        soc_ax = ax.twinx()
        soc_ax.set_ylim(_soc_ylim_for_zero_alignment(ax.get_ylim(), soc_upper=1.05))
        fig.canvas.draw()

        power_zero = ax.transData.transform((0.0, 0.0))[1]
        soc_zero = soc_ax.transData.transform((0.0, 0.0))[1]
        self.assertAlmostEqual(float(power_zero), float(soc_zero), places=6)
        plt.close(fig)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_plot_exports_outage_context_figure(self) -> None:
        summary, dispatch = self._sources()
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            context_paths = plot_outage_context_figure(out, "SG_UTown", summary, dispatch)

            suffixes = {path.suffix for path in context_paths.values()}
            self.assertTrue({".png", ".svg", ".pdf", ".csv"}.issubset(suffixes))
            for path in context_paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

            svg_text = next(path for path in context_paths.values() if path.suffix == ".svg").read_text()
            self.assertIn(">a</text>", svg_text)
            for label in "bcde":
                self.assertNotIn(f">{label}</text>", svg_text)


if __name__ == "__main__":
    unittest.main()
