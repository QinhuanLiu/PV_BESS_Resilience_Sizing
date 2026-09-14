from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from src.app import cli as cli_module
from src.app.workflow import run_selected
from src.figures.lolp_zero_three_year_uncertainty import make_lolp_zero_three_year_uncertainty_figures
from src.figures.lolp_zero_uncertainty import (
    _padded_x_limits,
    make_lolp_zero_uncertainty_figures,
    shared_lolp_zero_distribution_x_limits,
)
from src.io.config import DEFAULT_CONFIG
from src.io.data_loader import discover_input_cases, load_input_case
from src.lolp_zero_uncertainty import (
    DEFAULT_LOLP_ZERO_SCENARIO_COUNT,
    MANILA_LOLP_ZERO_CASE_KEY,
    select_lolp_zero_min_cost_design,
    summarize_lolp_zero_uncertainty_results,
)
from src.lolp_zero_three_year_uncertainty import (
    DEFAULT_LOLP_ZERO_THREE_YEAR_SCENARIO_COUNT,
    DEFAULT_LOLP_ZERO_THREE_YEAR_SIMULATION_YEARS,
    LOLP_ZERO_THREE_YEAR_FEASIBILITY_COLUMN,
    build_lolp_zero_three_year_case_frame,
    lolp_zero_three_year_seed_sequence,
    pv_bess_mean_annual_multiyear_cost,
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


def _lolp_zero_rows(count: int = 30) -> pd.DataFrame:
    rows = []
    for scenario_id in range(1, count + 1):
        pv = 3.5 + 0.1 * (scenario_id % 5)
        battery = 5.0 + 0.5 * (scenario_id % 4)
        cost = 1500.0 + 12.0 * scenario_id + 20.0 * (scenario_id % 3)
        rows.append(
            {
                "uncertainty_scenario_id": scenario_id,
                "outage_seed": 1000 + scenario_id - 1,
                "pv_capacity_kw": pv,
                "pv_capacity_mw": pv / 1000.0,
                "battery_energy_kwh": battery,
                "battery_energy_mwh": battery / 1000.0,
                "pv_bess_system_annual_cost_sgd": cost,
                "system_total_cost_sgd_per_year": cost,
                "system_total_cost_million_sgd_per_year": cost / 1e6,
                "unserved_energy_kwh_per_year": 0.0,
                "UE_kwh_per_year": 0.0,
                "scenario_LOLP": 0.0,
                "outage_weighted_load_kwh": 10.0,
            }
        )
    return pd.DataFrame(rows)


def _outage_rows(count: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_key": [MANILA_LOLP_ZERO_CASE_KEY] * count,
            "city_key": ["Manila"] * count,
            "building_key": ["SingleFamily"] * count,
            "uncertainty_scenario_id": list(range(1, count + 1)),
            "outage_seed": list(range(1000, 1000 + count)),
            "simulated_saidi_min_per_year": [120.0] * count,
            "simulated_saifi_per_year": [2.0] * count,
            "max_event_duration_min": [70.0] * count,
            "outage_weighted_load_kwh": [10.0] * count,
        }
    )


def _lolp_zero_three_year_rows(count: int = 30) -> pd.DataFrame:
    rows = _lolp_zero_rows(count)
    rows["simulation_years"] = 3
    rows["outage_seed"] = [1000 + (idx - 1) * 3 for idx in rows["uncertainty_scenario_id"].astype(int)]
    rows["outage_seed_sequence"] = [
        f"{1000 + (idx - 1) * 3};{1001 + (idx - 1) * 3};{1002 + (idx - 1) * 3}"
        for idx in rows["uncertainty_scenario_id"].astype(int)
    ]
    rows["unserved_energy_kwh_total_3yr"] = 0.0
    rows["outage_weighted_load_kwh_total_3yr"] = 30.0
    rows["outage_weighted_load_kwh_per_year"] = 10.0
    rows["dispatch_backend"] = "greedy"
    return rows


def _three_year_outage_rows(count: int = 30, years: int = 3) -> pd.DataFrame:
    rows = []
    for scenario_id in range(1, count + 1):
        seeds = lolp_zero_three_year_seed_sequence(1000, scenario_id, years)
        for year, seed in enumerate(seeds, start=1):
            rows.append(
                {
                    "case_key": MANILA_LOLP_ZERO_CASE_KEY,
                    "city_key": "Manila",
                    "building_key": "SingleFamily",
                    "uncertainty_scenario_id": scenario_id,
                    "simulation_year": year,
                    "simulation_years": years,
                    "outage_seed": seed,
                    "outage_seed_sequence": ";".join(str(value) for value in seeds),
                    "simulated_saidi_min_per_year": 120.0,
                    "simulated_saifi_per_year": 2.0,
                    "max_event_duration_min": 70.0,
                    "outage_weighted_load_kwh": 10.0,
                }
            )
    return pd.DataFrame(rows)


def _cv_summary() -> pd.DataFrame:
    return summarize_lolp_zero_uncertainty_results(_lolp_zero_rows())


def _parameter_rows() -> list[dict[str, object]]:
    rows = [
        ("pv_capex_sgd_per_kw", 900, "SGD/kW", 800, 1000, 200, "PV unit CAPEX.", "光伏单位装机投资成本。"),
        ("battery_capex_sgd_per_kwh", 300, "SGD/kWh", 200, 400, 100, "Battery energy CAPEX.", "电池储能单位能量容量投资成本。"),
        ("battery_power_capex_sgd_per_kw", 150, "SGD/kW", 120, 180, 60, "Battery power CAPEX.", "电池储能单位功率容量投资成本。"),
        ("saidi_min_per_year", 120, "min/year", 0, 120, 60, "Annual outage duration.", "年平均停电持续时间。"),
        ("saifi_per_year", 2, "events/year", 1, 2, 1, "Annual outage count.", "年平均停电次数。"),
        ("voll_sgd_per_kwh", 5, "SGD/kWh", 0, 10, 5, "Value of lost load.", "单位未供电电量对应的停电损失价值。"),
        ("city_name", "Manila", "-", "", "", "", "City key.", "城市或地区名称。"),
        ("building_name", "SingleFamily", "-", "", "", "", "Building key.", "建筑或建筑群名称。"),
        ("local_refine_window_kw", 0, "kW", "", "", "", "PV local refinement window.", "光伏容量局部细化窗口。"),
        ("local_refine_step_kw", 1, "kW", "", "", "", "PV local refinement step.", "光伏容量局部细化步长。"),
        ("rooftop_area_m2", 100, "m2", "", "", "", "Effective rooftop area.", "可安装光伏的有效屋顶面积。"),
        ("pv_module_power_density_kw_per_m2", 0.2, "kW/m2", "", "", "", "PV module power density.", "单位面积光伏组件装机功率。"),
        ("pv_om_fraction_per_year", 0.015, "fraction/year", "", "", "", "PV annual O&M.", "光伏年度运维成本占 CAPEX 比例。"),
        ("pv_project_years", 15, "years", "", "", "", "PV project life.", "光伏项目寿命。"),
        ("cooling_cop", 5.0, "-", "", "", "", "Cooling COP.", "制冷系统 COP。"),
        ("heating_cop", 3.5, "-", "", "", "", "Heating COP.", "供热系统 COP。"),
        ("battery_om_fraction_per_year", 0.02, "fraction/year", "", "", "", "Battery O&M.", "电池年度运维成本占 CAPEX 比例。"),
        ("battery_max_c_rate", 0.5, "1/h", "", "", "", "Battery max C-rate.", "电池最大充放电倍率。"),
        ("battery_roundtrip_efficiency", 0.9, "fraction", "", "", "", "Battery round-trip efficiency.", "电池往返效率。"),
        ("battery_soc_min_fraction", 0.1, "fraction", "", "", "", "Minimum SOC.", "电池最小 SOC。"),
        ("battery_soc_max_fraction", 0.9, "fraction", "", "", "", "Maximum SOC.", "电池最大 SOC。"),
        ("battery_initial_soc_fraction", 0.5, "fraction", "", "", "", "Initial SOC.", "初始 SOC。"),
        ("battery_eol_soh_fraction", 0.8, "fraction", "", "", "", "End-of-life SOH.", "电池寿命终止 SOH。"),
        ("battery_rated_cycle_life_to_eol", 9000, "cycles", "", "", "", "Rated cycle life.", "到寿命终止的额定循环次数。"),
        ("battery_calendar_loss_per_year", 0.008, "fraction/year", "", "", "", "Calendar degradation.", "电池日历衰减率。"),
        ("battery_max_replacement_years", 30, "years", "", "", "", "Maximum replacement interval.", "最长更换周期。"),
        ("local_refine_window_kwh", 0, "kWh", "", "", "", "Local refinement window.", "在粗筛最优容量附近继续细化的窗口。"),
        ("local_refine_step_kwh", 1, "kWh", "", "", "", "Local refinement step.", "细化搜索步长。"),
        ("outage_seed", 42, "-", "", "", "", "Outage random seed.", "停电随机生成种子。"),
        ("forced_event_mode", "match_saidi", "-", "", "", "", "Outage event generation mode.", "停电事件生成方式。"),
        ("uncertainty_baseline_scenario_count", 2, "scenarios", "", "", "", "Uncertainty baseline count.", "不确定性基准分析场景数量。"),
        ("uncertainty_sensitivity_scenario_count", 2, "scenarios", "", "", "", "Uncertainty sensitivity count.", "不确定性敏感性分析场景数量。"),
        ("uncertainty_seed_start", 1000, "-", "", "", "", "Uncertainty seed start.", "不确定性停电场景起始随机种子。"),
        ("parallel_workers", 1, "workers", "", "", "", "Parallel workers for uncertainty analysis.", "不确定性分析并行进程数。"),
        ("pv_during_outage_available", "true", "boolean", "", "", "", "PV availability during outage.", "停电期间 PV 是否可用。"),
    ]
    return [dict(zip(PARAMETER_COLUMNS, row, strict=True)) for row in rows]


def _write_manila_case(root: Path) -> Path:
    case_dir = root / "Input" / "Manila_SingleFamily"
    case_dir.mkdir(parents=True, exist_ok=True)
    load = pd.DataFrame(
        {
            "Cooling": [10.0] * 8760,
            "Heating": [0.0] * 8760,
            "Lighting": [1.5] * 8760,
            "Equipment": [1.0] * 8760,
        }
    )
    pv = pd.DataFrame({"PV_output_kWh_per_m2": [0.6] * 8760})
    tou = pd.DataFrame({"hour": range(24), "tou_sgd_per_kwh": [0.20] * 24})
    params = pd.DataFrame(_parameter_rows(), columns=PARAMETER_COLUMNS)
    load.to_csv(case_dir / "Manila_SingleFamily_Load.csv", index=False)
    pv.to_csv(case_dir / "Manila_SingleFamily_PV.csv", index=False)
    tou.to_csv(case_dir / "Manila_SingleFamily_TOU.csv", index=False)
    params.to_csv(case_dir / "Manila_SingleFamily_Parameter.csv", index=False)
    return root / "Input"


class LOLPZeroUncertaintyTests(unittest.TestCase):
    def test_selects_minimum_cost_zero_lolp_candidate(self) -> None:
        candidates = pd.DataFrame(
            [
                {"pv_capacity_kw": 4.0, "battery_energy_kwh": 0.0, "pv_bess_system_annual_cost_sgd": 100.0, "unserved_energy_kwh_per_year": 0.5},
                {"pv_capacity_kw": 5.0, "battery_energy_kwh": 6.0, "pv_bess_system_annual_cost_sgd": 90.0, "unserved_energy_kwh_per_year": 0.0},
                {"pv_capacity_kw": 4.0, "battery_energy_kwh": 6.0, "pv_bess_system_annual_cost_sgd": 80.0, "unserved_energy_kwh_per_year": 0.0},
            ]
        )
        selected = select_lolp_zero_min_cost_design(candidates)

        self.assertEqual(selected["pv_capacity_kw"], 4.0)
        self.assertEqual(selected["battery_energy_kwh"], 6.0)
        self.assertEqual(selected["pv_bess_system_annual_cost_sgd"], 80.0)

    def test_select_lolp_zero_reports_missing_feasible_candidate(self) -> None:
        candidates = pd.DataFrame(
            [
                {"pv_capacity_kw": 4.0, "battery_energy_kwh": 0.0, "pv_bess_system_annual_cost_sgd": 100.0, "unserved_energy_kwh_per_year": 0.5}
            ]
        )
        with self.assertRaisesRegex(ValueError, "No LOLP=0 candidate"):
            select_lolp_zero_min_cost_design(candidates)

    def test_three_year_seed_sequence_uses_consecutive_groups(self) -> None:
        self.assertEqual(lolp_zero_three_year_seed_sequence(1000, 1, 3), [1000, 1001, 1002])
        self.assertEqual(lolp_zero_three_year_seed_sequence(1000, 2, 3), [1003, 1004, 1005])

    def test_three_year_case_frame_repeats_typical_year_with_unique_hours(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _write_manila_case(root)
            config = deepcopy(DEFAULT_CONFIG)
            config.setdefault("dispatch", {})["backend"] = "greedy"
            case = next(case for case in discover_input_cases(input_dir) if case.case_key == MANILA_LOLP_ZERO_CASE_KEY)
            _, _, case_config, _ = load_input_case(case, config)

            frame, pv_meta = build_lolp_zero_three_year_case_frame(case, case_config, 2.0, simulation_years=3)

            self.assertEqual(len(frame), 8760 * 3)
            self.assertTrue(frame["hour_index"].is_unique)
            self.assertEqual(frame["hour_index"].iloc[0], 0)
            self.assertEqual(frame["hour_index"].iloc[-1], 8760 * 3 - 1)
            self.assertEqual(frame["simulation_year"].astype(int).unique().tolist(), [1, 2, 3])
            self.assertEqual(frame.groupby("simulation_year")["typical_year_hour_index"].min().tolist(), [0, 0, 0])
            self.assertEqual(frame.groupby("simulation_year")["typical_year_hour_index"].max().tolist(), [8759, 8759, 8759])
            self.assertEqual(float(pv_meta["simulation_years"]), 3.0)

    def test_multiyear_cost_averages_operating_terms_per_year(self) -> None:
        config = deepcopy(DEFAULT_CONFIG)
        config["active_scenario"] = {"voll_sgd_per_kwh": 5.0}
        config["pv"]["capex_sgd_per_kw"] = 100.0
        config["pv"]["om_fraction_per_year"] = 0.01
        config["pv"]["project_years"] = 10.0
        config["economics"] = {
            "battery_capex_sgd_per_kwh": 100.0,
            "battery_power_capex_sgd_per_kw": 20.0,
            "battery_om_fraction_per_year": 0.01,
        }
        config["battery"].update(
            {
                "eol_soh_fraction": 0.8,
                "rated_cycle_life_to_eol": 9000.0,
                "calendar_loss_per_year": 0.008,
                "max_replacement_years": 30.0,
            }
        )
        dispatch = pd.DataFrame(
            {
                "grid_import_kwh": [10.0] * 6,
                "tou_sgd_per_kwh": [2.0] * 6,
                "unmet_load_kwh_equiv": [1.0] * 6,
                "battery_charge_kwh": [1.0] * 6,
                "battery_discharge_kwh": [1.0] * 6,
                "battery_soc_kwh": [5.0] * 6,
            }
        )
        cost = pv_bess_mean_annual_multiyear_cost(
            pd.DataFrame(),
            dispatch,
            {"battery_energy_kwh": 10.0, "battery_power_kw": 5.0},
            {"pv_capacity_kw": 2.0, "pv_capacity_mw": 0.002},
            config,
            simulation_years=3,
        )

        self.assertAlmostEqual(cost["pv_bess_grid_cost_total_3yr_sgd"], 120.0)
        self.assertAlmostEqual(cost["pv_bess_grid_cost_sgd"], 40.0)
        self.assertAlmostEqual(cost["unserved_energy_kwh_total_3yr"], 6.0)
        self.assertAlmostEqual(cost["unserved_energy_kwh_per_year"], 2.0)
        self.assertAlmostEqual(cost["unserved_energy_cost_sgd_per_year"], 10.0)
        self.assertAlmostEqual(cost["equivalent_full_cycles_total_3yr"], 0.6)
        self.assertAlmostEqual(cost["equivalent_full_cycles"], 0.2)

    def test_three_year_selects_by_total_unserved_energy(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "pv_capacity_kw": 4.0,
                    "battery_energy_kwh": 5.0,
                    "pv_bess_system_annual_cost_sgd": 50.0,
                    "unserved_energy_kwh_total_3yr": 0.2,
                },
                {
                    "pv_capacity_kw": 5.0,
                    "battery_energy_kwh": 6.0,
                    "pv_bess_system_annual_cost_sgd": 90.0,
                    "unserved_energy_kwh_total_3yr": 0.0,
                },
                {
                    "pv_capacity_kw": 6.0,
                    "battery_energy_kwh": 7.0,
                    "pv_bess_system_annual_cost_sgd": 80.0,
                    "unserved_energy_kwh_total_3yr": 0.0,
                },
            ]
        )

        selected = select_lolp_zero_min_cost_design(
            candidates,
            feasibility_column=LOLP_ZERO_THREE_YEAR_FEASIBILITY_COLUMN,
        )

        self.assertEqual(selected["pv_capacity_kw"], 6.0)
        self.assertEqual(selected["battery_energy_kwh"], 7.0)

    def test_cv_summary_uses_population_standard_deviation(self) -> None:
        rows = _lolp_zero_rows(3)
        rows["system_total_cost_sgd_per_year"] = [100.0, 200.0, 300.0]
        rows["pv_capacity_kw"] = [1.0, 2.0, 3.0]
        rows["battery_energy_kwh"] = [10.0, 10.0, 10.0]

        summary = summarize_lolp_zero_uncertainty_results(rows)
        cost = summary[summary["metric"].eq("annualized_system_cost")].iloc[0]
        battery = summary[summary["metric"].eq("battery_energy")].iloc[0]

        self.assertAlmostEqual(float(cost["mean"]), 200.0)
        self.assertAlmostEqual(float(cost["std"]), float(np.std([100.0, 200.0, 300.0], ddof=0)))
        self.assertAlmostEqual(float(cost["coefficient_of_variation"]), float(np.std([100.0, 200.0, 300.0], ddof=0) / 200.0))
        self.assertAlmostEqual(float(battery["coefficient_of_variation"]), 0.0)

    def test_distribution_x_limits_keep_whitespace_around_data(self) -> None:
        lower, upper = _padded_x_limits(np.asarray([4.0, 4.5, 5.0], dtype=float))

        self.assertLess(lower, 4.0)
        self.assertGreater(upper, 5.0)
        self.assertAlmostEqual(lower, 3.82)
        self.assertAlmostEqual(upper, 5.18)

    def test_distribution_x_limits_do_not_go_negative_for_nonnegative_data(self) -> None:
        lower, upper = _padded_x_limits(np.asarray([0.0, 2.5, 5.5], dtype=float))

        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 5.5)

    def test_shared_distribution_x_limits_use_both_figure_sources(self) -> None:
        figure1 = _lolp_zero_rows(3)
        figure2 = _lolp_zero_three_year_rows(3)
        figure1["pv_capacity_kw"] = [4.0, 4.2, 4.4]
        figure2["pv_capacity_kw"] = [5.0, 5.2, 5.4]
        figure1["battery_energy_kwh"] = [2.0, 2.5, 3.0]
        figure2["battery_energy_kwh"] = [6.0, 6.5, 7.0]
        figure1["system_total_cost_sgd_per_year"] = [1000.0, 1100.0, 1200.0]
        figure2["system_total_cost_sgd_per_year"] = [2000.0, 2100.0, 2200.0]

        limits = shared_lolp_zero_distribution_x_limits(figure1, figure2)

        self.assertLess(limits["annualized_cost"][0], 1.0)
        self.assertGreater(limits["annualized_cost"][1], 2.2)
        self.assertLess(limits["pv_capacity"][0], 4.0)
        self.assertGreater(limits["pv_capacity"][1], 5.4)
        self.assertLess(limits["battery_capacity"][0], 2.0)
        self.assertGreater(limits["battery_capacity"][1], 7.0)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_lolp_zero_distribution_figures_export_three_single_plots(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = make_lolp_zero_uncertainty_figures(out, MANILA_LOLP_ZERO_CASE_KEY, _lolp_zero_rows())

            expected_stems = [
                "Manila_SingleFamily_LOLP_Zero_Uncertainty_Annualized_Cost_Distribution",
                "Manila_SingleFamily_LOLP_Zero_Uncertainty_PV_Capacity_Distribution",
                "Manila_SingleFamily_LOLP_Zero_Uncertainty_Battery_Capacity_Distribution",
            ]
            for stem in expected_stems:
                for suffix in ("png", "svg", "pdf"):
                    self.assertTrue((out / f"{stem}.{suffix}").exists())
                self.assertTrue((out / f"{stem}_Source_Data.csv").exists())
            svg = (out / f"{expected_stems[0]}.svg").read_text(encoding="utf-8")
            self.assertIn("Manila LOLP=0 annualized cost", svg)
            self.assertIn("Probability density", svg)
            self.assertIn("CV=", svg)
            self.assertIn("annualized_cost_source_data", paths)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_figure2_three_year_distribution_figures_export_three_single_plots(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = make_lolp_zero_three_year_uncertainty_figures(
                out,
                MANILA_LOLP_ZERO_CASE_KEY,
                _lolp_zero_three_year_rows(),
            )

            expected_stems = [
                "Manila_SingleFamily_Figure2_LOLP_Zero_Three_Year_Uncertainty_Annualized_Cost_Distribution",
                "Manila_SingleFamily_Figure2_LOLP_Zero_Three_Year_Uncertainty_PV_Capacity_Distribution",
                "Manila_SingleFamily_Figure2_LOLP_Zero_Three_Year_Uncertainty_Battery_Capacity_Distribution",
            ]
            for stem in expected_stems:
                for suffix in ("png", "svg", "pdf"):
                    self.assertTrue((out / f"{stem}.{suffix}").exists())
                self.assertTrue((out / f"{stem}_Source_Data.csv").exists())
            svg = (out / f"{expected_stems[0]}.svg").read_text(encoding="utf-8")
            self.assertIn("Manila three-year LOLP=0 annualized cost", svg)
            self.assertIn("Probability density", svg)
            self.assertIn("CV=", svg)
            self.assertIn("figure2_annualized_cost_source_data", paths)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_workflow_lolp_zero_uncertainty_defaults_to_manila_and_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _write_manila_case(root)
            output_dir = root / "outputs"
            config = deepcopy(DEFAULT_CONFIG)
            config.setdefault("dispatch", {})["backend"] = "greedy"
            with patch(
                "src.app.workflow.run_lolp_zero_uncertainty_analysis",
                return_value=(_lolp_zero_rows(), _outage_rows(), _cv_summary()),
            ) as analysis:
                paths = run_selected(
                    config,
                    input_dir,
                    output_dir,
                    run_baseline=False,
                    run_sensitivity=False,
                    run_lolp_zero_uncertainty=True,
                )

            self.assertEqual(analysis.call_args.kwargs["scenario_count"], DEFAULT_LOLP_ZERO_SCENARIO_COUNT)
            uncertainty = output_dir / "Manila" / "Manila_SingleFamily_Uncertainty_Analysis" / "Manila_SingleFamily_LOLP_Zero_Uncertainty"
            self.assertTrue(uncertainty.is_dir())
            all_results = pd.read_csv(uncertainty / "Manila_SingleFamily_LOLP_Zero_Uncertainty_All_Results.csv")
            cv_summary = pd.read_csv(uncertainty / "Manila_SingleFamily_LOLP_Zero_Uncertainty_CV_Summary.csv")
            self.assertEqual(len(all_results), 30)
            self.assertTrue((all_results["unserved_energy_kwh_per_year"].astype(float) <= 1e-6).all())
            self.assertEqual(set(cv_summary["metric"]), {"annualized_system_cost", "pv_capacity", "battery_energy"})
            self.assertTrue(any("LOLP_Zero_Uncertainty" in str(path) for path in paths.values()))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_workflow_figure2_three_year_uncertainty_defaults_to_manila_and_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _write_manila_case(root)
            output_dir = root / "outputs"
            config = deepcopy(DEFAULT_CONFIG)
            config.setdefault("dispatch", {})["backend"] = "greedy"
            with patch(
                "src.app.workflow.run_lolp_zero_three_year_uncertainty_analysis",
                return_value=(_lolp_zero_three_year_rows(), _three_year_outage_rows(), summarize_lolp_zero_uncertainty_results(_lolp_zero_three_year_rows())),
            ) as analysis:
                paths = run_selected(
                    config,
                    input_dir,
                    output_dir,
                    run_baseline=False,
                    run_sensitivity=False,
                    run_lolp_zero_three_year_uncertainty=True,
                )

            self.assertEqual(analysis.call_args.kwargs["scenario_count"], DEFAULT_LOLP_ZERO_THREE_YEAR_SCENARIO_COUNT)
            self.assertEqual(analysis.call_args.kwargs["simulation_years"], DEFAULT_LOLP_ZERO_THREE_YEAR_SIMULATION_YEARS)
            uncertainty = (
                output_dir
                / "Manila"
                / "Manila_SingleFamily_Uncertainty_Analysis"
                / "Manila_SingleFamily_Figure2_LOLP_Zero_Three_Year_Uncertainty"
            )
            self.assertTrue(uncertainty.is_dir())
            stem = "Manila_SingleFamily_Figure2_LOLP_Zero_Three_Year_Uncertainty"
            all_results = pd.read_csv(uncertainty / f"{stem}_All_Results.csv")
            outage_scenarios = pd.read_csv(uncertainty / f"{stem}_Outage_Scenarios.csv")
            cv_summary = pd.read_csv(uncertainty / f"{stem}_CV_Summary.csv")
            self.assertEqual(len(all_results), 30)
            self.assertEqual(len(outage_scenarios), 90)
            self.assertTrue((all_results["unserved_energy_kwh_total_3yr"].astype(float) <= 1e-6).all())
            self.assertEqual(set(cv_summary["metric"]), {"annualized_system_cost", "pv_capacity", "battery_energy"})
            self.assertTrue(any("Figure2_LOLP_Zero_Three_Year_Uncertainty" in str(path) for path in paths.values()))

    def test_cli_lolp_zero_uncertainty_flag_reaches_workflow(self) -> None:
        argv = ["pv_bess_sizing", "--analysis", "lolp-zero-uncertainty"]
        config = {"input": {"directory": "Input"}, "outputs": {"directory": "outputs"}}
        with (
            patch("sys.argv", argv),
            patch("src.app.cli.load_config", return_value=config),
            patch("src.app.cli.run_selected", return_value={}) as run_selected_mock,
        ):
            cli_module.main()

        kwargs = run_selected_mock.call_args.kwargs
        self.assertTrue(kwargs["run_lolp_zero_uncertainty"])
        self.assertFalse(kwargs["run_baseline"])
        self.assertFalse(kwargs["run_uncertainty_baseline"])
        self.assertFalse(kwargs["run_uncertainty_sensitivity"])

    def test_cli_figure2_three_year_lolp_zero_flag_reaches_workflow(self) -> None:
        argv = ["pv_bess_sizing", "--analysis", "lolp-zero-three-year-uncertainty"]
        config = {"input": {"directory": "Input"}, "outputs": {"directory": "outputs"}}
        with (
            patch("sys.argv", argv),
            patch("src.app.cli.load_config", return_value=config),
            patch("src.app.cli.run_selected", return_value={}) as run_selected_mock,
        ):
            cli_module.main()

        kwargs = run_selected_mock.call_args.kwargs
        self.assertTrue(kwargs["run_lolp_zero_three_year_uncertainty"])
        self.assertFalse(kwargs["run_lolp_zero_uncertainty"])
        self.assertFalse(kwargs["run_baseline"])


if __name__ == "__main__":
    unittest.main()
