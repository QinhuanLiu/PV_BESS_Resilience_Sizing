from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.app import cli as cli_module
from src.app.workflow import run_selected
from src.dispatch import solve_dispatch_greedy
from src.figures.single_outage_city_baseline import (
    BRISBANE_SENSITIVITY_DRIVER_ORDER,
    BRISBANE_SENSITIVITY_TOU_K_VALUES,
    AFFORDABLE_COST_DESIGN_CASES,
    CITY_ORDER,
    COST_DESIGN_CASES,
    CVAR_ALPHA,
    CVAR_LAMBDAS,
    DAY_NIGHT_COST_DESIGN_CASES,
    FIGURE_1_TO_4_CITY_ORDER,
    FIGURE_C_CITY_ORDER,
    FIGURE_D_CITY_ORDER,
    FIGURE_NAMES,
    SCENARIO_DISTRIBUTION_COUNT,
    SINGAPORE_BRISBANE_CITY_ORDER,
    SINGLE_OUTAGE_BAR_STEP,
    SINGLE_OUTAGE_BAR_WIDTH,
    SINGLE_OUTAGE_BASE_FONT_SIZE,
    SINGLE_OUTAGE_COLORBAR_LABEL_SIZE,
    SINGLE_OUTAGE_COST_XTICK_SIZE,
    SINGLE_OUTAGE_TICK_FONT_SIZE,
    SINGLE_OUTAGE_TITLE_FONT_SIZE,
    _apply_single_outage_readable_font_style,
    _contour_colormap,
    _contour_metric_specs,
    _cost_component_colors,
    _draw_contour_panel,
    _figure_c_outage_mode,
    _brisbane_sensitivity_tasks,
    _evaluate_brisbane_sensitivity_tasks_parallel,
    _evaluate_cvar_candidate_scenarios_parallel,
    _mean_preserving_tou_profile,
    _metric_levels,
    _single_outage_energy_balance_palette,
    _select_cvar_design_curve,
    _stack_city_design_cost_bars,
    _summarize_cvar_candidate_results,
    _tou_step_xy,
    build_single_outage_city_affordable_lolp_sources,
    build_single_outage_city_baseline_sources,
    build_single_outage_city_brisbane_sensitivity_source,
    build_single_outage_city_cvar_sources,
    build_single_outage_city_scenario_distribution_sources,
    build_single_outage_city_tou_profile_source,
    calculate_pv_self_use_metrics,
    make_single_outage_city_baseline_figures,
)
from src.figures.core import apply_nature_style, matplotlib_context
from src.figures.display_units import (
    DISPLAY_BATTERY_ENERGY,
    DISPLAY_PV_CAPACITY,
    add_display_unit_columns,
    figure_display_units,
)
from src.io.config import DEFAULT_CONFIG
from src.io.data_loader import load_case_parameters


INPUT_ROOT = Path("Input")


def _assert_city_label_order(testcase: unittest.TestCase, svg: str, labels: list[str]) -> None:
    positions = []
    for label in labels:
        position = svg.find(label)
        testcase.assertGreaterEqual(position, 0, f"{label} not found in SVG")
        positions.append(position)
    testcase.assertEqual(positions, sorted(positions))


class SingleOutageCityBaselineFigureTests(unittest.TestCase):
    def _fast_config(self) -> dict:
        config = deepcopy(DEFAULT_CONFIG)
        config.setdefault("dispatch", {})["backend"] = "greedy"
        config.setdefault("capacity_search", {})["coarse_intervals"] = 1
        config.setdefault("capacity_search", {})["local_refine_divisions"] = 1
        config.setdefault("capacity_search", {})["max_upper_expansions"] = 0
        return config

    def _fake_uncertainty_results(self, case, config, case_key, scenario_count=100, seed_start=1000):
        city_factor = float(len(case_key))
        rows = []
        for scenario_id in range(1, int(scenario_count) + 1):
            rows.append(
                {
                    "uncertainty_scenario_id": scenario_id,
                    "outage_seed": int(seed_start) + scenario_id - 1,
                    "pv_capacity_kw": city_factor + 0.1 * scenario_id,
                    "battery_energy_kwh": 2.0 * city_factor + 0.2 * scenario_id,
                    "unserved_energy_kwh_per_year": 1.0 + 0.05 * scenario_id,
                    "outage_weighted_load_kwh": 100.0 + scenario_id,
                    "simulated_saidi_min_per_year": 60.0 + scenario_id,
                    "simulated_saifi_per_year": 1.0 + 0.01 * scenario_id,
                    "max_event_duration_min": 30.0 + scenario_id,
                    "pv_bess_system_annual_cost_sgd": 1000.0 + 10.0 * scenario_id,
                }
            )
        return pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame()

    def _fake_cvar_parallel_eval(
        self,
        candidates,
        scenario_results,
        cases_by_city=None,
        case_configs_by_city=None,
        config=None,
    ):
        rows = []
        scenarios = scenario_results.sort_values(["city_key", "scenario_id"])
        for candidate in candidates.sort_values(["city_key", "candidate_order"]).to_dict("records"):
            city_scenarios = scenarios[scenarios["city_key"].astype(str) == str(candidate["city_key"])]
            for scenario in city_scenarios.to_dict("records"):
                scenario_id = int(scenario["scenario_id"])
                candidate_order = int(candidate["candidate_order"])
                pv = float(candidate["pv_capacity_kw"])
                battery = float(candidate["battery_energy_kwh"])
                fixed = 20.0 * pv + 10.0 * battery
                operation = max(1000.0 - 2.0 * pv - battery, 1.0)
                ue = max(0.0, 0.08 * scenario_id / max(candidate_order, 1))
                outage_loss = 100.0 * ue
                outage_load = 100.0 + scenario_id
                system_cost = fixed + operation + outage_loss
                rows.append(
                    {
                        "city_key": str(candidate["city_key"]),
                        "city_label": str(candidate["city_label"]),
                        "case_key": str(candidate["case_key"]),
                        "source_case_key": str(candidate.get("source_case_key", candidate["case_key"])),
                        "candidate_id": str(candidate["candidate_id"]),
                        "candidate_order": candidate_order,
                        "scenario_id": scenario_id,
                        "outage_seed": int(scenario["outage_seed"]),
                        "pv_capacity_kw": pv,
                        "pv_capacity_mw": pv / 1000.0,
                        "battery_energy_kwh": battery,
                        "battery_energy_mwh": battery / 1000.0,
                        "fixed_investment_cost_sgd_per_year": fixed,
                        "operation_cost_sgd_per_year": operation,
                        "outage_loss_sgd_per_year": outage_loss,
                        "system_total_cost_sgd_per_year": system_cost,
                        "system_total_cost_million_sgd_per_year": system_cost / 1e6,
                        "UE_kwh_per_year": ue,
                        "UE_mwh_per_year": ue / 1000.0,
                        "ue_kwh_per_year": ue,
                        "energy_based_lolp": min(ue / outage_load, 1.0),
                        "energy_based_lolp_percent": 100.0 * min(ue / outage_load, 1.0),
                        "outage_weighted_load_kwh": outage_load,
                        "annual_outage_hours": float(scenario.get("annual_outage_hours", 1.0)),
                        "simulated_saidi_min_per_year": 60.0,
                        "simulated_saifi_per_year": 1.0,
                        "max_event_duration_hours": 1.0,
                    }
                )
        return pd.DataFrame(rows)

    def _fake_cvar_sources(self, sources):
        candidates = (
            sources["design_space"]
            .sort_values(["city_key", "pv_capacity_kw", "battery_energy_kwh"])
            .groupby("city_key", group_keys=False)
            .head(2)
            .copy()
            .reset_index(drop=True)
        )
        candidate_rows = []
        for city_key, group in candidates.groupby("city_key", sort=False):
            for order, row in enumerate(group.to_dict("records"), start=1):
                candidate_rows.append(
                    {
                        "city_key": str(city_key),
                        "city_label": str(row["city_label"]),
                        "case_key": str(row["case_key"]),
                        "source_case_key": str(row.get("source_case_key", row["case_key"])),
                        "candidate_id": f"{city_key}_D{order:04d}",
                        "candidate_order": order,
                        "pv_capacity_kw": float(row["pv_capacity_kw"]),
                        "pv_capacity_mw": float(row["pv_capacity_kw"]) / 1000.0,
                        "battery_energy_kwh": float(row["battery_energy_kwh"]),
                        "battery_energy_mwh": float(row["battery_energy_kwh"]) / 1000.0,
                        "candidate_source": "test source",
                        "from_figure1_design_space": True,
                        "from_figure4_expost_optima": False,
                    }
                )
        candidate_source = pd.DataFrame(candidate_rows)
        scenario_eval = self._fake_cvar_parallel_eval(candidate_source, sources["scenario_design_results"])
        summary = _summarize_cvar_candidate_results(candidate_source, scenario_eval, alpha=CVAR_ALPHA)
        curve = _select_cvar_design_curve(summary, alpha=CVAR_ALPHA, lambdas=CVAR_LAMBDAS)
        return {
            "cvar_candidate_scenario_results": scenario_eval,
            "cvar_candidate_summary": summary,
            "cvar_design_curve": curve,
        }

    def _fake_brisbane_sensitivity_rows_from_tasks(self, tasks):
        rows = []
        for task in sorted(tasks, key=lambda item: (int(item["driver_order"]), float(item["x_value"]))):
            driver_order = int(task["driver_order"])
            x_value = float(task["x_value"])
            pv = 2.0 + 0.12 * x_value + driver_order
            battery = 1.0 + 0.18 * x_value + 2.0 * driver_order
            cost = 1800.0 + 25.0 * x_value + 90.0 * driver_order
            rows.append(
                {
                    "driver_order": driver_order,
                    "sensitivity_driver": str(task["sensitivity_driver"]),
                    "x_value": x_value,
                    "x_label": str(task["x_label"]),
                    "city_key": "Brisbane",
                    "city_label": "Brisbane",
                    "case_key": "Brisbane_SingleFamily",
                    "source_case_key": "Brisbane_SingleFamily",
                    "pv_capacity_kw": pv,
                    "pv_capacity_mw": pv / 1000.0,
                    "battery_energy_kwh": battery,
                    "battery_energy_mwh": battery / 1000.0,
                    "system_total_cost_sgd_per_year": cost,
                    "system_total_cost_million_sgd_per_year": cost / 1e6,
                    "UE_kwh_per_year": max(0.0, 2.0 - 0.02 * x_value),
                    "UE_mwh_per_year": max(0.0, 2.0 - 0.02 * x_value) / 1000.0,
                    "energy_based_LOLP": 0.01,
                    "energy_based_lolp": 0.01,
                    "energy_based_lolp_percent": 1.0,
                    "outage_weighted_load_kwh": 100.0,
                    "saidi_min_per_year": float(task["saidi_min_per_year"]),
                    "voll_sgd_per_kwh": float(task["voll_sgd_per_kwh"]),
                    "tou_spread_k": float(task["tou_spread_k"]),
                    "tou_mean_sgd_per_kwh": 0.2364541667,
                    "tou_min_sgd_per_kwh": max(0.0, 0.20 - 0.02 * float(task["tou_spread_k"])),
                    "tou_max_sgd_per_kwh": 0.24 + 0.02 * float(task["tou_spread_k"]),
                    "simulated_saidi_min_per_year": float(task["saidi_min_per_year"]),
                    "simulated_saifi_per_year": 1.0,
                }
            )
        return rows

    def _fake_brisbane_sensitivity_source(self):
        parameters = load_case_parameters(INPUT_ROOT / "Brisbane_SingleFamily" / "Brisbane_SingleFamily_Parameter.csv")
        return pd.DataFrame(self._fake_brisbane_sensitivity_rows_from_tasks(_brisbane_sensitivity_tasks(parameters)))

    def _fake_affordable_city_sources(self, city_key: str, city_order: int):
        label = {"Metro_Manila": "Metro Manila"}.get(city_key, city_key)
        case_key = {
            "Singapore": "SG_SingleFamily",
            "Brisbane": "Brisbane_SingleFamily",
            "Jakarta": "Jakarta_SingleFamily",
            "Metro_Manila": "Manila_SingleFamily",
        }[city_key]
        zero_zero_cost = 1000.0 + 80.0 * city_order
        selected_cost = zero_zero_cost + 120.0
        grid_points = [
            (0.0, 0.0, zero_zero_cost, 2.5, 0.25, True, False),
            (0.0, 2.0, zero_zero_cost + 35.0, 1.5, 0.15, False, False),
            (2.0, 0.0, zero_zero_cost + 55.0, 1.0, 0.10, False, False),
            (2.0, 2.0, selected_cost, 0.0, 0.0, False, True),
        ]
        design_rows = []
        for pv, battery, cost, ue, lolp, is_zero_ref, is_selected in grid_points:
            design_rows.append(
                {
                    "city_key": city_key,
                    "city_label": label,
                    "case_key": case_key,
                    "source_case_key": case_key,
                    "pv_capacity_kw": pv,
                    "pv_capacity_mw": pv / 1000.0,
                    "battery_energy_kwh": battery,
                    "battery_energy_mwh": battery / 1000.0,
                    "system_total_cost_sgd_per_year": cost,
                    "system_total_cost_million_sgd_per_year": cost / 1e6,
                    "grid_cost_sgd_per_year": cost * 0.68,
                    "pv_annual_cost_sgd_per_year": cost * (0.12 if pv > 0 else 0.0),
                    "battery_annual_cost_sgd_per_year": cost * (0.15 if battery > 0 else 0.0),
                    "lost_load_cost_sgd_per_year": max(cost - cost * 0.68 - cost * (0.12 if pv > 0 else 0.0) - cost * (0.15 if battery > 0 else 0.0), 0.0),
                    "UE_kwh_per_year": ue,
                    "UE_mwh_per_year": ue / 1000.0,
                    "outage_weighted_load_kwh": 10.0,
                    "energy_based_LOLP": lolp,
                    "energy_based_lolp_percent": 100.0 * lolp,
                    "self_sufficiency_rate": 0.2 + 0.1 * pv,
                    "self_consumption_rate": 0.7 + 0.05 * battery,
                    "is_minimum_cost_design": is_zero_ref,
                    "outage_mode": _figure_c_outage_mode(city_key),
                    "outage_seed": 202501 + city_order,
                    "zero_pv_zero_bess_cost_sgd_per_year": zero_zero_cost,
                    "affordable_boundary_cost_sgd_per_year": zero_zero_cost,
                    "cost_delta_vs_no_pv_no_bess_sgd_per_year": cost - zero_zero_cost,
                    "is_no_pv_no_bess_reference": is_zero_ref,
                    "is_lolp_zero_min_cost_design": is_selected,
                    "selected_lolp_zero_pv_capacity_kw": 2.0,
                    "selected_lolp_zero_battery_energy_kwh": 2.0,
                    "selected_lolp_zero_cost_sgd_per_year": selected_cost,
                    "selected_lolp_zero_UE_kwh_per_year": 0.0,
                    "selected_lolp_zero_energy_based_LOLP": 0.0,
                    "point_source": "existing_evaluated",
                }
            )

        component_specs = [
            ("grid_bill", "Grid bill", 1, 780.0),
            ("pv_annual_cost", "PV depreciation + O&M", 2, 120.0),
            ("battery_annual_cost", "Battery depreciation + O&M", 3, 95.0),
            ("lost_load_cost", "Lost load cost", 4, 5.0),
        ]
        cost_rows = []
        for design_order, (design_case, pv, battery, total, ue) in enumerate(
            [
                ("no_pv_no_bess", 0.0, 0.0, zero_zero_cost, 2.5),
                ("lolp_zero_min_cost", 2.0, 2.0, selected_cost, 0.0),
            ],
            start=1,
        ):
            scale = total / sum(item[3] for item in component_specs)
            for component_key, component_label, stack_order, value in component_specs:
                cost = value * scale
                cost_rows.append(
                    {
                        "city_key": city_key,
                        "city_label": label,
                        "case_key": case_key,
                        "source_case_key": case_key,
                        "design_case": design_case,
                        "design_case_label": {
                            "no_pv_no_bess": "0 PV / 0 BESS",
                            "lolp_zero_min_cost": "LOLP=0 min-cost design",
                        }[design_case],
                        "design_case_order": design_order,
                        "component_key": component_key,
                        "component_label": component_label,
                        "stack_order": stack_order,
                        "cost_sgd": cost,
                        "share_of_total_percent": 100.0 * cost / total,
                        "system_total_cost_sgd_per_year": total,
                        "pv_capacity_kw": pv,
                        "pv_capacity_mw": pv / 1000.0,
                        "battery_energy_kwh": battery,
                        "battery_energy_mwh": battery / 1000.0,
                        "UE_kwh_per_year": ue,
                        "outage_mode": _figure_c_outage_mode(city_key),
                        "outage_seed": 202501 + city_order,
                        "zero_pv_zero_bess_cost_sgd_per_year": zero_zero_cost,
                        "affordable_boundary_cost_sgd_per_year": zero_zero_cost,
                        "selected_lolp_zero_cost_sgd_per_year": selected_cost,
                        "cost_delta_vs_no_pv_no_bess_sgd_per_year": total - zero_zero_cost,
                    }
                )

        balance_rows = []
        for hour in range(72):
            in_outage = 24 <= hour < 32
            charge = 0.35 if hour < 20 else 0.0
            discharge = 0.45 if in_outage else 0.0
            soc = max(0.2, min(1.0, 0.82 + 0.003 * min(hour, 20) - 0.02 * max(0, min(hour - 24, 8))))
            balance_rows.append(
                {
                    "city_key": city_key,
                    "city_label": label,
                    "case_key": case_key,
                    "source_case_key": case_key,
                    "plot_hour": hour,
                    "hour_index": 1000 + hour,
                    "timestamp": f"2025-01-01 {hour % 24:02d}:00:00",
                    "event_start_hour": 1024.0,
                    "event_end_hour": 1032.0,
                    "event_duration_hours": 8.0,
                    "event_UE_kwh": 0.0,
                    "event_net_deficit_kwh": 4.0,
                    "grid_availability": 0.0 if in_outage else 1.0,
                    "outage_fraction": 1.0 if in_outage else 0.0,
                    "critical_load_kwh": 1.0,
                    "outage_weighted_load_kwh": 8.0,
                    "served_load_kwh": 1.0,
                    "pv_available_kwh": 0.8 if 8 <= hour % 24 <= 17 else 0.0,
                    "pv_used_kwh": 0.6 if 8 <= hour % 24 <= 17 else 0.0,
                    "grid_import_kwh": 0.4 if not in_outage else 0.0,
                    "battery_charge_kwh": charge,
                    "battery_discharge_kwh": discharge,
                    "pv_curtailed_kwh": 0.05 if 8 <= hour % 24 <= 17 else 0.0,
                    "conversion_loss_kwh": 0.02 if charge or discharge else 0.0,
                    "unmet_load_kwh_equiv": 0.0,
                    "battery_soc_kwh": 2.0 * soc,
                    "battery_soc_fraction": soc,
                    "battery_soc_delta_kwh": charge - discharge,
                    "battery_soc_expected_delta_kwh": charge - discharge,
                    "battery_soc_delta_residual_kwh": 0.0,
                    "simultaneous_charge_discharge_kwh": 0.0,
                    "minimum_soc_fraction": 0.2,
                    "pv_capacity_kw": 2.0,
                    "battery_energy_kwh": 2.0,
                    "design_case": "lolp_zero_min_cost",
                    "design_case_label": "LOLP=0 min-cost design",
                    "outage_mode": _figure_c_outage_mode(city_key),
                    "outage_seed": 202501 + city_order,
                    "zero_pv_zero_bess_cost_sgd_per_year": zero_zero_cost,
                    "affordable_boundary_cost_sgd_per_year": zero_zero_cost,
                    "selected_lolp_zero_cost_sgd_per_year": selected_cost,
                    "cost_delta_vs_no_pv_no_bess_sgd_per_year": selected_cost - zero_zero_cost,
                }
            )

        event_summary = pd.DataFrame(
            [
                {
                    "city_key": city_key,
                    "city_label": label,
                    "case_key": case_key,
                    "source_case_key": case_key,
                    "event_id": 1,
                    "event_start_hour": 1024.0,
                    "event_end_hour": 1032.0,
                    "event_duration_hours": 8.0,
                    "event_UE_kwh": 0.0,
                    "event_net_deficit_kwh": 4.0,
                    "window_start_hour": 1000,
                    "window_end_hour": 1071,
                    "window_UE_kwh": 0.0,
                    "pv_capacity_kw": 2.0,
                    "battery_energy_kwh": 2.0,
                    "outage_mode": _figure_c_outage_mode(city_key),
                    "outage_seed": 202501 + city_order,
                    "zero_pv_zero_bess_cost_sgd_per_year": zero_zero_cost,
                    "affordable_boundary_cost_sgd_per_year": zero_zero_cost,
                    "selected_lolp_zero_cost_sgd_per_year": selected_cost,
                    "cost_delta_vs_no_pv_no_bess_sgd_per_year": selected_cost - zero_zero_cost,
                }
            ]
        )
        return {
            "city_order": city_order,
            "design_space": pd.DataFrame(design_rows),
            "cost_distribution": pd.DataFrame(cost_rows),
            "energy_balance": pd.DataFrame(balance_rows),
            "event_summary": event_summary,
        }

    def _fake_affordable_sources(self):
        city_sources = [
            self._fake_affordable_city_sources(city_key, index)
            for index, city_key in enumerate(FIGURE_C_CITY_ORDER)
        ]
        design_space = pd.concat([item["design_space"] for item in city_sources], ignore_index=True)
        return {
            "affordable_design_space": design_space,
            "affordable_design_space_plot_grid": design_space.copy(),
            "affordable_cost_distribution": pd.concat([item["cost_distribution"] for item in city_sources], ignore_index=True),
            "affordable_energy_balance": pd.concat([item["energy_balance"] for item in city_sources], ignore_index=True),
            "affordable_event_summary": pd.concat([item["event_summary"] for item in city_sources], ignore_index=True),
        }

    def _fake_day_night_sources(self):
        affordable = self._fake_affordable_sources()["affordable_design_space"]
        frames = []
        for city_key in FIGURE_D_CITY_ORDER:
            for scenario_order, scenario_key in enumerate(("night", "day")):
                group = affordable[affordable["city_key"].astype(str) == city_key].copy()
                group["scenario_key"] = scenario_key
                group["scenario_order"] = scenario_order
                group["scenario_label"] = f"{group['city_label'].iloc[0]} {scenario_key}"
                group["outage_window_start_hour"] = 19.0 if scenario_key == "night" else 10.0
                group["outage_window_end_hour"] = 23.0 if scenario_key == "night" else 14.0
                group["day_window_start_hour"] = 10.0
                group["day_window_end_hour"] = 14.0
                group["axis_pv_max_kw"] = float(group["pv_capacity_kw"].max())
                group["axis_battery_max_kwh"] = float(group["battery_energy_kwh"].max())
                if scenario_key == "day":
                    group["outage_mode"] = "day_window_random"
                    group["system_total_cost_sgd_per_year"] = group["system_total_cost_sgd_per_year"].astype(float) + 40.0
                    group["grid_cost_sgd_per_year"] = group["grid_cost_sgd_per_year"].astype(float) + 25.0
                frames.append(group)
        design = pd.concat(frames, ignore_index=True, sort=False)
        cost_rows = []
        component_specs = [
            ("grid_bill", "Grid bill", 1),
            ("pv_annual_cost", "PV depreciation + O&M", 2),
            ("battery_annual_cost", "Battery depreciation + O&M", 3),
            ("lost_load_cost", "Lost load cost", 4),
        ]
        for city_key in FIGURE_D_CITY_ORDER:
            city = design[(design["city_key"].astype(str) == city_key) & (design["point_source"].astype(str) != "interpolated_for_plot")]
            night = city[city["scenario_key"].astype(str) == "night"]
            day = city[city["scenario_key"].astype(str) == "day"]
            night_zero = night[(night["pv_capacity_kw"].eq(0.0)) & (night["battery_energy_kwh"].eq(0.0))].iloc[0]
            night_selected = night[night["is_lolp_zero_min_cost_design"].astype(bool)].iloc[0]
            day_zero = day[(day["pv_capacity_kw"].eq(0.0)) & (day["battery_energy_kwh"].eq(0.0))].iloc[0]
            day_selected = day[day["is_lolp_zero_min_cost_design"].astype(bool)].iloc[0]
            night_under_day = day[
                np.isclose(day["pv_capacity_kw"].astype(float), float(night_selected["pv_capacity_kw"]))
                & np.isclose(day["battery_energy_kwh"].astype(float), float(night_selected["battery_energy_kwh"]))
            ].iloc[0]
            design_rows = [
                ("night_no_pv_no_bess", night_zero),
                ("night_lolp_zero_design", night_selected),
                ("day_no_pv_no_bess", day_zero),
                ("day_lolp_zero_design", day_selected),
                ("night_design_under_day_outage", night_under_day),
            ]
            for order, (design_case, row) in enumerate(design_rows, start=1):
                total = float(row["system_total_cost_sgd_per_year"])
                values = {
                    "grid_bill": float(row["grid_cost_sgd_per_year"]),
                    "pv_annual_cost": float(row["pv_annual_cost_sgd_per_year"]),
                    "battery_annual_cost": float(row["battery_annual_cost_sgd_per_year"]),
                    "lost_load_cost": float(row["lost_load_cost_sgd_per_year"]),
                }
                for component_key, component_label, stack_order in component_specs:
                    value = values[component_key]
                    axis_prefix = {
                        "night_no_pv_no_bess": "Night 0/0",
                        "night_lolp_zero_design": "Night LOLP=0",
                        "day_no_pv_no_bess": "Day 0/0",
                        "day_lolp_zero_design": "Day LOLP=0",
                        "night_design_under_day_outage": "Night design\nunder day",
                    }[design_case]
                    cost_rows.append(
                        {
                            "city_key": city_key,
                            "city_label": str(row["city_label"]),
                            "case_key": str(row["case_key"]),
                            "source_case_key": str(row["source_case_key"]),
                            "scenario_key": str(row["scenario_key"]),
                            "scenario_label": str(row["scenario_label"]),
                            "design_case": design_case,
                            "design_case_label": axis_prefix.replace("\n", " "),
                            "design_case_order": order,
                            "axis_label": f"{axis_prefix}\n{row['pv_capacity_kw']:g} kW PV\n{row['battery_energy_kwh']:g} kWh BESS",
                            "component_key": component_key,
                            "component_label": component_label,
                            "stack_order": stack_order,
                            "cost_sgd": value,
                            "share_of_total_percent": 100.0 * value / max(total, 1e-9),
                            "system_total_cost_sgd_per_year": total,
                            "pv_capacity_kw": float(row["pv_capacity_kw"]),
                            "pv_capacity_mw": float(row["pv_capacity_kw"]) / 1000.0,
                            "battery_energy_kwh": float(row["battery_energy_kwh"]),
                            "battery_energy_mwh": float(row["battery_energy_kwh"]) / 1000.0,
                            "UE_kwh_per_year": float(row["UE_kwh_per_year"]),
                            "energy_based_LOLP": float(row["energy_based_LOLP"]),
                            "outage_mode": str(row["outage_mode"]),
                            "outage_seed": int(row["outage_seed"]),
                        }
                    )
        events = pd.DataFrame(
            [
                {
                    "city_key": city_key,
                    "city_label": {"Jakarta": "Jakarta", "Singapore": "Singapore"}[city_key],
                    "case_key": "Jakarta_SingleFamily" if city_key == "Jakarta" else "SG_SingleFamily",
                    "source_case_key": "Jakarta_SingleFamily" if city_key == "Jakarta" else "SG_SingleFamily",
                    "scenario_key": scenario_key,
                    "scenario_order": 0 if scenario_key == "night" else 1,
                    "scenario_label": f"{city_key} {scenario_key}",
                    "outage_mode": "night_window_random" if scenario_key == "night" else "day_window_random",
                    "outage_seed": 42,
                    "event_start_hour": 43.0 if scenario_key == "night" else 34.0,
                    "event_end_hour": 44.0 if scenario_key == "night" else 35.0,
                    "event_start_hour_of_day": 19.0 if scenario_key == "night" else 10.0,
                    "event_duration_minutes": 60.0,
                    "outage_window_start_hour": 19.0 if scenario_key == "night" else 10.0,
                    "outage_window_end_hour": 23.0 if scenario_key == "night" else 14.0,
                    "day_window_start_hour": 10.0,
                    "day_window_end_hour": 14.0,
                }
                for city_key in FIGURE_D_CITY_ORDER
                for scenario_key in ("night", "day")
            ]
        )
        return {
            "day_night_affordable_design_space": design,
            "day_night_cost_distribution": pd.DataFrame(cost_rows),
            "day_night_event_summary": events,
        }

    def test_pv_source_tracking_excludes_grid_charged_battery_from_ssr_scr(self) -> None:
        config = {
            "technical": {"cooling_cop": 1.0, "heating_cop": 1.0},
            "battery": {
                "roundtrip_efficiency": 1.0,
                "max_c_rate": 1.0,
                "initial_soc_fraction": 0.0,
                "soc_min_fraction": 0.0,
                "soc_max_fraction": 1.0,
                "rated_cycle_life_to_eol": 9000.0,
            },
            "economics": {
                "battery_capex_sgd_per_kwh": 100.0,
                "battery_power_capex_sgd_per_kw": 10.0,
            },
            "dispatch": {"backend": "greedy", "include_degradation_penalty": False},
        }
        frame = pd.DataFrame(
            {
                "hour_index": [0, 1, 2],
                "grid_availability": [1.0, 1.0, 1.0],
                "outage_fraction": [0.0, 0.0, 0.0],
                "tou_sgd_per_kwh": [0.0, 0.5, 2.0],
                "pv_kwh": [0.0, 0.0, 0.0],
                "electric_load_kwh": [0.0, 0.0, 10.0],
                "cooling_load_kwh_th": [0.0, 0.0, 0.0],
                "heating_load_kwh_th": [0.0, 0.0, 0.0],
            }
        )

        dispatch = solve_dispatch_greedy(frame, {"battery_energy_kwh": 10.0, "battery_power_kw": 10.0}, config)
        metrics = calculate_pv_self_use_metrics(dispatch, annual_load_kwh=10.0)

        self.assertGreater(float(dispatch["grid_to_battery_kwh"].sum()), 0.0)
        self.assertGreater(float(dispatch["battery_discharge_from_grid_kwh"].sum()), 0.0)
        self.assertAlmostEqual(metrics["pv_self_served_load_kwh"], 0.0)
        self.assertAlmostEqual(metrics["self_sufficiency_rate"], 0.0)
        self.assertAlmostEqual(metrics["self_consumption_rate"], 0.0)

    def test_greedy_dispatch_does_not_cycle_battery_at_flat_tou_threshold(self) -> None:
        config = {
            "technical": {"cooling_cop": 1.0, "heating_cop": 1.0},
            "battery": {
                "roundtrip_efficiency": 1.0,
                "max_c_rate": 1.0,
                "initial_soc_fraction": 0.5,
                "soc_min_fraction": 0.0,
                "soc_max_fraction": 1.0,
                "rated_cycle_life_to_eol": 9000.0,
            },
            "economics": {
                "battery_capex_sgd_per_kwh": 100.0,
                "battery_power_capex_sgd_per_kw": 10.0,
            },
            "dispatch": {"backend": "greedy", "include_degradation_penalty": False},
        }
        frame = pd.DataFrame(
            {
                "hour_index": list(range(6)),
                "grid_availability": [1.0] * 6,
                "outage_fraction": [0.0] * 6,
                "tou_sgd_per_kwh": [0.2] * 6,
                "pv_kwh": [0.0] * 6,
                "electric_load_kwh": [1.0] * 6,
                "cooling_load_kwh_th": [0.0] * 6,
                "heating_load_kwh_th": [0.0] * 6,
            }
        )

        dispatch = solve_dispatch_greedy(frame, {"battery_energy_kwh": 10.0, "battery_power_kw": 10.0}, config)

        self.assertAlmostEqual(float(dispatch["simultaneous_charge_discharge_kwh"].sum()), 0.0)
        self.assertAlmostEqual(float(dispatch["battery_charge_kwh"].sum()), 0.0)
        self.assertAlmostEqual(float(dispatch["battery_discharge_kwh"].sum()), 0.0)

    def test_pv_origin_battery_discharge_counts_after_storage_loss(self) -> None:
        config = {
            "technical": {"cooling_cop": 1.0, "heating_cop": 1.0},
            "battery": {
                "roundtrip_efficiency": 0.81,
                "max_c_rate": 1.0,
                "initial_soc_fraction": 0.0,
                "soc_min_fraction": 0.0,
                "soc_max_fraction": 1.0,
                "rated_cycle_life_to_eol": 9000.0,
            },
            "economics": {
                "battery_capex_sgd_per_kwh": 100.0,
                "battery_power_capex_sgd_per_kw": 10.0,
            },
            "dispatch": {"backend": "greedy", "include_degradation_penalty": False},
        }
        frame = pd.DataFrame(
            {
                "hour_index": [0, 1],
                "grid_availability": [1.0, 1.0],
                "outage_fraction": [0.0, 0.0],
                "tou_sgd_per_kwh": [0.20, 0.50],
                "pv_kwh": [10.0, 0.0],
                "electric_load_kwh": [0.0, 8.1],
                "cooling_load_kwh_th": [0.0, 0.0],
                "heating_load_kwh_th": [0.0, 0.0],
            }
        )

        dispatch = solve_dispatch_greedy(frame, {"battery_energy_kwh": 10.0, "battery_power_kw": 10.0}, config)
        metrics = calculate_pv_self_use_metrics(dispatch, annual_load_kwh=8.1)

        self.assertAlmostEqual(float(dispatch["battery_discharge_from_pv_kwh"].sum()), 8.1, places=6)
        self.assertAlmostEqual(metrics["pv_self_served_load_kwh"], 8.1, places=6)
        self.assertAlmostEqual(metrics["self_sufficiency_rate"], 1.0, places=6)
        self.assertAlmostEqual(metrics["self_consumption_rate"], 0.81, places=6)

    def test_source_builder_uses_four_real_singlefamily_cases_without_uncertainty(self) -> None:
        with patch("src.figures.single_outage_city_baseline.run_uncertainty_analysis") as uncertainty:
            sources = build_single_outage_city_baseline_sources(INPUT_ROOT, config=self._fast_config())

        uncertainty.assert_not_called()
        self.assertEqual(set(sources), {"design_space", "cost_distribution", "energy_balance", "event_summary"})
        self.assertEqual(sources["design_space"]["city_key"].drop_duplicates().tolist(), list(CITY_ORDER))
        self.assertEqual(sources["energy_balance"].groupby("city_key").size().astype(int).to_dict(), {city: 72 for city in CITY_ORDER})
        self.assertTrue(sources["design_space"]["self_sufficiency_rate"].between(0.0, 1.0).all())
        self.assertTrue(sources["design_space"]["self_consumption_rate"].between(0.0, 1.0).all())
        costs = sources["cost_distribution"]
        self.assertEqual(set(costs.groupby("city_key").size().astype(int).tolist()), {12})
        self.assertEqual(costs["design_case"].drop_duplicates().astype(str).tolist(), list(COST_DESIGN_CASES))
        self.assertEqual(set(costs.groupby(["city_key", "design_case"]).size().astype(int).tolist()), {4})
        designs = costs[["city_key", "design_case", "pv_capacity_kw", "battery_energy_kwh"]].drop_duplicates()
        for city_key, group in designs.groupby("city_key"):
            by_case = group.set_index("design_case")
            self.assertAlmostEqual(float(by_case.loc["no_pv_no_bess", "pv_capacity_kw"]), 0.0)
            self.assertAlmostEqual(float(by_case.loc["no_pv_no_bess", "battery_energy_kwh"]), 0.0)
            self.assertAlmostEqual(float(by_case.loc["same_pv_no_bess", "battery_energy_kwh"]), 0.0)
            self.assertAlmostEqual(
                float(by_case.loc["same_pv_no_bess", "pv_capacity_kw"]),
                float(by_case.loc["optimal_pv_bess", "pv_capacity_kw"]),
            )

    def test_affordable_lolp_source_builder_uses_c_series_city_modes_and_parallel(self) -> None:
        fake_city_sources = [
            self._fake_affordable_city_sources(city_key, index)
            for index, city_key in enumerate(FIGURE_C_CITY_ORDER)
        ]

        def _fake_parallel(tasks, cases_by_city):
            self.assertEqual([task[0] for task in tasks], list(FIGURE_C_CITY_ORDER))
            self.assertEqual([task[3] for task in tasks], ["night_window_random", "random", "night_window_random", "random"])
            self.assertEqual(set(cases_by_city), set(FIGURE_C_CITY_ORDER))
            return list(reversed(fake_city_sources))

        with (
            patch("src.figures.single_outage_city_baseline.run_uncertainty_analysis") as uncertainty,
            patch(
                "src.figures.single_outage_city_baseline._build_affordable_lolp_city_sources_parallel",
                side_effect=_fake_parallel,
            ) as parallel_builder,
        ):
            sources = build_single_outage_city_affordable_lolp_sources(INPUT_ROOT, config=self._fast_config())

        uncertainty.assert_not_called()
        parallel_builder.assert_called_once()
        self.assertEqual(
            sources["affordable_design_space"]["city_key"].drop_duplicates().astype(str).tolist(),
            list(FIGURE_C_CITY_ORDER),
        )
        self.assertEqual(
            sources["affordable_cost_distribution"]["design_case"].drop_duplicates().astype(str).tolist(),
            list(AFFORDABLE_COST_DESIGN_CASES),
        )
        self.assertEqual(
            sources["affordable_energy_balance"].groupby("city_key").size().astype(int).to_dict(),
            {city: 72 for city in FIGURE_C_CITY_ORDER},
        )
        selected = sources["affordable_design_space"][
            sources["affordable_design_space"]["is_lolp_zero_min_cost_design"].astype(bool)
        ]
        self.assertEqual(len(selected), len(FIGURE_C_CITY_ORDER))
        self.assertTrue((selected["UE_kwh_per_year"].astype(float) <= 1e-6).all())

    def test_tou_profile_source_uses_real_four_city_inputs(self) -> None:
        source = build_single_outage_city_tou_profile_source(INPUT_ROOT)

        self.assertEqual(source["city_key"].drop_duplicates().astype(str).tolist(), list(FIGURE_1_TO_4_CITY_ORDER))
        self.assertEqual(source.groupby("city_key").size().astype(int).to_dict(), {city: 24 for city in FIGURE_1_TO_4_CITY_ORDER})
        for _, group in source.groupby("city_key", sort=False):
            self.assertEqual(group["hour"].astype(int).tolist(), list(range(24)))
            self.assertEqual(group["interval_start_hour"].astype(float).tolist(), [float(hour) for hour in range(24)])
            self.assertEqual(group["interval_end_hour"].astype(float).tolist(), [float(hour) for hour in range(1, 25)])

    def test_tou_step_xy_preserves_hourly_constant_intervals(self) -> None:
        source = pd.DataFrame(
            {
                "city_key": ["Singapore"] * 24,
                "city_label": ["Singapore"] * 24,
                "case_key": ["SG_SingleFamily"] * 24,
                "hour": list(range(24)),
                "interval_start_hour": [float(hour) for hour in range(24)],
                "interval_end_hour": [float(hour) for hour in range(1, 25)],
                "tou_sgd_per_kwh": [0.19] * 13 + [0.32] * 11,
            }
        )

        x, y = _tou_step_xy(source)

        self.assertEqual(x.astype(float).tolist(), [float(hour) for hour in range(25)])
        self.assertEqual(len(y), 25)
        self.assertAlmostEqual(float(y[12]), 0.19)
        self.assertAlmostEqual(float(y[13]), 0.32)
        self.assertAlmostEqual(float(y[-1]), 0.32)

    def test_figure_7_tou_spread_keeps_mean_and_nonnegative(self) -> None:
        tou = pd.read_csv(INPUT_ROOT / "Brisbane_SingleFamily" / "Brisbane_SingleFamily_TOU.csv")
        base = tou.sort_values("hour").reset_index(drop=True)

        same = _mean_preserving_tou_profile(tou, 1.0)
        expanded = _mean_preserving_tou_profile(tou, 1.25)

        self.assertEqual(same["hour"].astype(int).tolist(), list(range(24)))
        self.assertTrue(np.allclose(same["tou_sgd_per_kwh"].astype(float), base["tou_sgd_per_kwh"].astype(float)))
        self.assertAlmostEqual(float(expanded["tou_sgd_per_kwh"].mean()), float(base["tou_sgd_per_kwh"].mean()), places=12)
        self.assertGreaterEqual(float(expanded["tou_sgd_per_kwh"].min()), 0.0)
        with self.assertRaises(ValueError):
            _mean_preserving_tou_profile(tou, 2.0)

    def test_figure_7_brisbane_sensitivity_source_ranges_and_parallel(self) -> None:
        config = self._fast_config()
        parameters = load_case_parameters(INPUT_ROOT / "Brisbane_SingleFamily" / "Brisbane_SingleFamily_Parameter.csv")
        tasks = _brisbane_sensitivity_tasks(parameters)

        self.assertEqual(
            {
                driver: sum(1 for task in tasks if task["sensitivity_driver"] == driver)
                for driver in BRISBANE_SENSITIVITY_DRIVER_ORDER
            },
            {"tou_spread": 6, "saidi": 9, "voll": 7},
        )
        self.assertEqual(
            [float(task["x_value"]) for task in tasks if task["sensitivity_driver"] == "tou_spread"],
            list(BRISBANE_SENSITIVITY_TOU_K_VALUES),
        )
        self.assertEqual(
            [float(task["x_value"]) for task in tasks if task["sensitivity_driver"] == "saidi"],
            parameters.sensitivity["saidi_min_per_year"],
        )
        self.assertEqual(
            [float(task["x_value"]) for task in tasks if task["sensitivity_driver"] == "voll"],
            parameters.sensitivity["voll_sgd_per_kwh"],
        )

        with (
            patch("src.figures.single_outage_city_baseline.run_uncertainty_analysis") as uncertainty,
            patch(
                "src.figures.single_outage_city_baseline._evaluate_brisbane_sensitivity_tasks_parallel",
                side_effect=lambda task_arg, *_args, **_kwargs: self._fake_brisbane_sensitivity_rows_from_tasks(task_arg),
            ) as parallel_eval,
        ):
            source = build_single_outage_city_brisbane_sensitivity_source(INPUT_ROOT, config=config)

        uncertainty.assert_not_called()
        parallel_eval.assert_called_once()
        self.assertEqual(parallel_eval.call_args.kwargs["workers"], 6)
        self.assertEqual(len(source), 22)
        self.assertEqual(source["sensitivity_driver"].drop_duplicates().astype(str).tolist(), list(BRISBANE_SENSITIVITY_DRIVER_ORDER))
        self.assertEqual(set(source["city_key"].astype(str)), {"Brisbane"})

    def test_figure_7_parallel_executor_ordering(self) -> None:
        tasks = [
            {
                "driver_order": 2,
                "sensitivity_driver": "voll",
                "x_value": 100.0,
                "x_label": "100",
                "tou_spread_k": 1.0,
                "saidi_min_per_year": 90.0,
                "voll_sgd_per_kwh": 100.0,
            },
            {
                "driver_order": 0,
                "sensitivity_driver": "tou_spread",
                "x_value": 0.0,
                "x_label": "k=0",
                "tou_spread_k": 0.0,
                "saidi_min_per_year": 90.0,
                "voll_sgd_per_kwh": 150.0,
            },
        ]

        def _rows_from_payloads(_func, payloads, chunksize=1):
            payload_list = list(payloads)
            return self._fake_brisbane_sensitivity_rows_from_tasks(reversed(payload_list))

        with patch("src.figures.single_outage_city_baseline.ProcessPoolExecutor") as executor_cls:
            executor = executor_cls.return_value.__enter__.return_value
            executor.map.side_effect = _rows_from_payloads
            result = _evaluate_brisbane_sensitivity_tasks_parallel(tasks, INPUT_ROOT, self._fast_config(), workers=2)

        executor_cls.assert_called_once()
        self.assertEqual(executor_cls.call_args.kwargs["max_workers"], 2)
        self.assertEqual([(row["driver_order"], row["x_value"]) for row in result], [(0, 0.0), (2, 100.0)])

    def test_figure_4_scenario_source_uses_100_optimizations_per_city(self) -> None:
        config = self._fast_config()
        sources = build_single_outage_city_baseline_sources(INPUT_ROOT, config=config)
        with TemporaryDirectory() as tmp:
            config.setdefault("outputs", {})["directory"] = tmp
            with patch(
                "src.figures.single_outage_city_baseline.run_uncertainty_analysis",
                side_effect=self._fake_uncertainty_results,
            ) as uncertainty:
                scenario_sources = build_single_outage_city_scenario_distribution_sources(
                    INPUT_ROOT,
                    config=config,
                    baseline_sources=sources,
                )

        self.assertEqual(uncertainty.call_count, len(CITY_ORDER))
        self.assertEqual(set(scenario_sources), {"scenario_design_results", "reference_design_results"})
        scenario_results = scenario_sources["scenario_design_results"]
        self.assertEqual(
            scenario_results.groupby("city_key").size().astype(int).to_dict(),
            {city: SCENARIO_DISTRIBUTION_COUNT for city in CITY_ORDER},
        )
        self.assertEqual(len(scenario_sources["reference_design_results"]), len(CITY_ORDER))
        self.assertTrue(scenario_results["energy_based_lolp"].between(0.0, 1.0).all())

    def test_figure_5_cvar_source_uses_existing_scenarios_without_uncertainty_optimization(self) -> None:
        config = self._fast_config()
        sources = build_single_outage_city_baseline_sources(INPUT_ROOT, config=config)
        with TemporaryDirectory() as cache_tmp:
            config.setdefault("outputs", {})["directory"] = cache_tmp
            with patch(
                "src.figures.single_outage_city_baseline.run_uncertainty_analysis",
                side_effect=self._fake_uncertainty_results,
            ):
                scenario_sources = build_single_outage_city_scenario_distribution_sources(
                    INPUT_ROOT,
                    config=config,
                    baseline_sources=sources,
                )

        with (
            patch("src.figures.single_outage_city_baseline.run_uncertainty_analysis") as uncertainty,
            patch("src.figures.single_outage_city_baseline.resilience_design") as optimization,
            patch(
                "src.figures.single_outage_city_baseline._evaluate_cvar_candidate_scenarios_parallel",
                side_effect=self._fake_cvar_parallel_eval,
            ) as parallel_eval,
        ):
            cvar_sources = build_single_outage_city_cvar_sources(
                INPUT_ROOT,
                config=config,
                baseline_sources=sources,
                scenario_sources=scenario_sources,
            )

        uncertainty.assert_not_called()
        optimization.assert_not_called()
        parallel_eval.assert_called_once()
        self.assertEqual(
            set(cvar_sources),
            {"cvar_design_curve", "cvar_candidate_summary", "cvar_candidate_scenario_results"},
        )
        curve = cvar_sources["cvar_design_curve"]
        self.assertEqual(len(curve), len(CITY_ORDER) * len(CVAR_LAMBDAS))
        self.assertEqual(curve.groupby(["city_key", "risk_aversion_lambda"]).size().astype(int).unique().tolist(), [1])
        self.assertEqual(sorted(curve["risk_aversion_lambda"].drop_duplicates().round(1).tolist()), list(CVAR_LAMBDAS))
        scenario_eval = cvar_sources["cvar_candidate_scenario_results"]
        summary = cvar_sources["cvar_candidate_summary"]
        first = summary.iloc[0]
        first_losses = scenario_eval[
            (scenario_eval["city_key"].astype(str) == str(first["city_key"]))
            & (scenario_eval["candidate_id"].astype(str) == str(first["candidate_id"]))
        ]["outage_loss_sgd_per_year"].astype(float)
        expected_cvar = float(first_losses.sort_values().tail(10).mean())
        self.assertAlmostEqual(float(first["cvar90_outage_loss_sgd_per_year"]), expected_cvar)

    def test_figure_5_fixed_design_evaluation_uses_process_pool(self) -> None:
        scenario_rows = []
        for city_key in CITY_ORDER:
            for scenario_id in range(1, SCENARIO_DISTRIBUTION_COUNT + 1):
                scenario_rows.append(
                    {
                        "city_key": city_key,
                        "city_label": city_key,
                        "case_key": f"{city_key}_SingleFamily",
                        "source_case_key": f"{city_key}_SingleFamily",
                        "scenario_id": scenario_id,
                        "outage_seed": 1000 + scenario_id - 1,
                        "pv_capacity_kw": 1.0,
                        "pv_capacity_mw": 0.001,
                        "battery_energy_kwh": 0.0,
                        "battery_energy_mwh": 0.0,
                        "UE_kwh_per_year": 0.0,
                        "UE_mwh_per_year": 0.0,
                        "ue_kwh_per_year": 0.0,
                        "energy_based_lolp": 0.0,
                        "energy_based_lolp_percent": 0.0,
                        "outage_weighted_load_kwh": 1.0,
                        "annual_outage_hours": 1.0,
                        "system_total_cost_sgd_per_year": 1.0,
                        "system_total_cost_million_sgd_per_year": 0.000001,
                        "plot_jitter": 0.0,
                    }
                )
        scenario_results = pd.DataFrame(scenario_rows)
        candidates = pd.DataFrame(
            [
                {
                    "city_key": "Singapore",
                    "city_label": "Singapore",
                    "case_key": "SG_SingleFamily",
                    "source_case_key": "SG_SingleFamily",
                    "candidate_id": "Singapore_D0001",
                    "candidate_order": 1,
                    "pv_capacity_kw": 0.0,
                    "pv_capacity_mw": 0.0,
                    "battery_energy_kwh": 0.0,
                    "battery_energy_mwh": 0.0,
                    "candidate_source": "test",
                    "from_figure1_design_space": True,
                    "from_figure4_expost_optima": False,
                },
                {
                    "city_key": "Singapore",
                    "city_label": "Singapore",
                    "case_key": "SG_SingleFamily",
                    "source_case_key": "SG_SingleFamily",
                    "candidate_id": "Singapore_D0002",
                    "candidate_order": 2,
                    "pv_capacity_kw": 1.0,
                    "pv_capacity_mw": 0.001,
                    "battery_energy_kwh": 1.0,
                    "battery_energy_mwh": 0.001,
                    "candidate_source": "test",
                    "from_figure1_design_space": True,
                    "from_figure4_expost_optima": False,
                },
            ]
        )
        fake_rows = self._fake_cvar_parallel_eval(candidates, scenario_results)
        with (
            patch("src.figures.single_outage_city_baseline.os.cpu_count", return_value=2),
            patch("src.figures.single_outage_city_baseline.ProcessPoolExecutor") as executor_cls,
        ):
            executor = executor_cls.return_value.__enter__.return_value
            executor.map.return_value = fake_rows.to_dict("records")
            result = _evaluate_cvar_candidate_scenarios_parallel(
                candidates,
                scenario_results,
                cases_by_city={},
                case_configs_by_city={},
                config={},
            )

        executor_cls.assert_called_once()
        self.assertEqual(executor_cls.call_args.kwargs["max_workers"], 2)
        self.assertEqual(len(result), len(fake_rows))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_figure_5_reads_existing_source_data_without_rebuilding_baseline(self) -> None:
        config = self._fast_config()
        baseline_sources = build_single_outage_city_baseline_sources(INPUT_ROOT, config=config)
        with TemporaryDirectory() as cache_tmp:
            config.setdefault("outputs", {})["directory"] = cache_tmp
            with patch(
                "src.figures.single_outage_city_baseline.run_uncertainty_analysis",
                side_effect=self._fake_uncertainty_results,
            ):
                scenario_sources = build_single_outage_city_scenario_distribution_sources(
                    INPUT_ROOT,
                    config=config,
                    baseline_sources=baseline_sources,
                )

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            figure_1 = baseline_sources["design_space"].copy()
            figure_1.insert(0, "source_table", "design_space")
            figure_1.to_csv(output_dir / f"SingleFamily_{FIGURE_NAMES[1]}_Source_Data.csv", index=False)
            figure_4_frames = []
            for key in ("scenario_design_results", "reference_design_results"):
                frame = scenario_sources[key].copy()
                frame.insert(0, "source_table", key)
                figure_4_frames.append(frame)
            pd.concat(figure_4_frames, ignore_index=True, sort=False).to_csv(
                output_dir / f"SingleFamily_{FIGURE_NAMES[4]}_Source_Data.csv",
                index=False,
            )
            with (
                patch("src.figures.single_outage_city_baseline.build_single_outage_city_baseline_sources") as baseline_builder,
                patch("src.figures.single_outage_city_baseline.run_uncertainty_analysis") as uncertainty,
                patch("src.figures.single_outage_city_baseline.resilience_design") as optimization,
                patch(
                    "src.figures.single_outage_city_baseline._evaluate_cvar_candidate_scenarios_parallel",
                    side_effect=self._fake_cvar_parallel_eval,
                ),
            ):
                paths = make_single_outage_city_baseline_figures(output_dir, config=config, figures=[5])
            self.assertTrue(paths[f"SingleFamily_{FIGURE_NAMES[5]}_png"].exists())

        baseline_builder.assert_not_called()
        uncertainty.assert_not_called()
        optimization.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_affordable_lolp_figures_read_existing_source_data_without_rebuilding(self) -> None:
        fake_sources = self._fake_affordable_sources()
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            figure_tables = {
                "1c": ("affordable_design_space",),
                "2c": ("affordable_cost_distribution",),
                "3c": ("affordable_energy_balance", "affordable_event_summary"),
            }
            for figure_no, table_names in figure_tables.items():
                frames = []
                for table_name in table_names:
                    frame = fake_sources[table_name].copy()
                    frame.insert(0, "source_table", table_name)
                    frames.append(frame)
                pd.concat(frames, ignore_index=True, sort=False).to_csv(
                    output_dir / f"SingleFamily_{FIGURE_NAMES[figure_no]}_Source_Data.csv",
                    index=False,
                )
            with (
                patch("src.figures.single_outage_city_baseline.build_single_outage_city_affordable_lolp_sources") as builder,
                patch("src.figures.single_outage_city_baseline.resilience_design") as optimization,
                patch("src.figures.single_outage_city_baseline.evaluate_candidate") as evaluate,
                patch("src.figures.single_outage_city_baseline.run_uncertainty_analysis") as uncertainty,
            ):
                paths = make_single_outage_city_baseline_figures(
                    output_dir,
                    config=self._fast_config(),
                    figures=["1c", "2c", "3c"],
                )
            self.assertTrue(paths[f"SingleFamily_{FIGURE_NAMES['1c']}_png"].exists())
            self.assertTrue(paths[f"SingleFamily_{FIGURE_NAMES['2c']}_png"].exists())
            self.assertTrue(paths[f"SingleFamily_{FIGURE_NAMES['3c']}_png"].exists())

        builder.assert_not_called()
        optimization.assert_not_called()
        evaluate.assert_not_called()
        uncertainty.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_all_single_outage_city_figures_export(self) -> None:
        config = self._fast_config()
        sources = build_single_outage_city_baseline_sources(INPUT_ROOT, config=config)
        with TemporaryDirectory() as cache_tmp:
            config.setdefault("outputs", {})["directory"] = cache_tmp
            with patch(
                "src.figures.single_outage_city_baseline.run_uncertainty_analysis",
                side_effect=self._fake_uncertainty_results,
            ):
                sources.update(
                    build_single_outage_city_scenario_distribution_sources(
                        INPUT_ROOT,
                        config=config,
                        baseline_sources=sources,
                    )
                )
        sources.update(self._fake_cvar_sources(sources))
        sources["brisbane_sensitivity_results"] = self._fake_brisbane_sensitivity_source()
        sources.update(self._fake_affordable_sources())
        sources.update(self._fake_day_night_sources())
        with TemporaryDirectory() as tmp:
            paths = make_single_outage_city_baseline_figures(Path(tmp), sources=sources, figures=["all"])

            for figure_no, figure_name in FIGURE_NAMES.items():
                base = f"SingleFamily_{figure_name}"
                self.assertTrue(paths[f"{base}_png"].exists())
                self.assertTrue(paths[f"{base}_svg"].exists())
                self.assertTrue(paths[f"{base}_pdf"].exists())
                self.assertTrue(paths[f"figure_{figure_no}_source_data"].exists())
            figure_1_svg = paths[f"SingleFamily_{FIGURE_NAMES[1]}_svg"].read_text(encoding="utf-8")
            self.assertIn("SSR", figure_1_svg)
            self.assertIn("SCR", figure_1_svg)
            figure_1_to_4_labels = [CITY_LABEL for CITY_LABEL in ("Bangkok", "Singapore", "Metro Manila", "Brisbane")]
            self.assertEqual(list(FIGURE_1_TO_4_CITY_ORDER), ["Bangkok", "Singapore", "Metro_Manila", "Brisbane"])
            _assert_city_label_order(self, figure_1_svg, figure_1_to_4_labels)
            for city in ("Singapore", "Metro Manila", "Brisbane", "Bangkok"):
                self.assertIn(city, figure_1_svg)
            figure_2_svg = paths[f"SingleFamily_{FIGURE_NAMES[2]}_svg"].read_text(encoding="utf-8")
            self.assertNotIn("Opt.", figure_2_svg)
            self.assertIn("kW PV", figure_2_svg)
            self.assertIn("kWh BESS", figure_2_svg)
            self.assertNotIn("#cfcece", figure_2_svg.lower())
            _assert_city_label_order(self, figure_2_svg, figure_1_to_4_labels)
            cost_colors = _cost_component_colors()
            self.assertEqual(
                cost_colors,
                {
                    "grid_bill": "#6F7E86",
                    "pv_annual_cost": "#5F8A70",
                    "battery_annual_cost": "#4E5F8A",
                    "lost_load_cost": "#8F5B55",
                },
            )
            figure_3_svg = paths[f"SingleFamily_{FIGURE_NAMES[3]}_svg"].read_text(encoding="utf-8")
            self.assertIn("Served energy", figure_3_svg)
            self.assertIn("Unserved energy", figure_3_svg)
            self.assertNotIn("Served load", figure_3_svg)
            self.assertNotIn("Unserved load", figure_3_svg)
            _assert_city_label_order(self, figure_3_svg, figure_1_to_4_labels)
            balance_palette = _single_outage_energy_balance_palette()
            self.assertEqual(balance_palette["grid"], "#9FC7C9")
            self.assertEqual(balance_palette["load"], "#AAB2B8")
            figure_4_svg = paths[f"SingleFamily_{FIGURE_NAMES[4]}_svg"].read_text(encoding="utf-8")
            self.assertIn("100 scenarios", figure_4_svg)
            self.assertIn("Energy-based LOLP", figure_4_svg)
            self.assertIn("Bangkok", figure_4_svg)
            _assert_city_label_order(self, figure_4_svg, figure_1_to_4_labels)
            figure_5_svg = paths[f"SingleFamily_{FIGURE_NAMES[5]}_svg"].read_text(encoding="utf-8")
            self.assertIn("CVaR", figure_5_svg)
            self.assertIn("lambda", figure_5_svg)
            self.assertIn("Bangkok", figure_5_svg)
            _assert_city_label_order(self, figure_5_svg, ["Singapore", "Metro Manila", "Brisbane", "Bangkok"])
            self.assertEqual(list(SINGAPORE_BRISBANE_CITY_ORDER), ["Singapore", "Brisbane"])
            for figure_no in ("1b", "2b", "3b"):
                svg = paths[f"SingleFamily_{FIGURE_NAMES[figure_no]}_svg"].read_text(encoding="utf-8")
                self.assertIn("Singapore", svg)
                self.assertIn("Brisbane", svg)
                self.assertNotIn("Bangkok", svg)
                self.assertNotIn("Metro Manila", svg)
                source_data = pd.read_csv(paths[f"figure_{figure_no}_source_data"])
                self.assertEqual(
                    source_data["city_key"].drop_duplicates().astype(str).tolist(),
                    list(SINGAPORE_BRISBANE_CITY_ORDER),
                )
            figure_1b_svg = paths[f"SingleFamily_{FIGURE_NAMES['1b']}_svg"].read_text(encoding="utf-8")
            _assert_city_label_order(self, figure_1b_svg, ["Singapore", "Brisbane"])
            for label in ("Cost", "UE", "SSR", "SCR"):
                self.assertIn(label, figure_1b_svg)
            self.assertNotIn("LOLP", figure_1b_svg)
            figure_1c_svg = paths[f"SingleFamily_{FIGURE_NAMES['1c']}_svg"].read_text(encoding="utf-8")
            for label in ("Cost", "LOLP", "0 PV / 0 BESS cost", "LOLP=0 min-cost design"):
                self.assertIn(label, figure_1c_svg)
            _assert_city_label_order(self, figure_1c_svg, ["Singapore", "Brisbane", "Jakarta", "Metro Manila"])
            figure_1c_source = pd.read_csv(paths["figure_1c_source_data"])
            figure_1c_design = figure_1c_source[
                figure_1c_source["source_table"].astype(str) == "affordable_design_space"
            ]
            figure_1c_plot_grid = figure_1c_source[
                figure_1c_source["source_table"].astype(str) == "affordable_design_space_plot_grid"
            ]
            self.assertEqual(
                figure_1c_design["city_key"].drop_duplicates().astype(str).tolist(),
                list(FIGURE_C_CITY_ORDER),
            )
            self.assertEqual(
                int(figure_1c_design["is_lolp_zero_min_cost_design"].astype(bool).sum()),
                len(FIGURE_C_CITY_ORDER),
            )
            self.assertFalse(figure_1c_plot_grid["energy_based_LOLP"].isna().any())
            figure_1d_svg = paths[f"SingleFamily_{FIGURE_NAMES['1d']}_svg"].read_text(encoding="utf-8")
            for label in ("Singapore night", "Singapore day", "Jakarta night", "Jakarta day", "Cost", "LOLP"):
                self.assertIn(label, figure_1d_svg)
            figure_1d_source = pd.read_csv(paths["figure_1d_source_data"])
            self.assertEqual(
                set(figure_1d_source["city_key"].dropna().astype(str)),
                set(FIGURE_D_CITY_ORDER),
            )
            self.assertEqual(set(figure_1d_source["scenario_key"].dropna().astype(str)), {"night", "day"})
            figure_2c_svg = paths[f"SingleFamily_{FIGURE_NAMES['2c']}_svg"].read_text(encoding="utf-8")
            self.assertIn("kW PV", figure_2c_svg)
            self.assertIn("kWh BESS", figure_2c_svg)
            _assert_city_label_order(self, figure_2c_svg, ["Singapore", "Brisbane", "Jakarta", "Metro Manila"])
            figure_2c_source = pd.read_csv(paths["figure_2c_source_data"])
            self.assertEqual(len(figure_2c_source), len(FIGURE_C_CITY_ORDER) * len(AFFORDABLE_COST_DESIGN_CASES) * 4)
            self.assertEqual(
                figure_2c_source["design_case"].drop_duplicates().astype(str).tolist(),
                list(AFFORDABLE_COST_DESIGN_CASES),
            )
            figure_2d_svg = paths[f"SingleFamily_{FIGURE_NAMES['2d']}_svg"].read_text(encoding="utf-8")
            self.assertIn("Night design", figure_2d_svg)
            self.assertIn("under day", figure_2d_svg)
            figure_2d_source = pd.read_csv(paths["figure_2d_source_data"])
            figure_2d_costs = figure_2d_source[
                figure_2d_source["source_table"].astype(str) == "day_night_cost_distribution"
            ]
            self.assertEqual(len(figure_2d_costs), len(FIGURE_D_CITY_ORDER) * len(DAY_NIGHT_COST_DESIGN_CASES) * 4)
            self.assertEqual(
                figure_2d_costs["design_case"].drop_duplicates().astype(str).tolist(),
                list(DAY_NIGHT_COST_DESIGN_CASES),
            )
            figure_3c_svg = paths[f"SingleFamily_{FIGURE_NAMES['3c']}_svg"].read_text(encoding="utf-8")
            self.assertIn("Served energy", figure_3c_svg)
            self.assertIn("Unserved energy", figure_3c_svg)
            _assert_city_label_order(self, figure_3c_svg, ["Singapore", "Brisbane", "Jakarta", "Metro Manila"])
            figure_3c_source = pd.read_csv(paths["figure_3c_source_data"])
            figure_3c_balance = figure_3c_source[
                figure_3c_source["source_table"].astype(str) == "affordable_energy_balance"
            ]
            self.assertEqual(
                figure_3c_balance.groupby("city_key").size().astype(int).to_dict(),
                {city: 72 for city in FIGURE_C_CITY_ORDER},
            )
            figure_6a_svg = paths[f"SingleFamily_{FIGURE_NAMES['6a']}_svg"].read_text(encoding="utf-8")
            self.assertIn("TOU", figure_6a_svg)
            self.assertIn("Hour of day", figure_6a_svg)
            _assert_city_label_order(self, figure_6a_svg, figure_1_to_4_labels)
            figure_6b_svg = paths[f"SingleFamily_{FIGURE_NAMES['6b']}_svg"].read_text(encoding="utf-8")
            self.assertIn("TOU", figure_6b_svg)
            self.assertIn("Hour of day", figure_6b_svg)
            self.assertIn("Singapore", figure_6b_svg)
            self.assertIn("Brisbane", figure_6b_svg)
            self.assertNotIn("Bangkok", figure_6b_svg)
            self.assertNotIn("Metro Manila", figure_6b_svg)
            figure_6a_source = pd.read_csv(paths["figure_6a_source_data"])
            self.assertEqual(len(figure_6a_source), len(FIGURE_1_TO_4_CITY_ORDER) * 24)
            self.assertEqual(figure_6a_source["city_key"].drop_duplicates().astype(str).tolist(), list(FIGURE_1_TO_4_CITY_ORDER))
            figure_6b_source = pd.read_csv(paths["figure_6b_source_data"])
            self.assertEqual(len(figure_6b_source), len(SINGAPORE_BRISBANE_CITY_ORDER) * 24)
            self.assertEqual(figure_6b_source["city_key"].drop_duplicates().astype(str).tolist(), list(SINGAPORE_BRISBANE_CITY_ORDER))
            figure_7_svg = paths[f"SingleFamily_{FIGURE_NAMES[7]}_svg"].read_text(encoding="utf-8")
            for label in ("TOU spread", "SAIDI", "VoLL", "Optimal PV", "Optimal BESS", "Annualized cost"):
                self.assertIn(label, figure_7_svg)
            figure_7_source = pd.read_csv(paths["figure_7_source_data"])
            self.assertEqual(len(figure_7_source), 22)
            self.assertEqual(set(figure_7_source["city_key"].astype(str)), {"Brisbane"})

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_figure_1_contour_uses_battery_x_and_pv_y(self) -> None:
        source = pd.DataFrame(
            {
                "city_key": ["Singapore"] * 4,
                "city_label": ["Singapore"] * 4,
                "case_key": ["SG_SingleFamily"] * 4,
                "pv_capacity_kw": [0.0, 0.0, 10.0, 10.0],
                "pv_capacity_mw": [0.0, 0.0, 0.01, 0.01],
                "battery_energy_kwh": [0.0, 20.0, 0.0, 20.0],
                "battery_energy_mwh": [0.0, 0.02, 0.0, 0.02],
                "system_total_cost_sgd_per_year": [1.0, 2.0, 3.0, 4.0],
                "system_total_cost_million_sgd_per_year": [0.000001, 0.000002, 0.000003, 0.000004],
                "UE_kwh_per_year": [0.0, 0.0, 0.0, 0.0],
                "UE_mwh_per_year": [0.0, 0.0, 0.0, 0.0],
                "outage_weighted_load_kwh": [1.0, 1.0, 1.0, 1.0],
                "energy_based_LOLP": [0.0, 0.0, 0.0, 0.0],
                "self_sufficiency_rate": [0.0, 0.0, 0.0, 0.0],
                "self_consumption_rate": [0.0, 0.0, 0.0, 0.0],
                "is_minimum_cost_design": [True, False, False, False],
            }
        )
        display_source, units = add_display_unit_columns(source)
        spec = _contour_metric_specs(units)[0]
        mpl, plt = matplotlib_context()
        apply_nature_style(mpl)
        fig, ax = plt.subplots()
        try:
            _draw_contour_panel(mpl, ax, display_source, spec, levels=[0.0, 2.0, 4.0])
            self.assertEqual(tuple(round(value, 6) for value in ax.get_xlim()), (0.0, 20.0))
            self.assertEqual(tuple(round(value, 6) for value in ax.get_ylim()), (0.0, 10.0))
            self.assertEqual(float(display_source[DISPLAY_BATTERY_ENERGY].max()), 20.0)
            self.assertEqual(float(display_source[DISPLAY_PV_CAPACITY].max()), 10.0)
        finally:
            plt.close(fig)

    def test_figure_1b_bounded_ue_levels_follow_source_range(self) -> None:
        values = [0.0, 0.394851, 2.009607]
        levels = _metric_levels(values, "ue", bound_to_data=True)

        self.assertAlmostEqual(float(levels[0]), min(values), places=9)
        self.assertAlmostEqual(float(levels[-1]), max(values), places=9)
        self.assertTrue(all(float(level) >= min(values) for level in levels))
        self.assertTrue(all(float(level) <= max(values) for level in levels))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_figure_1_ue_colormap_low_end_is_not_white(self) -> None:
        mpl, _ = matplotlib_context()
        cmap = _contour_colormap(mpl, {"key": "ue", "cmap": "Reds"})
        low_color = cmap(0.0)[:3]

        self.assertLess(min(float(channel) for channel in low_color), 0.90)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_single_outage_figures_apply_readable_font_style(self) -> None:
        mpl, _ = matplotlib_context()
        apply_nature_style(mpl)
        _apply_single_outage_readable_font_style(mpl)

        self.assertAlmostEqual(float(mpl.rcParams["font.size"]), SINGLE_OUTAGE_BASE_FONT_SIZE)
        self.assertAlmostEqual(float(mpl.rcParams["axes.labelsize"]), SINGLE_OUTAGE_BASE_FONT_SIZE)
        self.assertAlmostEqual(float(mpl.rcParams["xtick.labelsize"]), SINGLE_OUTAGE_TICK_FONT_SIZE)
        self.assertAlmostEqual(float(mpl.rcParams["ytick.labelsize"]), SINGLE_OUTAGE_TICK_FONT_SIZE)
        self.assertAlmostEqual(float(mpl.rcParams["legend.fontsize"]), SINGLE_OUTAGE_TICK_FONT_SIZE)
        self.assertAlmostEqual(float(mpl.rcParams["axes.titlesize"]), SINGLE_OUTAGE_TITLE_FONT_SIZE)
        self.assertGreater(SINGLE_OUTAGE_COLORBAR_LABEL_SIZE, 7.0)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_figure_2_cost_stack_bars_use_narrow_width(self) -> None:
        cost_rows = []
        component_specs = [
            ("grid_bill", "Grid bill", 1),
            ("pv_annual_cost", "PV depreciation + O&M", 2),
            ("battery_annual_cost", "Battery depreciation + O&M", 3),
            ("lost_load_cost", "Lost load cost", 4),
        ]
        design_sizes = {
            "no_pv_no_bess": (0.0, 0.0),
            "same_pv_no_bess": (5.0, 0.0),
            "optimal_pv_bess": (5.0, 12.0),
        }
        for design_case, (pv_capacity, battery_energy) in design_sizes.items():
            for component_key, component_label, stack_order in component_specs:
                cost_rows.append(
                    {
                        "design_case": design_case,
                        "component_key": component_key,
                        "component_label": component_label,
                        "stack_order": stack_order,
                        "cost_sgd": 100.0,
                        "pv_capacity_kw": pv_capacity,
                        "battery_energy_kwh": battery_energy,
                    }
                )
        costs = pd.DataFrame(cost_rows)
        units = figure_display_units(
            pd.DataFrame(
                {
                    "pv_capacity_kw": [0.0, 5.0, 5.0],
                    "pv_capacity_mw": [0.0, 0.005, 0.005],
                    "battery_energy_kwh": [0.0, 0.0, 12.0],
                    "battery_energy_mwh": [0.0, 0.0, 0.012],
                    "UE_kwh_per_year": [0.0, 0.0, 0.0],
                    "UE_mwh_per_year": [0.0, 0.0, 0.0],
                    "system_total_cost_sgd_per_year": [1000.0, 1200.0, 1400.0],
                }
            )
        )
        mpl, plt = matplotlib_context()
        apply_nature_style(mpl)
        fig, ax = plt.subplots()
        try:
            _stack_city_design_cost_bars(ax, costs, _cost_component_colors(), divisor=1000.0, units=units)
            self.assertTrue(ax.patches)
            for patch in ax.patches:
                self.assertAlmostEqual(float(patch.get_width()), SINGLE_OUTAGE_BAR_WIDTH, places=6)
            bar_centers = sorted({round(float(patch.get_x() + patch.get_width() / 2.0), 6) for patch in ax.patches})
            self.assertGreaterEqual(len(bar_centers), 3)
            for step in np.diff(bar_centers[:3]):
                self.assertAlmostEqual(float(step), SINGLE_OUTAGE_BAR_STEP, places=6)
            for label in ax.get_xticklabels():
                self.assertAlmostEqual(float(label.get_fontsize()), SINGLE_OUTAGE_COST_XTICK_SIZE, places=6)
        finally:
            plt.close(fig)

    def test_workflow_single_outage_city_entrypoint(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = run_selected(
                self._fast_config(),
                INPUT_ROOT,
                Path(tmp),
                run_baseline=False,
                run_sensitivity=False,
                run_single_outage_city_baseline=True,
                single_outage_city_figures=[1],
            )

            self.assertTrue(any("Figure_1_Four_City_Baseline_Design_Space_Contours" in str(path) for path in paths.values()))
            self.assertFalse(any("Uncertainty" in str(path) for path in paths.values()))

    def test_cli_single_outage_city_flags_reach_workflow(self) -> None:
        argv = [
            "pv_bess_sizing",
            "--analysis",
            "single-outage-city-baseline",
            "--single-outage-city-figure",
            "1c",
            "--single-outage-city-figure",
            "1d",
            "--single-outage-city-figure",
            "2c",
            "--single-outage-city-figure",
            "2d",
            "--single-outage-city-figure",
            "3c",
            "--single-outage-city-figure",
            "6a",
            "--single-outage-city-figure",
            "6b",
            "--single-outage-city-figure",
            "7",
        ]
        config = {"input": {"directory": "Input"}, "outputs": {"directory": "outputs"}}
        with (
            patch("sys.argv", argv),
            patch("src.app.cli.load_config", return_value=config),
            patch("src.app.cli.run_selected", return_value={}) as run_selected_mock,
        ):
            cli_module.main()

        kwargs = run_selected_mock.call_args.kwargs
        self.assertTrue(kwargs["run_single_outage_city_baseline"])
        self.assertFalse(kwargs["run_baseline"])
        self.assertEqual(kwargs["single_outage_city_figures"], ["1c", "1d", "2c", "2d", "3c", "6a", "6b", "7"])


if __name__ == "__main__":
    unittest.main()
