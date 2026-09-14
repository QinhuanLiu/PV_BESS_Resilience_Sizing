from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.figures.core import plot_smooth_line
from src.figures.single_outage_city_baseline import CITY_LABELS
from src.figures.singlefamily_preliminary_result import (
    NIGHT_OUTAGE_MODE,
    PRELIMINARY_CASE_BY_CITY,
    PRELIMINARY_CITY_ORDER,
    PRELIMINARY_DATA_STATUS,
    PRELIMINARY_FIGURE_NAMES,
    PRELIMINARY_SENSITIVITY_LABELS,
    PRELIMINARY_SENSITIVITY_PARAMETERS,
    build_singlefamily_preliminary_lolp_zero_sensitivity_source,
    build_singlefamily_preliminary_tou_profile_source,
    make_singlefamily_preliminary_result_figures,
    plot_singlefamily_preliminary_lolp_zero_sensitivity,
)
from src.io.config import DEFAULT_CONFIG
from src.io.data_loader import load_case_parameters


INPUT_ROOT = Path("Input")


class SingleFamilyPreliminaryResultFigureTests(unittest.TestCase):
    def _fast_config(self) -> dict:
        config = deepcopy(DEFAULT_CONFIG)
        config.setdefault("dispatch", {})["backend"] = "greedy"
        config.setdefault("capacity_search", {})["coarse_intervals"] = 1
        config.setdefault("capacity_search", {})["local_refine_divisions"] = 1
        config.setdefault("capacity_search", {})["max_upper_expansions"] = 0
        return config

    def _fake_tou_source(self) -> pd.DataFrame:
        rows = []
        for city_order, city_key in enumerate(PRELIMINARY_CITY_ORDER):
            for hour in range(24):
                rows.append(
                    {
                        "city_key": city_key,
                        "city_label": CITY_LABELS[city_key],
                        "case_key": PRELIMINARY_CASE_BY_CITY[city_key],
                        "source_case_key": PRELIMINARY_CASE_BY_CITY[city_key],
                        "hour": hour,
                        "interval_start_hour": float(hour),
                        "interval_end_hour": float(hour + 1),
                        "tou_sgd_per_kwh": 0.12 + 0.01 * city_order + (0.05 if 18 <= hour <= 21 else 0.0),
                        "data_status": PRELIMINARY_DATA_STATUS,
                    }
                )
        return pd.DataFrame(rows)

    def _fake_affordable_sources(self) -> dict[str, pd.DataFrame]:
        city_sources = [
            self._fake_affordable_city_sources(city_key, index)
            for index, city_key in enumerate(PRELIMINARY_CITY_ORDER)
        ]
        design_space = pd.concat([item["design_space"] for item in city_sources], ignore_index=True)
        return {
            "affordable_design_space": design_space,
            "affordable_design_space_plot_grid": design_space.copy(),
            "affordable_cost_distribution": pd.concat(
                [item["cost_distribution"] for item in city_sources],
                ignore_index=True,
            ),
            "affordable_energy_balance": pd.concat(
                [item["energy_balance"] for item in city_sources],
                ignore_index=True,
            ),
            "affordable_event_summary": pd.concat(
                [item["event_summary"] for item in city_sources],
                ignore_index=True,
            ),
        }

    def _fake_affordable_city_sources(self, city_key: str, city_order: int) -> dict[str, pd.DataFrame]:
        case_key = PRELIMINARY_CASE_BY_CITY[city_key]
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
                    "city_label": CITY_LABELS[city_key],
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
                    "lost_load_cost_sgd_per_year": max(
                        cost
                        - cost * 0.68
                        - cost * (0.12 if pv > 0 else 0.0)
                        - cost * (0.15 if battery > 0 else 0.0),
                        0.0,
                    ),
                    "UE_kwh_per_year": ue,
                    "UE_mwh_per_year": ue / 1000.0,
                    "outage_weighted_load_kwh": 10.0,
                    "energy_based_LOLP": lolp,
                    "energy_based_lolp_percent": 100.0 * lolp,
                    "self_sufficiency_rate": 0.20 + 0.04 * pv,
                    "self_consumption_rate": 0.70 + 0.03 * battery,
                    "is_minimum_cost_design": is_zero_ref,
                    "outage_mode": NIGHT_OUTAGE_MODE,
                    "outage_seed": 202601 + city_order,
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
                    "data_status": PRELIMINARY_DATA_STATUS,
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
                        "city_label": CITY_LABELS[city_key],
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
                        "outage_mode": NIGHT_OUTAGE_MODE,
                        "outage_seed": 202601 + city_order,
                        "zero_pv_zero_bess_cost_sgd_per_year": zero_zero_cost,
                        "affordable_boundary_cost_sgd_per_year": zero_zero_cost,
                        "selected_lolp_zero_cost_sgd_per_year": selected_cost,
                        "cost_delta_vs_no_pv_no_bess_sgd_per_year": total - zero_zero_cost,
                        "data_status": PRELIMINARY_DATA_STATUS,
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
                    "city_label": CITY_LABELS[city_key],
                    "case_key": case_key,
                    "source_case_key": case_key,
                    "plot_hour": hour,
                    "hour_index": 1000 + hour,
                    "timestamp": f"2026-01-01 {hour % 24:02d}:00:00",
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
                    "outage_mode": NIGHT_OUTAGE_MODE,
                    "outage_seed": 202601 + city_order,
                    "zero_pv_zero_bess_cost_sgd_per_year": zero_zero_cost,
                    "affordable_boundary_cost_sgd_per_year": zero_zero_cost,
                    "selected_lolp_zero_cost_sgd_per_year": selected_cost,
                    "cost_delta_vs_no_pv_no_bess_sgd_per_year": selected_cost - zero_zero_cost,
                    "data_status": PRELIMINARY_DATA_STATUS,
                }
            )

        event_summary = pd.DataFrame(
            [
                {
                    "city_key": city_key,
                    "city_label": CITY_LABELS[city_key],
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
                    "outage_mode": NIGHT_OUTAGE_MODE,
                    "outage_seed": 202601 + city_order,
                    "zero_pv_zero_bess_cost_sgd_per_year": zero_zero_cost,
                    "affordable_boundary_cost_sgd_per_year": zero_zero_cost,
                    "selected_lolp_zero_cost_sgd_per_year": selected_cost,
                    "cost_delta_vs_no_pv_no_bess_sgd_per_year": selected_cost - zero_zero_cost,
                    "data_status": PRELIMINARY_DATA_STATUS,
                }
            ]
        )
        return {
            "design_space": pd.DataFrame(design_rows),
            "cost_distribution": pd.DataFrame(cost_rows),
            "energy_balance": pd.DataFrame(balance_rows),
            "event_summary": event_summary,
        }

    def _fake_sensitivity_source(self) -> pd.DataFrame:
        rows = []
        for city_order, city_key in enumerate(PRELIMINARY_CITY_ORDER):
            case_key = PRELIMINARY_CASE_BY_CITY[city_key]
            for parameter_order, parameter in enumerate(PRELIMINARY_SENSITIVITY_PARAMETERS):
                for value_index, x_value in enumerate((0.0, 0.5, 1.0)):
                    value = 100.0 * x_value + 10.0 * parameter_order
                    system_cost = 2000.0 + 80.0 * city_order + 30.0 * parameter_order + 25.0 * x_value
                    pv_capacity = 2.0 + city_order + parameter_order + 0.4 * x_value
                    battery_energy = 1.0 + city_order + 0.5 * parameter_order + 0.2 * x_value
                    rows.append(
                        {
                            "city_key": city_key,
                            "city_label": CITY_LABELS[city_key],
                            "city_order": city_order,
                            "case_key": case_key,
                            "source_case_key": case_key,
                            "sensitivity_parameter": parameter,
                            "sensitivity_parameter_order": parameter_order,
                            "sensitivity_label": PRELIMINARY_SENSITIVITY_LABELS[parameter],
                            "sensitivity_value": value,
                            "sensitivity_value_index": value_index,
                            "sensitivity_unit": "test",
                            "sensitivity_base_value": 50.0,
                            "sensitivity_min": 0.0,
                            "sensitivity_max": 100.0,
                            "sensitivity_x_normalized": x_value,
                            "pv_capacity_kw": pv_capacity,
                            "pv_capacity_mw": pv_capacity / 1000.0,
                            "battery_energy_kwh": battery_energy,
                            "battery_energy_mwh": battery_energy / 1000.0,
                            "system_total_cost_sgd_per_year": system_cost,
                            "system_total_cost_million_sgd_per_year": system_cost / 1e6,
                            "UE_kwh_per_year": 0.0,
                            "UE_mwh_per_year": 0.0,
                            "energy_based_LOLP": 0.0,
                            "energy_based_lolp": 0.0,
                            "energy_based_lolp_percent": 0.0,
                            "outage_weighted_load_kwh": 10.0,
                            "is_lolp_zero_min_cost_design": True,
                            "outage_mode": NIGHT_OUTAGE_MODE,
                            "outage_seed": 42,
                            "night_window_start_hour": 19.0,
                            "night_window_end_hour": 23.0,
                            "data_status": PRELIMINARY_DATA_STATUS,
                        }
                    )
        return pd.DataFrame(rows)

    def _fake_sensitivity_rows_from_tasks(self, tasks) -> list[dict]:
        rows = []
        for task in tasks:
            city_key = str(task["city_key"])
            case_key = PRELIMINARY_CASE_BY_CITY[city_key]
            value = float(task["sensitivity_value"])
            pv_capacity = 2.0 + int(task["city_order"]) + 0.01 * value
            battery_energy = 1.0 + int(task["sensitivity_parameter_order"]) + 0.02 * value
            system_cost = 2000.0 + 3.0 * value + 25.0 * int(task["sensitivity_parameter_order"])
            rows.append(
                {
                    "city_key": city_key,
                    "city_label": CITY_LABELS[city_key],
                    "city_order": int(task["city_order"]),
                    "case_key": case_key,
                    "source_case_key": case_key,
                    "sensitivity_parameter": str(task["sensitivity_parameter"]),
                    "sensitivity_parameter_order": int(task["sensitivity_parameter_order"]),
                    "sensitivity_label": str(task["sensitivity_label"]),
                    "sensitivity_value": value,
                    "sensitivity_value_index": int(task["sensitivity_value_index"]),
                    "sensitivity_unit": str(task["sensitivity_unit"]),
                    "sensitivity_base_value": float(task["sensitivity_base_value"]),
                    "sensitivity_min": float(task["sensitivity_min"]),
                    "sensitivity_max": float(task["sensitivity_max"]),
                    "sensitivity_x_normalized": float(task["sensitivity_x_normalized"]),
                    "pv_capacity_kw": pv_capacity,
                    "pv_capacity_mw": pv_capacity / 1000.0,
                    "battery_energy_kwh": battery_energy,
                    "battery_energy_mwh": battery_energy / 1000.0,
                    "system_total_cost_sgd_per_year": system_cost,
                    "system_total_cost_million_sgd_per_year": system_cost / 1e6,
                    "UE_kwh_per_year": 0.0,
                    "UE_mwh_per_year": 0.0,
                    "energy_based_LOLP": 0.0,
                    "energy_based_lolp": 0.0,
                    "energy_based_lolp_percent": 0.0,
                    "outage_weighted_load_kwh": 10.0,
                    "is_lolp_zero_min_cost_design": True,
                    "outage_mode": NIGHT_OUTAGE_MODE,
                    "outage_seed": 42,
                    "night_window_start_hour": 19.0,
                    "night_window_end_hour": 23.0,
                    "data_status": PRELIMINARY_DATA_STATUS,
                }
            )
        return rows

    def test_tou_source_uses_requested_city_order(self) -> None:
        source = build_singlefamily_preliminary_tou_profile_source(INPUT_ROOT)

        self.assertEqual(source["city_key"].drop_duplicates().astype(str).tolist(), list(PRELIMINARY_CITY_ORDER))
        self.assertEqual(len(source), 4 * 24)
        self.assertNotIn("Bangkok", set(source["city_key"].astype(str)))
        for _, group in source.groupby("city_key", sort=False):
            self.assertEqual(group["hour"].astype(int).tolist(), list(range(24)))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_all_preliminary_figures_export_with_night_sources(self) -> None:
        sources = self._fake_affordable_sources()
        sources["tou_profile"] = self._fake_tou_source()
        sources["sensitivity_results"] = self._fake_sensitivity_source()
        with TemporaryDirectory() as tmp:
            paths = make_singlefamily_preliminary_result_figures(Path(tmp), figures=["all"], sources=sources)

            for figure_no, figure_name in PRELIMINARY_FIGURE_NAMES.items():
                base = f"SingleFamily_{figure_name}"
                self.assertTrue(paths[f"{base}_png"].exists())
                self.assertTrue(paths[f"{base}_svg"].exists())
                self.assertTrue(paths[f"{base}_pdf"].exists())
                self.assertTrue(paths[f"figure_{figure_no}_source_data"].exists())

            figure1 = pd.read_csv(paths["figure_1_source_data"])
            self.assertEqual(figure1["city_key"].drop_duplicates().astype(str).tolist(), list(PRELIMINARY_CITY_ORDER))
            figure2 = pd.read_csv(paths["figure_2_source_data"])
            figure3 = pd.read_csv(paths["figure_3_source_data"])
            self.assertEqual(set(figure2["outage_mode"].dropna().astype(str)), {NIGHT_OUTAGE_MODE})
            self.assertEqual(set(figure3["outage_mode"].dropna().astype(str)), {NIGHT_OUTAGE_MODE})
            figure4 = pd.read_csv(paths["figure_4_source_data"])
            self.assertEqual(set(figure4["sensitivity_parameter"].astype(str)), set(PRELIMINARY_SENSITIVITY_PARAMETERS))

    def test_sensitivity_builder_uses_city_input_ranges(self) -> None:
        with patch(
            "src.figures.singlefamily_preliminary_result._evaluate_preliminary_sensitivity_tasks_parallel",
            side_effect=self._fake_sensitivity_rows_from_tasks,
        ):
            source = build_singlefamily_preliminary_lolp_zero_sensitivity_source(
                INPUT_ROOT,
                config=self._fast_config(),
            )

        for city_key in PRELIMINARY_CITY_ORDER:
            parameters = load_case_parameters(INPUT_ROOT / PRELIMINARY_CASE_BY_CITY[city_key] / f"{PRELIMINARY_CASE_BY_CITY[city_key]}_Parameter.csv")
            for parameter in PRELIMINARY_SENSITIVITY_PARAMETERS:
                expected_values = [float(value) for value in parameters.sensitivity[parameter]]
                actual = source[
                    (source["city_key"].astype(str) == city_key)
                    & (source["sensitivity_parameter"].astype(str) == parameter)
                ]["sensitivity_value"].astype(float).tolist()
                self.assertEqual(actual, expected_values)
        self.assertEqual(set(source["outage_mode"].astype(str)), {NIGHT_OUTAGE_MODE})
        self.assertTrue((source["UE_kwh_per_year"].astype(float) == 0.0).all())
        self.assertTrue(source["is_lolp_zero_min_cost_design"].astype(bool).all())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_figure4_uses_smooth_lines_without_raw_points(self) -> None:
        sources = {"sensitivity_results": self._fake_sensitivity_source()}
        with TemporaryDirectory() as tmp:
            with patch(
                "src.figures.singlefamily_preliminary_result.plot_smooth_line",
                wraps=plot_smooth_line,
            ) as smoother:
                paths = plot_singlefamily_preliminary_lolp_zero_sensitivity(Path(tmp), "SingleFamily", sources)

            self.assertEqual(smoother.call_count, len(PRELIMINARY_CITY_ORDER) * len(PRELIMINARY_SENSITIVITY_PARAMETERS) * 4)
            for call in smoother.call_args_list:
                self.assertFalse(call.kwargs["show_points"])
            svg = paths[f"SingleFamily_{PRELIMINARY_FIGURE_NAMES[4]}_svg"].read_text(encoding="utf-8")
            self.assertNotIn('marker: url("#', svg)


if __name__ == "__main__":
    unittest.main()
