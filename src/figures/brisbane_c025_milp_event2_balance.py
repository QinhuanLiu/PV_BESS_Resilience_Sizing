"""Compare the saved Brisbane event-2 Greedy dispatch with the current MILP backend.

The backend is a continuous linear program despite its historical name. It sees
the complete 72-hour window; this experiment does not re-optimize capacity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..brisbane_c025_15y_optimization import CASE_KEY, RESULTS_DIR, ROOT, _sha256
from ..dispatch import solve_dispatch_milp
from ..io.data_loader import build_case_frame, discover_input_cases
from ..optimizer import battery_sizing, scenario_config
from .brisbane_c025_optimized_outage_balance import (
    _read_csv,
    _validated_run,
    build_source_data as build_greedy_source_data,
)
from .core import write_source_data
from .typical_outage_72h_energy_balance import (
    ENERGY_TOLERANCE_KWH,
    WINDOW_HOURS,
    _balance_columns,
    plot_figure,
)


EVENT_ID = 2
BASE_NAME = "Brisbane_15Y_C0p25_Event2_72h_MILPBackend_ContinuousLP_Energy_Balance"
ACTIVITY_THRESHOLD_KWH = 0.01


def _dispatch_metrics(
    frame: pd.DataFrame, initial_soc_kwh: float, config: dict[str, Any],
    battery_power_kw: float, degradation_penalty_sgd_per_kwh: float,
) -> dict[str, float | int | list[int]]:
    charge = frame["battery_charge_kwh"].to_numpy(dtype=float)
    discharge = frame["battery_discharge_kwh"].to_numpy(dtype=float)
    charge_active = charge >= ACTIVITY_THRESHOLD_KWH
    discharge_active = discharge >= ACTIVITY_THRESHOLD_KWH
    directions = np.where(charge_active & ~discharge_active, 1,
                          np.where(discharge_active & ~charge_active, -1, 0))
    active_directions = directions[directions != 0]
    prices = frame["tou_sgd_per_kwh"].to_numpy(dtype=float)
    export_prices = frame["pv_export_sgd_per_kwh"].to_numpy(dtype=float)
    voll = float(config["active_scenario"]["voll_sgd_per_kwh"])
    penalty = degradation_penalty_sgd_per_kwh
    objective = (
        float(np.dot(prices, frame["grid_import_kwh"]))
        - float(np.dot(export_prices, frame["pv_export_kwh"]))
        + voll * float(frame["unmet_load_kwh_equiv"].sum())
        + penalty * float((frame["battery_charge_kwh"] + frame["battery_discharge_kwh"]).sum())
    )
    exceeded = frame.loc[
        frame["battery_charge_kwh"] > battery_power_kw + ENERGY_TOLERANCE_KWH, "hour_index"
    ].astype(int).tolist()
    return {
        "initial_soc_kwh": float(initial_soc_kwh),
        "final_soc_kwh": float(frame["battery_soc_kwh"].iloc[-1]),
        "charge_kwh": float(charge.sum()),
        "discharge_kwh": float(discharge.sum()),
        "maximum_hourly_charge_kwh": float(charge.max()),
        "maximum_hourly_discharge_kwh": float(discharge.max()),
        "charge_power_limit_exceeded_hour_indices": exceeded,
        "charge_power_limit_exceeded_hour_count": len(exceeded),
        "active_charge_hour_count": int(charge_active.sum()),
        "active_discharge_hour_count": int(discharge_active.sum()),
        "same_hour_charge_and_discharge_count": int((charge_active & discharge_active).sum()),
        "nonidle_direction_reversal_count": int(np.count_nonzero(np.diff(active_directions)))
        if len(active_directions) > 1 else 0,
        "unserved_kwh": float(frame["unmet_load_kwh_equiv"].sum()),
        "operational_objective_sgd": objective,
    }


def build_source_data(results_dir: Path = RESULTS_DIR) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    results_dir = Path(results_dir)
    config, manifest = _validated_run(results_dir)
    greedy_source, greedy_summary = build_greedy_source_data(results_dir)
    if int(greedy_summary["event_id"]) != EVENT_ID or not greedy_summary["plotted_design_is_global_optimum"]:
        raise ValueError("The saved selected event/design differs from the fixed comparison")
    if (float(greedy_summary["pv_capacity_kw"]), float(greedy_summary["battery_energy_kwh"])) != (3.5, 11.6):
        raise ValueError("The fixed 3.5 kW / 11.6 kWh design is missing")
    start = int(greedy_summary["window_start_hour"])
    end = int(greedy_summary["window_end_exclusive_hour"])
    if (start, end) != (11280, 11352):
        raise ValueError("The selected event no longer has the expected 72-hour window")
    events = _read_csv(results_dir, "Outage_Events")
    overlapping = events.loc[(events["end_hour"] > start) & (events["start_hour"] < end)]
    if overlapping["event_id"].astype(int).tolist() != [EVENT_ID]:
        raise ValueError("The 72-hour window contains another outage")

    saved_dispatch = _read_csv(results_dir, "Dispatch")
    initial_soc = float(saved_dispatch.iloc[start - 1]["battery_soc_kwh"])
    greedy = saved_dispatch.iloc[start:end].reset_index(drop=True)
    case = next(item for item in discover_input_cases(ROOT / "Input") if item.case_key == CASE_KEY)
    case_frame, _ = build_case_frame(case, config, 3.5)
    profile = _read_csv(results_dir, "Outage_Profile")
    if len(case_frame) != len(profile) or len(case_frame) != len(saved_dispatch):
        raise ValueError("The case frame, outage profile and Greedy dispatch do not align")
    window = case_frame.iloc[start:end].copy().reset_index(drop=True)
    outage = profile.iloc[start:end].reset_index(drop=True)
    if not np.array_equal(window["hour_index"].to_numpy(), outage["hour_index"].to_numpy()):
        raise ValueError("The saved outage profile is misaligned")
    window["grid_availability"] = outage["grid_availability"].to_numpy(dtype=float)
    window["outage_fraction"] = outage["outage_fraction"].to_numpy(dtype=float)
    technical = config["technical"]
    equivalent_load = (
        window["electric_load_kwh"]
        + window["cooling_load_kwh_th"] / float(technical["cooling_cop"])
        + window["heating_load_kwh_th"] / float(technical["heating_cop"])
    )
    for actual, saved, name in (
        (window["pv_kwh"], greedy["pv_available_kwh"], "PV"),
        (equivalent_load, greedy["total_load_e_equiv_kwh"], "load"),
        (window["tou_sgd_per_kwh"], greedy["tou_sgd_per_kwh"], "TOU price"),
        (window["grid_availability"], greedy["grid_availability"], "grid availability"),
    ):
        if not np.allclose(actual, saved, atol=ENERGY_TOLERANCE_KWH, rtol=0):
            raise ValueError(f"MILP and Greedy input mismatch: {name}")

    scenario = scenario_config(config, CASE_KEY)
    sizing = battery_sizing(11.6, config)
    milp = solve_dispatch_milp(window, sizing, scenario, initial_soc_kwh=initial_soc)
    if len(milp) != WINDOW_HOURS or not milp["solver_status"].eq("Optimal").all():
        raise ValueError("The 72-hour MILP backend did not solve to optimality")
    if not np.array_equal(milp["hour_index"].to_numpy(dtype=int), greedy["hour_index"].to_numpy(dtype=int)):
        raise ValueError("The MILP result hours differ from Greedy")
    if (milp[["battery_charge_kwh", "battery_discharge_kwh"]].to_numpy() > 2.9 + ENERGY_TOLERANCE_KWH).any():
        raise ValueError("The MILP result exceeds the 0.25C power limit")
    soc_min = 11.6 * float(config["battery"]["soc_min_fraction"])
    soc_max = 11.6 * float(config["battery"]["soc_max_fraction"])
    if ((milp["battery_soc_kwh"] < soc_min - ENERGY_TOLERANCE_KWH)
            | (milp["battery_soc_kwh"] > soc_max + ENERGY_TOLERANCE_KWH)).any():
        raise ValueError("The MILP result violates the SOC limits")
    if float(milp["battery_soc_kwh"].iloc[-1]) < initial_soc - ENERGY_TOLERANCE_KWH:
        raise ValueError("The MILP terminal SOC is below its initial SOC")

    source = milp.copy()
    source.insert(0, "plot_hour", np.arange(WINDOW_HOURS, dtype=int))
    source["analysis_year"] = window["analysis_year"].to_numpy()
    source["hour_of_year"] = window["hour_of_year"].to_numpy()
    source["day_label"] = np.repeat(["Previous day", "Outage day", "Recovery day"], 24)
    event_start = float(greedy_summary["event_start_hour"])
    event_end = float(greedy_summary["event_end_hour"])
    source["event_outage_fraction"] = np.maximum(
        0.0, np.minimum(event_end, source["hour_index"].to_numpy() + 1.0)
        - np.maximum(event_start, source["hour_index"].to_numpy()),
    )
    if not np.allclose(source["event_outage_fraction"], source["outage_fraction"],
                       atol=ENERGY_TOLERANCE_KWH, rtol=0):
        raise ValueError("The MILP outage differs from saved event 2")
    source["is_event_hour"] = source["event_outage_fraction"] > 0
    source = _balance_columns(source, config, 11.6)
    source["outage_load_kwh"] = source["critical_load_kwh"] * source["event_outage_fraction"]
    source["battery_to_outage_kwh"] = source["battery_discharge_outage_kwh"]
    source["outage_balance_residual_kwh"] = (
        source["outage_load_kwh"] - source["pv_to_outage_kwh"]
        - source["battery_to_outage_kwh"] - source["unmet_outage_kwh"]
    )
    affected = source.loc[source["is_event_hour"]]
    if affected.empty or float(affected["outage_balance_residual_kwh"].abs().max()) > ENERGY_TOLERANCE_KWH:
        raise ValueError("The MILP outage balance does not close")
    source["event_id"] = EVENT_ID
    source["event_start_hour"] = event_start
    source["event_end_hour"] = event_end

    penalty = float(milp["degradation_penalty_sgd_per_kwh"].iloc[0])
    greedy_metrics = _dispatch_metrics(greedy, initial_soc, scenario, 2.9, penalty)
    milp_metrics = _dispatch_metrics(milp, initial_soc, scenario, 2.9, penalty)
    comparison = {
        "case_key": CASE_KEY,
        "event_id": EVENT_ID,
        "window_start_hour": start,
        "window_end_exclusive_hour": end,
        "pv_capacity_kw": 3.5,
        "battery_energy_kwh": 11.6,
        "battery_power_limit_kw": 2.9,
        "activity_threshold_kwh": ACTIVITY_THRESHOLD_KWH,
        "direction_reversal_rule": "classify single-direction hours with flow >= 0.01 kWh, ignore idle and simultaneous hours, count changes between successive classified hours",
        "greedy": greedy_metrics,
        "milp_backend_continuous_lp": milp_metrics,
        "milp_same_hour_bidirectional_details": milp.loc[
            (milp["battery_charge_kwh"] >= ACTIVITY_THRESHOLD_KWH)
            & (milp["battery_discharge_kwh"] >= ACTIVITY_THRESHOLD_KWH),
            ["hour_index", "grid_availability", "outage_fraction", "grid_to_battery_kwh",
             "pv_to_battery_kwh", "battery_charge_kwh", "battery_discharge_kwh",
             "battery_discharge_outage_kwh"],
        ].to_dict(orient="records"),
        "same_hour_note": "An hour can contain grid-available and outage portions; both directions within that hour do not establish simultaneous instantaneous power flow.",
        "comparison_limit": "The current MILP backend has full knowledge of all 72 hours; Greedy is myopic. This is a fixed-design dispatch illustration, not a 15-year MILP capacity optimization.",
        "original_input_sha256": manifest["source_input_sha256"],
    }
    summary = dict(greedy_summary)
    summary.update({
        "dispatch_backend": "milp",
        "dispatch_model_class": "continuous_linear_program_no_integer_variables",
        "plotted_design_annual_cost_source": "saved_15_year_greedy_optimization",
        "dispatch_horizon_hours": WINDOW_HOURS,
        "initial_soc_kwh": initial_soc,
        "terminal_soc_kwh": float(milp["battery_soc_kwh"].iloc[-1]),
        "terminal_soc_rule": "not_below_initial_soc",
        "forecast_information": "full_72_hour_perfect_foresight",
        "outage_load_kwh": float(affected["outage_load_kwh"].sum()),
        "pv_to_outage_kwh": float(affected["pv_to_outage_kwh"].sum()),
        "battery_to_outage_kwh": float(affected["battery_to_outage_kwh"].sum()),
        "unserved_outage_kwh": float(affected["unmet_outage_kwh"].sum()),
        "maximum_hourly_ac_balance_residual_kwh": float(source["ac_balance_residual_kwh"].abs().max()),
        "maximum_outage_balance_residual_kwh": float(affected["outage_balance_residual_kwh"].abs().max()),
        "solver_status": "Optimal",
        "solver_objective_sgd": float(milp["milp_objective_sgd"].iloc[0]),
    })
    return source, summary, comparison


def run(results_dir: Path = RESULTS_DIR) -> dict[str, Path]:
    source, summary, comparison = build_source_data(results_dir)
    output_dir = Path(results_dir) / "figures"
    paths = plot_figure(
        source, summary, output_dir, base_name=BASE_NAME,
        pv_capacity_kw=3.5, battery_energy_kwh=11.6,
        design_status="Seed 42, event 2 | Fixed design | MILP backend (continuous LP)",
    )
    base = output_dir / BASE_NAME
    paths["source_data"] = write_source_data(base, source)
    summary["source_data_sha256"] = _sha256(paths["source_data"])
    summary["figure_sha256"] = {key: _sha256(path) for key, path in paths.items() if key != "source_data"}
    summary_path = base.with_name(base.name + "_Event_Summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    paths["event_summary"] = summary_path
    comparison_path = base.with_name(base.name + "_Greedy_Comparison.json")
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
    paths["comparison"] = comparison_path
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    for label, path in run(args.results_dir).items():
        print(f"{label}: {path}")
