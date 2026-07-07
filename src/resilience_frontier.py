from __future__ import annotations

from typing import Any

import pandas as pd

from .data_loader import CaseParameters, InputCase, build_case_frame
from .optimizer import evaluate_candidate, scenario_config
from .outage import simulate_outages, summarize_outages
from .resilience_metrics import (
    DEFAULT_LOLP_THRESHOLD,
    achieved_lolp,
    chance_constraint_probability,
    electric_equivalent_building_load,
    eue,
    outage_load_served_ratio,
    outage_weighted_load_kwh,
    scenario_lolp,
)


DEFAULT_FRONTIER_OUTAGE_SCENARIO_COUNT = 100
DEFAULT_FRONTIER_SEED_START = 1000


def run_resilience_frontier(
    case: InputCase,
    config: dict[str, Any],
    parameters: CaseParameters,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frontier_cfg = _frontier_config(parameters)
    outage_scenarios = _generate_outage_scenarios(case, config, frontier_cfg)
    scenario_results = _evaluate_frontier_candidates(case, config, frontier_cfg, outage_scenarios)
    all_candidates = _aggregate_candidate_results(scenario_results, frontier_cfg)
    optimal_designs = _optimal_designs(all_candidates, frontier_cfg)
    return all_candidates, scenario_results, optimal_designs, outage_scenarios


def _frontier_config(parameters: CaseParameters) -> dict[str, float | int]:
    base = parameters.base
    return {
        "frontier_outage_scenario_count": _positive_int(
            base.get("frontier_outage_scenario_count"),
            DEFAULT_FRONTIER_OUTAGE_SCENARIO_COUNT,
            "frontier_outage_scenario_count",
        ),
        "frontier_seed_start": _nonnegative_int(
            base.get("frontier_seed_start"),
            DEFAULT_FRONTIER_SEED_START,
            "frontier_seed_start",
        ),
        "lolp_threshold": _nonnegative_float(
            base.get("lolp_threshold"),
            DEFAULT_LOLP_THRESHOLD,
            "lolp_threshold",
        ),
    }


def _generate_outage_scenarios(
    case: InputCase,
    config: dict[str, Any],
    frontier_cfg: dict[str, float | int],
) -> pd.DataFrame:
    scenario = config["scenarios"][case.case_key]
    rows: list[dict[str, Any]] = []
    scenario_count = int(frontier_cfg["frontier_outage_scenario_count"])
    seed_start = int(frontier_cfg["frontier_seed_start"])
    for offset in range(scenario_count):
        scenario_id = offset + 1
        seed = seed_start + offset
        profile, events = simulate_outages(
            saifi_per_year=float(scenario["saifi_per_year"]),
            saidi_min_per_year=float(scenario["saidi_min_per_year"]),
            hours=8760,
            seed=seed,
            forced_event_mode=str(config.get("outage", {}).get("forced_event_mode", "match_saidi")),
        )
        summary = summarize_outages(profile, events)
        max_duration = 0.0
        if not events.empty and "duration_minutes" in events.columns:
            max_duration = float(events["duration_minutes"].astype(float).max())
        rows.append(
            {
                "case_key": case.case_key,
                "city_key": case.city_key,
                "building_key": case.building_key,
                "frontier_scenario_id": scenario_id,
                "outage_seed": seed,
                "max_event_duration_min": max_duration,
                "outage_weight_sum_hours": float(profile["outage_fraction"].astype(float).sum()),
                **summary,
                "_outage_profile": profile,
                "_outage_events": events,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_frontier_candidates(
    case: InputCase,
    config: dict[str, Any],
    frontier_cfg: dict[str, float | int],
    outage_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    aware_cfg = scenario_config(config, case.case_key, resilience_informed=True)
    technical = aware_cfg["technical"]
    cooling_cop = float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0)))
    heating_cop = float(technical.get("heating_cop", 3.5))
    voll = float(aware_cfg["active_scenario"]["voll_sgd_per_kwh"])
    rows: list[dict[str, Any]] = []

    for pv_capacity in [float(value) for value in config["pv"]["capacity_kwp_candidates"]]:
        case_frame, pv_meta = build_case_frame(case, aware_cfg, pv_capacity)
        total_load = electric_equivalent_building_load(case_frame, cooling_cop, heating_cop)
        for battery_energy in [float(value) for value in config["battery"]["energy_kwh_candidates"]]:
            for scenario_row in outage_scenarios.to_dict("records"):
                outage_profile = scenario_row["_outage_profile"]
                frame = case_frame.merge(outage_profile, on="hour_index", how="left")
                result, dispatch = evaluate_candidate(frame, battery_energy, pv_meta, aware_cfg)
                outage_weighted_load = outage_weighted_load_kwh(total_load, frame["outage_fraction"])
                scenario_eue = eue(dispatch)
                scenario_unserved_cost = scenario_eue * voll
                rows.append(
                    {
                        "case_key": case.case_key,
                        "city_key": case.city_key,
                        "building_key": case.building_key,
                        "frontier_scenario_id": int(scenario_row["frontier_scenario_id"]),
                        "outage_seed": int(scenario_row["outage_seed"]),
                        "lolp_threshold": float(frontier_cfg["lolp_threshold"]),
                        "pv_capacity_kwp": float(result["pv_capacity_kwp"]),
                        "pv_capacity_mwp": float(result["pv_capacity_mwp"]),
                        "battery_energy_kwh": float(result["battery_energy_kwh"]),
                        "battery_power_kw": float(result["battery_power_kw"]),
                        "battery_c_rate": float(result["battery_c_rate"]),
                        "outage_weighted_load_kwh": float(outage_weighted_load.sum()),
                        "EUE_kwh_per_year": scenario_eue,
                        "outage_load_served_ratio": outage_load_served_ratio(dispatch, outage_weighted_load),
                        "scenario_LOLP": scenario_lolp(dispatch, frame["outage_fraction"]),
                        "unserved_energy_cost_sgd_per_year": scenario_unserved_cost,
                        "annualized_battery_cost_sgd": float(result["annualized_battery_cost_sgd"]),
                        "battery_om_cost_sgd": float(result["battery_om_cost_sgd"]),
                        "pv_total_annual_cost_sgd": float(result["pv_total_annual_cost_sgd"]),
                        "annualized_pv_cost_sgd": float(result["annualized_pv_cost_sgd"]),
                        "pv_om_cost_sgd": float(result["pv_om_cost_sgd"]),
                        "pv_bess_grid_energy_kwh": float(result["pv_bess_grid_energy_kwh"]),
                        "pv_bess_grid_cost_sgd": float(result["pv_bess_grid_cost_sgd"]),
                        "pv_bess_private_annual_cost_sgd": float(result["pv_bess_private_annual_cost_sgd"]),
                        "pv_bess_system_annual_cost_sgd": float(result["pv_bess_system_annual_cost_sgd"]),
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_candidate_results(
    scenario_results: pd.DataFrame,
    frontier_cfg: dict[str, float | int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["case_key", "city_key", "building_key", "pv_capacity_kwp", "battery_energy_kwh"]
    for keys, group in scenario_results.groupby(group_columns, sort=True):
        case_key, city_key, building_key, pv_capacity, battery_energy = keys
        scenario_lolps = group["scenario_LOLP"].astype(float).tolist()
        rows.append(
            {
                "case_key": case_key,
                "city_key": city_key,
                "building_key": building_key,
                "pv_capacity_kwp": float(pv_capacity),
                "pv_capacity_mwp": float(group["pv_capacity_mwp"].iloc[0]),
                "battery_energy_kwh": float(battery_energy),
                "battery_power_kw": float(group["battery_power_kw"].iloc[0]),
                "battery_c_rate": float(group["battery_c_rate"].iloc[0]),
                "frontier_outage_scenario_count": int(frontier_cfg["frontier_outage_scenario_count"]),
                "lolp_threshold": float(frontier_cfg["lolp_threshold"]),
                "mean_EUE_kwh_per_year": float(group["EUE_kwh_per_year"].astype(float).mean()),
                "mean_unserved_energy_cost_sgd_per_year": float(
                    group["unserved_energy_cost_sgd_per_year"].astype(float).mean()
                ),
                "mean_outage_load_served_ratio": float(group["outage_load_served_ratio"].astype(float).mean()),
                "achieved_LOLP": achieved_lolp(scenario_lolps),
                "chance_constraint_probability": chance_constraint_probability(
                    scenario_lolps,
                    lolp_threshold=float(frontier_cfg["lolp_threshold"]),
                ),
                "annualized_battery_cost_sgd_per_year": float(group["annualized_battery_cost_sgd"].astype(float).mean()),
                "battery_om_cost_sgd_per_year": float(group["battery_om_cost_sgd"].astype(float).mean()),
                "pv_total_annual_cost_sgd_per_year": float(group["pv_total_annual_cost_sgd"].astype(float).mean()),
                "annualized_pv_cost_sgd_per_year": float(group["annualized_pv_cost_sgd"].astype(float).mean()),
                "pv_om_cost_sgd_per_year": float(group["pv_om_cost_sgd"].astype(float).mean()),
                "mean_grid_cost_sgd_per_year": float(group["pv_bess_grid_cost_sgd"].astype(float).mean()),
                "mean_private_annual_cost_sgd_per_year": float(
                    group["pv_bess_private_annual_cost_sgd"].astype(float).mean()
                ),
                "system_total_cost_sgd_per_year": float(group["pv_bess_system_annual_cost_sgd"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("system_total_cost_sgd_per_year").reset_index(drop=True)


def _optimal_designs(
    all_candidates: pd.DataFrame,
    frontier_cfg: dict[str, float | int],
) -> pd.DataFrame:
    minimum = dict(all_candidates.sort_values("system_total_cost_sgd_per_year").iloc[0])
    rows = [
        {
            "design_type": "minimum_total_cost_design",
            "available": True,
            "unavailable_reason": "",
            **minimum,
        }
    ]
    feasible = all_candidates[
        all_candidates["achieved_LOLP"].astype(float) <= float(frontier_cfg["lolp_threshold"])
    ].copy()
    if feasible.empty:
        unavailable = {column: pd.NA for column in all_candidates.columns}
        unavailable.update(
            {
                "design_type": "reliability_constrained_design",
                "available": False,
                "unavailable_reason": "No candidate satisfies achieved_LOLP <= lolp_threshold.",
                "lolp_threshold": float(frontier_cfg["lolp_threshold"]),
            }
        )
        rows.append(unavailable)
    else:
        constrained = dict(feasible.sort_values("system_total_cost_sgd_per_year").iloc[0])
        rows.append(
            {
                "design_type": "reliability_constrained_design",
                "available": True,
                "unavailable_reason": "",
                **constrained,
            }
        )
    return pd.DataFrame(rows)


def frontier_outage_scenarios_for_csv(outage_scenarios: pd.DataFrame) -> pd.DataFrame:
    return outage_scenarios.drop(columns=["_outage_profile", "_outage_events"], errors="ignore")


def _positive_int(value: Any, default: int, name: str) -> int:
    result = int(float(default if value is None else value))
    if result <= 0 or float(default if value is None else value) != float(result):
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: Any, default: int, name: str) -> int:
    result = int(float(default if value is None else value))
    if result < 0 or float(default if value is None else value) != float(result):
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _nonnegative_float(value: Any, default: float, name: str) -> float:
    result = float(default if value is None else value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result
