from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import pandas as pd

from .costing import baseline_no_battery_cost, pv_bess_annual_cost
from .dispatch import outage_free_frame, solve_dispatch_greedy

PvCapacityFrameBuilder = Callable[[float], tuple[pd.DataFrame, dict[str, float | str]]]


def scenario_config(config: dict[str, Any], scenario_key: str, resilience_informed: bool = True) -> dict[str, Any]:
    resolved = deepcopy(config)
    scenario = deepcopy(config["scenarios"][scenario_key])
    resolved["active_scenario"] = scenario
    resolved["active_scenario_key"] = scenario_key
    resolved["economics"] = deepcopy(config["economics"])
    resolved["economics"].update(scenario.get("battery_economics", {}))
    if not resilience_informed:
        resolved["active_scenario"]["voll_sgd_per_kwh"] = 0.0
    return resolved


def battery_sizing(energy_kwh: float, config: dict[str, Any]) -> dict[str, float]:
    energy = float(energy_kwh)
    max_c = float(config["battery"]["max_c_rate"])
    return {
        "battery_energy_kwh": energy,
        "battery_power_kw": energy * max_c if energy > 0 else 0.0,
        "battery_c_rate": max_c if energy > 0 else 0.0,
    }


def evaluate_candidate(
    frame: pd.DataFrame,
    battery_energy_kwh: float,
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    sizing = battery_sizing(battery_energy_kwh, config)
    dispatch = solve_dispatch_greedy(frame, sizing, config)
    baseline = baseline_no_battery_cost(frame, pv_meta, config)
    pv_bess = pv_bess_annual_cost(frame, dispatch, sizing, pv_meta, config)
    result = {
        **sizing,
        **baseline,
        **pv_bess,
        "avoided_unserved_energy_kwh_per_year": (
            baseline["baseline_unserved_energy_kwh_per_year"] - pv_bess["unserved_energy_kwh_per_year"]
        ),
        "system_annual_cost_delta_vs_baseline_sgd": (
            pv_bess["pv_bess_system_annual_cost_sgd"] - baseline["baseline_system_annual_cost_sgd"]
        ),
    }
    return result, dispatch


def optimize_battery(
    frame: pd.DataFrame,
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
    candidates: list[float] | None = None,
    row_metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    values = [float(x) for x in (candidates or config["battery"]["energy_kwh_candidates"])]
    rows: list[dict[str, Any]] = []
    dispatch_by_energy: dict[float, pd.DataFrame] = {}
    metadata = row_metadata or {}
    for value in values:
        result, dispatch = evaluate_candidate(frame, value, pv_meta, config)
        result.update(metadata)
        rows.append(result)
        dispatch_by_energy[value] = dispatch
    summary = pd.DataFrame(rows).sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    best_energy = float(summary.iloc[0]["battery_energy_kwh"])
    return summary, dispatch_by_energy[best_energy], dict(summary.iloc[0])


def optimize_battery_with_refinement(
    frame: pd.DataFrame,
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    coarse, _, coarse_best = optimize_battery(frame, pv_meta, config, row_metadata={"search_stage": "coarse", "search_iteration": 0})
    anchor = float(coarse_best["battery_energy_kwh"])
    window = float(config["battery"]["local_refine_window_kwh"])
    step = float(config["battery"]["local_refine_step_kwh"])
    start = max(0.0, anchor - window)
    end = max(start, anchor + window)
    count = int(round((end - start) / step))
    refined_values = sorted(set(round(start + i * step, 6) for i in range(count + 1)) | {anchor})
    refined, dispatch, best = optimize_battery(
        frame,
        pv_meta,
        config,
        refined_values,
        row_metadata={"search_stage": "optimal_region", "search_iteration": 0},
    )
    combined = pd.concat([coarse, refined], ignore_index=True)
    combined = combined.drop_duplicates(subset=["battery_energy_kwh"], keep="last")
    combined = combined.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    return combined, dispatch, best


def _refined_values(anchor: float, window: float, step: float) -> list[float]:
    start = max(0.0, anchor - window)
    end = max(start, anchor + window)
    count = int(round((end - start) / step))
    return sorted(set(round(start + i * step, 6) for i in range(count + 1)) | {anchor})


def _expanded_axis(values: list[float], factor: float = 1.5) -> list[float]:
    values = sorted({float(value) for value in values})
    if len(values) < 2:
        return values
    upper = max(values)
    if upper <= 0.0:
        return values
    intervals = len(values) - 1
    step = upper * float(factor) / intervals
    return [round(i * step, 6) for i in range(intervals + 1)]


def _is_upper_boundary(value: float, values: list[float]) -> bool:
    if len(values) < 2:
        return False
    upper = max(float(item) for item in values)
    if upper <= 0.0:
        return False
    return abs(float(value) - upper) <= 1e-6


def _is_axis_edge(value: float, values: list[float]) -> bool:
    if len(values) < 3:
        return False
    axis = sorted(float(item) for item in values)
    value_f = float(value)
    return abs(value_f - axis[0]) <= 1e-6 or abs(value_f - axis[-1]) <= 1e-6


def optimize_pv_battery_with_refinement(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pv_candidates = [float(value) for value in config["pv"]["capacity_kwp_candidates"]]
    battery_candidates = [float(value) for value in config["battery"]["energy_kwh_candidates"]]
    search_cfg = config.get("capacity_search", {})
    expansion_factor = float(search_cfg.get("upper_expansion_factor", 1.5))
    max_upper_expansions = int(search_cfg.get("max_upper_expansions", 2))
    refine_divisions = max(1.0, float(search_cfg.get("local_refine_divisions", 6)))
    coarse_history: list[pd.DataFrame] = []
    coarse_best: dict[str, Any] = {}
    final_coarse_iteration = 0
    for iteration in range(max_upper_expansions + 1):
        rows: list[pd.DataFrame] = []
        for pv_capacity in pv_candidates:
            frame, pv_meta = pv_capacity_frame_builder(float(pv_capacity))
            summary, _, _ = optimize_battery(
                frame,
                pv_meta,
                config,
                battery_candidates,
                row_metadata={"search_stage": "coarse", "search_iteration": iteration},
            )
            rows.append(summary)
        coarse = pd.concat(rows, ignore_index=True)
        coarse = coarse.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
        coarse_history.append(coarse)
        coarse_best = dict(coarse.iloc[0])
        expand_pv = _is_upper_boundary(float(coarse_best["pv_capacity_kwp"]), pv_candidates)
        expand_battery = _is_upper_boundary(float(coarse_best["battery_energy_kwh"]), battery_candidates)
        final_coarse_iteration = iteration
        if iteration >= max_upper_expansions or not (expand_pv or expand_battery):
            break
        if expand_pv:
            pv_candidates = _expanded_axis(pv_candidates, factor=expansion_factor)
            config["pv"]["capacity_kwp_candidates"] = pv_candidates
            config["pv"]["local_refine_window_kwp"] = pv_candidates[1] - pv_candidates[0]
            config["pv"]["local_refine_step_kwp"] = max((pv_candidates[1] - pv_candidates[0]) / refine_divisions, 1e-6)
        if expand_battery:
            battery_candidates = _expanded_axis(battery_candidates, factor=expansion_factor)
            config["battery"]["energy_kwh_candidates"] = battery_candidates
            config["battery"]["local_refine_window_kwh"] = battery_candidates[1] - battery_candidates[0]
            config["battery"]["local_refine_step_kwh"] = max((battery_candidates[1] - battery_candidates[0]) / refine_divisions, 1e-6)

    coarse = coarse_history[-1]
    best_pv_capacity = float(coarse_best["pv_capacity_kwp"])
    best_battery = float(coarse_best["battery_energy_kwh"])
    refined_history: list[pd.DataFrame] = []
    refined: pd.DataFrame | None = None
    for iteration in range(2):
        refined_pv_values = _refined_values(
            best_pv_capacity,
            float(config["pv"]["local_refine_window_kwp"]),
            float(config["pv"]["local_refine_step_kwp"]),
        )
        refined_battery_values = _refined_values(
            best_battery,
            float(config["battery"]["local_refine_window_kwh"]),
            float(config["battery"]["local_refine_step_kwh"]),
        )

        refined_rows: list[pd.DataFrame] = []
        for pv_capacity in refined_pv_values:
            refine_frame, refine_pv_meta = pv_capacity_frame_builder(float(pv_capacity))
            local_refined, _, _ = optimize_battery(
                refine_frame,
                refine_pv_meta,
                config,
                refined_battery_values,
                row_metadata={"search_stage": "optimal_region", "search_iteration": iteration},
            )
            refined_rows.append(local_refined)
        refined = pd.concat(refined_rows, ignore_index=True)
        refined = refined.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
        refined_history.append(refined)
        refined_best = dict(refined.iloc[0])
        best_pv_capacity = float(refined_best["pv_capacity_kwp"])
        best_battery = float(refined_best["battery_energy_kwh"])
        if not (_is_axis_edge(best_pv_capacity, refined_pv_values) or _is_axis_edge(best_battery, refined_battery_values)):
            break

    if refined is None:
        raise ValueError("PV-BESS optimization failed to generate a refined candidate grid")
    combined = pd.concat([coarse, refined], ignore_index=True)
    combined = combined.drop_duplicates(subset=["pv_capacity_kwp", "battery_energy_kwh"], keep="last")
    combined = combined.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    best = dict(combined.iloc[0])
    best["coarse_search_iteration"] = final_coarse_iteration
    best["optimal_region_iteration"] = int(refined["search_iteration"].max())
    best_frame, best_pv_meta = pv_capacity_frame_builder(float(best["pv_capacity_kwp"]))
    _, dispatch = evaluate_candidate(best_frame, float(best["battery_energy_kwh"]), best_pv_meta, config)
    all_rows = pd.concat(coarse_history + refined_history, ignore_index=True)
    all_rows = all_rows.drop_duplicates(subset=["pv_capacity_kwp", "battery_energy_kwh", "search_stage", "search_iteration"], keep="last")
    all_rows = all_rows.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    return all_rows, dispatch, best


def surface_search(
    base_frame: pd.DataFrame,
    frame_builder,
    scenario_key: str,
    outage_profile: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    scenario_cfg = scenario_config(config, scenario_key, resilience_informed=True)
    for pv_capacity in config["pv"]["capacity_kwp_candidates"]:
        frame, pv_meta = frame_builder(base_frame, scenario_key, config, float(pv_capacity))
        frame = frame.merge(outage_profile, on="hour_index", how="left")
        for battery in config["battery"]["energy_kwh_candidates"]:
            result, _ = evaluate_candidate(frame, float(battery), pv_meta, scenario_cfg)
            rows.append(
                {
                    "scenario": scenario_key,
                    **result,
                }
            )
    surface = pd.DataFrame(rows)
    matrix = surface.pivot_table(
        index="pv_capacity_kwp",
        columns="battery_energy_kwh",
        values="pv_bess_system_annual_cost_sgd",
        aggfunc="min",
    ).sort_index()
    return surface, matrix


def resilience_design_pair(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    config: dict[str, Any],
    scenario_key: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    aware_cfg = scenario_config(config, scenario_key, resilience_informed=True)
    flex_cfg = scenario_config(config, scenario_key, resilience_informed=False)

    def no_outage_builder(pv_capacity: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = pv_capacity_frame_builder(pv_capacity)
        return outage_free_frame(frame), pv_meta

    flex_summary, _, flex_opt_best = optimize_pv_battery_with_refinement(no_outage_builder, flex_cfg)
    flex_frame, flex_pv_meta = pv_capacity_frame_builder(float(flex_opt_best["pv_capacity_kwp"]))
    flex_eval, flex_dispatch = evaluate_candidate(
        flex_frame,
        float(flex_opt_best["battery_energy_kwh"]),
        flex_pv_meta,
        aware_cfg,
    )
    flex_eval.update(
        {
            "scenario": scenario_key,
            "design_case": "flexibility_only",
            "optimization_outage_mode": "no_outage",
            "optimization_voll_sgd_per_kwh": 0.0,
            "evaluation_voll_sgd_per_kwh": aware_cfg["active_scenario"]["voll_sgd_per_kwh"],
        }
    )

    aware_summary, aware_dispatch, aware_best = optimize_pv_battery_with_refinement(pv_capacity_frame_builder, aware_cfg)
    aware_best.update(
        {
            "scenario": scenario_key,
            "design_case": "resilience_informed",
            "optimization_outage_mode": "city_outage",
            "optimization_voll_sgd_per_kwh": aware_cfg["active_scenario"]["voll_sgd_per_kwh"],
            "evaluation_voll_sgd_per_kwh": aware_cfg["active_scenario"]["voll_sgd_per_kwh"],
        }
    )

    long_rows = pd.DataFrame([flex_eval, aware_best])
    dispatches = {"flexibility_only": flex_dispatch, "resilience_informed": aware_dispatch}
    summaries = {"flexibility_only": flex_summary, "resilience_informed": aware_summary}
    return long_rows, dispatches, summaries
