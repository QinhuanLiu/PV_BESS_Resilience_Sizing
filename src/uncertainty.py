from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .data_loader import InputCase, build_case_frame
from .dispatch import outage_free_frame
from .optimizer import evaluate_candidate, optimize_pv_battery_with_refinement, scenario_config
from .outage import simulate_outages, summarize_outages


UNCERTAINTY_METRICS = [
    "pv_capacity_kwp",
    "battery_energy_kwh",
    "pv_bess_private_annual_cost_sgd",
    "unserved_energy_kwh_per_year",
    "unserved_energy_cost_sgd_per_year",
    "pv_bess_system_annual_cost_sgd",
]


def run_uncertainty_analysis(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    scenario_count: int = 100,
    seed_start: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if int(scenario_count) <= 0:
        raise ValueError("uncertainty scenario count must be positive")

    scenario_records: list[dict[str, Any]] = []
    scenario_payloads: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for offset in range(int(scenario_count)):
        scenario_id = offset + 1
        seed = int(seed_start) + offset
        profile, events = _simulate_uncertainty_outage(config, case_key, seed)
        metrics = _outage_scenario_metrics(case, config, case_key, scenario_id, seed, profile, events)
        scenario_records.append(metrics)
        scenario_payloads.append((metrics, profile))

    outage_scenarios = pd.DataFrame(scenario_records)
    flex_best = _optimize_flexibility_design_once(case, config, case_key)

    workers = int(config.get("uncertainty", {}).get("parallel_workers", 1))
    if workers <= 1 or len(scenario_payloads) <= 1:
        scenario_results = _evaluate_uncertainty_scenarios_serial(case, config, case_key, flex_best, scenario_payloads)
    else:
        max_workers = min(workers, len(scenario_payloads))
        tasks = [(case, config, case_key, flex_best, metrics, outage_profile) for metrics, outage_profile in scenario_payloads]
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
        except OSError as exc:
            warnings.warn(
                f"Could not start uncertainty ProcessPoolExecutor ({exc}); falling back to serial execution.",
                RuntimeWarning,
                stacklevel=2,
            )
            scenario_results = _evaluate_uncertainty_scenarios_serial(case, config, case_key, flex_best, scenario_payloads)
        else:
            with executor:
                scenario_results = list(executor.map(_evaluate_uncertainty_scenario_pair, tasks))

    rows = [row for scenario_pair in scenario_results for row in scenario_pair]
    all_results = _sort_uncertainty_rows(pd.DataFrame(rows))
    summary = summarize_uncertainty_results(all_results)
    return all_results, outage_scenarios, summary


def _evaluate_uncertainty_scenarios_serial(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    flex_best: dict[str, Any],
    scenario_payloads: list[tuple[dict[str, Any], pd.DataFrame]],
) -> list[list[dict[str, Any]]]:
    return [
        _evaluate_uncertainty_scenario_pair((case, config, case_key, flex_best, metrics, outage_profile))
        for metrics, outage_profile in scenario_payloads
    ]


def _evaluate_uncertainty_scenario_pair(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    case, config, case_key, flex_best, scenario_metrics, outage_profile = args
    worker_config = deepcopy(config)
    worker_flex_best = deepcopy(flex_best)
    worker_metrics = deepcopy(scenario_metrics)
    return [
        _evaluate_flexibility_design(case, worker_config, case_key, worker_flex_best, outage_profile.copy(), worker_metrics),
        _optimize_resilience_design_for_scenario(
            case,
            worker_config,
            case_key,
            outage_profile.copy(),
            deepcopy(worker_metrics),
        ),
    ]


def _sort_uncertainty_rows(all_results: pd.DataFrame) -> pd.DataFrame:
    if all_results.empty:
        return all_results
    design_order = {"flexibility_only": 0, "resilience_informed": 1}
    sorted_rows = all_results.copy()
    sorted_rows["_design_order"] = sorted_rows["design_case"].map(design_order).fillna(99).astype(int)
    sorted_rows = sorted_rows.sort_values(["uncertainty_scenario_id", "_design_order"]).drop(columns="_design_order")
    return sorted_rows.reset_index(drop=True)


def summarize_uncertainty_results(all_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if all_results.empty:
        return pd.DataFrame()
    for design_case, group in all_results.groupby("design_case", sort=False):
        label = str(group["design_case_label"].iloc[0]) if "design_case_label" in group.columns else str(design_case)
        for metric in UNCERTAINTY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "design_case": design_case,
                    "design_case_label": label,
                    "metric": metric,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "p10": float(values.quantile(0.10)),
                    "p25": float(values.quantile(0.25)),
                    "p75": float(values.quantile(0.75)),
                    "p90": float(values.quantile(0.90)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "std": float(values.std(ddof=0)),
                }
            )
    return pd.DataFrame(rows)


def _simulate_uncertainty_outage(
    config: dict[str, Any],
    case_key: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = config["scenarios"][case_key]
    return simulate_outages(
        saifi_per_year=float(scenario["saifi_per_year"]),
        saidi_min_per_year=float(scenario["saidi_min_per_year"]),
        hours=8760,
        seed=int(seed),
        forced_event_mode=str(config.get("outage", {}).get("forced_event_mode", "match_saidi")),
    )


def _outage_scenario_metrics(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    scenario_id: int,
    seed: int,
    outage_profile: pd.DataFrame,
    outage_events: pd.DataFrame,
) -> dict[str, Any]:
    reference_pv_capacity = float(max(config["pv"]["capacity_kwp_candidates"]))
    frame, _ = build_case_frame(case, config, reference_pv_capacity)
    merged = frame.merge(outage_profile, on="hour_index", how="left")
    outage_fraction = merged["outage_fraction"].astype(float)
    cooling_cop = max(float(config["technical"].get("cooling_cop", config["technical"].get("chiller_cop", 5.0))), 1e-9)
    heating_cop = max(float(config["technical"].get("heating_cop", 3.5)), 1e-9)
    load_e = (
        merged["electric_load_kwh"].astype(float)
        + merged["cooling_load_kwh_th"].astype(float) / cooling_cop
        + merged["heating_load_kwh_th"].astype(float) / heating_cop
    )
    high_price = float(merged["tou_sgd_per_kwh"].astype(float).quantile(0.75))
    outage_summary = summarize_outages(outage_profile, outage_events)
    max_event_duration = 0.0
    if not outage_events.empty and "duration_minutes" in outage_events.columns:
        max_event_duration = float(outage_events["duration_minutes"].astype(float).max())
    return {
        "uncertainty_scenario_id": int(scenario_id),
        "outage_seed": int(seed),
        "simulated_saidi_min_per_year": float(outage_summary["simulated_saidi_min_per_year"]),
        "simulated_saifi_per_year": float(outage_summary["simulated_saifi_per_year"]),
        "max_event_duration_min": max_event_duration,
        "outage_hours_peak_tou": float(outage_fraction[merged["tou_sgd_per_kwh"].astype(float) >= high_price].sum()),
        "outage_weighted_load_kwh": float((outage_fraction * load_e).sum()),
        "outage_weighted_pv_available_kwh": float((outage_fraction * merged["pv_kwh"].astype(float)).sum()),
        "reference_pv_capacity_kwp": reference_pv_capacity,
        "peak_tou_threshold_sgd_per_kwh": high_price,
    }


def _optimize_flexibility_design_once(case: InputCase, config: dict[str, Any], case_key: str) -> dict[str, Any]:
    flex_cfg = scenario_config(deepcopy(config), case_key, resilience_informed=False)

    def no_outage_builder(pv_capacity_kwp: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, flex_cfg, pv_capacity_kwp)
        return outage_free_frame(frame), pv_meta

    _, _, best = optimize_pv_battery_with_refinement(no_outage_builder, flex_cfg)
    return {
        "pv_capacity_kwp": float(best["pv_capacity_kwp"]),
        "battery_energy_kwh": float(best["battery_energy_kwh"]),
    }


def _evaluate_flexibility_design(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    flex_best: dict[str, Any],
    outage_profile: pd.DataFrame,
    scenario_metrics: dict[str, Any],
) -> dict[str, Any]:
    aware_cfg = scenario_config(deepcopy(config), case_key, resilience_informed=True)
    frame, pv_meta = build_case_frame(case, aware_cfg, float(flex_best["pv_capacity_kwp"]))
    frame = frame.merge(outage_profile, on="hour_index", how="left")
    result, _ = evaluate_candidate(frame, float(flex_best["battery_energy_kwh"]), pv_meta, aware_cfg)
    result.update(
        _result_metadata(
            config,
            case_key,
            scenario_metrics,
            design_case="flexibility_only",
            optimization_outage_mode="no_outage",
            optimization_voll_sgd_per_kwh=0.0,
            optimization_reused=True,
        )
    )
    return result


def _optimize_resilience_design_for_scenario(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    outage_profile: pd.DataFrame,
    scenario_metrics: dict[str, Any],
) -> dict[str, Any]:
    aware_cfg = scenario_config(deepcopy(config), case_key, resilience_informed=True)

    def outage_builder(pv_capacity_kwp: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, aware_cfg, pv_capacity_kwp)
        return frame.merge(outage_profile, on="hour_index", how="left"), pv_meta

    _, _, best = optimize_pv_battery_with_refinement(outage_builder, aware_cfg)
    best.update(
        _result_metadata(
            config,
            case_key,
            scenario_metrics,
            design_case="resilience_informed",
            optimization_outage_mode="scenario_outage",
            optimization_voll_sgd_per_kwh=float(aware_cfg["active_scenario"]["voll_sgd_per_kwh"]),
            optimization_reused=False,
        )
    )
    return best


def _result_metadata(
    config: dict[str, Any],
    case_key: str,
    scenario_metrics: dict[str, Any],
    design_case: str,
    optimization_outage_mode: str,
    optimization_voll_sgd_per_kwh: float,
    optimization_reused: bool,
) -> dict[str, Any]:
    scenario = config["scenarios"][case_key]
    city_key, _, building_key = case_key.partition("_")
    return {
        "case_key": case_key,
        "city_key": city_key,
        "building_key": building_key,
        "scenario_name": scenario["name"],
        "city_or_service_area": scenario["city_or_service_area"],
        "design_case": design_case,
        "design_case_label": "Flexibility-only" if design_case == "flexibility_only" else "Resilience-informed",
        "optimization_outage_mode": optimization_outage_mode,
        "optimization_voll_sgd_per_kwh": float(optimization_voll_sgd_per_kwh),
        "evaluation_voll_sgd_per_kwh": float(scenario["voll_sgd_per_kwh"]),
        "optimization_reused_across_uncertainty_scenarios": bool(optimization_reused),
        "saidi_min_per_year": float(scenario["saidi_min_per_year"]),
        "saifi_per_year": float(scenario["saifi_per_year"]),
        "voll_sgd_per_kwh": float(scenario["voll_sgd_per_kwh"]),
        "battery_capex_sgd_per_kwh": float(config["economics"]["battery_capex_sgd_per_kwh"]),
        "battery_power_capex_sgd_per_kw": float(config["economics"]["battery_power_capex_sgd_per_kw"]),
        "battery_om_fraction_per_year": float(config["economics"]["battery_om_fraction_per_year"]),
        "pv_capex_sgd_per_kwp": float(config["pv"]["capex_sgd_per_kwp"]),
        **scenario_metrics,
    }
