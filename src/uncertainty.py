from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .io.data_loader import InputCase, build_case_frame
from .optimizer import CandidateEvaluator, CandidateStore, optimize_pv_battery_with_refinement, scenario_config
from .outage_evaluation import generate_outage_scenarios, outage_profile_diagnostics


UNCERTAINTY_METRICS = [
    "pv_capacity_kw",
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

    scenario_payloads: list[tuple[dict[str, Any], pd.DataFrame]] = []
    generated_scenarios = generate_outage_scenarios(
        case,
        config,
        scenario_count=int(scenario_count),
        seed_start=int(seed_start),
        scenario_id_column="uncertainty_scenario_id",
    )
    scenario_records: list[dict[str, Any]] = []
    for scenario_row in generated_scenarios.to_dict("records"):
        metrics = _outage_scenario_metrics(case, config, scenario_row)
        scenario_records.append(metrics)
        scenario_payloads.append((metrics, scenario_row["_outage_profile"]))
    outage_scenarios = pd.DataFrame(scenario_records)
    workers = int(config.get("uncertainty", {}).get("parallel_workers", 1))
    if workers <= 1 or len(scenario_payloads) <= 1:
        rows = _evaluate_uncertainty_scenarios_serial(case, config, case_key, scenario_payloads)
    else:
        max_workers = min(workers, len(scenario_payloads))
        tasks = [(case, config, case_key, metrics, outage_profile) for metrics, outage_profile in scenario_payloads]
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
        except OSError as exc:
            warnings.warn(
                f"Could not start uncertainty ProcessPoolExecutor ({exc}); falling back to serial execution.",
                RuntimeWarning,
                stacklevel=2,
            )
            rows = _evaluate_uncertainty_scenarios_serial(case, config, case_key, scenario_payloads)
        else:
            with executor:
                rows = list(executor.map(_evaluate_uncertainty_scenario, tasks))

    all_results = _sort_uncertainty_rows(pd.DataFrame(rows))
    summary = summarize_uncertainty_results(all_results)
    return all_results, outage_scenarios, summary


def _evaluate_uncertainty_scenarios_serial(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    scenario_payloads: list[tuple[dict[str, Any], pd.DataFrame]],
) -> list[dict[str, Any]]:
    return [
        _evaluate_uncertainty_scenario((case, config, case_key, metrics, outage_profile))
        for metrics, outage_profile in scenario_payloads
    ]


def _evaluate_uncertainty_scenario(args: tuple[Any, ...]) -> dict[str, Any]:
    case, config, case_key, scenario_metrics, outage_profile = args
    worker_config = deepcopy(config)
    worker_metrics = deepcopy(scenario_metrics)
    return _optimize_resilience_design_for_scenario(
        case,
        worker_config,
        case_key,
        outage_profile.copy(),
        worker_metrics,
    )


def _sort_uncertainty_rows(all_results: pd.DataFrame) -> pd.DataFrame:
    if all_results.empty:
        return all_results
    sorted_rows = all_results.copy()
    sorted_rows = sorted_rows.sort_values("uncertainty_scenario_id")
    return sorted_rows.reset_index(drop=True)


def summarize_uncertainty_results(all_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if all_results.empty:
        return pd.DataFrame()
    for metric in UNCERTAINTY_METRICS:
        values = pd.to_numeric(all_results[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
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


def _outage_scenario_metrics(
    case: InputCase,
    config: dict[str, Any],
    scenario_row: dict[str, Any],
) -> dict[str, Any]:
    reference_pv_capacity = float(max(config["pv"]["capacity_kw_candidates"]))
    diagnostics = outage_profile_diagnostics(
        case,
        config,
        scenario_row["_outage_profile"],
        reference_pv_capacity,
    )
    return {
        "uncertainty_scenario_id": int(scenario_row["uncertainty_scenario_id"]),
        "outage_seed": int(scenario_row["outage_seed"]),
        "simulated_saidi_min_per_year": float(scenario_row["simulated_saidi_min_per_year"]),
        "simulated_saifi_per_year": float(scenario_row["simulated_saifi_per_year"]),
        "max_event_duration_min": float(scenario_row["max_event_duration_min"]),
        **diagnostics,
    }


def _optimize_resilience_design_for_scenario(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    outage_profile: pd.DataFrame,
    scenario_metrics: dict[str, Any],
) -> dict[str, Any]:
    aware_cfg = scenario_config(deepcopy(config), case_key)

    def outage_builder(pv_capacity_kw: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, aware_cfg, pv_capacity_kw)
        return frame.merge(outage_profile, on="hour_index", how="left"), pv_meta

    evaluator = CandidateEvaluator(aware_cfg, CandidateStore(retain_dispatch=True))
    _, _, best = optimize_pv_battery_with_refinement(outage_builder, aware_cfg, evaluator=evaluator)
    best.update(
        _result_metadata(
            config,
            case_key,
            scenario_metrics,
            optimization_outage_mode="scenario_outage",
            optimization_voll_sgd_per_kwh=float(aware_cfg["active_scenario"]["voll_sgd_per_kwh"]),
        )
    )
    return best


def _result_metadata(
    config: dict[str, Any],
    case_key: str,
    scenario_metrics: dict[str, Any],
    optimization_outage_mode: str,
    optimization_voll_sgd_per_kwh: float,
) -> dict[str, Any]:
    scenario = config["scenarios"][case_key]
    city_key, _, building_key = case_key.partition("_")
    return {
        "case_key": case_key,
        "city_key": city_key,
        "building_key": building_key,
        "scenario_name": scenario["name"],
        "city_or_service_area": scenario["city_or_service_area"],
        "optimization_outage_mode": optimization_outage_mode,
        "optimization_voll_sgd_per_kwh": float(optimization_voll_sgd_per_kwh),
        "evaluation_voll_sgd_per_kwh": float(scenario["voll_sgd_per_kwh"]),
        "optimization_reused_across_uncertainty_scenarios": False,
        "saidi_min_per_year": float(scenario["saidi_min_per_year"]),
        "saifi_per_year": float(scenario["saifi_per_year"]),
        "voll_sgd_per_kwh": float(scenario["voll_sgd_per_kwh"]),
        "battery_capex_sgd_per_kwh": float(config["economics"]["battery_capex_sgd_per_kwh"]),
        "battery_power_capex_sgd_per_kw": float(config["economics"]["battery_power_capex_sgd_per_kw"]),
        "battery_om_fraction_per_year": float(config["economics"]["battery_om_fraction_per_year"]),
        "pv_capex_sgd_per_kw": float(config["pv"]["capex_sgd_per_kw"]),
        **scenario_metrics,
    }
