from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .io.data_loader import InputCase, build_case_frame
from .optimizer import (
    CandidateEvaluator,
    CandidateStore,
    PvCapacityFrameBuilder,
    _expanded_axis,
    _is_axis_edge,
    _is_upper_boundary,
    _refined_values,
    scenario_config,
)
from .outage_evaluation import generate_outage_scenarios, outage_profile_diagnostics


MANILA_LOLP_ZERO_CASE_KEY = "Manila_SingleFamily"
DEFAULT_LOLP_ZERO_SCENARIO_COUNT = 30
LOLP_ZERO_TOLERANCE_KWH = 1e-6

LOLP_ZERO_CV_METRICS = (
    ("system_total_cost_sgd_per_year", "annualized_system_cost", "SGD/year"),
    ("pv_capacity_kw", "pv_capacity", "kW"),
    ("battery_energy_kwh", "battery_energy", "kWh"),
)


def run_lolp_zero_uncertainty_analysis(
    case: InputCase,
    config: dict[str, Any],
    case_key: str | None = None,
    scenario_count: int = DEFAULT_LOLP_ZERO_SCENARIO_COUNT,
    seed_start: int | None = None,
    tolerance_kwh: float = LOLP_ZERO_TOLERANCE_KWH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run per-outage LOLP=0 sizing for one typical year."""
    resolved_case_key = case_key or case.case_key
    if resolved_case_key != case.case_key:
        raise ValueError(f"LOLP=0 uncertainty case_key {resolved_case_key} does not match input case {case.case_key}.")
    if int(scenario_count) <= 0:
        raise ValueError("LOLP=0 uncertainty scenario count must be positive.")
    if float(tolerance_kwh) < 0.0:
        raise ValueError("LOLP=0 tolerance must be non-negative.")
    if seed_start is None:
        seed_start = int(config.get("uncertainty", {}).get("seed_start", 1000))

    generated_scenarios = generate_outage_scenarios(
        case,
        config,
        scenario_count=int(scenario_count),
        seed_start=int(seed_start),
        scenario_id_column="uncertainty_scenario_id",
    )
    scenario_payloads: list[tuple[dict[str, Any], pd.DataFrame]] = []
    scenario_records: list[dict[str, Any]] = []
    for scenario_row in generated_scenarios.to_dict("records"):
        metrics = _outage_scenario_metrics(case, config, scenario_row)
        scenario_records.append(metrics)
        scenario_payloads.append((metrics, scenario_row["_outage_profile"]))

    workers = int(config.get("uncertainty", {}).get("parallel_workers", 1))
    if workers <= 1 or len(scenario_payloads) <= 1:
        rows = _evaluate_lolp_zero_scenarios_serial(
            case,
            config,
            resolved_case_key,
            scenario_payloads,
            float(tolerance_kwh),
        )
    else:
        max_workers = min(workers, len(scenario_payloads))
        tasks = [
            (case, config, resolved_case_key, metrics, outage_profile, float(tolerance_kwh))
            for metrics, outage_profile in scenario_payloads
        ]
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
        except OSError as exc:
            warnings.warn(
                f"Could not start LOLP=0 uncertainty ProcessPoolExecutor ({exc}); falling back to serial execution.",
                RuntimeWarning,
                stacklevel=2,
            )
            rows = _evaluate_lolp_zero_scenarios_serial(
                case,
                config,
                resolved_case_key,
                scenario_payloads,
                float(tolerance_kwh),
            )
        else:
            with executor:
                rows = list(executor.map(_evaluate_lolp_zero_uncertainty_scenario, tasks))

    all_results = _sort_lolp_zero_rows(pd.DataFrame(rows))
    outage_scenarios = _sort_lolp_zero_rows(pd.DataFrame(scenario_records))
    cv_summary = summarize_lolp_zero_uncertainty_results(all_results)
    return all_results, outage_scenarios, cv_summary


def optimize_lolp_zero_design_with_refinement(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    config: dict[str, Any],
    evaluator: CandidateEvaluator | None = None,
    tolerance_kwh: float = LOLP_ZERO_TOLERANCE_KWH,
    feasibility_column: str = "unserved_energy_kwh_per_year",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return evaluated candidates and the minimum-cost candidate satisfying UE=0."""
    if float(tolerance_kwh) < 0.0:
        raise ValueError("LOLP=0 tolerance must be non-negative.")
    search_config = deepcopy(config)
    active_evaluator = evaluator or CandidateEvaluator(search_config, CandidateStore(retain_dispatch=False))
    search_cfg = search_config.get("capacity_search", {})
    expansion_factor = float(search_cfg.get("upper_expansion_factor", 1.5))
    max_upper_expansions = int(search_cfg.get("max_upper_expansions", 2))
    refine_divisions = max(1.0, float(search_cfg.get("local_refine_divisions", 6)))

    pv_candidates = [float(value) for value in search_config["pv"]["capacity_kw_candidates"]]
    battery_candidates = [float(value) for value in search_config["battery"]["energy_kwh_candidates"]]
    history: list[pd.DataFrame] = []
    selected: dict[str, Any] | None = None
    final_coarse_iteration = 0

    for iteration in range(max_upper_expansions + 1):
        search_config["pv"]["capacity_kw_candidates"] = pv_candidates
        search_config["battery"]["energy_kwh_candidates"] = battery_candidates
        coarse = _evaluate_lolp_zero_candidate_grid(
            pv_capacity_frame_builder,
            pv_candidates,
            battery_candidates,
            search_config,
            search_stage="coarse",
            search_iteration=iteration,
            evaluator=active_evaluator,
        )
        history.append(coarse)
        final_coarse_iteration = iteration
        try:
            selected = select_lolp_zero_min_cost_design(
                coarse,
                tolerance_kwh=float(tolerance_kwh),
                feasibility_column=feasibility_column,
            )
        except ValueError:
            selected = None

        if selected is not None:
            expand_pv = _is_upper_boundary(float(selected["pv_capacity_kw"]), pv_candidates)
            expand_battery = _is_upper_boundary(float(selected["battery_energy_kwh"]), battery_candidates)
            if iteration >= max_upper_expansions or not (expand_pv or expand_battery):
                break
        else:
            expand_pv = True
            expand_battery = True
            if iteration >= max_upper_expansions:
                combined = pd.concat(history, ignore_index=True)
                _raise_no_lolp_zero_candidate(combined, float(tolerance_kwh), feasibility_column=feasibility_column)

        if expand_pv:
            pv_candidates = _expand_axis_and_update_refinement(
                search_config,
                "pv",
                pv_candidates,
                expansion_factor,
                refine_divisions,
            )
        if expand_battery:
            battery_candidates = _expand_axis_and_update_refinement(
                search_config,
                "battery",
                battery_candidates,
                expansion_factor,
                refine_divisions,
            )

    if selected is None:
        combined = pd.concat(history, ignore_index=True)
        _raise_no_lolp_zero_candidate(combined, float(tolerance_kwh), feasibility_column=feasibility_column)

    best_pv_capacity = float(selected["pv_capacity_kw"])
    best_battery = float(selected["battery_energy_kwh"])
    refined_history: list[pd.DataFrame] = []
    for iteration in range(2):
        refined_pv_values = _refined_values(
            best_pv_capacity,
            float(search_config["pv"]["local_refine_window_kw"]),
            float(search_config["pv"]["local_refine_step_kw"]),
        )
        refined_battery_values = _refined_values(
            best_battery,
            float(search_config["battery"]["local_refine_window_kwh"]),
            float(search_config["battery"]["local_refine_step_kwh"]),
        )
        refined = _evaluate_lolp_zero_candidate_grid(
            pv_capacity_frame_builder,
            refined_pv_values,
            refined_battery_values,
            search_config,
            search_stage="lolp_zero_region",
            search_iteration=iteration,
            evaluator=active_evaluator,
        )
        refined_history.append(refined)
        refined_best = select_lolp_zero_min_cost_design(
            refined,
            tolerance_kwh=float(tolerance_kwh),
            feasibility_column=feasibility_column,
        )
        best_pv_capacity = float(refined_best["pv_capacity_kw"])
        best_battery = float(refined_best["battery_energy_kwh"])
        if not (
            _is_axis_edge(best_pv_capacity, refined_pv_values)
            or _is_axis_edge(best_battery, refined_battery_values)
        ):
            break

    combined = pd.concat(history + refined_history, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["pv_capacity_kw", "battery_energy_kwh", "search_stage", "search_iteration"],
        keep="last",
    )
    combined = combined.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    final_best = select_lolp_zero_min_cost_design(
        combined,
        tolerance_kwh=float(tolerance_kwh),
        feasibility_column=feasibility_column,
    )
    final_best["coarse_search_iteration"] = int(final_coarse_iteration)
    final_best["optimal_region_iteration"] = int(
        max((float(frame["search_iteration"].max()) for frame in refined_history if not frame.empty), default=0.0)
    )
    return combined, final_best


def select_lolp_zero_min_cost_design(
    candidate_summary: pd.DataFrame,
    tolerance_kwh: float = LOLP_ZERO_TOLERANCE_KWH,
    feasibility_column: str = "unserved_energy_kwh_per_year",
) -> dict[str, Any]:
    """Select the least-cost candidate whose annual unserved energy is effectively zero."""
    if candidate_summary.empty:
        raise ValueError("LOLP=0 design selection received an empty candidate summary.")
    if feasibility_column not in candidate_summary.columns:
        raise ValueError(f"LOLP=0 design selection requires {feasibility_column}.")
    if "pv_bess_system_annual_cost_sgd" not in candidate_summary.columns:
        raise ValueError("LOLP=0 design selection requires pv_bess_system_annual_cost_sgd.")
    feasible = candidate_summary[
        pd.to_numeric(candidate_summary[feasibility_column], errors="coerce")
        <= float(tolerance_kwh)
    ].copy()
    if feasible.empty:
        _raise_no_lolp_zero_candidate(candidate_summary, float(tolerance_kwh), feasibility_column=feasibility_column)
    selected = feasible.sort_values(
        ["pv_bess_system_annual_cost_sgd", "pv_capacity_kw", "battery_energy_kwh"]
    ).iloc[0]
    return selected.to_dict()


def summarize_lolp_zero_uncertainty_results(all_results: pd.DataFrame) -> pd.DataFrame:
    """Return mean, population standard deviation, and CV for the required metrics."""
    rows: list[dict[str, Any]] = []
    if all_results.empty:
        return pd.DataFrame()
    for column, metric, unit in LOLP_ZERO_CV_METRICS:
        if column not in all_results.columns:
            raise ValueError(f"LOLP=0 uncertainty results are missing {column}.")
        values = pd.to_numeric(all_results[column], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if abs(mean) <= 1e-12:
            cv = 0.0 if std <= 1e-12 else np.nan
        else:
            cv = float(std / mean)
        rows.append(
            {
                "metric": metric,
                "source_column": column,
                "unit": unit,
                "count": int(values.count()),
                "mean": mean,
                "std": std,
                "coefficient_of_variation": cv,
                "coefficient_of_variation_percent": cv * 100.0 if np.isfinite(cv) else np.nan,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_lolp_zero_scenarios_serial(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    scenario_payloads: list[tuple[dict[str, Any], pd.DataFrame]],
    tolerance_kwh: float,
) -> list[dict[str, Any]]:
    return [
        _evaluate_lolp_zero_uncertainty_scenario((case, config, case_key, metrics, outage_profile, tolerance_kwh))
        for metrics, outage_profile in scenario_payloads
    ]


def _evaluate_lolp_zero_uncertainty_scenario(args: tuple[Any, ...]) -> dict[str, Any]:
    case, config, case_key, scenario_metrics, outage_profile, tolerance_kwh = args
    worker_config = deepcopy(config)
    worker_metrics = deepcopy(scenario_metrics)
    return _optimize_lolp_zero_design_for_scenario(
        case,
        worker_config,
        case_key,
        outage_profile.copy(),
        worker_metrics,
        float(tolerance_kwh),
    )


def _optimize_lolp_zero_design_for_scenario(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    outage_profile: pd.DataFrame,
    scenario_metrics: dict[str, Any],
    tolerance_kwh: float,
) -> dict[str, Any]:
    aware_cfg = scenario_config(deepcopy(config), case_key)

    def outage_builder(pv_capacity_kw: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, aware_cfg, pv_capacity_kw)
        return frame.merge(outage_profile, on="hour_index", how="left"), pv_meta

    evaluator = CandidateEvaluator(aware_cfg, CandidateStore(retain_dispatch=False))
    _, selected = optimize_lolp_zero_design_with_refinement(
        outage_builder,
        aware_cfg,
        evaluator=evaluator,
        tolerance_kwh=float(tolerance_kwh),
    )
    outage_load = float(scenario_metrics.get("outage_weighted_load_kwh", 0.0))
    ue_kwh = float(selected["unserved_energy_kwh_per_year"])
    scenario_lolp = 0.0 if outage_load <= 0.0 else float(np.clip(ue_kwh / outage_load, 0.0, 1.0))
    result = dict(selected)
    result.update(
        _result_metadata(
            config,
            case_key,
            scenario_metrics,
            optimization_outage_mode="scenario_outage_lolp_zero",
            optimization_voll_sgd_per_kwh=float(aware_cfg["active_scenario"]["voll_sgd_per_kwh"]),
            tolerance_kwh=float(tolerance_kwh),
        )
    )
    result["system_total_cost_sgd_per_year"] = float(result["pv_bess_system_annual_cost_sgd"])
    result["system_total_cost_million_sgd_per_year"] = float(result["pv_bess_system_annual_cost_sgd"]) / 1e6
    result["UE_kwh_per_year"] = ue_kwh
    result["UE_mwh_per_year"] = ue_kwh / 1000.0
    result["scenario_LOLP"] = scenario_lolp
    result["outage_load_served_ratio"] = 1.0 - scenario_lolp
    result["lolp_zero_feasible"] = ue_kwh <= float(tolerance_kwh)
    return result


def _evaluate_lolp_zero_candidate_grid(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    pv_values: list[float],
    battery_values: list[float],
    config: dict[str, Any],
    search_stage: str,
    search_iteration: int,
    evaluator: CandidateEvaluator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pv_capacity in sorted({float(value) for value in pv_values}):
        frame, pv_meta = pv_capacity_frame_builder(float(pv_capacity))
        for battery_energy in sorted({float(value) for value in battery_values}):
            result, _ = evaluator.evaluate(frame, float(battery_energy), pv_meta, require_dispatch=False)
            row = dict(result)
            row.update({"search_stage": search_stage, "search_iteration": int(search_iteration)})
            rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)


def _expand_axis_and_update_refinement(
    config: dict[str, Any],
    axis: str,
    values: list[float],
    expansion_factor: float,
    refine_divisions: float,
) -> list[float]:
    expanded = _expanded_axis([float(value) for value in values], factor=float(expansion_factor))
    if len(expanded) >= 2:
        step = float(expanded[1] - expanded[0])
    else:
        step = 1.0
    if axis == "pv":
        config["pv"]["capacity_kw_candidates"] = expanded
        config["pv"]["local_refine_window_kw"] = step
        config["pv"]["local_refine_step_kw"] = max(step / float(refine_divisions), 1e-6)
    elif axis == "battery":
        config["battery"]["energy_kwh_candidates"] = expanded
        config["battery"]["local_refine_window_kwh"] = step
        config["battery"]["local_refine_step_kwh"] = max(step / float(refine_divisions), 1e-6)
    else:
        raise ValueError(f"Unknown capacity axis: {axis}")
    return expanded


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
        "case_key": case.case_key,
        "city_key": case.city_key,
        "building_key": case.building_key,
        "uncertainty_scenario_id": int(scenario_row["uncertainty_scenario_id"]),
        "outage_seed": int(scenario_row["outage_seed"]),
        "simulated_saidi_min_per_year": float(scenario_row["simulated_saidi_min_per_year"]),
        "simulated_saifi_per_year": float(scenario_row["simulated_saifi_per_year"]),
        "max_event_duration_min": float(scenario_row["max_event_duration_min"]),
        "outage_weight_sum_hours": float(scenario_row["outage_weight_sum_hours"]),
        "simulated_events": float(scenario_row["simulated_events"]),
        "min_grid_availability": float(scenario_row["min_grid_availability"]),
        **diagnostics,
    }


def _result_metadata(
    config: dict[str, Any],
    case_key: str,
    scenario_metrics: dict[str, Any],
    optimization_outage_mode: str,
    optimization_voll_sgd_per_kwh: float,
    tolerance_kwh: float,
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
        "lolp_zero_tolerance_kwh": float(tolerance_kwh),
        "saidi_min_per_year": float(scenario["saidi_min_per_year"]),
        "saifi_per_year": float(scenario["saifi_per_year"]),
        "voll_sgd_per_kwh": float(scenario["voll_sgd_per_kwh"]),
        "battery_capex_sgd_per_kwh": float(config["economics"]["battery_capex_sgd_per_kwh"]),
        "battery_power_capex_sgd_per_kw": float(config["economics"]["battery_power_capex_sgd_per_kw"]),
        "battery_om_fraction_per_year": float(config["economics"]["battery_om_fraction_per_year"]),
        "pv_capex_sgd_per_kw": float(config["pv"]["capex_sgd_per_kw"]),
        **scenario_metrics,
    }


def _sort_lolp_zero_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "uncertainty_scenario_id" not in frame.columns:
        return frame
    return frame.sort_values("uncertainty_scenario_id").reset_index(drop=True)


def _raise_no_lolp_zero_candidate(
    candidate_summary: pd.DataFrame,
    tolerance_kwh: float,
    feasibility_column: str = "unserved_energy_kwh_per_year",
) -> None:
    min_ue = np.nan
    if feasibility_column in candidate_summary.columns and not candidate_summary.empty:
        min_ue = float(pd.to_numeric(candidate_summary[feasibility_column], errors="coerce").min())
    raise ValueError(
        "No LOLP=0 candidate satisfies "
        f"{feasibility_column} <= {float(tolerance_kwh):g}. "
        f"Minimum evaluated unserved energy was {min_ue:g} kWh; "
        "expand the PV/BESS capacity search range."
    )
