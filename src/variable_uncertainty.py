from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .design_selection import select_minimum_total_cost_design, select_reliability_constrained_design
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
from .outage_evaluation import (
    OutageEvaluationContext,
    evaluate_outage_candidate,
    generate_outage_scenarios,
    outage_profile_diagnostics,
)
from .resilience_metrics import reliability_from_ue


DEFAULT_VARIABLE_UNCERTAINTY_SCENARIO_COUNT = 100
VARIABLE_UNCERTAINTY_SUMMARY_METRICS = (
    ("pv_capacity_kw", "pv_capacity", "kW"),
    ("battery_energy_kwh", "battery_energy", "kWh"),
    ("pv_bess_private_annual_cost_sgd", "private_annual_cost", "SGD/year"),
    ("pv_bess_system_annual_cost_sgd", "system_total_cost", "SGD/year"),
    ("unserved_energy_kwh_per_year", "unserved_energy", "kWh/year"),
    ("scenario_LOLP", "scenario_LOLP", "fraction"),
)


def run_variable_uncertainty_analysis(
    case: InputCase,
    config: dict[str, Any],
    case_key: str | None = None,
    scenario_count: int = DEFAULT_VARIABLE_UNCERTAINTY_SCENARIO_COUNT,
    seed_start: int | None = None,
    lolp_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run configurable outage uncertainty analysis for one input case."""
    resolved_case_key = case_key or case.case_key
    if resolved_case_key != case.case_key:
        raise ValueError(
            f"Variable uncertainty case_key {resolved_case_key} does not match input case {case.case_key}."
        )
    if int(scenario_count) <= 0:
        raise ValueError("variable uncertainty scenario count must be positive.")
    if lolp_threshold is not None and float(lolp_threshold) < 0.0:
        raise ValueError("LOLP threshold must be non-negative.")
    if seed_start is None:
        seed_start = int(config.get("uncertainty", {}).get("seed_start", 1000))

    outage_scenarios = generate_outage_scenarios(
        case,
        config,
        scenario_count=int(scenario_count),
        seed_start=int(seed_start),
        scenario_id_column="uncertainty_scenario_id",
    )
    scenario_payloads: list[tuple[dict[str, Any], pd.DataFrame]] = []
    scenario_records: list[dict[str, Any]] = []
    for scenario_row in outage_scenarios.to_dict("records"):
        metrics = _outage_scenario_metrics(case, config, scenario_row)
        scenario_records.append(metrics)
        scenario_payloads.append((metrics, scenario_row["_outage_profile"]))

    workers = int(config.get("uncertainty", {}).get("parallel_workers", 1))
    if workers <= 1 or len(scenario_payloads) <= 1:
        scenario_rows = _evaluate_variable_scenarios_serial(
            case,
            config,
            resolved_case_key,
            scenario_payloads,
            lolp_threshold,
        )
    else:
        max_workers = min(workers, len(scenario_payloads))
        tasks = [
            (case, config, resolved_case_key, metrics, outage_profile, lolp_threshold)
            for metrics, outage_profile in scenario_payloads
        ]
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
        except OSError as exc:
            warnings.warn(
                f"Could not start variable uncertainty ProcessPoolExecutor ({exc}); falling back to serial execution.",
                RuntimeWarning,
                stacklevel=2,
            )
            scenario_rows = _evaluate_variable_scenarios_serial(
                case,
                config,
                resolved_case_key,
                scenario_payloads,
                lolp_threshold,
            )
        else:
            with executor:
                scenario_rows = list(executor.map(_evaluate_variable_scenario, tasks))

    scenario_optima = _sort_variable_rows(pd.DataFrame(scenario_rows))
    outage_table = _sort_variable_rows(pd.DataFrame(scenario_records))
    distribution_summary = summarize_variable_uncertainty_results(scenario_optima)
    candidate_scenario_results = evaluate_variable_uncertainty_candidate_scenarios(
        case,
        config,
        outage_scenarios,
        lolp_threshold=lolp_threshold,
    )
    candidate_summary = aggregate_variable_uncertainty_candidates(
        candidate_scenario_results,
        lolp_threshold=lolp_threshold,
    )
    selected_designs = select_variable_uncertainty_designs(candidate_summary, lolp_threshold=lolp_threshold)
    return (
        scenario_optima,
        outage_table,
        distribution_summary,
        candidate_scenario_results,
        candidate_summary,
        selected_designs,
    )


def optimize_variable_uncertainty_design_with_refinement(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    config: dict[str, Any],
    outage_weighted_load_kwh: float,
    evaluator: CandidateEvaluator | None = None,
    lolp_threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return evaluated candidates and the selected design for one outage scenario."""
    if lolp_threshold is not None and float(lolp_threshold) < 0.0:
        raise ValueError("LOLP threshold must be non-negative.")
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
        coarse = _evaluate_variable_candidate_grid(
            pv_capacity_frame_builder,
            pv_candidates,
            battery_candidates,
            search_config,
            search_stage="coarse",
            search_iteration=iteration,
            evaluator=active_evaluator,
            outage_weighted_load_kwh=float(outage_weighted_load_kwh),
        )
        history.append(coarse)
        final_coarse_iteration = iteration
        selected = _try_select_variable_design(coarse, lolp_threshold=lolp_threshold)

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
                return combined, _unavailable_scenario_design(combined, lolp_threshold)

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
        return combined, _unavailable_scenario_design(combined, lolp_threshold)

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
        refined = _evaluate_variable_candidate_grid(
            pv_capacity_frame_builder,
            refined_pv_values,
            refined_battery_values,
            search_config,
            search_stage="optimal_region",
            search_iteration=iteration,
            evaluator=active_evaluator,
            outage_weighted_load_kwh=float(outage_weighted_load_kwh),
        )
        refined_history.append(refined)
        refined_best = select_variable_uncertainty_scenario_design(refined, lolp_threshold=lolp_threshold)
        best_pv_capacity = float(refined_best["pv_capacity_kw"])
        best_battery = float(refined_best["battery_energy_kwh"])
        selected = refined_best
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
    final_best = select_variable_uncertainty_scenario_design(combined, lolp_threshold=lolp_threshold)
    final_best["coarse_search_iteration"] = int(final_coarse_iteration)
    final_best["optimal_region_iteration"] = int(
        max((float(frame["search_iteration"].max()) for frame in refined_history if not frame.empty), default=0.0)
    )
    final_best["available"] = True
    final_best["unavailable_reason"] = ""
    return combined, final_best


def select_variable_uncertainty_scenario_design(
    candidate_summary: pd.DataFrame,
    lolp_threshold: float | None = None,
) -> dict[str, Any]:
    """Select a scenario-level design, optionally constrained by scenario LOLP."""
    if candidate_summary.empty:
        raise ValueError("Variable uncertainty design selection received an empty candidate summary.")
    for column in ("pv_bess_system_annual_cost_sgd", "pv_capacity_kw", "battery_energy_kwh", "scenario_LOLP"):
        if column not in candidate_summary.columns:
            raise ValueError(f"Variable uncertainty design selection requires {column}.")
    candidates = candidate_summary.copy()
    if lolp_threshold is not None:
        candidates = candidates[candidates["scenario_LOLP"].astype(float) <= float(lolp_threshold)].copy()
        if candidates.empty:
            raise ValueError("No candidate satisfies scenario_LOLP <= lolp_threshold.")
    selected = candidates.sort_values(
        ["pv_bess_system_annual_cost_sgd", "pv_capacity_kw", "battery_energy_kwh"]
    ).iloc[0]
    result = selected.to_dict()
    result["available"] = True
    result["unavailable_reason"] = ""
    return result


def summarize_variable_uncertainty_results(scenario_optima: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scenario_optima.empty:
        return pd.DataFrame()
    available = scenario_optima.copy()
    if "available" in available.columns:
        available = available[available["available"].map(lambda value: False if pd.isna(value) else bool(value))]
    for column, metric, unit in VARIABLE_UNCERTAINTY_SUMMARY_METRICS:
        if column not in available.columns:
            continue
        values = pd.to_numeric(available[column], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        cv = 0.0 if abs(mean) <= 1e-12 and std <= 1e-12 else float(std / mean) if abs(mean) > 1e-12 else np.nan
        rows.append(
            {
                "metric": metric,
                "source_column": column,
                "unit": unit,
                "count": int(values.count()),
                "mean": mean,
                "median": float(values.median()),
                "p10": float(values.quantile(0.10)),
                "p25": float(values.quantile(0.25)),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "p95": float(values.quantile(0.95)),
                "min": float(values.min()),
                "max": float(values.max()),
                "std": std,
                "coefficient_of_variation": cv,
                "coefficient_of_variation_percent": cv * 100.0 if np.isfinite(cv) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_variable_uncertainty_candidate_scenarios(
    case: InputCase,
    config: dict[str, Any],
    outage_scenarios: pd.DataFrame,
    lolp_threshold: float | None = None,
) -> pd.DataFrame:
    aware_cfg = scenario_config(deepcopy(config), case.case_key)
    rows: list[dict[str, Any]] = []
    for pv_capacity in [float(value) for value in aware_cfg["pv"]["capacity_kw_candidates"]]:
        context = OutageEvaluationContext(case, aware_cfg, pv_capacity, retain_dispatch=False)
        for battery_energy in [float(value) for value in aware_cfg["battery"]["energy_kwh_candidates"]]:
            for scenario_row in outage_scenarios.to_dict("records"):
                evaluation = evaluate_outage_candidate(
                    context,
                    scenario_row["_outage_profile"],
                    battery_energy,
                    require_dispatch=False,
                )
                result = evaluation.result
                metrics = _clipped_reliability_metrics(evaluation.metrics)
                rows.append(
                    {
                        "case_key": case.case_key,
                        "city_key": case.city_key,
                        "building_key": case.building_key,
                        "uncertainty_scenario_id": int(scenario_row["uncertainty_scenario_id"]),
                        "outage_seed": int(scenario_row["outage_seed"]),
                        "lolp_threshold": np.nan if lolp_threshold is None else float(lolp_threshold),
                        "pv_capacity_kw": float(result["pv_capacity_kw"]),
                        "pv_capacity_mw": float(result["pv_capacity_mw"]),
                        "battery_energy_kwh": float(result["battery_energy_kwh"]),
                        "battery_power_kw": float(result["battery_power_kw"]),
                        "battery_c_rate": float(result["battery_c_rate"]),
                        **metrics,
                        "unserved_energy_kwh_per_year": float(result["unserved_energy_kwh_per_year"]),
                        "unserved_energy_cost_sgd_per_year": float(result["unserved_energy_cost_sgd_per_year"]),
                        "pv_bess_grid_cost_sgd": float(result["pv_bess_grid_cost_sgd"]),
                        "pv_bess_private_annual_cost_sgd": float(result["pv_bess_private_annual_cost_sgd"]),
                        "pv_bess_system_annual_cost_sgd": float(result["pv_bess_system_annual_cost_sgd"]),
                    }
                )
    return _sort_variable_rows(pd.DataFrame(rows))


def aggregate_variable_uncertainty_candidates(
    candidate_scenario_results: pd.DataFrame,
    lolp_threshold: float | None = None,
) -> pd.DataFrame:
    if candidate_scenario_results.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["case_key", "city_key", "building_key", "pv_capacity_kw", "battery_energy_kwh"]
    for keys, group in candidate_scenario_results.groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        scenario_lolps = group["scenario_LOLP"].astype(float)
        row: dict[str, Any] = {column: value for column, value in zip(group_columns, key_values, strict=True)}
        row.update(
            {
                "pv_capacity_mw": float(group["pv_capacity_mw"].iloc[0]),
                "battery_power_kw": float(group["battery_power_kw"].iloc[0]),
                "battery_c_rate": float(group["battery_c_rate"].iloc[0]),
                "variable_uncertainty_scenario_count": int(len(group)),
                "scenario_count": int(len(group)),
                "lolp_threshold": np.nan if lolp_threshold is None else float(lolp_threshold),
                "mean_UE_kwh_per_year": float(group["UE_kwh_per_year"].astype(float).mean()),
                "mean_unserved_energy_cost_sgd_per_year": float(
                    group["unserved_energy_cost_sgd_per_year"].astype(float).mean()
                ),
                "mean_outage_load_served_ratio": float(group["outage_load_served_ratio"].astype(float).mean()),
                "mean_scenario_LOLP": float(scenario_lolps.mean()),
                "chance_constraint_probability": (
                    np.nan if lolp_threshold is None else float((scenario_lolps <= float(lolp_threshold)).mean())
                ),
                "mean_grid_cost_sgd_per_year": float(group["pv_bess_grid_cost_sgd"].astype(float).mean()),
                "mean_private_annual_cost_sgd_per_year": float(
                    group["pv_bess_private_annual_cost_sgd"].astype(float).mean()
                ),
                "system_total_cost_sgd_per_year": float(group["pv_bess_system_annual_cost_sgd"].astype(float).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("system_total_cost_sgd_per_year").reset_index(drop=True)


def select_variable_uncertainty_designs(
    candidate_summary: pd.DataFrame,
    lolp_threshold: float | None = None,
) -> pd.DataFrame:
    if candidate_summary.empty:
        return pd.DataFrame()
    rows = [select_minimum_total_cost_design(candidate_summary)]
    if lolp_threshold is not None:
        rows.append(select_reliability_constrained_design(candidate_summary, lolp_threshold=float(lolp_threshold)))
    return pd.DataFrame(rows)


def variable_uncertainty_outage_scenarios_for_csv(outage_scenarios: pd.DataFrame) -> pd.DataFrame:
    return outage_scenarios.drop(columns=["_outage_profile", "_outage_events"], errors="ignore")


def _evaluate_variable_scenarios_serial(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    scenario_payloads: list[tuple[dict[str, Any], pd.DataFrame]],
    lolp_threshold: float | None,
) -> list[dict[str, Any]]:
    return [
        _evaluate_variable_scenario((case, config, case_key, metrics, outage_profile, lolp_threshold))
        for metrics, outage_profile in scenario_payloads
    ]


def _evaluate_variable_scenario(args: tuple[Any, ...]) -> dict[str, Any]:
    case, config, case_key, scenario_metrics, outage_profile, lolp_threshold = args
    worker_config = deepcopy(config)
    worker_metrics = deepcopy(scenario_metrics)
    return _optimize_variable_design_for_scenario(
        case,
        worker_config,
        case_key,
        outage_profile.copy(),
        worker_metrics,
        lolp_threshold,
    )


def _optimize_variable_design_for_scenario(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    outage_profile: pd.DataFrame,
    scenario_metrics: dict[str, Any],
    lolp_threshold: float | None,
) -> dict[str, Any]:
    aware_cfg = scenario_config(deepcopy(config), case_key)

    def outage_builder(pv_capacity_kw: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, aware_cfg, pv_capacity_kw)
        return frame.merge(outage_profile, on="hour_index", how="left"), pv_meta

    evaluator = CandidateEvaluator(aware_cfg, CandidateStore(retain_dispatch=False))
    _, selected = optimize_variable_uncertainty_design_with_refinement(
        outage_builder,
        aware_cfg,
        outage_weighted_load_kwh=float(scenario_metrics.get("outage_weighted_load_kwh", 0.0)),
        evaluator=evaluator,
        lolp_threshold=lolp_threshold,
    )
    result = dict(selected)
    result.update(
        _result_metadata(
            aware_cfg,
            case_key,
            scenario_metrics,
            lolp_threshold=lolp_threshold,
        )
    )
    if bool(result.get("available", True)):
        result["system_total_cost_sgd_per_year"] = float(result["pv_bess_system_annual_cost_sgd"])
        result["system_total_cost_million_sgd_per_year"] = float(result["pv_bess_system_annual_cost_sgd"]) / 1e6
        result["UE_kwh_per_year"] = float(result.get("UE_kwh_per_year", result["unserved_energy_kwh_per_year"]))
        result["UE_mwh_per_year"] = float(result["UE_kwh_per_year"]) / 1000.0
    return result


def _evaluate_variable_candidate_grid(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    pv_values: list[float],
    battery_values: list[float],
    config: dict[str, Any],
    search_stage: str,
    search_iteration: int,
    evaluator: CandidateEvaluator,
    outage_weighted_load_kwh: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pv_capacity in sorted({float(value) for value in pv_values}):
        frame, pv_meta = pv_capacity_frame_builder(float(pv_capacity))
        for battery_energy in sorted({float(value) for value in battery_values}):
            result, _ = evaluator.evaluate(frame, float(battery_energy), pv_meta, require_dispatch=False)
            scenario_ue = float(result["unserved_energy_kwh_per_year"])
            metrics = _clipped_reliability_metrics(reliability_from_ue(scenario_ue, float(outage_weighted_load_kwh)))
            row = dict(result)
            row.update(metrics)
            row.update({"search_stage": search_stage, "search_iteration": int(search_iteration)})
            rows.append(row)
    return pd.DataFrame(rows).sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)


def _try_select_variable_design(
    candidate_summary: pd.DataFrame,
    lolp_threshold: float | None,
) -> dict[str, Any] | None:
    try:
        return select_variable_uncertainty_scenario_design(candidate_summary, lolp_threshold=lolp_threshold)
    except ValueError:
        return None


def _unavailable_scenario_design(candidate_summary: pd.DataFrame, lolp_threshold: float | None) -> dict[str, Any]:
    row: dict[str, Any] = {column: pd.NA for column in candidate_summary.columns}
    row.update(
        {
            "available": False,
            "unavailable_reason": "No candidate satisfies scenario_LOLP <= lolp_threshold.",
            "lolp_threshold": np.nan if lolp_threshold is None else float(lolp_threshold),
        }
    )
    if not candidate_summary.empty and "scenario_LOLP" in candidate_summary.columns:
        closest = candidate_summary.sort_values(["scenario_LOLP", "pv_bess_system_annual_cost_sgd"]).iloc[0]
        row["min_evaluated_scenario_LOLP"] = float(closest["scenario_LOLP"])
        row["min_evaluated_UE_kwh_per_year"] = float(closest["UE_kwh_per_year"])
    return row


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


def _clipped_reliability_metrics(metrics: dict[str, float]) -> dict[str, float]:
    result = dict(metrics)
    scenario_lolp = float(np.clip(float(result.get("scenario_LOLP", 0.0)), 0.0, 1.0))
    result["scenario_LOLP"] = scenario_lolp
    result["outage_load_served_ratio"] = 1.0 - scenario_lolp
    return result


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
    lolp_threshold: float | None,
) -> dict[str, Any]:
    scenario = config["scenarios"][case_key]
    city_key, _, building_key = case_key.partition("_")
    return {
        "case_key": case_key,
        "city_key": city_key,
        "building_key": building_key,
        "scenario_name": scenario["name"],
        "city_or_service_area": scenario["city_or_service_area"],
        "optimization_outage_mode": "variable_uncertainty_scenario_outage",
        "optimization_voll_sgd_per_kwh": float(config["active_scenario"]["voll_sgd_per_kwh"]),
        "evaluation_voll_sgd_per_kwh": float(scenario["voll_sgd_per_kwh"]),
        "lolp_threshold": np.nan if lolp_threshold is None else float(lolp_threshold),
        "saidi_min_per_year": float(scenario["saidi_min_per_year"]),
        "saifi_per_year": float(scenario["saifi_per_year"]),
        "voll_sgd_per_kwh": float(scenario["voll_sgd_per_kwh"]),
        "battery_capex_sgd_per_kwh": float(config["economics"]["battery_capex_sgd_per_kwh"]),
        "battery_power_capex_sgd_per_kw": float(config["economics"]["battery_power_capex_sgd_per_kw"]),
        "battery_om_fraction_per_year": float(config["economics"]["battery_om_fraction_per_year"]),
        "pv_capex_sgd_per_kw": float(config["pv"]["capex_sgd_per_kw"]),
        **scenario_metrics,
    }


def _sort_variable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "uncertainty_scenario_id" not in frame.columns:
        return frame
    columns = ["uncertainty_scenario_id"]
    for column in ("pv_capacity_kw", "battery_energy_kwh"):
        if column in frame.columns:
            columns.append(column)
    return frame.sort_values(columns).reset_index(drop=True)
