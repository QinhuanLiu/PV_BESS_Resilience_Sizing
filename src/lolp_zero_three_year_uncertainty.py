from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .costing import pv_annual_cost
from .degradation import annual_battery_soh_loss, annualized_battery_cost
from .dispatch import solve_dispatch
from .io.data_loader import InputCase, build_case_frame
from .lolp_zero_uncertainty import (
    DEFAULT_LOLP_ZERO_SCENARIO_COUNT,
    LOLP_ZERO_TOLERANCE_KWH,
    MANILA_LOLP_ZERO_CASE_KEY,
    _outage_scenario_metrics,
    _result_metadata,
    optimize_lolp_zero_design_with_refinement,
    summarize_lolp_zero_uncertainty_results,
)
from .optimizer import battery_sizing, scenario_config
from .outage_evaluation import generate_outage_scenarios


DEFAULT_LOLP_ZERO_THREE_YEAR_SCENARIO_COUNT = DEFAULT_LOLP_ZERO_SCENARIO_COUNT
DEFAULT_LOLP_ZERO_THREE_YEAR_SIMULATION_YEARS = 3
LOLP_ZERO_THREE_YEAR_FEASIBILITY_COLUMN = "unserved_energy_kwh_total_3yr"


def run_lolp_zero_three_year_uncertainty_analysis(
    case: InputCase,
    config: dict[str, Any],
    case_key: str | None = None,
    scenario_count: int = DEFAULT_LOLP_ZERO_THREE_YEAR_SCENARIO_COUNT,
    simulation_years: int = DEFAULT_LOLP_ZERO_THREE_YEAR_SIMULATION_YEARS,
    seed_start: int | None = None,
    tolerance_kwh: float = LOLP_ZERO_TOLERANCE_KWH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run 30 three-year Manila LOLP=0 sizing paths with greedy dispatch."""
    resolved_case_key = case_key or case.case_key
    if resolved_case_key != case.case_key:
        raise ValueError(
            f"Three-year LOLP=0 uncertainty case_key {resolved_case_key} does not match input case {case.case_key}."
        )
    if int(scenario_count) <= 0:
        raise ValueError("Three-year LOLP=0 uncertainty scenario count must be positive.")
    if int(simulation_years) <= 0:
        raise ValueError("Three-year LOLP=0 simulation_years must be positive.")
    if float(tolerance_kwh) < 0.0:
        raise ValueError("LOLP=0 tolerance must be non-negative.")
    if seed_start is None:
        seed_start = int(config.get("uncertainty", {}).get("seed_start", 1000))

    greedy_config = deepcopy(config)
    greedy_config.setdefault("dispatch", {})["backend"] = "greedy"

    scenario_payloads, scenario_records = build_lolp_zero_three_year_outage_paths(
        case,
        greedy_config,
        scenario_count=int(scenario_count),
        simulation_years=int(simulation_years),
        seed_start=int(seed_start),
    )

    workers = int(greedy_config.get("uncertainty", {}).get("parallel_workers", 1))
    if workers <= 1 or len(scenario_payloads) <= 1:
        rows = _evaluate_lolp_zero_three_year_scenarios_serial(
            case,
            greedy_config,
            resolved_case_key,
            scenario_payloads,
            int(simulation_years),
            float(tolerance_kwh),
        )
    else:
        max_workers = min(workers, len(scenario_payloads))
        tasks = [
            (case, greedy_config, resolved_case_key, metrics, outage_profile, int(simulation_years), float(tolerance_kwh))
            for metrics, outage_profile in scenario_payloads
        ]
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
        except OSError as exc:
            warnings.warn(
                f"Could not start three-year LOLP=0 ProcessPoolExecutor ({exc}); falling back to serial execution.",
                RuntimeWarning,
                stacklevel=2,
            )
            rows = _evaluate_lolp_zero_three_year_scenarios_serial(
                case,
                greedy_config,
                resolved_case_key,
                scenario_payloads,
                int(simulation_years),
                float(tolerance_kwh),
            )
        else:
            with executor:
                rows = list(executor.map(_evaluate_lolp_zero_three_year_uncertainty_scenario, tasks))

    all_results = _sort_three_year_rows(pd.DataFrame(rows))
    outage_scenarios = _sort_three_year_rows(pd.DataFrame(scenario_records))
    cv_summary = summarize_lolp_zero_uncertainty_results(all_results)
    return all_results, outage_scenarios, cv_summary


def lolp_zero_three_year_seed_sequence(seed_start: int, scenario_id: int, simulation_years: int = 3) -> list[int]:
    """Return the deterministic consecutive outage seeds for one multi-year path."""
    if int(scenario_id) <= 0:
        raise ValueError("scenario_id must be positive.")
    if int(simulation_years) <= 0:
        raise ValueError("simulation_years must be positive.")
    first = int(seed_start) + (int(scenario_id) - 1) * int(simulation_years)
    return [first + offset for offset in range(int(simulation_years))]


def build_lolp_zero_three_year_case_frame(
    case: InputCase,
    config: dict[str, Any],
    pv_capacity_kw: float,
    simulation_years: int = DEFAULT_LOLP_ZERO_THREE_YEAR_SIMULATION_YEARS,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Repeat the 8760 Manila typical year into one continuous multi-year dispatch frame."""
    if int(simulation_years) <= 0:
        raise ValueError("simulation_years must be positive.")
    base, pv_meta = build_case_frame(case, config, float(pv_capacity_kw))
    frames: list[pd.DataFrame] = []
    hours_per_year = len(base)
    for year in range(1, int(simulation_years) + 1):
        year_frame = base.copy()
        original_hour = year_frame["hour_index"].astype(int)
        year_frame["typical_year_hour_index"] = original_hour
        year_frame["simulation_year"] = int(year)
        year_frame["hour_index"] = original_hour + (int(year) - 1) * hours_per_year
        if "timestamp" in year_frame.columns:
            year_frame["timestamp"] = pd.to_datetime(year_frame["timestamp"]) + pd.to_timedelta(
                (int(year) - 1) * hours_per_year,
                unit="h",
            )
        frames.append(year_frame)
    combined = pd.concat(frames, ignore_index=True)
    if not combined["hour_index"].is_unique:
        raise ValueError("Three-year case frame generated duplicate hour_index values.")
    pv_meta = dict(pv_meta)
    pv_meta["simulation_years"] = float(simulation_years)
    return combined, pv_meta


def build_lolp_zero_three_year_outage_paths(
    case: InputCase,
    config: dict[str, Any],
    scenario_count: int,
    simulation_years: int,
    seed_start: int,
) -> tuple[list[tuple[dict[str, Any], pd.DataFrame]], list[dict[str, Any]]]:
    """Create 30 multi-year outage paths and their year-level records."""
    scenario_payloads: list[tuple[dict[str, Any], pd.DataFrame]] = []
    scenario_records: list[dict[str, Any]] = []
    for scenario_id in range(1, int(scenario_count) + 1):
        seeds = lolp_zero_three_year_seed_sequence(int(seed_start), scenario_id, int(simulation_years))
        generated = generate_outage_scenarios(
            case,
            config,
            scenario_count=int(simulation_years),
            seed_start=seeds[0],
            scenario_id_column="simulation_year",
        )
        combined_profiles: list[pd.DataFrame] = []
        year_metrics: list[dict[str, Any]] = []
        for generated_row in generated.to_dict("records"):
            year = int(generated_row["simulation_year"])
            row = dict(generated_row)
            row["uncertainty_scenario_id"] = int(scenario_id)
            metrics = _outage_scenario_metrics(case, config, row)
            metrics["simulation_year"] = int(year)
            metrics["simulation_years"] = int(simulation_years)
            metrics["three_year_path_seed_start"] = int(seeds[0])
            metrics["outage_seed_sequence"] = ";".join(str(seed) for seed in seeds)
            year_metrics.append(metrics)
            scenario_records.append(metrics)

            profile = row["_outage_profile"].copy()
            typical_hour = profile["hour_index"].astype(int)
            profile["typical_year_hour_index"] = typical_hour
            profile["simulation_year"] = int(year)
            profile["hour_index"] = typical_hour + (int(year) - 1) * 8760
            combined_profiles.append(profile)

        path_profile = pd.concat(combined_profiles, ignore_index=True)
        path_metrics = _three_year_path_metrics(case, config, int(scenario_id), seeds, year_metrics)
        scenario_payloads.append((path_metrics, path_profile))
    return scenario_payloads, scenario_records


class MeanAnnualMultiyearCandidateEvaluator:
    """Candidate evaluator that uses continuous multi-year dispatch with annualized operating terms."""

    def __init__(self, config: dict[str, Any], simulation_years: int) -> None:
        self.config = config
        self.simulation_years = int(simulation_years)
        self._no_bess_references: dict[tuple[int, float], dict[str, Any]] = {}

    def evaluate(
        self,
        frame: pd.DataFrame,
        battery_energy_kwh: float,
        pv_meta: dict[str, float | str],
        require_dispatch: bool = True,
    ) -> tuple[dict[str, Any], pd.DataFrame | None]:
        reference_key = (id(frame), round(float(pv_meta["pv_capacity_kw"]), 6))
        no_bess_reference = self._no_bess_references.get(reference_key)
        result, dispatch = evaluate_mean_annual_multiyear_candidate(
            frame,
            float(battery_energy_kwh),
            pv_meta,
            self.config,
            simulation_years=self.simulation_years,
            no_bess_reference=no_bess_reference,
        )
        self._no_bess_references[reference_key] = _no_bess_reference_from_multiyear_result(result)
        return dict(result), dispatch.copy() if require_dispatch else None


def evaluate_mean_annual_multiyear_candidate(
    frame: pd.DataFrame,
    battery_energy_kwh: float,
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
    simulation_years: int,
    no_bess_reference: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    sizing = battery_sizing(float(battery_energy_kwh), config)
    dispatch = solve_dispatch(frame, sizing, config)
    pv_bess = pv_bess_mean_annual_multiyear_cost(frame, dispatch, sizing, pv_meta, config, int(simulation_years))
    if no_bess_reference is None:
        if abs(float(battery_energy_kwh)) <= 1e-9:
            no_bess_reference = _no_bess_reference_from_multiyear_cost(pv_bess)
        else:
            zero_sizing = battery_sizing(0.0, config)
            zero_dispatch = solve_dispatch(frame, zero_sizing, config)
            zero_cost = pv_bess_mean_annual_multiyear_cost(
                frame,
                zero_dispatch,
                zero_sizing,
                pv_meta,
                config,
                int(simulation_years),
            )
            no_bess_reference = _no_bess_reference_from_multiyear_cost(zero_cost)
    result = {
        **sizing,
        **no_bess_reference,
        **pv_bess,
        "dispatch_backend": str(dispatch["dispatch_backend"].iloc[0]) if "dispatch_backend" in dispatch.columns else "unknown",
        "solver_status": str(dispatch["solver_status"].iloc[0]) if "solver_status" in dispatch.columns else "",
        "milp_objective_sgd": (
            float(dispatch["milp_objective_sgd"].iloc[0])
            if "milp_objective_sgd" in dispatch.columns and pd.notna(dispatch["milp_objective_sgd"].iloc[0])
            else pd.NA
        ),
        "degradation_penalty_sgd_per_kwh": (
            float(dispatch["degradation_penalty_sgd_per_kwh"].iloc[0])
            if "degradation_penalty_sgd_per_kwh" in dispatch.columns
            else 0.0
        ),
        "simultaneous_charge_discharge_kwh": (
            float(dispatch["simultaneous_charge_discharge_kwh"].sum()) / max(float(simulation_years), 1e-9)
            if "simultaneous_charge_discharge_kwh" in dispatch.columns
            else 0.0
        ),
        "simultaneous_charge_discharge_kwh_total_3yr": (
            float(dispatch["simultaneous_charge_discharge_kwh"].sum())
            if "simultaneous_charge_discharge_kwh" in dispatch.columns
            else 0.0
        ),
        "avoided_unserved_energy_kwh_per_year": (
            no_bess_reference["no_bess_reference_unserved_energy_kwh_per_year"]
            - pv_bess["unserved_energy_kwh_per_year"]
        ),
        "avoided_unserved_energy_kwh_total_3yr": (
            no_bess_reference["no_bess_reference_unserved_energy_kwh_total_3yr"]
            - pv_bess["unserved_energy_kwh_total_3yr"]
        ),
        "system_annual_cost_delta_vs_no_bess_reference_sgd": (
            pv_bess["pv_bess_system_annual_cost_sgd"]
            - no_bess_reference["no_bess_reference_system_annual_cost_sgd"]
        ),
    }
    return result, dispatch


def pv_bess_mean_annual_multiyear_cost(
    frame: pd.DataFrame,
    dispatch: pd.DataFrame,
    sizing: dict[str, float],
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
    simulation_years: int,
) -> dict[str, float]:
    """Return annualized PV/BESS cost with multi-year operating terms averaged per year."""
    years = max(float(simulation_years), 1e-9)
    pv_cost = pv_annual_cost(pv_meta, config)
    degradation = _mean_annual_multiyear_degradation(dispatch, sizing["battery_energy_kwh"], config, years)
    battery_cost = annualized_battery_cost(
        sizing["battery_energy_kwh"],
        sizing["battery_power_kw"],
        degradation["annual_soh_loss"],
        config,
    )
    grid_energy_total = float(dispatch["grid_import_kwh"].sum())
    grid_cost_total = float((dispatch["grid_import_kwh"] * dispatch["tou_sgd_per_kwh"]).sum())
    lost_total = float(dispatch["unmet_load_kwh_equiv"].sum())
    grid_energy_per_year = grid_energy_total / years
    grid_cost_per_year = grid_cost_total / years
    lost_per_year = lost_total / years
    lost_cost_per_year = lost_per_year * float(config["active_scenario"]["voll_sgd_per_kwh"])
    private_cost = (
        grid_cost_per_year
        + pv_cost["pv_total_annual_cost_sgd"]
        + battery_cost["annualized_battery_cost_sgd"]
        + battery_cost["battery_om_cost_sgd"]
    )
    return {
        **pv_cost,
        **degradation,
        **battery_cost,
        "simulation_years": float(simulation_years),
        "pv_bess_grid_energy_kwh": float(grid_energy_per_year),
        "pv_bess_grid_energy_kwh_total_3yr": float(grid_energy_total),
        "pv_bess_grid_cost_sgd": float(grid_cost_per_year),
        "pv_bess_grid_cost_total_3yr_sgd": float(grid_cost_total),
        "pv_bess_private_annual_cost_sgd": float(private_cost),
        "unserved_energy_kwh_per_year": float(lost_per_year),
        "unserved_energy_kwh_total_3yr": float(lost_total),
        "unserved_energy_cost_sgd_per_year": float(lost_cost_per_year),
        "unserved_energy_cost_total_3yr_sgd": float(lost_cost_per_year * years),
        "pv_bess_system_annual_cost_sgd": float(private_cost + lost_cost_per_year),
    }


def _evaluate_lolp_zero_three_year_scenarios_serial(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    scenario_payloads: list[tuple[dict[str, Any], pd.DataFrame]],
    simulation_years: int,
    tolerance_kwh: float,
) -> list[dict[str, Any]]:
    return [
        _evaluate_lolp_zero_three_year_uncertainty_scenario(
            (case, config, case_key, metrics, outage_profile, int(simulation_years), float(tolerance_kwh))
        )
        for metrics, outage_profile in scenario_payloads
    ]


def _evaluate_lolp_zero_three_year_uncertainty_scenario(args: tuple[Any, ...]) -> dict[str, Any]:
    case, config, case_key, scenario_metrics, outage_profile, simulation_years, tolerance_kwh = args
    worker_config = deepcopy(config)
    worker_config.setdefault("dispatch", {})["backend"] = "greedy"
    return _optimize_lolp_zero_three_year_design_for_scenario(
        case,
        worker_config,
        case_key,
        outage_profile.copy(),
        deepcopy(scenario_metrics),
        int(simulation_years),
        float(tolerance_kwh),
    )


def _optimize_lolp_zero_three_year_design_for_scenario(
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    outage_profile: pd.DataFrame,
    scenario_metrics: dict[str, Any],
    simulation_years: int,
    tolerance_kwh: float,
) -> dict[str, Any]:
    aware_cfg = scenario_config(deepcopy(config), case_key)
    aware_cfg.setdefault("dispatch", {})["backend"] = "greedy"

    def outage_builder(pv_capacity_kw: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_lolp_zero_three_year_case_frame(
            case,
            aware_cfg,
            float(pv_capacity_kw),
            simulation_years=int(simulation_years),
        )
        return frame.merge(outage_profile, on="hour_index", how="left"), pv_meta

    evaluator = MeanAnnualMultiyearCandidateEvaluator(aware_cfg, int(simulation_years))
    _, selected = optimize_lolp_zero_design_with_refinement(
        outage_builder,
        aware_cfg,
        evaluator=evaluator,
        tolerance_kwh=float(tolerance_kwh),
        feasibility_column=LOLP_ZERO_THREE_YEAR_FEASIBILITY_COLUMN,
    )
    outage_load = float(scenario_metrics.get("outage_weighted_load_kwh_total_3yr", 0.0))
    ue_total = float(selected["unserved_energy_kwh_total_3yr"])
    scenario_lolp = 0.0 if outage_load <= 0.0 else float(np.clip(ue_total / outage_load, 0.0, 1.0))
    result = dict(selected)
    result.update(
        _result_metadata(
            aware_cfg,
            case_key,
            scenario_metrics,
            optimization_outage_mode="figure2_three_year_scenario_outage_lolp_zero",
            optimization_voll_sgd_per_kwh=float(aware_cfg["active_scenario"]["voll_sgd_per_kwh"]),
            tolerance_kwh=float(tolerance_kwh),
        )
    )
    result["system_total_cost_sgd_per_year"] = float(result["pv_bess_system_annual_cost_sgd"])
    result["system_total_cost_million_sgd_per_year"] = float(result["pv_bess_system_annual_cost_sgd"]) / 1e6
    result["UE_kwh_per_year"] = float(result["unserved_energy_kwh_per_year"])
    result["UE_mwh_per_year"] = float(result["UE_kwh_per_year"]) / 1000.0
    result["UE_kwh_total_3yr"] = ue_total
    result["UE_mwh_total_3yr"] = ue_total / 1000.0
    result["scenario_LOLP"] = scenario_lolp
    result["outage_load_served_ratio"] = 1.0 - scenario_lolp
    result["lolp_zero_feasible"] = ue_total <= float(tolerance_kwh)
    return result


def _mean_annual_multiyear_degradation(
    dispatch: pd.DataFrame,
    capacity_kwh: float,
    config: dict[str, Any],
    simulation_years: float,
) -> dict[str, float]:
    total = annual_battery_soh_loss(dispatch, float(capacity_kwh), config)
    years = max(float(simulation_years), 1e-9)
    annual_equivalent_cycles = float(total["equivalent_full_cycles"]) / years
    annual_cycle_loss = float(total["cycle_aging_loss"]) / years
    calendar_loss = float(total["calendar_aging_loss"])
    return {
        "annual_soh_loss": float(annual_cycle_loss + calendar_loss),
        "cycle_aging_loss": annual_cycle_loss,
        "calendar_aging_loss": calendar_loss,
        "equivalent_full_cycles": annual_equivalent_cycles,
        "equivalent_full_cycles_total_3yr": float(total["equivalent_full_cycles"]),
        "cycle_aging_loss_total_3yr": float(total["cycle_aging_loss"]),
    }


def _no_bess_reference_from_multiyear_cost(pv_bess: dict[str, Any]) -> dict[str, Any]:
    return {
        "no_bess_reference_grid_energy_kwh": float(pv_bess["pv_bess_grid_energy_kwh"]),
        "no_bess_reference_grid_energy_kwh_total_3yr": float(pv_bess["pv_bess_grid_energy_kwh_total_3yr"]),
        "no_bess_reference_grid_cost_sgd": float(pv_bess["pv_bess_grid_cost_sgd"]),
        "no_bess_reference_grid_cost_total_3yr_sgd": float(pv_bess["pv_bess_grid_cost_total_3yr_sgd"]),
        "no_bess_reference_private_annual_cost_sgd": float(pv_bess["pv_bess_private_annual_cost_sgd"]),
        "no_bess_reference_unserved_energy_kwh_per_year": float(pv_bess["unserved_energy_kwh_per_year"]),
        "no_bess_reference_unserved_energy_kwh_total_3yr": float(pv_bess["unserved_energy_kwh_total_3yr"]),
        "no_bess_reference_unserved_energy_cost_sgd_per_year": float(
            pv_bess["unserved_energy_cost_sgd_per_year"]
        ),
        "no_bess_reference_unserved_energy_cost_total_3yr_sgd": float(
            pv_bess["unserved_energy_cost_total_3yr_sgd"]
        ),
        "no_bess_reference_system_annual_cost_sgd": float(pv_bess["pv_bess_system_annual_cost_sgd"]),
    }


def _no_bess_reference_from_multiyear_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "no_bess_reference_grid_energy_kwh": float(result["no_bess_reference_grid_energy_kwh"]),
        "no_bess_reference_grid_energy_kwh_total_3yr": float(
            result["no_bess_reference_grid_energy_kwh_total_3yr"]
        ),
        "no_bess_reference_grid_cost_sgd": float(result["no_bess_reference_grid_cost_sgd"]),
        "no_bess_reference_grid_cost_total_3yr_sgd": float(
            result["no_bess_reference_grid_cost_total_3yr_sgd"]
        ),
        "no_bess_reference_private_annual_cost_sgd": float(
            result["no_bess_reference_private_annual_cost_sgd"]
        ),
        "no_bess_reference_unserved_energy_kwh_per_year": float(
            result["no_bess_reference_unserved_energy_kwh_per_year"]
        ),
        "no_bess_reference_unserved_energy_kwh_total_3yr": float(
            result["no_bess_reference_unserved_energy_kwh_total_3yr"]
        ),
        "no_bess_reference_unserved_energy_cost_sgd_per_year": float(
            result["no_bess_reference_unserved_energy_cost_sgd_per_year"]
        ),
        "no_bess_reference_unserved_energy_cost_total_3yr_sgd": float(
            result["no_bess_reference_unserved_energy_cost_total_3yr_sgd"]
        ),
        "no_bess_reference_system_annual_cost_sgd": float(
            result["no_bess_reference_system_annual_cost_sgd"]
        ),
    }


def _three_year_path_metrics(
    case: InputCase,
    config: dict[str, Any],
    scenario_id: int,
    seeds: list[int],
    year_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    years = max(len(year_metrics), 1)
    scenario = config["scenarios"][case.case_key]

    def _sum(key: str) -> float:
        return float(sum(float(row.get(key, 0.0)) for row in year_metrics))

    def _mean(key: str) -> float:
        return _sum(key) / float(years)

    outage_weighted_load_total = _sum("outage_weighted_load_kwh")
    outage_weighted_pv_total = _sum("outage_weighted_pv_available_kwh")
    return {
        "case_key": case.case_key,
        "city_key": case.city_key,
        "building_key": case.building_key,
        "uncertainty_scenario_id": int(scenario_id),
        "outage_seed": int(seeds[0]),
        "outage_seed_sequence": ";".join(str(seed) for seed in seeds),
        "simulation_years": int(years),
        "three_year_path_seed_start": int(seeds[0]),
        "simulated_saidi_min_per_year": _mean("simulated_saidi_min_per_year"),
        "simulated_saifi_per_year": _mean("simulated_saifi_per_year"),
        "simulated_saidi_min_total_3yr": _sum("simulated_saidi_min_per_year"),
        "simulated_saifi_total_3yr": _sum("simulated_saifi_per_year"),
        "max_event_duration_min": float(max(float(row.get("max_event_duration_min", 0.0)) for row in year_metrics)),
        "outage_weight_sum_hours": _sum("outage_weight_sum_hours"),
        "outage_weight_sum_hours_per_year": _mean("outage_weight_sum_hours"),
        "outage_weight_sum_hours_total_3yr": _sum("outage_weight_sum_hours"),
        "simulated_events": _sum("simulated_events"),
        "simulated_events_per_year": _mean("simulated_events"),
        "min_grid_availability": float(min(float(row.get("min_grid_availability", 1.0)) for row in year_metrics)),
        "outage_hours_peak_tou": _sum("outage_hours_peak_tou"),
        "outage_hours_peak_tou_per_year": _mean("outage_hours_peak_tou"),
        "outage_weighted_load_kwh": outage_weighted_load_total,
        "outage_weighted_load_kwh_per_year": outage_weighted_load_total / float(years),
        "outage_weighted_load_kwh_total_3yr": outage_weighted_load_total,
        "outage_weighted_pv_available_kwh": outage_weighted_pv_total,
        "outage_weighted_pv_available_kwh_per_year": outage_weighted_pv_total / float(years),
        "outage_weighted_pv_available_kwh_total_3yr": outage_weighted_pv_total,
        "reference_pv_capacity_kw": float(max(config["pv"]["capacity_kw_candidates"])),
        "peak_tou_threshold_sgd_per_kwh": float(year_metrics[0].get("peak_tou_threshold_sgd_per_kwh", np.nan)),
        "saidi_min_per_year": float(scenario["saidi_min_per_year"]),
        "saifi_per_year": float(scenario["saifi_per_year"]),
    }


def _sort_three_year_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "uncertainty_scenario_id" not in frame.columns:
        return frame
    by = ["uncertainty_scenario_id"]
    if "simulation_year" in frame.columns:
        by.append("simulation_year")
    return frame.sort_values(by).reset_index(drop=True)
