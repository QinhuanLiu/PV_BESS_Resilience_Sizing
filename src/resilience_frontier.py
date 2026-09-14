from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io.data_loader import CaseParameters, InputCase
from .design_selection import (
    aggregate_reliability_by_candidate,
    is_available,
    select_affordable_backup_design,
    select_minimum_total_cost_design,
    select_reliability_constrained_design,
)
from .optimizer import scenario_config
from .outage_evaluation import (
    OutageEvaluationContext,
    evaluate_outage_candidate,
    generate_outage_scenarios,
    rehydrate_outage_scenarios,
)
from .resilience_metrics import (
    DEFAULT_LOLP_THRESHOLD,
)


DEFAULT_FRONTIER_OUTAGE_SCENARIO_COUNT = 100
DEFAULT_FRONTIER_SEED_START = 1000
AFFORDABILITY_TOLERANCE_SGD = 1e-6


def run_resilience_frontier(
    case: InputCase,
    config: dict[str, Any],
    parameters: CaseParameters,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frontier_cfg = _frontier_config(parameters)
    outage_scenarios = generate_outage_scenarios(
        case,
        config,
        scenario_count=int(frontier_cfg["frontier_outage_scenario_count"]),
        seed_start=int(frontier_cfg["frontier_seed_start"]),
        scenario_id_column="frontier_scenario_id",
    )
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


def _evaluate_frontier_candidates(
    case: InputCase,
    config: dict[str, Any],
    frontier_cfg: dict[str, float | int],
    outage_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    aware_cfg = scenario_config(config, case.case_key)
    baseline_by_scenario = _no_pv_no_bess_baseline_by_scenario(case, aware_cfg, outage_scenarios)
    rows: list[dict[str, Any]] = []

    for pv_capacity in [float(value) for value in config["pv"]["capacity_kw_candidates"]]:
        context = OutageEvaluationContext(case, aware_cfg, pv_capacity, retain_dispatch=False)
        for battery_energy in [float(value) for value in config["battery"]["energy_kwh_candidates"]]:
            for scenario_row in outage_scenarios.to_dict("records"):
                evaluation = evaluate_outage_candidate(
                    context,
                    scenario_row["_outage_profile"],
                    battery_energy,
                    require_dispatch=False,
                )
                result = evaluation.result
                baseline = baseline_by_scenario[int(scenario_row["frontier_scenario_id"])]
                baseline_cost = float(baseline["baseline_generalized_cost_sgd_per_year"])
                system_cost = float(result["pv_bess_system_annual_cost_sgd"])
                affordability_margin = baseline_cost - system_cost
                rows.append(
                    {
                        "case_key": case.case_key,
                        "city_key": case.city_key,
                        "building_key": case.building_key,
                        "frontier_scenario_id": int(scenario_row["frontier_scenario_id"]),
                        "outage_seed": int(scenario_row["outage_seed"]),
                        "lolp_threshold": float(frontier_cfg["lolp_threshold"]),
                        "pv_capacity_kw": float(result["pv_capacity_kw"]),
                        "pv_capacity_mw": float(result["pv_capacity_mw"]),
                        "battery_energy_kwh": float(result["battery_energy_kwh"]),
                        "battery_power_kw": float(result["battery_power_kw"]),
                        "battery_c_rate": float(result["battery_c_rate"]),
                        **evaluation.metrics,
                        "annualized_battery_cost_sgd": float(result["annualized_battery_cost_sgd"]),
                        "battery_om_cost_sgd": float(result["battery_om_cost_sgd"]),
                        "pv_total_annual_cost_sgd": float(result["pv_total_annual_cost_sgd"]),
                        "annualized_pv_cost_sgd": float(result["annualized_pv_cost_sgd"]),
                        "pv_om_cost_sgd": float(result["pv_om_cost_sgd"]),
                        "pv_bess_grid_energy_kwh": float(result["pv_bess_grid_energy_kwh"]),
                        "pv_bess_grid_cost_sgd": float(result["pv_bess_grid_cost_sgd"]),
                        "pv_bess_private_annual_cost_sgd": float(result["pv_bess_private_annual_cost_sgd"]),
                        "pv_bess_system_annual_cost_sgd": system_cost,
                        **baseline,
                        "affordability_margin_sgd_per_year": affordability_margin,
                        "cost_ratio_vs_baseline": _cost_ratio(system_cost, baseline_cost),
                        "is_affordable": bool(affordability_margin >= -AFFORDABILITY_TOLERANCE_SGD),
                    }
                )
    return pd.DataFrame(rows)


def _no_pv_no_bess_baseline_by_scenario(
    case: InputCase,
    config: dict[str, Any],
    outage_scenarios: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    context = OutageEvaluationContext(case, config, 0.0, retain_dispatch=False)
    baseline_by_scenario: dict[int, dict[str, float]] = {}
    for scenario_row in outage_scenarios.to_dict("records"):
        scenario_id = int(scenario_row["frontier_scenario_id"])
        evaluation = evaluate_outage_candidate(
            context,
            scenario_row["_outage_profile"],
            0.0,
            require_dispatch=False,
        )
        result = evaluation.result
        baseline_by_scenario[scenario_id] = {
            "baseline_grid_cost_sgd_per_year": float(result["pv_bess_grid_cost_sgd"]),
            "baseline_unserved_energy_kwh_per_year": float(result["unserved_energy_kwh_per_year"]),
            "baseline_unserved_energy_cost_sgd_per_year": float(result["unserved_energy_cost_sgd_per_year"]),
            "baseline_generalized_cost_sgd_per_year": float(result["pv_bess_system_annual_cost_sgd"]),
        }
    return baseline_by_scenario


def _aggregate_candidate_results(
    scenario_results: pd.DataFrame,
    frontier_cfg: dict[str, float | int],
) -> pd.DataFrame:
    aggregated = aggregate_reliability_by_candidate(
        scenario_results,
        group_columns=["case_key", "city_key", "building_key", "pv_capacity_kw", "battery_energy_kwh"],
        lolp_threshold=float(frontier_cfg["lolp_threshold"]),
        first_columns={
            "pv_capacity_mw": "pv_capacity_mw",
            "battery_power_kw": "battery_power_kw",
            "battery_c_rate": "battery_c_rate",
        },
        mean_columns={
            "mean_UE_kwh_per_year": "UE_kwh_per_year",
            "mean_unserved_energy_cost_sgd_per_year": "unserved_energy_cost_sgd_per_year",
            "mean_outage_load_served_ratio": "outage_load_served_ratio",
            "baseline_grid_cost_sgd_per_year": "baseline_grid_cost_sgd_per_year",
            "baseline_unserved_energy_kwh_per_year": "baseline_unserved_energy_kwh_per_year",
            "baseline_unserved_energy_cost_sgd_per_year": "baseline_unserved_energy_cost_sgd_per_year",
            "baseline_generalized_cost_sgd_per_year": "baseline_generalized_cost_sgd_per_year",
            "annualized_battery_cost_sgd_per_year": "annualized_battery_cost_sgd",
            "battery_om_cost_sgd_per_year": "battery_om_cost_sgd",
            "pv_total_annual_cost_sgd_per_year": "pv_total_annual_cost_sgd",
            "annualized_pv_cost_sgd_per_year": "annualized_pv_cost_sgd",
            "pv_om_cost_sgd_per_year": "pv_om_cost_sgd",
            "mean_grid_cost_sgd_per_year": "pv_bess_grid_cost_sgd",
            "mean_private_annual_cost_sgd_per_year": "pv_bess_private_annual_cost_sgd",
            "system_total_cost_sgd_per_year": "pv_bess_system_annual_cost_sgd",
        },
        constant_columns={
            "frontier_outage_scenario_count": int(frontier_cfg["frontier_outage_scenario_count"]),
            "lolp_threshold": float(frontier_cfg["lolp_threshold"]),
        },
        sort_columns=["system_total_cost_sgd_per_year"],
    )
    aggregated = _with_affordability_columns(aggregated)
    column_order = [
        "case_key",
        "city_key",
        "building_key",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_power_kw",
        "battery_c_rate",
        "frontier_outage_scenario_count",
        "lolp_threshold",
        "mean_UE_kwh_per_year",
        "mean_unserved_energy_cost_sgd_per_year",
        "mean_outage_load_served_ratio",
        "mean_scenario_LOLP",
        "chance_constraint_probability",
        "baseline_grid_cost_sgd_per_year",
        "baseline_unserved_energy_kwh_per_year",
        "baseline_unserved_energy_cost_sgd_per_year",
        "baseline_generalized_cost_sgd_per_year",
        "affordability_margin_sgd_per_year",
        "cost_ratio_vs_baseline",
        "is_affordable",
        "annualized_battery_cost_sgd_per_year",
        "battery_om_cost_sgd_per_year",
        "pv_total_annual_cost_sgd_per_year",
        "annualized_pv_cost_sgd_per_year",
        "pv_om_cost_sgd_per_year",
        "mean_grid_cost_sgd_per_year",
        "mean_private_annual_cost_sgd_per_year",
        "system_total_cost_sgd_per_year",
    ]
    return aggregated[[column for column in column_order if column in aggregated.columns]]


def _with_affordability_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    baseline = pd.to_numeric(result["baseline_generalized_cost_sgd_per_year"], errors="coerce")
    system_cost = pd.to_numeric(result["system_total_cost_sgd_per_year"], errors="coerce")
    result["affordability_margin_sgd_per_year"] = baseline - system_cost
    result["cost_ratio_vs_baseline"] = [
        _cost_ratio(system, base)
        for system, base in zip(system_cost.tolist(), baseline.tolist(), strict=True)
    ]
    result["is_affordable"] = result["affordability_margin_sgd_per_year"].astype(float) >= -AFFORDABILITY_TOLERANCE_SGD
    return result


def _cost_ratio(system_cost: float, baseline_cost: float) -> float:
    system = float(system_cost)
    baseline = float(baseline_cost)
    if baseline > 1e-9:
        return system / baseline
    if abs(system) <= AFFORDABILITY_TOLERANCE_SGD:
        return 1.0
    return float("inf")


def _optimal_designs(
    all_candidates: pd.DataFrame,
    frontier_cfg: dict[str, float | int],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            select_minimum_total_cost_design(all_candidates),
            select_affordable_backup_design(all_candidates),
            select_reliability_constrained_design(
                all_candidates,
                lolp_threshold=float(frontier_cfg["lolp_threshold"]),
            ),
        ]
    )


def frontier_outage_scenarios_for_csv(outage_scenarios: pd.DataFrame) -> pd.DataFrame:
    return outage_scenarios.drop(columns=["_outage_profile", "_outage_events"], errors="ignore")


def load_frontier_outputs(
    case: InputCase,
    config: dict[str, Any],
    frontier_dir: Path,
    case_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    optimal_path = frontier_dir / f"{case_key}_Frontier_Optimal_Designs.csv"
    outage_path = frontier_dir / f"{case_key}_Frontier_Outage_Scenarios.csv"
    if not optimal_path.exists() or not outage_path.exists():
        return None
    optimal_designs = pd.read_csv(optimal_path)
    outage_table = pd.read_csv(outage_path)
    required_optimal = {"design_type", "available", "pv_capacity_kw", "battery_energy_kwh"}
    required_outage = {"frontier_scenario_id", "outage_seed"}
    if not required_optimal.issubset(optimal_designs.columns) or not required_outage.issubset(outage_table.columns):
        return None
    outage_scenarios = rehydrate_outage_scenarios(
        case,
        config,
        outage_table,
        scenario_id_column="frontier_scenario_id",
    )
    return optimal_designs, outage_scenarios


def result_design_space_source(all_candidates: pd.DataFrame, optimal_designs: pd.DataFrame) -> pd.DataFrame:
    """Return source data for PV-BESS design-space result figures."""
    source = all_candidates.copy()
    source["battery_energy_mwh"] = source["battery_energy_kwh"].astype(float) / 1000.0
    source["mean_UE_mwh_per_year"] = source["mean_UE_kwh_per_year"].astype(float) / 1000.0
    source["system_total_cost_million_sgd_per_year"] = source["system_total_cost_sgd_per_year"].astype(float) / 1e6
    source["private_cost_million_sgd_per_year"] = source["mean_private_annual_cost_sgd_per_year"].astype(float) / 1e6
    source["energy_based_LOLP"] = source["mean_scenario_LOLP"].astype(float)
    source["is_minimum_total_cost_design"] = False
    source["is_affordable_backup_design"] = False
    source["is_reliability_constrained_design"] = False
    for design_type, column in (
        ("minimum_total_cost_design", "is_minimum_total_cost_design"),
        ("affordable_backup_design", "is_affordable_backup_design"),
        ("reliability_constrained_design", "is_reliability_constrained_design"),
    ):
        matches = optimal_designs[
            (optimal_designs["design_type"].astype(str) == design_type)
            & (optimal_designs.get("available", pd.Series(True, index=optimal_designs.index)).map(is_available))
        ]
        if matches.empty:
            continue
        row = matches.iloc[0]
        mask = (
            np.isclose(source["pv_capacity_kw"].astype(float), float(row["pv_capacity_kw"]))
            & np.isclose(source["battery_energy_kwh"].astype(float), float(row["battery_energy_kwh"]))
        )
        source.loc[mask, column] = True
    return source.sort_values(["pv_capacity_kw", "battery_energy_kwh"]).reset_index(drop=True)


def result_pareto_source(all_candidates: pd.DataFrame, optimal_designs: pd.DataFrame) -> pd.DataFrame:
    """Return cost-resilience Pareto source data using private cost and UE."""
    source = result_design_space_source(all_candidates, optimal_designs)
    source["is_pareto_efficient"] = _pareto_minimize(
        source["mean_private_annual_cost_sgd_per_year"].astype(float),
        source["mean_UE_kwh_per_year"].astype(float),
    )
    return source.sort_values(
        ["is_pareto_efficient", "mean_private_annual_cost_sgd_per_year", "mean_UE_kwh_per_year"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def result_capacity_slice_source(all_candidates: pd.DataFrame, optimal_designs: pd.DataFrame) -> pd.DataFrame:
    """Return low/optimal/high PV capacity slices for BESS-capacity scans."""
    source = result_design_space_source(all_candidates, optimal_designs)
    selected = _selected_slice_pv_values(source, optimal_designs)
    labels = {
        selected[0]: "Low PV",
        selected[len(selected) // 2]: "Optimal-region PV",
        selected[-1]: "High PV",
    }
    sliced = source[source["pv_capacity_kw"].astype(float).round(6).isin([round(value, 6) for value in selected])].copy()
    sliced["pv_slice_label"] = sliced["pv_capacity_kw"].astype(float).map(
        {value: label for value, label in labels.items()}
    )
    return sliced.sort_values(["pv_capacity_kw", "battery_energy_kwh"]).reset_index(drop=True)


def _pareto_minimize(cost: pd.Series, risk: pd.Series) -> pd.Series:
    values = pd.DataFrame({"cost": cost.astype(float), "risk": risk.astype(float)})
    efficient: list[bool] = []
    for row in values.itertuples(index=False):
        dominated = (
            ((values["cost"] <= float(row.cost)) & (values["risk"] <= float(row.risk)))
            & ((values["cost"] < float(row.cost)) | (values["risk"] < float(row.risk)))
        ).any()
        efficient.append(not bool(dominated))
    return pd.Series(efficient, index=cost.index)


def _selected_slice_pv_values(source: pd.DataFrame, optimal_designs: pd.DataFrame) -> list[float]:
    pv_values = sorted(source["pv_capacity_kw"].dropna().astype(float).unique().tolist())
    if len(pv_values) <= 3:
        return pv_values
    available = optimal_designs[
        optimal_designs.get("available", pd.Series(True, index=optimal_designs.index)).map(is_available)
    ]
    if not available.empty and "pv_capacity_kw" in available.columns:
        target = float(available.sort_values("system_total_cost_sgd_per_year").iloc[0]["pv_capacity_kw"])
    else:
        target = float(source.sort_values("system_total_cost_sgd_per_year").iloc[0]["pv_capacity_kw"])
    nearest = min(pv_values, key=lambda value: abs(value - target))
    selected = sorted({pv_values[0], nearest, pv_values[-1]})
    for value in pv_values:
        if len(selected) >= 3:
            break
        selected.append(value)
        selected = sorted(set(selected))
    return selected


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
