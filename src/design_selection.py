from __future__ import annotations

from typing import Any

import pandas as pd

from .resilience_metrics import DEFAULT_LOLP_THRESHOLD, chance_constraint_probability, mean_scenario_lolp


def aggregate_reliability_by_candidate(
    scenario_results: pd.DataFrame,
    group_columns: list[str],
    lolp_threshold: float = DEFAULT_LOLP_THRESHOLD,
    first_columns: dict[str, str] | None = None,
    mean_columns: dict[str, str] | None = None,
    constant_columns: dict[str, Any] | None = None,
    scenario_count_column: str | None = None,
    sort_columns: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    first_columns = first_columns or {}
    mean_columns = mean_columns or {}
    constant_columns = constant_columns or {}
    for keys, group in scenario_results.groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = {column: value for column, value in zip(group_columns, key_values, strict=True)}
        row.update(constant_columns)
        if scenario_count_column is not None:
            row[scenario_count_column] = int(len(group))
        for output_column, source_column in first_columns.items():
            row[output_column] = group[source_column].iloc[0]
        for output_column, source_column in mean_columns.items():
            row[output_column] = float(group[source_column].astype(float).mean())
        scenario_lolps = group["scenario_LOLP"].astype(float).tolist()
        row["mean_scenario_LOLP"] = mean_scenario_lolp(scenario_lolps)
        row["chance_constraint_probability"] = chance_constraint_probability(
            scenario_lolps,
            lolp_threshold=float(lolp_threshold),
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if sort_columns and not result.empty:
        result = result.sort_values(sort_columns)
    return result.reset_index(drop=True)


def select_minimum_total_cost_design(
    all_candidates: pd.DataFrame,
    cost_column: str = "system_total_cost_sgd_per_year",
    design_type: str = "minimum_total_cost_design",
) -> dict[str, Any]:
    row = dict(all_candidates.sort_values(cost_column).iloc[0])
    return {"design_type": design_type, "available": True, "unavailable_reason": "", **row}


def select_affordable_backup_design(
    all_candidates: pd.DataFrame,
    affordability_column: str = "is_affordable",
    backup_column: str = "mean_outage_load_served_ratio",
    cost_column: str = "system_total_cost_sgd_per_year",
    pv_column: str = "pv_capacity_kw",
    battery_column: str = "battery_energy_kwh",
    design_type: str = "affordable_backup_design",
    unavailable_reason: str = "No candidate satisfies system_total_cost_sgd_per_year <= baseline_generalized_cost_sgd_per_year.",
) -> dict[str, Any]:
    affordable = all_candidates[_truthy_series(all_candidates[affordability_column])].copy()
    if affordable.empty:
        unavailable = {column: pd.NA for column in all_candidates.columns}
        unavailable.update(
            {
                "design_type": design_type,
                "available": False,
                "unavailable_reason": unavailable_reason,
            }
        )
        return unavailable
    row = dict(
        affordable.sort_values(
            [backup_column, cost_column, pv_column, battery_column],
            ascending=[False, True, True, True],
        ).iloc[0]
    )
    return {"design_type": design_type, "available": True, "unavailable_reason": "", **row}


def select_reliability_constrained_design(
    all_candidates: pd.DataFrame,
    lolp_threshold: float,
    reliability_column: str = "mean_scenario_LOLP",
    cost_column: str = "system_total_cost_sgd_per_year",
    design_type: str = "reliability_constrained_design",
    unavailable_reason: str = "No candidate satisfies mean_scenario_LOLP <= lolp_threshold.",
) -> dict[str, Any]:
    feasible = all_candidates[all_candidates[reliability_column].astype(float) <= float(lolp_threshold)].copy()
    if feasible.empty:
        unavailable = {column: pd.NA for column in all_candidates.columns}
        unavailable.update(
            {
                "design_type": design_type,
                "available": False,
                "unavailable_reason": unavailable_reason,
                "lolp_threshold": float(lolp_threshold),
            }
        )
        return unavailable
    row = dict(feasible.sort_values(cost_column).iloc[0])
    return {"design_type": design_type, "available": True, "unavailable_reason": "", **row}


def available_design_row(
    optimal_designs: pd.DataFrame,
    design_type: str,
    missing_message: str,
    unavailable_message: str,
    required_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    matches = optimal_designs[optimal_designs["design_type"].astype(str) == design_type]
    if matches.empty:
        raise ValueError(missing_message)
    row = matches.iloc[0].to_dict()
    if not is_available(row.get("available", True)):
        reason = row.get("unavailable_reason", "")
        raise ValueError(f"{unavailable_message} {reason}".rstrip())
    for column in required_columns:
        if pd.isna(row.get(column)):
            raise ValueError(f"Requested reference design {design_type} is missing {column}")
    row["design_type"] = design_type
    return row


def is_available(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "available"}
    return bool(value)


def _truthy_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: False if pd.isna(value) else is_available(value))
