from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
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
from .visualization import _matplotlib, _save_all, _style, _write_source_data


DEFAULT_RESERVE_MIN_FRACTION = 0.20
DEFAULT_RESERVE_MAX_FRACTION = 0.80
DEFAULT_RESERVE_STEP_FRACTION = 0.10
DEFAULT_RESERVE_REFERENCE_DESIGN = "minimum_total_cost"


def run_reserve_sensitivity(
    case: InputCase,
    config: dict[str, Any],
    parameters: CaseParameters,
    optimal_designs: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reserve_cfg = _reserve_config(parameters, config)
    reference = _select_reference_design(optimal_designs, str(reserve_cfg["reserve_reference_design"]))
    scenario_results = _evaluate_reserve_levels(case, config, reserve_cfg, reference, outage_scenarios)
    all_results = _aggregate_reserve_results(scenario_results, reserve_cfg, reference)
    cost_components = _cost_components(all_results)
    reliability_metrics = _reliability_metrics(all_results)
    marginal_cost = _marginal_cost(all_results)
    return all_results, cost_components, reliability_metrics, marginal_cost


def load_frontier_outputs_for_reserve(
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
    required_optimal = {"design_type", "available", "pv_capacity_kwp", "battery_energy_kwh"}
    required_outage = {"frontier_scenario_id", "outage_seed"}
    if not required_optimal.issubset(optimal_designs.columns) or not required_outage.issubset(outage_table.columns):
        return None
    outage_scenarios = _rehydrate_outage_scenarios(case, config, outage_table)
    return optimal_designs, outage_scenarios


def make_reserve_sensitivity_figures(
    reserve_dir: Path,
    case_key: str,
    cost_components: pd.DataFrame,
    reliability_metrics: pd.DataFrame,
    marginal_cost: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths.update(_figure_13_cost_components(reserve_dir, case_key, cost_components))
    paths.update(_figure_14_reliability_metrics(reserve_dir, case_key, reliability_metrics))
    paths.update(_figure_15_marginal_cost(reserve_dir, case_key, marginal_cost))
    return paths


def _reserve_config(parameters: CaseParameters, config: dict[str, Any]) -> dict[str, float | str]:
    base = parameters.base
    reserve_min = _fraction(base.get("reserve_min_fraction"), DEFAULT_RESERVE_MIN_FRACTION, "reserve_min_fraction")
    reserve_max = _fraction(base.get("reserve_max_fraction"), DEFAULT_RESERVE_MAX_FRACTION, "reserve_max_fraction")
    reserve_step = _positive_fraction(
        base.get("reserve_step_fraction"),
        DEFAULT_RESERVE_STEP_FRACTION,
        "reserve_step_fraction",
    )
    if reserve_max < reserve_min:
        raise ValueError("reserve_max_fraction must be greater than or equal to reserve_min_fraction")
    soc_max = float(config["battery"]["soc_max_fraction"])
    if reserve_max > soc_max:
        raise ValueError("reserve_fraction must not exceed battery_soc_max_fraction")
    return {
        "reserve_min_fraction": reserve_min,
        "reserve_max_fraction": reserve_max,
        "reserve_step_fraction": reserve_step,
        "reserve_reference_design": str(base.get("reserve_reference_design") or DEFAULT_RESERVE_REFERENCE_DESIGN),
        "lolp_threshold": _fraction(base.get("lolp_threshold"), DEFAULT_LOLP_THRESHOLD, "lolp_threshold"),
    }


def _evaluate_reserve_levels(
    case: InputCase,
    config: dict[str, Any],
    reserve_cfg: dict[str, float | str],
    reference: dict[str, Any],
    outage_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    if outage_scenarios.empty:
        raise ValueError("Reserve sensitivity requires at least one outage scenario")
    if "_outage_profile" not in outage_scenarios.columns:
        raise ValueError("Reserve sensitivity requires outage_scenarios with in-memory _outage_profile objects")

    aware_cfg = scenario_config(config, case.case_key, resilience_informed=True)
    technical = aware_cfg["technical"]
    cooling_cop = float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0)))
    heating_cop = float(technical.get("heating_cop", 3.5))
    voll = float(aware_cfg["active_scenario"]["voll_sgd_per_kwh"])
    pv_capacity = float(reference["pv_capacity_kwp"])
    battery_energy = float(reference["battery_energy_kwh"])
    case_frame, pv_meta = build_case_frame(case, aware_cfg, pv_capacity)
    total_load = electric_equivalent_building_load(case_frame, cooling_cop, heating_cop)
    rows: list[dict[str, Any]] = []

    for reserve_fraction in _reserve_values(reserve_cfg):
        reserve_cfg_for_dispatch = deepcopy(aware_cfg)
        reserve_cfg_for_dispatch["battery"] = deepcopy(aware_cfg["battery"])
        reserve_cfg_for_dispatch["battery"]["normal_operation_reserve_fraction"] = reserve_fraction
        for scenario_row in outage_scenarios.to_dict("records"):
            outage_profile = scenario_row["_outage_profile"]
            frame = case_frame.merge(outage_profile, on="hour_index", how="left")
            result, dispatch = evaluate_candidate(frame, battery_energy, pv_meta, reserve_cfg_for_dispatch)
            outage_weighted_load = outage_weighted_load_kwh(total_load, frame["outage_fraction"])
            scenario_eue = eue(dispatch)
            scenario_unserved_cost = scenario_eue * voll
            rows.append(
                {
                    "case_key": case.case_key,
                    "city_key": case.city_key,
                    "building_key": case.building_key,
                    "reserve_reference_design": reference["design_type"],
                    "reference_pv_capacity_kwp": pv_capacity,
                    "reference_battery_energy_kwh": battery_energy,
                    "reference_battery_power_kw": float(result["battery_power_kw"]),
                    "reserve_fraction": reserve_fraction,
                    "reserve_percent": reserve_fraction * 100.0,
                    "frontier_scenario_id": int(scenario_row["frontier_scenario_id"]),
                    "outage_seed": int(scenario_row["outage_seed"]),
                    "lolp_threshold": float(reserve_cfg["lolp_threshold"]),
                    "normal_operation_bill_cost_sgd_per_year": float(result["pv_bess_grid_cost_sgd"]),
                    "outage_unserved_energy_cost_sgd_per_year": scenario_unserved_cost,
                    "net_cost_sgd_per_year": float(result["pv_bess_grid_cost_sgd"]) + scenario_unserved_cost,
                    "EUE_kwh_per_year": scenario_eue,
                    "outage_load_served_ratio": outage_load_served_ratio(dispatch, outage_weighted_load),
                    "scenario_LOLP": scenario_lolp(dispatch, frame["outage_fraction"]),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_reserve_results(
    scenario_results: pd.DataFrame,
    reserve_cfg: dict[str, float | str],
    reference: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for reserve_fraction, group in scenario_results.groupby("reserve_fraction", sort=True):
        scenario_lolps = group["scenario_LOLP"].astype(float).tolist()
        rows.append(
            {
                "case_key": group["case_key"].iloc[0],
                "city_key": group["city_key"].iloc[0],
                "building_key": group["building_key"].iloc[0],
                "reserve_reference_design": reference["design_type"],
                "reference_pv_capacity_kwp": float(reference["pv_capacity_kwp"]),
                "reference_pv_capacity_mwp": float(reference["pv_capacity_kwp"]) / 1000.0,
                "reference_battery_energy_kwh": float(reference["battery_energy_kwh"]),
                "reference_battery_energy_mwh": float(reference["battery_energy_kwh"]) / 1000.0,
                "reference_battery_power_kw": float(group["reference_battery_power_kw"].iloc[0]),
                "reserve_fraction": float(reserve_fraction),
                "reserve_percent": float(reserve_fraction) * 100.0,
                "reserve_outage_scenario_count": int(len(group)),
                "lolp_threshold": float(reserve_cfg["lolp_threshold"]),
                "normal_operation_bill_cost_sgd_per_year": float(
                    group["normal_operation_bill_cost_sgd_per_year"].astype(float).mean()
                ),
                "outage_unserved_energy_cost_sgd_per_year": float(
                    group["outage_unserved_energy_cost_sgd_per_year"].astype(float).mean()
                ),
                "net_cost_sgd_per_year": float(group["net_cost_sgd_per_year"].astype(float).mean()),
                "EUE_kwh_per_year": float(group["EUE_kwh_per_year"].astype(float).mean()),
                "outage_load_served_ratio": float(group["outage_load_served_ratio"].astype(float).mean()),
                "achieved_LOLP": achieved_lolp(scenario_lolps),
                "chance_constraint_probability": chance_constraint_probability(
                    scenario_lolps,
                    lolp_threshold=float(reserve_cfg["lolp_threshold"]),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("reserve_fraction").reset_index(drop=True)


def _cost_components(all_results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_key",
        "city_key",
        "building_key",
        "reserve_reference_design",
        "reference_pv_capacity_kwp",
        "reference_battery_energy_kwh",
        "reserve_fraction",
        "reserve_percent",
        "normal_operation_bill_cost_sgd_per_year",
        "outage_unserved_energy_cost_sgd_per_year",
        "net_cost_sgd_per_year",
    ]
    return all_results[columns].copy()


def _reliability_metrics(all_results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_key",
        "city_key",
        "building_key",
        "reserve_reference_design",
        "reference_pv_capacity_kwp",
        "reference_battery_energy_kwh",
        "reserve_fraction",
        "reserve_percent",
        "EUE_kwh_per_year",
        "outage_load_served_ratio",
        "achieved_LOLP",
        "chance_constraint_probability",
        "lolp_threshold",
    ]
    return all_results[columns].copy()


def _marginal_cost(all_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = all_results.sort_values("reserve_fraction").reset_index(drop=True)
    for idx in range(1, len(ordered)):
        previous = ordered.iloc[idx - 1]
        current = ordered.iloc[idx]
        start = float(previous["reserve_fraction"])
        end = float(current["reserve_fraction"])
        rows.append(
            {
                "case_key": current["case_key"],
                "city_key": current["city_key"],
                "building_key": current["building_key"],
                "reserve_reference_design": current["reserve_reference_design"],
                "reference_pv_capacity_kwp": float(current["reference_pv_capacity_kwp"]),
                "reference_battery_energy_kwh": float(current["reference_battery_energy_kwh"]),
                "reserve_start_fraction": start,
                "reserve_end_fraction": end,
                "reserve_interval": f"{start:.0%} to {end:.0%}",
                "reserve_step_fraction": end - start,
                "previous_net_cost_sgd_per_year": float(previous["net_cost_sgd_per_year"]),
                "current_net_cost_sgd_per_year": float(current["net_cost_sgd_per_year"]),
                "marginal_net_cost_change_sgd_per_year": float(current["net_cost_sgd_per_year"])
                - float(previous["net_cost_sgd_per_year"]),
            }
        )
    return pd.DataFrame(rows)


def _select_reference_design(optimal_designs: pd.DataFrame, requested: str) -> dict[str, Any]:
    normalized = requested.strip().lower().replace("-", "_")
    if normalized in {"minimum", "minimum_total_cost", "minimum_total_cost_design"}:
        design_type = "minimum_total_cost_design"
    elif normalized in {"reliability", "reliability_constrained", "reliability_constrained_design"}:
        design_type = "reliability_constrained_design"
    else:
        raise ValueError(
            "reserve_reference_design must be minimum_total_cost or reliability_constrained"
        )
    matches = optimal_designs[optimal_designs["design_type"].astype(str) == design_type]
    if matches.empty:
        raise ValueError(f"Frontier optimal designs do not include {design_type}")
    row = matches.iloc[0].to_dict()
    if not _available(row.get("available", True)):
        reason = row.get("unavailable_reason", "")
        raise ValueError(f"Requested reserve reference design {design_type} is unavailable. {reason}")
    for column in ("pv_capacity_kwp", "battery_energy_kwh"):
        if pd.isna(row.get(column)):
            raise ValueError(f"Requested reserve reference design {design_type} is missing {column}")
    row["design_type"] = design_type
    return row


def _rehydrate_outage_scenarios(case: InputCase, config: dict[str, Any], outage_table: pd.DataFrame) -> pd.DataFrame:
    scenario = config["scenarios"][case.case_key]
    rows: list[dict[str, Any]] = []
    for row in outage_table.sort_values("frontier_scenario_id").to_dict("records"):
        seed = int(row["outage_seed"])
        profile, events = simulate_outages(
            saifi_per_year=float(scenario["saifi_per_year"]),
            saidi_min_per_year=float(scenario["saidi_min_per_year"]),
            hours=8760,
            seed=seed,
            forced_event_mode=str(config.get("outage", {}).get("forced_event_mode", "match_saidi")),
        )
        summary = summarize_outages(profile, events)
        hydrated = dict(row)
        hydrated.update(summary)
        hydrated["_outage_profile"] = profile
        hydrated["_outage_events"] = events
        rows.append(hydrated)
    return pd.DataFrame(rows)


def _reserve_values(reserve_cfg: dict[str, float | str]) -> list[float]:
    start = float(reserve_cfg["reserve_min_fraction"])
    end = float(reserve_cfg["reserve_max_fraction"])
    step = float(reserve_cfg["reserve_step_fraction"])
    values: list[float] = []
    value = start
    while value <= end + 1e-10:
        values.append(round(value, 10))
        value += step
    if not np.isclose(values[-1], end):
        values.append(round(end, 10))
    return values


def _figure_13_cost_components(out: Path, case_key: str, cost_components: pd.DataFrame) -> dict[str, Path]:
    source = cost_components.copy()
    source["normal_operation_bill_cost_million_sgd_per_year"] = source["normal_operation_bill_cost_sgd_per_year"] / 1e6
    source["outage_unserved_energy_cost_million_sgd_per_year"] = source["outage_unserved_energy_cost_sgd_per_year"] / 1e6
    source["net_cost_million_sgd_per_year"] = source["net_cost_sgd_per_year"] / 1e6
    base = out / f"{case_key}_Figure_13_Reserve_vs_Cost_Components"
    _write_source_data(base, source)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, ax = plt.subplots(figsize=(140 / 25.4, 84 / 25.4), constrained_layout=True)
    fig.suptitle("Reserve level and cost components", y=1.05, fontsize=8.5, fontweight="bold")
    ax.plot(source["reserve_percent"], source["normal_operation_bill_cost_million_sgd_per_year"], marker="o", label="Bill cost", color="#0f4d92")
    ax.plot(source["reserve_percent"], source["outage_unserved_energy_cost_million_sgd_per_year"], marker="o", label="Outage unserved cost", color="#b64342")
    ax.plot(source["reserve_percent"], source["net_cost_million_sgd_per_year"], marker="o", label="Net cost", color="#0f766e")
    ax.set_xlabel("Reserve level (%)")
    ax.set_ylabel("Annual cost (million SGD/year)")
    _style_reserve_axes([ax])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _figure_14_reliability_metrics(out: Path, case_key: str, reliability_metrics: pd.DataFrame) -> dict[str, Path]:
    source = reliability_metrics.copy()
    source["EUE_mwh_per_year"] = source["EUE_kwh_per_year"] / 1000.0
    base = out / f"{case_key}_Figure_14_Reserve_vs_Reliability_Metrics"
    _write_source_data(base, source)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, axes = plt.subplots(1, 3, figsize=(174 / 25.4, 74 / 25.4), constrained_layout=True)
    fig.suptitle("Reserve level and reliability metrics", y=1.08, fontsize=8.5, fontweight="bold")
    axes[0].plot(source["reserve_percent"], source["EUE_mwh_per_year"], marker="o", color="#0f4d92")
    axes[0].set_ylabel("EUE (MWh/year)")
    axes[1].plot(source["reserve_percent"], source["achieved_LOLP"], marker="o", color="#b64342")
    if "lolp_threshold" in source.columns:
        axes[1].axhline(float(source["lolp_threshold"].dropna().iloc[0]), color="#767676", linestyle="--", linewidth=0.85)
    axes[1].set_ylabel("Achieved LOLP")
    axes[2].plot(source["reserve_percent"], source["chance_constraint_probability"], marker="o", color="#0f766e")
    axes[2].set_ylabel("Chance probability")
    for ax, title in zip(axes, ["a  Expected unserved energy", "b  Achieved LOLP", "c  Chance constraint"], strict=True):
        ax.set_xlabel("Reserve level (%)")
        ax.set_title(title, pad=12)
    _style_reserve_axes(axes)
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _figure_15_marginal_cost(out: Path, case_key: str, marginal_cost: pd.DataFrame) -> dict[str, Path]:
    source = marginal_cost.copy()
    base = out / f"{case_key}_Figure_15_Reserve_Marginal_Cost"
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, ax = plt.subplots(figsize=(136 / 25.4, 82 / 25.4), constrained_layout=True)
    fig.suptitle("Marginal net cost of reserve", y=1.05, fontsize=8.5, fontweight="bold")
    if source.empty:
        _write_source_data(base, source)
        ax.text(0.5, 0.5, "No reserve intervals", ha="center", va="center", transform=ax.transAxes)
    else:
        source["marginal_net_cost_change_million_sgd_per_year"] = source["marginal_net_cost_change_sgd_per_year"] / 1e6
        _write_source_data(base, source)
        ax.bar(source["reserve_interval"], source["marginal_net_cost_change_million_sgd_per_year"], color="#0f4d92", width=0.72)
    ax.set_xlabel("Reserve interval")
    ax.set_ylabel("Marginal net cost change (million SGD/year)")
    ax.tick_params(axis="x", rotation=35)
    _style_reserve_axes([ax])
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _style_reserve_axes(axes: Any) -> None:
    for ax in list(np.ravel(axes)):
        ax.grid(axis="both", color="#e5e7eb", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.margins(x=0.05, y=0.12)


def _available(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _fraction(value: Any, default: float, name: str) -> float:
    result = float(default if value is None else value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_fraction(value: Any, default: float, name: str) -> float:
    result = float(default if value is None else value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result
