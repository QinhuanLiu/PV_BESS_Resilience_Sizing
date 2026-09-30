from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import pandas as pd

from .io.data_loader import build_case_frame
from .optimizer import resilience_design
from .outage import simulate_outages, summarize_outages, summarize_outages_by_year
from .resilience_metrics import electric_equivalent_building_load, outage_weighted_load_kwh


FrameTransform = Callable[[pd.DataFrame], pd.DataFrame]


def _add_outage(
    frame: pd.DataFrame,
    scenario_key: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    scenario = config["scenarios"][scenario_key]
    analysis_years = max(1, int(config.get("analysis", {}).get("years", 1)))
    outage_profile, outage_events = simulate_outages(
        saifi_per_year=float(scenario["saifi_per_year"]),
        saidi_min_per_year=float(scenario["saidi_min_per_year"]),
        hours=len(frame),
        seed=int(config["outage"].get("seed", 42)),
        forced_event_mode=str(config["outage"].get("forced_event_mode", "match_saidi")),
        night_window_start_hour=float(config["outage"].get("night_window_start_hour", 19.0)),
        night_window_end_hour=float(config["outage"].get("night_window_end_hour", 23.0)),
        analysis_years=analysis_years,
        aggregate_horizon=analysis_years > 1,
    )
    merged = frame.merge(outage_profile[["hour_index", "grid_availability", "outage_fraction"]], on="hour_index", how="left")
    return merged, outage_profile, outage_events, summarize_outages(
        outage_profile,
        outage_events,
        analysis_years=analysis_years,
        target_saidi_min_per_year=float(scenario["saidi_min_per_year"]),
        target_saifi_per_year=float(scenario["saifi_per_year"]),
    )


def _outage_weighted_load_total(frame: pd.DataFrame, outage_profile: pd.DataFrame, config: dict[str, Any]) -> float:
    analysis_years = max(1.0, float(config.get("analysis", {}).get("years", 1)))
    technical = config["technical"]
    total_load = electric_equivalent_building_load(
        frame,
        float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0))),
        float(technical.get("heating_cop", 3.5)),
    )
    outage = frame[["hour_index"]].merge(
        outage_profile[["hour_index", "outage_fraction"]],
        on="hour_index",
        how="left",
    )["outage_fraction"].fillna(0.0)
    return float(outage_weighted_load_kwh(total_load, outage).sum()) / analysis_years


def _annual_reliability_summary(
    frame: pd.DataFrame,
    outage_profile: pd.DataFrame,
    dispatch: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    analysis_years = max(1, int(config.get("analysis", {}).get("years", 1)))
    technical = config["technical"]
    total_load = electric_equivalent_building_load(
        frame,
        float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0))),
        float(technical.get("heating_cop", 3.5)),
    )
    merged = frame[["hour_index", "analysis_year"]].copy()
    merged["outage_weighted_load_kwh"] = outage_weighted_load_kwh(
        total_load,
        merged[["hour_index"]].merge(
            outage_profile[["hour_index", "outage_fraction"]],
            on="hour_index",
            how="left",
        )["outage_fraction"].fillna(0.0),
    )
    dispatch_by_hour = dispatch[["hour_index", "unmet_load_kwh_equiv"]].copy()
    dispatch_by_hour["hour_index"] = dispatch_by_hour["hour_index"].astype(int)
    merged = merged.merge(dispatch_by_hour, on="hour_index", how="left")
    merged["unmet_load_kwh_equiv"] = merged["unmet_load_kwh_equiv"].fillna(0.0)
    grouped = merged.groupby("analysis_year", as_index=False).agg(
        outage_weighted_load_kwh=("outage_weighted_load_kwh", "sum"),
        unserved_energy_kwh=("unmet_load_kwh_equiv", "sum"),
    )
    present = set(grouped["analysis_year"].astype(int).tolist())
    missing = [
        {"analysis_year": year, "outage_weighted_load_kwh": 0.0, "unserved_energy_kwh": 0.0}
        for year in range(1, analysis_years + 1)
        if year not in present
    ]
    if missing:
        grouped = pd.concat([grouped, pd.DataFrame(missing)], ignore_index=True)
    grouped = grouped.sort_values("analysis_year").reset_index(drop=True)
    grouped["energy_based_LOLP"] = grouped.apply(
        lambda row: 0.0
        if float(row["outage_weighted_load_kwh"]) <= 0.0
        else float(row["unserved_energy_kwh"]) / float(row["outage_weighted_load_kwh"]),
        axis=1,
    )
    grouped["outage_load_served_ratio"] = 1.0 - grouped["energy_based_LOLP"]
    return grouped


def _annotate_design_rows(
    rows: pd.DataFrame,
    case_key: str,
    config: dict[str, Any],
    outage_summary: dict[str, float],
    outage_weighted_load_total_kwh: float,
) -> pd.DataFrame:
    scenario = config["scenarios"][case_key]
    analysis_years = max(1, int(config.get("analysis", {}).get("years", 1)))
    annotated = rows.copy()
    annotated.insert(0, "building_key", case_key.split("_", 1)[1] if "_" in case_key else "")
    annotated.insert(0, "city_key", case_key.split("_", 1)[0])
    annotated.insert(0, "case_key", case_key)
    annotated["scenario_name"] = scenario["name"]
    annotated["city_or_service_area"] = scenario["city_or_service_area"]
    annotated["saidi_min_per_year"] = scenario["saidi_min_per_year"]
    annotated["saifi_per_year"] = scenario["saifi_per_year"]
    annotated["voll_sgd_per_kwh"] = scenario["voll_sgd_per_kwh"]
    annotated["analysis_years"] = analysis_years
    annotated["battery_capex_sgd_per_kwh"] = config["economics"]["battery_capex_sgd_per_kwh"]
    annotated["battery_power_capex_sgd_per_kw"] = config["economics"]["battery_power_capex_sgd_per_kw"]
    annotated["battery_om_fraction_per_year"] = config["economics"]["battery_om_fraction_per_year"]
    annotated["discount_rate"] = config["economics"].get("discount_rate", 0.0)
    annotated["pv_capex_sgd_per_kw"] = config["pv"]["capex_sgd_per_kw"]
    for key, value in outage_summary.items():
        annotated[key] = value
    annotated["outage_weighted_load_kwh"] = float(outage_weighted_load_total_kwh)
    if "battery_energy_kwh" in annotated.columns:
        annotated["battery_energy_mwh"] = annotated["battery_energy_kwh"].astype(float) / 1000.0
    if "unserved_energy_kwh_per_year" in annotated.columns:
        annotated["UE_kwh_per_year"] = annotated["unserved_energy_kwh_per_year"].astype(float)
        annotated["UE_mwh_per_year"] = annotated["UE_kwh_per_year"] / 1000.0
        annotated["energy_based_LOLP"] = (
            0.0
            if float(outage_weighted_load_total_kwh) <= 0.0
            else annotated["UE_kwh_per_year"] / float(outage_weighted_load_total_kwh)
        )
        annotated["outage_load_served_ratio"] = 1.0 - annotated["energy_based_LOLP"].astype(float)
    if "pv_bess_system_annual_cost_sgd" in annotated.columns:
        annotated["system_total_cost_sgd_per_year"] = annotated["pv_bess_system_annual_cost_sgd"].astype(float)
        annotated["system_total_cost_million_sgd_per_year"] = annotated["system_total_cost_sgd_per_year"] / 1e6
    return annotated


def evaluate_design_case(
    case: Any,
    config: dict[str, Any],
    case_key: str,
    frame_transform: FrameTransform | None = None,
    warm_start_best: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, int, float, float], pd.DataFrame], dict[str, float], pd.DataFrame]:
    """Optimize one configured case and return its model outputs without writing files."""
    run_config = deepcopy(config)
    preview_frame, _ = build_case_frame(case, run_config, max(run_config["pv"]["capacity_kw_candidates"]))
    if frame_transform is not None:
        preview_frame = frame_transform(preview_frame)
    _, outage_profile, outage_events, outage_summary = _add_outage(preview_frame, case_key, run_config)
    outage_weighted_total = _outage_weighted_load_total(preview_frame, outage_profile, run_config)

    def pv_capacity_frame_builder(pv_capacity_kw: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, run_config, pv_capacity_kw)
        if frame_transform is not None:
            frame = frame_transform(frame)
        return frame.merge(
            outage_profile[["hour_index", "grid_availability", "outage_fraction"]],
            on="hour_index",
            how="left",
        ), pv_meta

    rows, dispatch, summary, candidate_dispatches = resilience_design(
        pv_capacity_frame_builder,
        run_config,
        case_key,
        warm_start_best=warm_start_best,
    )
    annotated = _annotate_design_rows(rows, case_key, run_config, outage_summary, outage_weighted_total)
    scenario = run_config["scenarios"][case_key]
    outage_yearly = summarize_outages_by_year(
        outage_profile,
        outage_events,
        max(1, int(run_config.get("analysis", {}).get("years", 1))),
        target_saidi_min_per_year=float(scenario["saidi_min_per_year"]),
        target_saifi_per_year=float(scenario["saifi_per_year"]),
    )
    reliability_yearly = _annual_reliability_summary(preview_frame, outage_profile, dispatch, run_config)
    yearly_summary = outage_yearly.merge(reliability_yearly, on="analysis_year", how="left")
    return annotated, outage_profile, outage_events, dispatch, summary, candidate_dispatches, outage_summary, yearly_summary
