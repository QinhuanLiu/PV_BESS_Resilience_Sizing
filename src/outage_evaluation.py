from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .io.data_loader import InputCase, build_case_frame
from .optimizer import CandidateEvaluator, CandidateStore
from .outage import simulate_outages, summarize_outages
from .resilience_metrics import (
    ue,
    electric_equivalent_building_load,
    outage_weighted_load_kwh,
    reliability_from_ue,
)


@dataclass
class OutageCandidateEvaluation:
    result: dict[str, Any]
    frame: pd.DataFrame
    dispatch: pd.DataFrame | None
    metrics: dict[str, float]


class OutageEvaluationContext:
    def __init__(
        self,
        case: InputCase,
        config: dict[str, Any],
        pv_capacity_kw: float,
        retain_dispatch: bool = False,
        evaluator: CandidateEvaluator | None = None,
    ) -> None:
        self.case = case
        self.config = config
        self.pv_capacity_kw = float(pv_capacity_kw)
        self.case_frame, self.pv_meta = build_case_frame(case, config, self.pv_capacity_kw)
        technical = config["technical"]
        cooling_cop = float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0)))
        heating_cop = float(technical.get("heating_cop", 3.5))
        self.total_load_e_equiv_kwh = electric_equivalent_building_load(self.case_frame, cooling_cop, heating_cop)
        scenario = config.get("active_scenario") or config["scenarios"][case.case_key]
        self.voll_sgd_per_kwh = float(scenario["voll_sgd_per_kwh"])
        self.evaluator = evaluator or CandidateEvaluator(config, CandidateStore(retain_dispatch=retain_dispatch))

    def frame_with_outage(self, outage_profile: pd.DataFrame) -> pd.DataFrame:
        return self.case_frame.merge(outage_profile, on="hour_index", how="left")


def generate_outage_scenarios(
    case: InputCase,
    config: dict[str, Any],
    scenario_count: int,
    seed_start: int,
    scenario_id_column: str,
    hours: int = 8760,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenario = config.get("active_scenario") or config["scenarios"][case.case_key]
    for offset in range(int(scenario_count)):
        scenario_id = offset + 1
        seed = int(seed_start) + offset
        profile, events = simulate_outages(
            saifi_per_year=float(scenario["saifi_per_year"]),
            saidi_min_per_year=float(scenario["saidi_min_per_year"]),
            hours=int(hours),
            seed=seed,
            forced_event_mode=str(config.get("outage", {}).get("forced_event_mode", "match_saidi")),
            night_window_start_hour=float(config.get("outage", {}).get("night_window_start_hour", 19.0)),
            night_window_end_hour=float(config.get("outage", {}).get("night_window_end_hour", 23.0)),
        )
        rows.append(
            {
                "case_key": case.case_key,
                "city_key": case.city_key,
                "building_key": case.building_key,
                scenario_id_column: scenario_id,
                "outage_seed": seed,
                **outage_scenario_summary(profile, events),
                "_outage_profile": profile,
                "_outage_events": events,
            }
        )
    return pd.DataFrame(rows)


def rehydrate_outage_scenarios(
    case: InputCase,
    config: dict[str, Any],
    outage_table: pd.DataFrame,
    scenario_id_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenario = config.get("active_scenario") or config["scenarios"][case.case_key]
    for row in outage_table.sort_values(scenario_id_column).to_dict("records"):
        seed = int(row["outage_seed"])
        profile, events = simulate_outages(
            saifi_per_year=float(scenario["saifi_per_year"]),
            saidi_min_per_year=float(scenario["saidi_min_per_year"]),
            hours=8760,
            seed=seed,
            forced_event_mode=str(config.get("outage", {}).get("forced_event_mode", "match_saidi")),
            night_window_start_hour=float(config.get("outage", {}).get("night_window_start_hour", 19.0)),
            night_window_end_hour=float(config.get("outage", {}).get("night_window_end_hour", 23.0)),
        )
        hydrated = dict(row)
        hydrated.update(outage_scenario_summary(profile, events))
        hydrated["_outage_profile"] = profile
        hydrated["_outage_events"] = events
        rows.append(hydrated)
    return pd.DataFrame(rows)


def outage_scenario_summary(outage_profile: pd.DataFrame, outage_events: pd.DataFrame) -> dict[str, float]:
    summary = summarize_outages(outage_profile, outage_events)
    max_duration = 0.0
    if not outage_events.empty and "duration_minutes" in outage_events.columns:
        max_duration = float(outage_events["duration_minutes"].astype(float).max())
    return {
        "max_event_duration_min": max_duration,
        "outage_weight_sum_hours": float(outage_profile["outage_fraction"].astype(float).sum()),
        **summary,
    }


def outage_profile_diagnostics(
    case: InputCase,
    config: dict[str, Any],
    outage_profile: pd.DataFrame,
    reference_pv_capacity_kw: float,
) -> dict[str, float]:
    frame, _ = build_case_frame(case, config, float(reference_pv_capacity_kw))
    merged = frame.merge(outage_profile, on="hour_index", how="left")
    outage_fraction = merged["outage_fraction"].astype(float)
    technical = config["technical"]
    cooling_cop = max(float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0))), 1e-9)
    heating_cop = max(float(technical.get("heating_cop", 3.5)), 1e-9)
    load_e = electric_equivalent_building_load(merged, cooling_cop, heating_cop)
    high_price = float(merged["tou_sgd_per_kwh"].astype(float).quantile(0.75))
    return {
        "outage_hours_peak_tou": float(outage_fraction[merged["tou_sgd_per_kwh"].astype(float) >= high_price].sum()),
        "outage_weighted_load_kwh": float((outage_fraction * load_e).sum()),
        "outage_weighted_pv_available_kwh": float((outage_fraction * merged["pv_kwh"].astype(float)).sum()),
        "reference_pv_capacity_kw": float(reference_pv_capacity_kw),
        "peak_tou_threshold_sgd_per_kwh": high_price,
    }


def evaluate_outage_candidate(
    context: OutageEvaluationContext,
    outage_profile: pd.DataFrame,
    battery_energy_kwh: float,
    require_dispatch: bool = False,
) -> OutageCandidateEvaluation:
    frame = context.frame_with_outage(outage_profile)
    result, dispatch = context.evaluator.evaluate(
        frame,
        float(battery_energy_kwh),
        context.pv_meta,
        require_dispatch=require_dispatch,
    )
    if require_dispatch and dispatch is None:
        raise ValueError("Candidate evaluator did not return dispatch for outage evaluation")
    outage_weighted_load = outage_weighted_load_kwh(context.total_load_e_equiv_kwh, frame["outage_fraction"])
    scenario_ue = ue(dispatch) if dispatch is not None else float(result["unserved_energy_kwh_per_year"])
    metrics = reliability_from_ue(scenario_ue, float(outage_weighted_load.sum()))
    metrics["unserved_energy_cost_sgd_per_year"] = scenario_ue * context.voll_sgd_per_kwh
    return OutageCandidateEvaluation(result=result, frame=frame, dispatch=dispatch, metrics=metrics)
