from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import math

import numpy as np
import pandas as pd

from ..io.config import load_config
from ..io.data_loader import (
    CaseParameters,
    InputCase,
    build_case_frame,
    case_config_from_parameters,
    discover_input_cases,
    load_case_parameters,
    read_load_csv,
    read_pv_csv,
    read_tou_csv,
)
from ..optimizer import CandidateEvaluator, CandidateStore, resilience_design, scenario_config
from ..outage import simulate_outages
from ..outage_evaluation import generate_outage_scenarios
from ..resilience_metrics import electric_equivalent_building_load, outage_weighted_load_kwh
from ..uncertainty import run_uncertainty_analysis
from .core import NATURE_PALETTE, _matplotlib, _save_all, _style, _write_source_data
from .outage_balance import (
    WINDOW_HOURS as OUTAGE_BALANCE_WINDOW_HOURS,
    _draw_day_boundaries as _draw_outage_balance_day_boundaries,
    _palette as _outage_balance_palette,
    _panel_label as _outage_balance_panel_label,
    _plot_outage_bounds as _plot_outage_balance_bounds,
    _shade_outage as _shade_outage_balance,
    _soc_ylim_for_zero_alignment,
    _stack_negative_balance,
    _stack_positive_balance,
)


APARTMENT_CITY_KEY = "SG_Apartment"
APARTMENT_CASE_KEY = "SG_Apartment_Apartment"
DATA_STATUS = "synthetic_proxy"
REAL_SINGLEFAMILY_DATA_STATUS = "real_singlefamily_simulated"
DEFAULT_SCENARIO_COUNT = 100
DEFAULT_SYNTHETIC_SEED = 20260805
DEFAULT_ANALYSIS_VOLL_SGD_PER_KWH = 150.0
FIGURE_NUMBERS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
FIGURE_7B_ID = "7b"
FIGURE_7B_NAME = "Figure_7b_City_Scenario_Distribution_No_LOLP"
FIGURE_7B_BATTERY_ADJUSTMENTS_KWH = {
    "Metro_Manila": -30.0,
    "Brisbane": 10.0,
}
SUPPORTED_FIGURE_NUMBERS = (*FIGURE_NUMBERS, FIGURE_7B_ID)
FIGURE_NAMES = {
    1: "Figure_1_City_Comparison_Framework",
    2: "Figure_2_City_Specific_Input_Conditions",
    3: "Figure_3_Outage_Scenario_Generation_Validation",
    4: "Figure_4_Cross_City_Optimal_Sizing_Performance",
    5: "Figure_5_Representative_Outage_Operation",
    6: "Figure_6_Sensitivity_Design_Implications",
    7: "Figure_7_City_Scenario_Distribution",
    8: "Figure_8_Four_City_Outage_Energy_Balance",
    9: "Figure_9_Optimal_Design_Cost_Breakdown",
}


@dataclass(frozen=True)
class CityComparisonSpec:
    """Synthetic city descriptor for the Apartment comparison module."""

    city_key: str
    city_label: str
    role: str
    latitude: float
    longitude: float
    pv_yield_multiplier: float
    load_energy_multiplier: float
    tariff_multiplier: float
    tariff_spread_multiplier: float
    saidi_min_per_year: float
    saifi_per_year: float
    local_voll_proxy_sgd_per_kwh: float
    outage_tail_factor: float
    pv_peak_month: int
    load_peak_month: int
    load_seasonality: float = 0.05
    pv_seasonality: float = 0.08
    load_diurnal_strength: float = 0.0
    load_diurnal_peak_hour: int = 17


@dataclass(frozen=True)
class SingleFamilyDesignRun:
    best: pd.DataFrame
    dispatch: pd.DataFrame
    summary: pd.DataFrame
    outage_profile: pd.DataFrame
    outage_events: pd.DataFrame
    candidate_dispatches: dict[tuple[str, int, float, float], pd.DataFrame]


DEFAULT_CITY_SPECS = (
    CityComparisonSpec(
        city_key="Singapore",
        city_label="Singapore",
        role="High-reliability baseline",
        latitude=1.3521,
        longitude=103.8198,
        pv_yield_multiplier=1.00,
        load_energy_multiplier=1.00,
        tariff_multiplier=1.00,
        tariff_spread_multiplier=1.00,
        saidi_min_per_year=0.26,
        saifi_per_year=0.006,
        local_voll_proxy_sgd_per_kwh=5.0,
        outage_tail_factor=1.00,
        pv_peak_month=3,
        load_peak_month=5,
        load_seasonality=0.03,
        pv_seasonality=0.04,
        load_diurnal_strength=0.00,
        load_diurnal_peak_hour=17,
    ),
    CityComparisonSpec(
        city_key="Jakarta",
        city_label="Jakarta",
        role="Hot-humid medium-reliability case",
        latitude=-6.2088,
        longitude=106.8456,
        pv_yield_multiplier=1.06,
        load_energy_multiplier=1.08,
        tariff_multiplier=0.78,
        tariff_spread_multiplier=0.72,
        saidi_min_per_year=25.80,
        saifi_per_year=0.65,
        local_voll_proxy_sgd_per_kwh=60.0,
        outage_tail_factor=1.25,
        pv_peak_month=8,
        load_peak_month=10,
        load_seasonality=0.05,
        pv_seasonality=0.07,
        load_diurnal_strength=0.04,
        load_diurnal_peak_hour=15,
    ),
    CityComparisonSpec(
        city_key="Metro_Manila",
        city_label="Metro Manila",
        role="Long-tail outage case",
        latitude=14.5995,
        longitude=120.9842,
        pv_yield_multiplier=0.95,
        load_energy_multiplier=1.12,
        tariff_multiplier=0.92,
        tariff_spread_multiplier=0.85,
        saidi_min_per_year=108.21,
        saifi_per_year=1.04,
        local_voll_proxy_sgd_per_kwh=70.0,
        outage_tail_factor=3.00,
        pv_peak_month=4,
        load_peak_month=5,
        load_seasonality=0.07,
        pv_seasonality=0.12,
        load_diurnal_strength=0.07,
        load_diurnal_peak_hour=20,
    ),
    CityComparisonSpec(
        city_key="Brisbane",
        city_label="Brisbane",
        role="Subtropical VoLL/VCR benchmark",
        latitude=-27.4698,
        longitude=153.0251,
        pv_yield_multiplier=1.18,
        load_energy_multiplier=0.86,
        tariff_multiplier=1.15,
        tariff_spread_multiplier=1.30,
        saidi_min_per_year=71.36,
        saifi_per_year=0.647,
        local_voll_proxy_sgd_per_kwh=36.7,
        outage_tail_factor=1.45,
        pv_peak_month=1,
        load_peak_month=1,
        load_seasonality=0.18,
        pv_seasonality=0.18,
        load_diurnal_strength=0.10,
        load_diurnal_peak_hour=17,
    ),
)
SINGLEFAMILY_CASE_BY_CITY = {
    "Singapore": "SG_SingleFamily",
    "Metro_Manila": "Manila_SingleFamily",
    "Brisbane": "Brisbane_SingleFamily",
    "Jakarta": "Jakarta_SingleFamily",
}
SINGLEFAMILY_CITY_ORDER = ("Singapore", "Metro_Manila", "Brisbane", "Jakarta")
SINGLEFAMILY_CASE_KEY = "SingleFamily"
DEFAULT_SENSITIVITY_VOLL_VALUES = (0.0, 50.0, 100.0, 150.0, 200.0, 300.0)
DEFAULT_SENSITIVITY_BATTERY_CAPEX_VALUES = (200.0, 400.0, 600.0, 800.0, 1000.0)
DEFAULT_SENSITIVITY_SAIDI_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


CITY_COLORS = {
    "Singapore": NATURE_PALETTE["blue"],
    "Jakarta": NATURE_PALETTE["teal"],
    "Metro_Manila": NATURE_PALETTE["red"],
    "Brisbane": NATURE_PALETTE["gold"],
}
DESIGN_COLORS = {
    "economic_only": NATURE_PALETTE["neutral_mid"],
    "resilience_aware": NATURE_PALETTE["blue"],
}
DESIGN_LABELS = {
    "economic_only": "Economic only",
    "resilience_aware": "Resilience aware",
}


def build_apartment_city_comparison_sources(
    input_root: Path | str = "Input",
    *,
    city_specs: Sequence[CityComparisonSpec] = DEFAULT_CITY_SPECS,
    scenario_count: int = DEFAULT_SCENARIO_COUNT,
    seed: int = DEFAULT_SYNTHETIC_SEED,
    analysis_voll_sgd_per_kwh: float = DEFAULT_ANALYSIS_VOLL_SGD_PER_KWH,
) -> dict[str, pd.DataFrame]:
    """Build deterministic synthetic city-comparison source tables for SG Apartment."""
    input_paths = _apartment_input_paths(Path(input_root))
    load = pd.read_csv(input_paths["load"])
    pv = pd.read_csv(input_paths["pv"])
    tou = pd.read_csv(input_paths["tou"])
    parameters = load_case_parameters(input_paths["parameter"]).base

    base = _base_apartment_series(load, pv, tou, parameters)
    city_profiles = _build_city_profiles(base, city_specs)
    city_inputs = _build_city_inputs(city_profiles, city_specs, analysis_voll_sgd_per_kwh)
    monthly_profiles = _build_monthly_profiles(city_profiles)
    hourly_profiles = _build_hourly_profiles(city_profiles)
    outage_scenarios, outage_events = _build_outage_sources(
        city_specs,
        scenario_count=scenario_count,
        seed=seed,
    )
    anchors = _apartment_design_anchors(base)
    design_results = _build_design_results(
        city_inputs,
        hourly_profiles,
        anchors,
        parameters,
        analysis_voll_sgd_per_kwh,
    )
    cost_breakdown = _build_cost_breakdown_sources(design_results)
    scenario_design_results = _build_scenario_design_results(city_inputs, outage_scenarios, design_results, seed)
    operation_profiles = _build_operation_profiles(city_profiles, design_results)
    city_operation_profiles = _build_city_operation_profiles(city_profiles, city_inputs, design_results)
    sensitivity_curves, regime_map = _build_sensitivity_sources(city_inputs, design_results, anchors)

    return {
        "city_inputs": city_inputs,
        "monthly_profiles": monthly_profiles,
        "hourly_profiles": hourly_profiles,
        "outage_scenarios": outage_scenarios,
        "outage_events": outage_events,
        "design_results": design_results,
        "cost_breakdown": cost_breakdown,
        "scenario_design_results": scenario_design_results,
        "operation_profiles": operation_profiles,
        "city_operation_profiles": city_operation_profiles,
        "sensitivity_curves": sensitivity_curves,
        "regime_map": regime_map,
    }


def build_singlefamily_city_comparison_sources(
    input_root: Path | str = "Input",
    *,
    config: dict[str, Any] | None = None,
    city_specs: Sequence[CityComparisonSpec] | None = None,
    scenario_count: int = DEFAULT_SCENARIO_COUNT,
    seed: int = DEFAULT_SYNTHETIC_SEED,
    trace_output_root: Path | str | None = None,
    parameter_overrides: dict[str, Any] | None = None,
    sensitivity_voll_values: Sequence[float] = DEFAULT_SENSITIVITY_VOLL_VALUES,
    sensitivity_battery_capex_values: Sequence[float] = DEFAULT_SENSITIVITY_BATTERY_CAPEX_VALUES,
    sensitivity_saidi_multipliers: Sequence[float] = DEFAULT_SENSITIVITY_SAIDI_MULTIPLIERS,
) -> dict[str, pd.DataFrame]:
    """Build city-comparison source tables from real SingleFamily inputs and simulations."""
    specs = _singlefamily_city_specs(city_specs)
    cases = _singlefamily_cases_by_city(Path(input_root), specs)
    global_config = _singlefamily_runtime_config(config)
    trace_root = Path(trace_output_root) if trace_output_root is not None else None

    payloads: dict[str, dict[str, Any]] = {}
    city_profiles: dict[str, pd.DataFrame] = {}
    for spec in specs:
        case = cases[spec.city_key]
        parameters = load_case_parameters(case.parameter_csv)
        case_config = _singlefamily_config_for_spec(
            global_config,
            case,
            parameters,
            spec,
            parameter_overrides=parameter_overrides,
        )
        profile = _singlefamily_case_profile(case, case_config, spec)
        payloads[spec.city_key] = {
            "case": case,
            "parameters": parameters,
            "config": case_config,
        }
        city_profiles[spec.city_key] = profile

    city_inputs = _build_singlefamily_city_inputs(city_profiles, specs)
    monthly_profiles = _build_monthly_profiles(city_profiles)
    hourly_profiles = _build_hourly_profiles(city_profiles)
    anchors = _singlefamily_design_anchors(city_profiles, specs)

    design_rows: list[dict[str, Any]] = []
    aware_runs: dict[str, SingleFamilyDesignRun] = {}
    economic_dispatches: dict[str, pd.DataFrame] = {}
    for spec in specs:
        payload = payloads[spec.city_key]
        case = payload["case"]
        parameters = payload["parameters"]
        case_config = payload["config"]
        city_row = city_inputs[city_inputs["city_key"].astype(str) == spec.city_key].iloc[0]

        aware_run = _run_singlefamily_design(case, case_config)
        aware_runs[spec.city_key] = aware_run
        design_rows.append(
            _singlefamily_design_result_row(
                aware_run.best.iloc[0],
                spec,
                city_row,
                case_config,
                aware_run.outage_profile,
                design_case="resilience_aware",
            )
        )

        economic_config = _singlefamily_config_for_spec(
            global_config,
            case,
            parameters,
            spec,
            parameter_overrides=parameter_overrides,
            overrides={"voll_sgd_per_kwh": 0.0},
        )
        economic_run = _run_singlefamily_design(
            case,
            economic_config,
            outage_profile=aware_run.outage_profile,
            outage_events=aware_run.outage_events,
        )
        evaluated_economic, economic_dispatch = _evaluate_singlefamily_fixed_design(
            case,
            case_config,
            aware_run.outage_profile,
            float(economic_run.best.iloc[0]["pv_capacity_kw"]),
            float(economic_run.best.iloc[0]["battery_energy_kwh"]),
        )
        economic_dispatches[spec.city_key] = economic_dispatch
        design_rows.append(
            _singlefamily_design_result_row(
                evaluated_economic,
                spec,
                city_row,
                case_config,
                aware_run.outage_profile,
                design_case="economic_only",
            )
        )

        if trace_root is not None:
            _write_singlefamily_baseline_trace(trace_root, case, parameters, case_config, aware_run)

    design_results = _with_city_order(pd.DataFrame(design_rows), specs)
    cost_breakdown = _build_cost_breakdown_sources(design_results)
    scenario_design_results, outage_scenarios, outage_events = _build_singlefamily_scenario_results(
        payloads,
        specs,
        design_results,
        scenario_count=int(scenario_count),
        seed=int(seed),
        trace_output_root=trace_root,
    )
    operation_profiles = _build_singlefamily_operation_profiles(aware_runs, design_results)
    city_operation_profiles = _build_singlefamily_city_operation_profiles(aware_runs, design_results)
    sensitivity_curves, regime_map = _build_singlefamily_sensitivity_sources(
        global_config,
        payloads,
        specs,
        city_inputs,
        anchors,
        parameter_overrides=parameter_overrides,
        voll_values=sensitivity_voll_values,
        battery_capex_values=sensitivity_battery_capex_values,
        saidi_multipliers=sensitivity_saidi_multipliers,
    )
    if trace_root is not None:
        _write_singlefamily_sensitivity_trace(trace_root, payloads, sensitivity_curves)

    return {
        "city_inputs": city_inputs,
        "monthly_profiles": monthly_profiles,
        "hourly_profiles": hourly_profiles,
        "outage_scenarios": outage_scenarios,
        "outage_events": outage_events,
        "design_results": design_results,
        "cost_breakdown": cost_breakdown,
        "scenario_design_results": scenario_design_results,
        "operation_profiles": operation_profiles,
        "city_operation_profiles": city_operation_profiles,
        "sensitivity_curves": sensitivity_curves,
        "regime_map": regime_map,
    }


def _singlefamily_city_specs(city_specs: Sequence[CityComparisonSpec] | None) -> tuple[CityComparisonSpec, ...]:
    if city_specs is not None:
        return tuple(city_specs)
    lookup = {spec.city_key: spec for spec in DEFAULT_CITY_SPECS}
    return tuple(lookup[city_key] for city_key in SINGLEFAMILY_CITY_ORDER)


def _singlefamily_runtime_config(config: dict[str, Any] | None) -> dict[str, Any]:
    runtime = deepcopy(config if config is not None else load_config("config.yaml"))
    if str(runtime.get("dispatch", {}).get("backend", "auto")).lower() == "auto":
        runtime.setdefault("dispatch", {})["backend"] = "greedy"
    return runtime


def _singlefamily_cases_by_city(input_root: Path, specs: Sequence[CityComparisonSpec]) -> dict[str, InputCase]:
    discovered = {case.case_key: case for case in discover_input_cases(input_root)}
    cases: dict[str, InputCase] = {}
    missing: list[str] = []
    for spec in specs:
        case_key = SINGLEFAMILY_CASE_BY_CITY.get(spec.city_key)
        if case_key is None or case_key not in discovered:
            missing.append(f"{spec.city_key}:{case_key}")
            continue
        cases[spec.city_key] = discovered[case_key]
    if missing:
        available = sorted(discovered)
        raise ValueError(f"Missing SingleFamily city input case(s): {missing}. Available cases: {available}")
    return cases


def _singlefamily_config_for_spec(
    global_config: dict[str, Any],
    case: InputCase,
    parameters: CaseParameters,
    spec: CityComparisonSpec,
    *,
    parameter_overrides: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = dict(parameters.base)
    params.update(
        {
            "city_name": spec.city_label,
            "building_name": "SingleFamily",
            "saidi_min_per_year": spec.saidi_min_per_year,
            "saifi_per_year": spec.saifi_per_year,
            "voll_sgd_per_kwh": spec.local_voll_proxy_sgd_per_kwh,
        }
    )
    if parameter_overrides:
        params.update(parameter_overrides)
    if overrides:
        params.update(overrides)
    return case_config_from_parameters(global_config, case, params)


def _singlefamily_case_profile(case: InputCase, config: dict[str, Any], spec: CityComparisonSpec) -> pd.DataFrame:
    frame, _ = build_case_frame(case, config, 1.0)
    technical = config["technical"]
    cooling_cop = float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0)))
    heating_cop = float(technical.get("heating_cop", 3.5))
    load = electric_equivalent_building_load(frame, cooling_cop, heating_cop)
    timestamp = pd.to_datetime(frame["timestamp"])
    profile = pd.DataFrame(
        {
            "hour_index": frame["hour_index"].astype(int),
            "timestamp": timestamp,
            "month": timestamp.dt.month.astype(int),
            "hour": timestamp.dt.hour.astype(int),
            "load_kwh": load.astype(float),
            "pv_kwh_per_kw": frame["pv_kwh"].astype(float),
            "tou_sgd_per_kwh": frame["tou_sgd_per_kwh"].astype(float),
            "city_key": spec.city_key,
            "city_label": spec.city_label,
            "data_status": REAL_SINGLEFAMILY_DATA_STATUS,
        }
    )
    if profile[["load_kwh", "pv_kwh_per_kw", "tou_sgd_per_kwh"]].isna().any().any():
        raise ValueError(f"SingleFamily city profile produced missing values for {case.case_key}")
    return profile


def _build_singlefamily_city_inputs(
    city_profiles: dict[str, pd.DataFrame],
    city_specs: Sequence[CityComparisonSpec],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_pv = None
    base_load = None
    for spec in city_specs:
        profile = city_profiles[spec.city_key]
        annual_load_mwh = float(profile["load_kwh"].sum()) / 1000.0
        annual_pv_yield = float(profile["pv_kwh_per_kw"].sum())
        if spec.city_key == "Singapore":
            base_pv = annual_pv_yield
            base_load = annual_load_mwh
        price_by_hour = profile.groupby("hour")["tou_sgd_per_kwh"].mean()
        rows.append(
            {
                "city_key": spec.city_key,
                "city_label": spec.city_label,
                "role": spec.role,
                "latitude": spec.latitude,
                "longitude": spec.longitude,
                "annual_load_mwh": annual_load_mwh,
                "peak_load_kw": float(profile["load_kwh"].max()),
                "annual_pv_yield_kwh_per_kw": annual_pv_yield,
                "mean_tariff_sgd_per_kwh": float(profile["tou_sgd_per_kwh"].mean()),
                "tariff_spread_sgd_per_kwh": float(price_by_hour.max() - price_by_hour.min()),
                "saidi_min_per_year": spec.saidi_min_per_year,
                "saifi_per_year": spec.saifi_per_year,
                "local_voll_proxy_sgd_per_kwh": spec.local_voll_proxy_sgd_per_kwh,
                "analysis_voll_sgd_per_kwh": spec.local_voll_proxy_sgd_per_kwh,
                "outage_tail_factor": spec.outage_tail_factor,
                "data_status": REAL_SINGLEFAMILY_DATA_STATUS,
                "source_note": "Real SingleFamily load/PV/TOU inputs with city reliability settings from city-comparison specs.",
            }
        )
    source = pd.DataFrame(rows)
    base_pv = float(base_pv or source["annual_pv_yield_kwh_per_kw"].iloc[0])
    base_load = float(base_load or source["annual_load_mwh"].iloc[0])
    source["pv_yield_index"] = source["annual_pv_yield_kwh_per_kw"] / base_pv
    source["load_index"] = source["annual_load_mwh"] / base_load
    return _with_city_order(source, city_specs)


def _singlefamily_design_anchors(
    city_profiles: dict[str, pd.DataFrame],
    specs: Sequence[CityComparisonSpec],
) -> dict[str, float]:
    base_key = "Singapore" if "Singapore" in city_profiles else specs[0].city_key
    base = city_profiles[base_key]
    annual_load = float(base["load_kwh"].sum())
    annual_pv_per_kw = float(base["pv_kwh_per_kw"].sum())
    peak_load = float(base["load_kwh"].max())
    pv_anchor = 0.75 * annual_load / max(annual_pv_per_kw, 1e-9)
    bess_anchor = max(1.0, 0.45 * peak_load)
    return {
        "annual_load_kwh": annual_load,
        "peak_load_kw": peak_load,
        "annual_pv_per_kw": annual_pv_per_kw,
        "base_pv_anchor_kw": float(pv_anchor),
        "base_bess_anchor_kwh": float(bess_anchor),
    }


def _run_singlefamily_design(
    case: InputCase,
    config: dict[str, Any],
    *,
    outage_profile: pd.DataFrame | None = None,
    outage_events: pd.DataFrame | None = None,
) -> SingleFamilyDesignRun:
    if outage_profile is None or outage_events is None:
        preview_frame, _ = build_case_frame(case, config, max(config["pv"]["capacity_kw_candidates"]))
        scenario = config["scenarios"][case.case_key]
        outage_profile, outage_events = simulate_outages(
            saifi_per_year=float(scenario["saifi_per_year"]),
            saidi_min_per_year=float(scenario["saidi_min_per_year"]),
            hours=len(preview_frame),
            seed=int(config.get("outage", {}).get("seed", 42)),
            forced_event_mode=str(config.get("outage", {}).get("forced_event_mode", "match_saidi")),
        )

    def pv_capacity_frame_builder(pv_capacity_kw: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, config, pv_capacity_kw)
        return frame.merge(outage_profile, on="hour_index", how="left"), pv_meta

    best, dispatch, summary, candidate_dispatches = resilience_design(
        pv_capacity_frame_builder,
        config,
        case.case_key,
    )
    return SingleFamilyDesignRun(
        best=best,
        dispatch=dispatch,
        summary=summary,
        outage_profile=outage_profile,
        outage_events=outage_events,
        candidate_dispatches=candidate_dispatches,
    )


def _evaluate_singlefamily_fixed_design(
    case: InputCase,
    config: dict[str, Any],
    outage_profile: pd.DataFrame,
    pv_capacity_kw: float,
    battery_energy_kwh: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    active_config = scenario_config(config, case.case_key)
    frame, pv_meta = build_case_frame(case, active_config, pv_capacity_kw)
    frame = frame.merge(outage_profile, on="hour_index", how="left")
    evaluator = CandidateEvaluator(active_config, CandidateStore(retain_dispatch=True))
    result, dispatch = evaluator.evaluate(frame, battery_energy_kwh, pv_meta, require_dispatch=True)
    if dispatch is None:
        raise ValueError("Fixed SingleFamily design evaluation did not return dispatch")
    result.update(
        {
            "scenario": case.case_key,
            "search_stage": "fixed_economic_capacity",
            "search_iteration": 0,
            "optimization_outage_mode": "city_outage",
            "optimization_voll_sgd_per_kwh": 0.0,
            "evaluation_voll_sgd_per_kwh": active_config["active_scenario"]["voll_sgd_per_kwh"],
        }
    )
    return result, dispatch


def _singlefamily_design_result_row(
    result_row: pd.Series | dict[str, Any],
    spec: CityComparisonSpec,
    city_input: pd.Series,
    config: dict[str, Any],
    outage_profile: pd.DataFrame,
    *,
    design_case: str,
) -> dict[str, Any]:
    result = dict(result_row)
    battery_energy = float(result["battery_energy_kwh"])
    battery_power = float(result["battery_power_kw"])
    econ = config["economics"]
    pv_initial = float(result["pv_capex_sgd"])
    battery_energy_initial = battery_energy * float(econ["battery_capex_sgd_per_kwh"])
    battery_power_initial = battery_power * float(econ["battery_power_capex_sgd_per_kw"])
    replacement_years = max(float(result.get("battery_replacement_years", 1.0)), 1e-9)
    ue_kwh = float(result["unserved_energy_kwh_per_year"])
    outage_load = _singlefamily_outage_weighted_load_from_profile(city_input, outage_profile)
    energy_based_lolp = 0.0 if outage_load <= 0.0 else float(np.clip(ue_kwh / outage_load, 0.0, 1.0))
    annualized_pv = float(result["annualized_pv_cost_sgd"])
    pv_om = float(result["pv_om_cost_sgd"])
    annualized_battery_energy = battery_energy_initial / replacement_years
    annualized_battery_power = battery_power_initial / replacement_years
    battery_om = float(result["battery_om_cost_sgd"])
    return {
        "city_key": spec.city_key,
        "city_label": spec.city_label,
        "design_case": design_case,
        "design_case_label": DESIGN_LABELS[design_case],
        "pv_capacity_kw": float(result["pv_capacity_kw"]),
        "battery_energy_kwh": battery_energy,
        "battery_power_kw": battery_power,
        "pv_initial_investment_sgd": pv_initial,
        "battery_energy_initial_investment_sgd": battery_energy_initial,
        "battery_power_initial_investment_sgd": battery_power_initial,
        "total_initial_investment_sgd": pv_initial + battery_energy_initial + battery_power_initial,
        "annualized_pv_capex_sgd_per_year": annualized_pv,
        "pv_om_sgd_per_year": pv_om,
        "annualized_battery_energy_capex_sgd_per_year": annualized_battery_energy,
        "annualized_battery_power_capex_sgd_per_year": annualized_battery_power,
        "battery_om_sgd_per_year": battery_om,
        "annualized_capex_sgd_per_year": annualized_pv + annualized_battery_energy + annualized_battery_power,
        "annual_om_sgd_per_year": pv_om + battery_om,
        "annualized_pv_bess_cost_sgd_per_year": (
            annualized_pv + pv_om + annualized_battery_energy + annualized_battery_power + battery_om
        ),
        "grid_cost_sgd_per_year": float(result["pv_bess_grid_cost_sgd"]),
        "outage_cost_sgd_per_year": float(result["unserved_energy_cost_sgd_per_year"]),
        "annualized_total_cost_sgd_per_year": float(result["pv_bess_system_annual_cost_sgd"]),
        "ue_kwh_per_year": ue_kwh,
        "energy_based_lolp": energy_based_lolp,
        "customer_coverage_probability": max(0.0, 1.0 - energy_based_lolp),
        "analysis_voll_sgd_per_kwh": float(config["scenarios"][next(iter(config["scenarios"]))]["voll_sgd_per_kwh"]),
        "mean_tariff_sgd_per_kwh": float(city_input["mean_tariff_sgd_per_kwh"]),
        "tariff_spread_sgd_per_kwh": float(city_input["tariff_spread_sgd_per_kwh"]),
        "saidi_min_per_year": float(spec.saidi_min_per_year),
        "saifi_per_year": float(spec.saifi_per_year),
        "outage_tail_factor": float(spec.outage_tail_factor),
        "mean_hourly_load_kw": float(city_input["annual_load_mwh"]) * 1000.0 / 8760.0,
        "outage_weighted_load_kwh": outage_load,
        "data_status": REAL_SINGLEFAMILY_DATA_STATUS,
    }


def _singlefamily_outage_weighted_load_from_profile(city_input: pd.Series, outage_profile: pd.DataFrame) -> float:
    average_load_kw = float(city_input["annual_load_mwh"]) * 1000.0 / 8760.0
    outage_hours = float(outage_profile["outage_fraction"].astype(float).sum())
    return average_load_kw * outage_hours


def _build_singlefamily_scenario_results(
    payloads: dict[str, dict[str, Any]],
    specs: Sequence[CityComparisonSpec],
    design_results: pd.DataFrame,
    *,
    scenario_count: int,
    seed: int,
    trace_output_root: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive")
    scenario_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    design_rows: list[pd.DataFrame] = []
    rng = np.random.default_rng(int(seed) + 3109)
    aware_lookup = (
        design_results[design_results["design_case"].astype(str) == "resilience_aware"]
        .set_index("city_key", drop=False)
    )
    for city_index, spec in enumerate(specs):
        payload = payloads[spec.city_key]
        case = payload["case"]
        config = payload["config"]
        seed_start = int(seed) + city_index * 1000
        generated = generate_outage_scenarios(
            case,
            config,
            scenario_count=scenario_count,
            seed_start=seed_start,
            scenario_id_column="scenario_id",
        )
        for scenario in generated.to_dict("records"):
            scenario_id = int(scenario["scenario_id"])
            scenario_rows.append(
                {
                    "city_key": spec.city_key,
                    "city_label": spec.city_label,
                    "scenario_id": scenario_id,
                    "input_saidi_min_per_year": spec.saidi_min_per_year,
                    "input_saifi_per_year": spec.saifi_per_year,
                    "simulated_saidi_min_per_year": float(scenario["simulated_saidi_min_per_year"]),
                    "simulated_saifi_per_year": float(scenario["simulated_saifi_per_year"]),
                    "annual_outage_hours": float(scenario["simulated_saidi_min_per_year"]) / 60.0,
                    "max_event_duration_hours": float(scenario["max_event_duration_min"]) / 60.0,
                    "event_count_for_plot": int(float(scenario["simulated_events"])),
                    "data_status": REAL_SINGLEFAMILY_DATA_STATUS,
                }
            )
            events = scenario["_outage_events"]
            for event in events.to_dict("records"):
                start_hour = float(event["start_hour"])
                event_rows.append(
                    {
                        "city_key": spec.city_key,
                        "city_label": spec.city_label,
                        "scenario_id": scenario_id,
                        "event_id": int(event["event_id"]),
                        "event_start_hour": start_hour,
                        "event_start_hour_of_day": int(start_hour) % 24,
                        "event_month": int((int(start_hour) // 730) % 12 + 1),
                        "duration_hours": float(event["duration_minutes"]) / 60.0,
                        "scenario_annual_outage_hours": float(scenario["simulated_saidi_min_per_year"]) / 60.0,
                        "data_status": REAL_SINGLEFAMILY_DATA_STATUS,
                    }
                )
        all_results, outage_scenarios, uncertainty_summary = run_uncertainty_analysis(
            case,
            config,
            case.case_key,
            scenario_count=scenario_count,
            seed_start=seed_start,
        )
        reference = aware_lookup.loc[spec.city_key]
        converted = _singlefamily_uncertainty_to_scenario_designs(
            all_results,
            spec,
            reference,
            rng,
        )
        design_rows.append(converted)
        if trace_output_root is not None:
            _write_singlefamily_uncertainty_trace(
                trace_output_root,
                case,
                all_results,
                outage_scenarios,
                uncertainty_summary,
            )
    return (
        _with_city_order(pd.concat(design_rows, ignore_index=True), specs),
        _with_city_order(pd.DataFrame(scenario_rows), specs),
        _with_city_order(pd.DataFrame(event_rows), specs),
    )


def _singlefamily_uncertainty_to_scenario_designs(
    all_results: pd.DataFrame,
    spec: CityComparisonSpec,
    reference: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in all_results.sort_values("uncertainty_scenario_id").to_dict("records"):
        ue = float(row["unserved_energy_kwh_per_year"])
        outage_load = float(row.get("outage_weighted_load_kwh", 0.0) or 0.0)
        energy_based_lolp = 0.0 if outage_load <= 0.0 else float(np.clip(ue / outage_load, 0.0, 1.0))
        rows.append(
            {
                "city_key": spec.city_key,
                "city_label": spec.city_label,
                "scenario_id": int(row["uncertainty_scenario_id"]),
                "pv_capacity_kw": float(row["pv_capacity_kw"]),
                "battery_energy_kwh": float(row["battery_energy_kwh"]),
                "ue_kwh_per_year": ue,
                "energy_based_lolp": energy_based_lolp,
                "energy_based_lolp_percent": 100.0 * energy_based_lolp,
                "annual_outage_hours": float(row["simulated_saidi_min_per_year"]) / 60.0,
                "simulated_saifi_per_year": float(row["simulated_saifi_per_year"]),
                "max_event_duration_hours": float(row["max_event_duration_min"]) / 60.0,
                "hours_stress": _safe_ratio(float(row["simulated_saidi_min_per_year"]), spec.saidi_min_per_year),
                "frequency_stress": _safe_ratio(float(row["simulated_saifi_per_year"]), spec.saifi_per_year),
                "duration_stress": _safe_ratio(float(row["max_event_duration_min"]) / 60.0, 1.0),
                "combined_stress": _safe_ratio(float(row["simulated_saidi_min_per_year"]), spec.saidi_min_per_year),
                "outage_tail_factor": float(spec.outage_tail_factor),
                "plot_jitter": float(rng.uniform(-0.16, 0.16)),
                "reference_pv_capacity_kw": float(reference["pv_capacity_kw"]),
                "reference_battery_energy_kwh": float(reference["battery_energy_kwh"]),
                "reference_ue_kwh_per_year": float(reference["ue_kwh_per_year"]),
                "reference_energy_based_lolp": float(reference["energy_based_lolp"]),
                "reference_outage_load_kwh_per_year": float(reference["outage_weighted_load_kwh"]),
                "scenario_outage_load_kwh_per_year": outage_load,
                "data_status": REAL_SINGLEFAMILY_DATA_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _safe_ratio(value: float, reference: float) -> float:
    return float(value) / max(float(reference), 1e-9)


def _build_singlefamily_operation_profiles(
    aware_runs: dict[str, SingleFamilyDesignRun],
    design_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for city_key in ("Singapore", "Metro_Manila"):
        if city_key not in aware_runs:
            continue
        design = _design_ordered(design_results, [city_key], "resilience_aware").iloc[0]
        rows.append(
            _singlefamily_dispatch_window(
                aware_runs[city_key],
                design,
                city_key,
                total_hours=37,
                outage_start_window_hour=12,
                include_city_operation_columns=False,
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_singlefamily_city_operation_profiles(
    aware_runs: dict[str, SingleFamilyDesignRun],
    design_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for city_key in _city_order_from(design_results[design_results["design_case"].astype(str) == "resilience_aware"]):
        design = _design_ordered(design_results, [city_key], "resilience_aware").iloc[0]
        rows.append(
            _singlefamily_dispatch_window(
                aware_runs[city_key],
                design,
                city_key,
                total_hours=72,
                outage_start_window_hour=36,
                include_city_operation_columns=True,
            )
        )
    return pd.concat(rows, ignore_index=True)


def _singlefamily_dispatch_window(
    run: SingleFamilyDesignRun,
    design: pd.Series,
    city_key: str,
    *,
    total_hours: int,
    outage_start_window_hour: int,
    include_city_operation_columns: bool,
) -> pd.DataFrame:
    event = _max_duration_event(run.outage_events)
    event_start_hour = float(event.get("start_hour", outage_start_window_hour))
    event_duration_hours = float(event.get("duration_minutes", 60.0)) / 60.0
    anchor = int(math.floor(event_start_hour))
    start_hour_index = anchor - int(outage_start_window_hour)
    dispatch = run.dispatch.sort_values("hour_index").set_index(run.dispatch["hour_index"].astype(int), drop=False)
    total_available_hours = len(dispatch)
    tou_prices = dispatch["tou_sgd_per_kwh"].astype(float)
    low_price_threshold = float(tou_prices.quantile(0.35))
    high_price_threshold = float(tou_prices.quantile(0.70))
    tou_spread = float(tou_prices.max() - tou_prices.min())
    records: list[dict[str, Any]] = []
    for window_hour in range(int(total_hours)):
        hour_index = int((start_hour_index + window_hour) % total_available_hours)
        row = dispatch.loc[hour_index]
        relative_hour = int(window_hour - outage_start_window_hour)
        load_kw = float(row["total_load_e_equiv_kwh"])
        unserved_kw = float(row["unmet_load_kwh_equiv"])
        served_load_kw = max(0.0, load_kw - unserved_kw)
        battery_capacity = float(row["battery_capacity_kwh"])
        soc_fraction = 0.0 if battery_capacity <= 0.0 else float(row["battery_soc_kwh"]) / battery_capacity
        tou_price = float(row["tou_sgd_per_kwh"])
        is_low_price = tou_price <= low_price_threshold + 1e-12
        is_high_price = tou_price >= high_price_threshold - 1e-12
        base_record = {
            "city_key": str(design["city_key"]),
            "city_label": str(design["city_label"]),
            "relative_hour": relative_hour,
            "outage_start_relative_hour": 0,
            "outage_end_relative_hour": event_duration_hours,
            "grid_available": 0 if float(row["outage_fraction"]) > 1e-9 else 1,
            "load_kw": load_kw,
            "pv_supply_kw": float(row["pv_used_kwh"]),
            "battery_discharge_kw": float(row["battery_discharge_kwh"]),
            "battery_charge_kw": float(row["battery_charge_kwh"]),
            "grid_supply_kw": float(row["grid_import_kwh"]),
            "unserved_kw": unserved_kw,
            "battery_soc_fraction": float(np.clip(soc_fraction, 0.0, 1.0)),
            "battery_energy_kwh": float(design["battery_energy_kwh"]),
            "pv_capacity_kw": float(design["pv_capacity_kw"]),
            "data_status": REAL_SINGLEFAMILY_DATA_STATUS,
        }
        if include_city_operation_columns:
            base_record.update(
                {
                    "window_hour": int(window_hour),
                    "hour_index": hour_index,
                    "clock_hour": hour_index % 24,
                    "representative_month": int((hour_index // 730) % 12 + 1),
                    "outage_start_window_hour": int(outage_start_window_hour),
                    "outage_end_window_hour": float(outage_start_window_hour) + event_duration_hours,
                    "representative_outage_duration_hours": event_duration_hours,
                    "tou_sgd_per_kwh": tou_price,
                    "tou_period": "low" if is_low_price else "high" if is_high_price else "shoulder",
                    "is_low_price": int(is_low_price),
                    "is_high_price": int(is_high_price),
                    "low_price_threshold_sgd_per_kwh": low_price_threshold,
                    "high_price_threshold_sgd_per_kwh": high_price_threshold,
                    "tou_spread_sgd_per_kwh": tou_spread,
                    "served_load_kw": served_load_kw,
                    "pv_available_kw": float(row["pv_available_kwh"]),
                    "pv_to_load_kw": min(load_kw, float(row["pv_used_kwh"])),
                    "pv_to_battery_kw": max(0.0, float(row["pv_used_kwh"]) - min(load_kw, float(row["pv_used_kwh"]))),
                    "grid_to_load_kw": min(served_load_kw, float(row["grid_import_kwh"])),
                    "grid_to_battery_kw": max(0.0, float(row["grid_import_kwh"]) - min(served_load_kw, float(row["grid_import_kwh"]))),
                    "grid_import_kw": float(row["grid_import_kwh"]),
                    "pv_curtailment_kw": float(row["pv_curtailed_kwh"]),
                    "reserve_floor_fraction": 0.10,
                    "target_soc_fraction": 0.50,
                    "reliability_target_soc_fraction": 0.50,
                    "pv_capacity_kw": float(design["pv_capacity_kw"]),
                    "ue_kwh_in_window": unserved_kw,
                    "saidi_min_per_year": float(design.get("saidi_min_per_year", np.nan)),
                    "saifi_per_year": float(design.get("saifi_per_year", np.nan)),
                    "outage_tail_factor": float(design.get("outage_tail_factor", np.nan)),
                }
            )
        records.append(base_record)
    return pd.DataFrame(records)


def _max_duration_event(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {"start_hour": 0.0, "duration_minutes": 15.0}
    ordered = events.sort_values("duration_minutes", ascending=False).reset_index(drop=True)
    return dict(ordered.iloc[0])


def _build_singlefamily_sensitivity_sources(
    global_config: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    specs: Sequence[CityComparisonSpec],
    city_inputs: pd.DataFrame,
    anchors: dict[str, float],
    *,
    parameter_overrides: dict[str, Any] | None,
    voll_values: Sequence[float],
    battery_capex_values: Sequence[float],
    saidi_multipliers: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    city_lookup = city_inputs.set_index(city_inputs["city_key"].astype(str))
    for spec in specs:
        payload = payloads[spec.city_key]
        case = payload["case"]
        parameters = payload["parameters"]
        city = city_lookup.loc[spec.city_key]
        for voll in voll_values:
            config = _singlefamily_config_for_spec(
                global_config,
                case,
                parameters,
                spec,
                parameter_overrides=parameter_overrides,
                overrides={"voll_sgd_per_kwh": float(voll)},
            )
            run = _run_singlefamily_design(case, config)
            rows.append(_singlefamily_sensitivity_result_row(run.best.iloc[0], spec, "voll_sgd_per_kwh", float(voll)))
        for capex in battery_capex_values:
            config = _singlefamily_config_for_spec(
                global_config,
                case,
                parameters,
                spec,
                parameter_overrides=parameter_overrides,
                overrides={"battery_capex_sgd_per_kwh": float(capex)},
            )
            run = _run_singlefamily_design(case, config)
            rows.append(_singlefamily_sensitivity_result_row(run.best.iloc[0], spec, "battery_capex_sgd_per_kwh", float(capex)))
        for multiplier in saidi_multipliers:
            config = _singlefamily_config_for_spec(
                global_config,
                case,
                parameters,
                spec,
                parameter_overrides=parameter_overrides,
                overrides={"saidi_min_per_year": float(spec.saidi_min_per_year) * float(multiplier)},
            )
            run = _run_singlefamily_design(case, config)
            rows.append(
                _singlefamily_sensitivity_result_row(
                    run.best.iloc[0],
                    spec,
                    "saidi_multiplier",
                    float(multiplier),
                    saidi_min_per_year=float(city["saidi_min_per_year"]) * float(multiplier),
                )
            )
    return pd.DataFrame(rows), _build_regime_map(city_inputs, anchors)


def _singlefamily_sensitivity_result_row(
    result: pd.Series,
    spec: CityComparisonSpec,
    sensitivity_type: str,
    x_value: float,
    saidi_min_per_year: float | None = None,
) -> dict[str, Any]:
    return {
        "city_key": spec.city_key,
        "city_label": spec.city_label,
        "sensitivity_type": sensitivity_type,
        "x_value": float(x_value),
        "battery_energy_kwh": float(result["battery_energy_kwh"]),
        "ue_kwh_per_year": float(result["unserved_energy_kwh_per_year"]),
        "saidi_min_per_year": saidi_min_per_year,
        "data_status": REAL_SINGLEFAMILY_DATA_STATUS,
    }


def _write_singlefamily_baseline_trace(
    output_root: Path,
    case: InputCase,
    parameters: CaseParameters,
    config: dict[str, Any],
    run: SingleFamilyDesignRun,
) -> None:
    from src.app import outputs as workflow_outputs
    from src.app.paths import case_output_paths
    from src.io.reporting import write_csv

    output_paths = case_output_paths(output_root, case.city_key, case.case_key)
    base_dir = output_paths.baseline
    frame, _ = build_case_frame(case, config, max(config["pv"]["capacity_kw_candidates"]))
    parameter_table = parameters.raw.copy()
    parameter_table.insert(0, "building_key", case.building_key)
    parameter_table.insert(0, "city_key", case.city_key)
    parameter_table.insert(0, "case_key", case.case_key)
    write_csv(base_dir / f"{case.case_key}_Baseline_Timeseries_Max_PV_Capacity.csv", frame)
    write_csv(base_dir / f"{case.case_key}_Baseline_Parameter_Table.csv", parameter_table)
    workflow_outputs.write_case_outputs(
        base_dir,
        case.case_key,
        config,
        run.best,
        run.outage_profile,
        run.outage_events,
        run.dispatch,
        run.summary,
        run.candidate_dispatches,
    )


def _write_singlefamily_uncertainty_trace(
    output_root: Path,
    case: InputCase,
    all_results: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
    uncertainty_summary: pd.DataFrame,
) -> None:
    from src.app import outputs as workflow_outputs
    from src.app.paths import case_output_paths

    output_paths = case_output_paths(output_root, case.city_key, case.case_key)
    workflow_outputs.write_uncertainty_baseline_outputs(
        output_paths.uncertainty_baseline,
        case.case_key,
        all_results,
        outage_scenarios,
        uncertainty_summary,
    )


def _write_singlefamily_sensitivity_trace(
    output_root: Path,
    payloads: dict[str, dict[str, Any]],
    sensitivity_curves: pd.DataFrame,
) -> None:
    from src.app.paths import case_output_paths
    from src.io.reporting import write_csv

    for city_key, payload in payloads.items():
        case = payload["case"]
        output_paths = case_output_paths(output_root, case.city_key, case.case_key)
        subset = sensitivity_curves[sensitivity_curves["city_key"].astype(str) == city_key].copy()
        if subset.empty:
            continue
        write_csv(
            output_paths.sensitivity / f"{case.case_key}_City_Comparison_Sensitivity_Curves.csv",
            subset,
        )


def make_city_comparison_figures(
    out: Path | str,
    *,
    input_root: Path | str = "Input",
    case_key: str = APARTMENT_CASE_KEY,
    figures: Sequence[int | str] | None = None,
    sources: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """Create Apartment city-comparison Figures 1-9 and optional Figure 7b source-data CSV files."""
    selected = _expand_city_comparison_figures(figures)
    unsupported = [figure for figure in selected if figure not in SUPPORTED_FIGURE_NUMBERS]
    if unsupported:
        raise ValueError(f"Unsupported city-comparison figure number(s): {unsupported}")
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = sources or build_apartment_city_comparison_sources(input_root)

    paths: dict[str, Path] = {}
    for figure_no in selected:
        base = output_dir / f"{case_key}_{_figure_name(figure_no)}"
        paths[f"figure_{figure_no}_source_data"] = _write_source_data(base, _figure_source(prepared, figure_no))
        plotter = {
            1: _plot_figure_1,
            2: _plot_figure_2,
            3: _plot_figure_3,
            4: _plot_figure_4,
            5: _plot_figure_5,
            6: _plot_figure_6,
            7: _plot_figure_7,
            8: _plot_figure_8,
            9: _plot_figure_9,
            FIGURE_7B_ID: _plot_figure_7b,
        }[figure_no]
        paths.update(plotter(output_dir, case_key, prepared))
    return paths


def _expand_city_comparison_figures(figures: Sequence[int | str] | None) -> tuple[int | str, ...]:
    if figures is None:
        return tuple(FIGURE_NUMBERS)
    selected: list[int | str] = []
    for figure in figures:
        if isinstance(figure, str) and figure.strip().lower() == "all":
            selected.extend([*FIGURE_NUMBERS, FIGURE_7B_ID])
            continue
        selected.append(_normalize_city_comparison_figure(figure))
    return tuple(dict.fromkeys(selected))


def _normalize_city_comparison_figure(figure: int | str) -> int | str:
    if isinstance(figure, str):
        normalized = figure.strip().lower()
        if normalized == FIGURE_7B_ID:
            return FIGURE_7B_ID
        if normalized.isdigit():
            return int(normalized)
        return normalized
    return figure


def _figure_name(figure_no: int | str) -> str:
    if figure_no == FIGURE_7B_ID:
        return FIGURE_7B_NAME
    return FIGURE_NAMES[int(figure_no)]


def _apartment_input_paths(input_root: Path) -> dict[str, Path]:
    case_dir = input_root / APARTMENT_CITY_KEY
    paths = {
        "load": case_dir / f"{APARTMENT_CASE_KEY}_Load.csv",
        "pv": case_dir / f"{APARTMENT_CASE_KEY}_PV.csv",
        "tou": case_dir / f"{APARTMENT_CASE_KEY}_TOU.csv",
        "parameter": case_dir / f"{APARTMENT_CASE_KEY}_Parameter.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Apartment city-comparison input file(s): " + ", ".join(missing))
    return paths


def _base_apartment_series(
    load: pd.DataFrame,
    pv: pd.DataFrame,
    tou: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    required_load = {"Cooling", "Heating", "Lighting", "Equipment"}
    if not required_load.issubset(load.columns):
        raise ValueError("Apartment load input is missing column(s): " + ", ".join(sorted(required_load - set(load.columns))))
    if "PV_output_kWh_per_m2" not in pv.columns:
        raise ValueError("Apartment PV input must include PV_output_kWh_per_m2")
    if not {"hour", "tou_sgd_per_kwh"}.issubset(tou.columns):
        raise ValueError("Apartment TOU input must include hour and tou_sgd_per_kwh")
    if len(load) != 8760 or len(pv) != 8760 or len(tou) != 24:
        raise ValueError("Apartment city comparison requires 8760 load/PV rows and 24 TOU rows.")

    timestamp = _timestamps(load)
    cooling_cop = float(parameters.get("cooling_cop", 5.0) or 5.0)
    heating_cop = float(parameters.get("heating_cop", 3.5) or 3.5)
    module_density = float(parameters.get("pv_module_power_density_kw_per_m2", 0.135) or 0.135)
    hour_to_price = dict(zip(tou["hour"].astype(int), tou["tou_sgd_per_kwh"].astype(float)))
    base_load = (
        load["Lighting"].astype(float)
        + load["Equipment"].astype(float)
        + load["Cooling"].astype(float) / max(cooling_cop, 1e-9)
        + load["Heating"].astype(float) / max(heating_cop, 1e-9)
    )
    base_pv_per_kw = pv["PV_output_kWh_per_m2"].astype(float) / max(module_density, 1e-9)
    frame = pd.DataFrame(
        {
            "hour_index": np.arange(8760, dtype=int),
            "timestamp": timestamp,
            "month": timestamp.dt.month.astype(int),
            "hour": timestamp.dt.hour.astype(int),
            "base_load_kwh": base_load.astype(float),
            "base_pv_kwh_per_kw": base_pv_per_kw.astype(float),
        }
    )
    frame["base_tou_sgd_per_kwh"] = frame["hour"].map(hour_to_price).astype(float)
    if frame[["base_load_kwh", "base_pv_kwh_per_kw", "base_tou_sgd_per_kwh"]].isna().any().any():
        raise ValueError("Apartment city comparison produced missing base timeseries values.")
    return frame


def _timestamps(load: pd.DataFrame) -> pd.Series:
    for column in ("timestamp", "Date/Time", "date_time", "datetime"):
        if column in load.columns:
            parsed = pd.to_datetime(load[column], errors="coerce")
            if not parsed.isna().any():
                return pd.Series(parsed)
    return pd.Series(pd.date_range("2026-01-01", periods=8760, freq="h"))


def _build_city_profiles(
    base: pd.DataFrame,
    city_specs: Sequence[CityComparisonSpec],
) -> dict[str, pd.DataFrame]:
    profiles: dict[str, pd.DataFrame] = {}
    base_annual_load = float(base["base_load_kwh"].sum())
    base_annual_pv = float(base["base_pv_kwh_per_kw"].sum())
    base_price_mean = float(base["base_tou_sgd_per_kwh"].mean())
    for spec in city_specs:
        frame = base[["hour_index", "timestamp", "month", "hour"]].copy()
        load_factor = _seasonal_factor(frame["month"], spec.load_peak_month, spec.load_seasonality) * _diurnal_factor(
            frame["hour"],
            spec.load_diurnal_peak_hour,
            spec.load_diurnal_strength,
        )
        pv_factor = _seasonal_factor(frame["month"], spec.pv_peak_month, spec.pv_seasonality)
        load_series = base["base_load_kwh"].astype(float) * load_factor
        pv_series = base["base_pv_kwh_per_kw"].astype(float) * pv_factor
        load_target = base_annual_load * spec.load_energy_multiplier
        pv_target = base_annual_pv * spec.pv_yield_multiplier
        frame["load_kwh"] = load_series * (load_target / max(float(load_series.sum()), 1e-9))
        frame["pv_kwh_per_kw"] = pv_series * (pv_target / max(float(pv_series.sum()), 1e-9))
        centered_price = base["base_tou_sgd_per_kwh"].astype(float) - base_price_mean
        frame["tou_sgd_per_kwh"] = (
            base_price_mean * spec.tariff_multiplier
            + centered_price * spec.tariff_multiplier * spec.tariff_spread_multiplier
        ).clip(lower=0.05)
        frame["city_key"] = spec.city_key
        frame["city_label"] = spec.city_label
        frame["data_status"] = DATA_STATUS
        profiles[spec.city_key] = frame
    return profiles


def _seasonal_factor(months: pd.Series, peak_month: int, strength: float) -> np.ndarray:
    angle = 2.0 * np.pi * (months.astype(float).to_numpy() - float(peak_month)) / 12.0
    return np.maximum(0.25, 1.0 + float(strength) * np.cos(angle))


def _diurnal_factor(hours: pd.Series, peak_hour: int, strength: float) -> np.ndarray:
    angle = 2.0 * np.pi * (hours.astype(float).to_numpy() - float(peak_hour)) / 24.0
    return np.maximum(0.50, 1.0 + float(strength) * np.cos(angle))


def _build_city_inputs(
    city_profiles: dict[str, pd.DataFrame],
    city_specs: Sequence[CityComparisonSpec],
    analysis_voll_sgd_per_kwh: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_pv = None
    base_load = None
    for spec in city_specs:
        profile = city_profiles[spec.city_key]
        annual_load_mwh = float(profile["load_kwh"].sum()) / 1000.0
        annual_pv_yield = float(profile["pv_kwh_per_kw"].sum())
        if spec.city_key == "Singapore":
            base_pv = annual_pv_yield
            base_load = annual_load_mwh
        price_by_hour = profile.groupby("hour")["tou_sgd_per_kwh"].mean()
        rows.append(
            {
                "city_key": spec.city_key,
                "city_label": spec.city_label,
                "role": spec.role,
                "latitude": spec.latitude,
                "longitude": spec.longitude,
                "annual_load_mwh": annual_load_mwh,
                "peak_load_kw": float(profile["load_kwh"].max()),
                "annual_pv_yield_kwh_per_kw": annual_pv_yield,
                "mean_tariff_sgd_per_kwh": float(profile["tou_sgd_per_kwh"].mean()),
                "tariff_spread_sgd_per_kwh": float(price_by_hour.max() - price_by_hour.min()),
                "saidi_min_per_year": spec.saidi_min_per_year,
                "saifi_per_year": spec.saifi_per_year,
                "local_voll_proxy_sgd_per_kwh": spec.local_voll_proxy_sgd_per_kwh,
                "analysis_voll_sgd_per_kwh": analysis_voll_sgd_per_kwh,
                "outage_tail_factor": spec.outage_tail_factor,
                "data_status": DATA_STATUS,
                "source_note": "Synthetic city proxy derived from SG Apartment load/PV/TOU inputs.",
            }
        )
    source = pd.DataFrame(rows)
    base_pv = float(base_pv or source["annual_pv_yield_kwh_per_kw"].iloc[0])
    base_load = float(base_load or source["annual_load_mwh"].iloc[0])
    source["pv_yield_index"] = source["annual_pv_yield_kwh_per_kw"] / base_pv
    source["load_index"] = source["annual_load_mwh"] / base_load
    return _with_city_order(source, city_specs)


def _build_monthly_profiles(city_profiles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for city_key, profile in city_profiles.items():
        data_status = str(profile["data_status"].iloc[0]) if "data_status" in profile.columns else DATA_STATUS
        monthly = (
            profile.groupby(["city_key", "city_label", "month"], as_index=False)
            .agg(
                load_mwh=("load_kwh", lambda values: float(values.sum()) / 1000.0),
                pv_yield_kwh_per_kw_month=("pv_kwh_per_kw", "sum"),
            )
            .reset_index(drop=True)
        )
        monthly["data_status"] = data_status
        rows.append(monthly)
    return pd.concat(rows, ignore_index=True)


def _build_hourly_profiles(city_profiles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for city_key, profile in city_profiles.items():
        data_status = str(profile["data_status"].iloc[0]) if "data_status" in profile.columns else DATA_STATUS
        hourly = (
            profile.groupby(["city_key", "city_label", "hour"], as_index=False)
            .agg(
                mean_load_kw=("load_kwh", "mean"),
                mean_pv_kw_per_kw=("pv_kwh_per_kw", "mean"),
                tou_sgd_per_kwh=("tou_sgd_per_kwh", "mean"),
            )
            .reset_index(drop=True)
        )
        hourly["normalized_load"] = hourly["mean_load_kw"] / max(float(hourly["mean_load_kw"].max()), 1e-9)
        hourly["data_status"] = data_status
        rows.append(hourly)
    return pd.concat(rows, ignore_index=True)


def _build_outage_sources(
    city_specs: Sequence[CityComparisonSpec],
    *,
    scenario_count: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive")
    scenario_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for city_index, spec in enumerate(city_specs):
        rng = np.random.default_rng(int(seed) + city_index * 997)
        sigma = 0.16 + 0.12 * float(spec.outage_tail_factor)
        saidi = _positive_mean_rescaled_lognormal(rng, spec.saidi_min_per_year, sigma, scenario_count)
        saifi = _positive_mean_rescaled_lognormal(rng, spec.saifi_per_year, sigma * 0.65, scenario_count)
        for scenario_index in range(scenario_count):
            scenario_id = scenario_index + 1
            simulated_saidi = float(saidi[scenario_index])
            simulated_saifi = float(saifi[scenario_index])
            annual_hours = simulated_saidi / 60.0
            event_count = 0 if annual_hours <= 0.0 else max(1, int(math.ceil(simulated_saifi)))
            durations = _event_durations(rng, annual_hours, event_count, spec.outage_tail_factor)
            max_duration = float(max(durations) if durations else 0.0)
            scenario_rows.append(
                {
                    "city_key": spec.city_key,
                    "city_label": spec.city_label,
                    "scenario_id": scenario_id,
                    "input_saidi_min_per_year": spec.saidi_min_per_year,
                    "input_saifi_per_year": spec.saifi_per_year,
                    "simulated_saidi_min_per_year": simulated_saidi,
                    "simulated_saifi_per_year": simulated_saifi,
                    "annual_outage_hours": annual_hours,
                    "max_event_duration_hours": max_duration,
                    "event_count_for_plot": event_count,
                    "data_status": DATA_STATUS,
                }
            )
            for event_id, duration in enumerate(durations, start=1):
                start_hour = float(rng.uniform(0.0, max(1.0, 8760.0 - duration)))
                start_hour_of_day = int(start_hour) % 24
                start_month = int((int(start_hour) // 730) % 12 + 1)
                event_rows.append(
                    {
                        "city_key": spec.city_key,
                        "city_label": spec.city_label,
                        "scenario_id": scenario_id,
                        "event_id": event_id,
                        "event_start_hour": start_hour,
                        "event_start_hour_of_day": start_hour_of_day,
                        "event_month": start_month,
                        "duration_hours": float(duration),
                        "scenario_annual_outage_hours": annual_hours,
                        "data_status": DATA_STATUS,
                    }
                )
    return pd.DataFrame(scenario_rows), pd.DataFrame(event_rows)


def _positive_mean_rescaled_lognormal(
    rng: np.random.Generator,
    target_mean: float,
    sigma: float,
    count: int,
) -> np.ndarray:
    if target_mean <= 0.0:
        return np.zeros(count, dtype=float)
    raw = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=count)
    return raw * (float(target_mean) / max(float(raw.mean()), 1e-12))


def _event_durations(
    rng: np.random.Generator,
    annual_hours: float,
    event_count: int,
    tail_factor: float,
) -> list[float]:
    if annual_hours <= 0.0 or event_count <= 0:
        return []
    if event_count == 1:
        return [float(annual_hours)]
    shape = max(0.16, 1.0 / max(float(tail_factor), 1e-9))
    shares = rng.gamma(shape=shape, scale=1.0, size=event_count)
    shares = shares / max(float(shares.sum()), 1e-12)
    return [float(annual_hours * share) for share in shares]


def _apartment_design_anchors(base: pd.DataFrame) -> dict[str, float]:
    annual_load = float(base["base_load_kwh"].sum())
    annual_pv_per_kw = float(base["base_pv_kwh_per_kw"].sum())
    peak_load = float(base["base_load_kwh"].max())
    pv_anchor = 0.75 * annual_load / max(annual_pv_per_kw, 1e-9)
    bess_anchor = max(25.0, 0.45 * peak_load)
    return {
        "annual_load_kwh": annual_load,
        "peak_load_kw": peak_load,
        "annual_pv_per_kw": annual_pv_per_kw,
        "base_pv_anchor_kw": float(pv_anchor),
        "base_bess_anchor_kwh": float(bess_anchor),
    }


def _build_design_results(
    city_inputs: pd.DataFrame,
    hourly_profiles: pd.DataFrame,
    anchors: dict[str, float],
    parameters: dict[str, Any],
    analysis_voll_sgd_per_kwh: float,
) -> pd.DataFrame:
    pv_capex = float(parameters.get("pv_capex_sgd_per_kw", 1200.0) or 1200.0)
    battery_capex = float(parameters.get("battery_capex_sgd_per_kwh", 600.0) or 600.0)
    battery_power_capex = float(parameters.get("battery_power_capex_sgd_per_kw", 180.0) or 180.0)
    pv_project_years = float(parameters.get("pv_project_years", 15.0) or 15.0)
    pv_om = float(parameters.get("pv_om_fraction_per_year", 0.015) or 0.015)
    battery_om = float(parameters.get("battery_om_fraction_per_year", 0.02) or 0.02)
    rows: list[dict[str, Any]] = []
    max_saidi = max(float(city_inputs["saidi_min_per_year"].max()), 1.0)
    max_saifi = max(float(city_inputs["saifi_per_year"].max()), 1.0)
    max_tail = max(float(city_inputs["outage_tail_factor"].max()), 1.0)
    for city in city_inputs.to_dict("records"):
        city_key = str(city["city_key"])
        hourly = hourly_profiles[hourly_profiles["city_key"].astype(str) == city_key]
        tariff_spread = float(city["tariff_spread_sgd_per_kwh"])
        tariff_mean = float(city["mean_tariff_sgd_per_kwh"])
        tariff_index = tariff_mean / max(float(city_inputs["mean_tariff_sgd_per_kwh"].mean()), 1e-9)
        spread_index = tariff_spread / max(float(city_inputs["tariff_spread_sgd_per_kwh"].mean()), 1e-9)
        saidi_norm = math.log1p(float(city["saidi_min_per_year"])) / math.log1p(max_saidi)
        saifi_norm = math.log1p(float(city["saifi_per_year"])) / math.log1p(max_saifi)
        tail_norm = float(city["outage_tail_factor"]) / max_tail
        resilience_index = 0.55 * saidi_norm + 0.25 * saifi_norm + 0.20 * saidi_norm * tail_norm
        pv_economic = anchors["base_pv_anchor_kw"] * (0.90 + 0.10 * tariff_index) * (
            0.94 + 0.06 * float(city["pv_yield_index"])
        )
        bess_economic = (
            anchors["base_bess_anchor_kwh"]
            * (0.58 + 0.42 * min(spread_index, 1.8))
            * (0.70 + 0.30 * float(city["load_index"]))
        )
        cases = (
            ("economic_only", pv_economic, bess_economic),
            (
                "resilience_aware",
                pv_economic * (1.0 + 0.10 * resilience_index + 0.03 * tail_norm),
                bess_economic * (1.0 + 4.15 * resilience_index + 0.32 * max(float(city["outage_tail_factor"]) - 1.0, 0.0)),
            ),
        )
        annual_load_kwh = float(city["annual_load_mwh"]) * 1000.0
        average_load_kw = annual_load_kwh / 8760.0
        for design_case, pv_kw, battery_kwh in cases:
            pv_kw = float(max(pv_kw, 0.0))
            battery_kwh = float(max(battery_kwh, 0.0))
            battery_power_kw = 0.5 * battery_kwh
            pv_generation_kwh = pv_kw * float(city["annual_pv_yield_kwh_per_kw"])
            self_consumed_fraction = min(0.82, 0.46 + 0.0013 * battery_kwh)
            solar_value = min(annual_load_kwh * tariff_mean * 0.58, pv_generation_kwh * tariff_mean * self_consumed_fraction)
            arbitrage_saving = battery_kwh * tariff_spread * 32.0
            grid_cost = max(0.0, annual_load_kwh * tariff_mean - solar_value - arbitrage_saving)
            pv_initial_investment = pv_kw * pv_capex
            battery_energy_initial_investment = battery_kwh * battery_capex
            battery_power_initial_investment = battery_power_kw * battery_power_capex
            total_initial_investment = pv_initial_investment + battery_energy_initial_investment + battery_power_initial_investment
            annualized_pv_capex = pv_initial_investment / max(pv_project_years, 1e-9)
            pv_om_cost = pv_initial_investment * pv_om
            annualized_battery_energy_capex = battery_energy_initial_investment / 10.5
            annualized_battery_power_capex = battery_power_initial_investment / 10.5
            battery_om_cost = battery_energy_initial_investment * battery_om
            pv_cost = annualized_pv_capex + pv_om_cost
            battery_cost = annualized_battery_energy_capex + annualized_battery_power_capex + battery_om_cost
            avg_duration = (float(city["saidi_min_per_year"]) / 60.0) / max(float(city["saifi_per_year"]), 0.05)
            outage_energy = average_load_kw * (float(city["saidi_min_per_year"]) / 60.0)
            deficit_scale = max(1.0, average_load_kw * max(avg_duration, 0.10) * (0.75 + 0.45 * float(city["outage_tail_factor"])))
            pv_resilience_credit = 1.0 - min(0.34, 0.055 * (pv_kw / max(anchors["base_pv_anchor_kw"], 1e-9)) * float(city["pv_yield_index"]))
            tail_penalty = 1.0 + 0.34 * max(float(city["outage_tail_factor"]) - 1.0, 0.0)
            ue = outage_energy * tail_penalty * math.exp(-battery_kwh / deficit_scale) * pv_resilience_credit
            ue = max(0.0, ue)
            outage_cost = ue * analysis_voll_sgd_per_kwh
            total_cost = pv_cost + battery_cost + grid_cost + outage_cost
            outage_load_reference = max(outage_energy * tail_penalty, 1e-9)
            lolp = min(1.0, ue / outage_load_reference)
            rows.append(
                {
                    "city_key": city_key,
                    "city_label": city["city_label"],
                    "design_case": design_case,
                    "design_case_label": DESIGN_LABELS[design_case],
                    "pv_capacity_kw": pv_kw,
                    "battery_energy_kwh": battery_kwh,
                    "battery_power_kw": battery_power_kw,
                    "pv_initial_investment_sgd": pv_initial_investment,
                    "battery_energy_initial_investment_sgd": battery_energy_initial_investment,
                    "battery_power_initial_investment_sgd": battery_power_initial_investment,
                    "total_initial_investment_sgd": total_initial_investment,
                    "annualized_pv_capex_sgd_per_year": annualized_pv_capex,
                    "pv_om_sgd_per_year": pv_om_cost,
                    "annualized_battery_energy_capex_sgd_per_year": annualized_battery_energy_capex,
                    "annualized_battery_power_capex_sgd_per_year": annualized_battery_power_capex,
                    "battery_om_sgd_per_year": battery_om_cost,
                    "annualized_capex_sgd_per_year": annualized_pv_capex + annualized_battery_energy_capex + annualized_battery_power_capex,
                    "annual_om_sgd_per_year": pv_om_cost + battery_om_cost,
                    "annualized_pv_bess_cost_sgd_per_year": pv_cost + battery_cost,
                    "grid_cost_sgd_per_year": grid_cost,
                    "outage_cost_sgd_per_year": outage_cost,
                    "annualized_total_cost_sgd_per_year": total_cost,
                    "ue_kwh_per_year": ue,
                    "energy_based_lolp": lolp,
                    "customer_coverage_probability": max(0.0, 1.0 - lolp),
                    "analysis_voll_sgd_per_kwh": analysis_voll_sgd_per_kwh,
                    "mean_tariff_sgd_per_kwh": tariff_mean,
                    "tariff_spread_sgd_per_kwh": tariff_spread,
                    "mean_hourly_load_kw": float(hourly["mean_load_kw"].mean()),
                    "data_status": DATA_STATUS,
                }
            )
    return pd.DataFrame(rows)


def _build_cost_breakdown_sources(design_results: pd.DataFrame) -> pd.DataFrame:
    aware = design_results[design_results["design_case"].astype(str) == "resilience_aware"].copy()
    annualized_components = (
        ("pv_capex", "PV initial investment", "Initial investment", "PV", "annualized_pv_capex_sgd_per_year", 1),
        ("battery_energy_capex", "BESS energy initial investment", "Initial investment", "BESS", "annualized_battery_energy_capex_sgd_per_year", 2),
        ("battery_power_capex", "BESS power initial investment", "Initial investment", "BESS", "annualized_battery_power_capex_sgd_per_year", 3),
        ("pv_om", "PV O&M", "O&M", "PV", "pv_om_sgd_per_year", 4),
        ("battery_om", "BESS O&M", "O&M", "BESS", "battery_om_sgd_per_year", 5),
        ("electricity_bill", "Electricity bill", "Electricity", "Grid", "grid_cost_sgd_per_year", 6),
        ("outage_loss", "Expected outage loss", "Reliability", "Outage", "outage_cost_sgd_per_year", 7),
    )
    upfront_components = (
        ("pv_upfront_capex", "PV upfront CAPEX", "PV", "pv_initial_investment_sgd", 1),
        ("battery_energy_upfront_capex", "BESS energy upfront CAPEX", "BESS", "battery_energy_initial_investment_sgd", 2),
        ("battery_power_upfront_capex", "BESS power upfront CAPEX", "BESS", "battery_power_initial_investment_sgd", 3),
    )
    rows: list[dict[str, Any]] = []
    for row in aware.to_dict("records"):
        data_status = str(row.get("data_status", DATA_STATUS))
        total_annualized = float(row["annualized_total_cost_sgd_per_year"])
        total_upfront = float(row["total_initial_investment_sgd"])
        for component_key, component_label, component_group, asset_group, column, order in annualized_components:
            value = float(row[column])
            rows.append(
                {
                    "city_key": row["city_key"],
                    "city_label": row["city_label"],
                    "design_case": row["design_case"],
                    "design_case_label": row["design_case_label"],
                    "cost_basis": "annualized",
                    "component_key": component_key,
                    "component_label": component_label,
                    "component_group": component_group,
                    "asset_group": asset_group,
                    "stack_order": order,
                    "cost_sgd": value,
                    "cost_ksgd": value / 1000.0,
                    "share_of_total_percent": 100.0 * value / max(total_annualized, 1e-9),
                    "total_cost_sgd": total_annualized,
                    "total_cost_ksgd": total_annualized / 1000.0,
                    "annual_load_mwh": float(row["mean_hourly_load_kw"]) * 8.76,
                    "total_cost_sgd_per_kwh_load": total_annualized / max(float(row["mean_hourly_load_kw"]) * 8760.0, 1e-9),
                    "pv_capacity_kw": row["pv_capacity_kw"],
                    "battery_energy_kwh": row["battery_energy_kwh"],
                    "data_status": data_status,
                }
            )
        for component_key, component_label, asset_group, column, order in upfront_components:
            value = float(row[column])
            rows.append(
                {
                    "city_key": row["city_key"],
                    "city_label": row["city_label"],
                    "design_case": row["design_case"],
                    "design_case_label": row["design_case_label"],
                    "cost_basis": "upfront",
                    "component_key": component_key,
                    "component_label": component_label,
                    "component_group": "Initial investment",
                    "asset_group": asset_group,
                    "stack_order": order,
                    "cost_sgd": value,
                    "cost_ksgd": value / 1000.0,
                    "share_of_total_percent": 100.0 * value / max(total_upfront, 1e-9),
                    "total_cost_sgd": total_upfront,
                    "total_cost_ksgd": total_upfront / 1000.0,
                    "annual_load_mwh": float(row["mean_hourly_load_kw"]) * 8.76,
                    "total_cost_sgd_per_kwh_load": total_annualized / max(float(row["mean_hourly_load_kw"]) * 8760.0, 1e-9),
                    "pv_capacity_kw": row["pv_capacity_kw"],
                    "battery_energy_kwh": row["battery_energy_kwh"],
                    "data_status": data_status,
                }
            )
    return pd.DataFrame(rows)


def _build_scenario_design_results(
    city_inputs: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
    design_results: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    aware = design_results[design_results["design_case"].astype(str) == "resilience_aware"].copy()
    city_lookup = city_inputs.set_index(city_inputs["city_key"].astype(str))
    design_lookup = aware.set_index(aware["city_key"].astype(str))
    scenario_stats = outage_scenarios.groupby("city_key").agg(
        mean_annual_outage_hours=("annual_outage_hours", "mean"),
        mean_saifi=("simulated_saifi_per_year", "mean"),
        mean_max_event_duration_hours=("max_event_duration_hours", "mean"),
    )
    rng = np.random.default_rng(int(seed) + 3109)
    rows: list[dict[str, Any]] = []
    for scenario in outage_scenarios.to_dict("records"):
        city_key = str(scenario["city_key"])
        city = city_lookup.loc[city_key]
        reference = design_lookup.loc[city_key]
        stats = scenario_stats.loc[city_key]
        annual_hours = float(max(scenario["annual_outage_hours"], 0.0))
        simulated_saifi = float(max(scenario["simulated_saifi_per_year"], 0.0))
        max_duration = float(max(scenario["max_event_duration_hours"], 0.0))
        mean_hours = float(max(stats["mean_annual_outage_hours"], 1e-9))
        mean_saifi = float(max(stats["mean_saifi"], 1e-9))
        mean_max_duration = float(max(stats["mean_max_event_duration_hours"], 1e-9))
        hours_stress = annual_hours / mean_hours
        frequency_stress = simulated_saifi / mean_saifi
        duration_stress = max_duration / mean_max_duration
        tail_factor = float(city["outage_tail_factor"])
        combined_stress = 0.42 * hours_stress + 0.36 * duration_stress + 0.22 * frequency_stress

        reference_pv = float(reference["pv_capacity_kw"])
        reference_bess = float(reference["battery_energy_kwh"])
        reference_ue = float(reference["ue_kwh_per_year"])
        pv_multiplier = np.clip(
            1.0 + 0.045 * (combined_stress - 1.0) + 0.012 * (hours_stress - 1.0) - 0.006 * max(tail_factor - 1.0, 0.0),
            0.84,
            1.20,
        )
        bess_multiplier = np.clip(
            1.0
            + 0.20 * (duration_stress - 1.0)
            + 0.12 * (hours_stress - 1.0)
            + 0.07 * (frequency_stress - 1.0)
            + 0.018 * max(tail_factor - 1.0, 0.0) * (duration_stress - 1.0),
            0.52,
            1.75,
        )
        pv_capacity = float(reference_pv * pv_multiplier)
        battery_energy = float(reference_bess * bess_multiplier)
        mean_load = float(reference["mean_hourly_load_kw"])
        tail_penalty = 1.0 + 0.34 * max(tail_factor - 1.0, 0.0)
        outage_load_reference = max(mean_load * mean_hours * tail_penalty, 1e-9)
        outage_load = max(mean_load * annual_hours * tail_penalty, 1e-9)
        deficit_scale = max(1.0, mean_load * max(max_duration, 0.10) * (0.75 + 0.45 * tail_factor))
        pv_ratio = pv_capacity / max(reference_pv, 1e-9)
        battery_ratio = battery_energy / max(reference_bess, 1e-9)
        pv_resilience_credit = 1.0 - min(0.34, 0.050 * pv_ratio * float(city["pv_yield_index"]))
        physical_ue = outage_load * math.exp(-battery_energy / deficit_scale) * pv_resilience_credit
        reference_scaled_ue = (
            max(reference_ue, 1e-6)
            * max(hours_stress, 0.0) ** 1.05
            * max(duration_stress, 0.25) ** 0.32
            * math.exp(-0.22 * (battery_ratio - 1.0))
            * math.exp(-0.05 * (pv_ratio - 1.0))
        )
        ue = min(outage_load * 0.98, max(0.0, 0.55 * physical_ue + 0.45 * reference_scaled_ue))
        energy_based_lolp = float(np.clip(ue / outage_load, 0.0, 1.0))
        rows.append(
            {
                "city_key": city_key,
                "city_label": scenario["city_label"],
                "scenario_id": int(scenario["scenario_id"]),
                "pv_capacity_kw": pv_capacity,
                "battery_energy_kwh": battery_energy,
                "ue_kwh_per_year": float(ue),
                "energy_based_lolp": energy_based_lolp,
                "energy_based_lolp_percent": 100.0 * energy_based_lolp,
                "annual_outage_hours": annual_hours,
                "simulated_saifi_per_year": simulated_saifi,
                "max_event_duration_hours": max_duration,
                "hours_stress": float(hours_stress),
                "frequency_stress": float(frequency_stress),
                "duration_stress": float(duration_stress),
                "combined_stress": float(combined_stress),
                "outage_tail_factor": tail_factor,
                "plot_jitter": float(rng.uniform(-0.16, 0.16)),
                "reference_pv_capacity_kw": reference_pv,
                "reference_battery_energy_kwh": reference_bess,
                "reference_ue_kwh_per_year": reference_ue,
                "reference_energy_based_lolp": float(reference["energy_based_lolp"]),
                "reference_outage_load_kwh_per_year": outage_load_reference,
                "scenario_outage_load_kwh_per_year": outage_load,
                "data_status": DATA_STATUS,
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _build_operation_profiles(
    city_profiles: dict[str, pd.DataFrame],
    design_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for city_key, duration_hours, initial_soc in (("Singapore", 1.0, 0.64), ("Metro_Manila", 10.0, 0.90)):
        profile = city_profiles[city_key]
        hourly = profile.groupby("hour", as_index=False).agg(load_kw=("load_kwh", "mean"), pv_kw_per_kw=("pv_kwh_per_kw", "mean"))
        design = design_results[
            (design_results["city_key"].astype(str) == city_key)
            & (design_results["design_case"].astype(str) == "resilience_aware")
        ].iloc[0]
        pv_kw = float(design["pv_capacity_kw"])
        battery_kwh = float(design["battery_energy_kwh"])
        min_soc = 0.10
        max_soc = 0.90
        soc_kwh = initial_soc * battery_kwh
        records: list[dict[str, Any]] = []
        for relative_hour in range(-12, 25):
            hour = (18 + relative_hour) % 24
            row = hourly[hourly["hour"].astype(int) == hour].iloc[0]
            load_kw = float(row["load_kw"])
            pv_available_kw = float(row["pv_kw_per_kw"]) * pv_kw
            in_outage = 0 <= relative_hour < duration_hours
            target_soc = max_soc * battery_kwh if relative_hour < 0 else 0.72 * battery_kwh
            pv_supply = min(load_kw, pv_available_kw)
            battery_discharge = 0.0
            grid_supply = 0.0
            unserved = 0.0
            battery_charge = 0.0
            if in_outage:
                deficit = max(0.0, load_kw - pv_supply)
                available = max(0.0, soc_kwh - min_soc * battery_kwh)
                battery_discharge = min(deficit, available, 0.5 * battery_kwh)
                soc_kwh -= battery_discharge
                unserved = max(0.0, deficit - battery_discharge)
            else:
                surplus_pv = max(0.0, pv_available_kw - pv_supply)
                reserve_gap = max(0.0, target_soc - soc_kwh)
                battery_charge = min(reserve_gap, surplus_pv + 0.18 * battery_kwh, 0.5 * battery_kwh)
                soc_kwh = min(max_soc * battery_kwh, soc_kwh + battery_charge * 0.90)
                grid_supply = max(0.0, load_kw - pv_supply + max(0.0, battery_charge - surplus_pv))
            records.append(
                {
                    "city_key": city_key,
                    "city_label": str(design["city_label"]),
                    "relative_hour": relative_hour,
                    "outage_start_relative_hour": 0,
                    "outage_end_relative_hour": duration_hours,
                    "grid_available": 0 if in_outage else 1,
                    "load_kw": load_kw,
                    "pv_supply_kw": pv_supply,
                    "battery_discharge_kw": battery_discharge,
                    "battery_charge_kw": battery_charge,
                    "grid_supply_kw": grid_supply,
                    "unserved_kw": unserved,
                    "battery_soc_fraction": soc_kwh / max(battery_kwh, 1e-9),
                    "battery_energy_kwh": battery_kwh,
                    "pv_capacity_kw": pv_kw,
                    "data_status": DATA_STATUS,
                }
            )
        rows.append(pd.DataFrame(records))
    return pd.concat(rows, ignore_index=True)


def _build_city_operation_profiles(
    city_profiles: dict[str, pd.DataFrame],
    city_inputs: pd.DataFrame,
    design_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    city_lookup = city_inputs.set_index(city_inputs["city_key"].astype(str))
    max_saidi = max(float(city_inputs["saidi_min_per_year"].max()), 1.0)
    max_saifi = max(float(city_inputs["saifi_per_year"].max()), 1.0)
    for city_key in _city_order_from(city_inputs):
        city = city_lookup.loc[city_key]
        profile = city_profiles[city_key].sort_values("hour_index").reset_index(drop=True)
        peak_month = int(profile.groupby("month")["load_kwh"].sum().idxmax())
        peak_profile = profile[profile["month"].astype(int) == peak_month]
        outage_anchor_index = int(peak_profile.loc[peak_profile["load_kwh"].idxmax(), "hour_index"])
        outage_start_window_hour = 36
        window = _profile_hour_window(profile, outage_anchor_index - outage_start_window_hour, 72)
        design = design_results[
            (design_results["city_key"].astype(str) == city_key)
            & (design_results["design_case"].astype(str) == "resilience_aware")
        ].iloc[0]
        saidi = float(city["saidi_min_per_year"])
        saifi = float(city["saifi_per_year"])
        tail_factor = float(city["outage_tail_factor"])
        outage_duration = _representative_outage_duration(saidi, saifi, tail_factor)
        risk_soc = float(
            np.clip(
                0.56
                + 0.16 * math.log1p(saidi) / math.log1p(max_saidi)
                + 0.10 * math.log1p(saifi) / math.log1p(max_saifi)
                + 0.035 * max(tail_factor - 1.0, 0.0),
                0.56,
                0.92,
            )
        )
        start_soc = max(0.44, risk_soc - 0.16)
        rows.append(
            _dispatch_representative_outage(
                window,
                design,
                city,
                peak_month=peak_month,
                outage_start_window_hour=outage_start_window_hour,
                outage_duration_hours=outage_duration,
                target_soc_fraction=risk_soc,
                initial_soc_fraction=start_soc,
            )
        )
    return pd.concat(rows, ignore_index=True)


def _profile_hour_window(profile: pd.DataFrame, start_hour_index: int, hours: int) -> pd.DataFrame:
    profile_by_hour = profile.set_index(profile["hour_index"].astype(int), drop=False)
    total_hours = len(profile_by_hour)
    indices = [int((start_hour_index + offset) % total_hours) for offset in range(hours)]
    return profile_by_hour.loc[indices].reset_index(drop=True)


def _representative_outage_duration(saidi_min_per_year: float, saifi_per_year: float, tail_factor: float) -> float:
    del tail_factor
    if saifi_per_year <= 0.0:
        return 0.25
    average_duration = (float(saidi_min_per_year) / 60.0) / float(saifi_per_year)
    return float(np.clip(average_duration, 0.25, 2.0))


def _dispatch_representative_outage(
    window: pd.DataFrame,
    design: pd.Series,
    city: pd.Series,
    *,
    peak_month: int,
    outage_start_window_hour: int,
    outage_duration_hours: float,
    target_soc_fraction: float,
    initial_soc_fraction: float,
) -> pd.DataFrame:
    pv_kw = float(design["pv_capacity_kw"])
    battery_kwh = float(design["battery_energy_kwh"])
    min_soc = 0.10
    max_soc = 0.92
    charge_efficiency = 0.90
    normal_discharge_cap = 0.24 * battery_kwh
    outage_discharge_cap = 0.50 * battery_kwh
    hourly_charge_cap = 0.30 * battery_kwh
    tou_prices = window["tou_sgd_per_kwh"].astype(float).to_numpy()
    low_price_threshold = float(np.quantile(tou_prices, 0.35))
    high_price_threshold = float(np.quantile(tou_prices, 0.70))
    tou_spread = float(max(tou_prices) - min(tou_prices))
    reliability_buffer = max(0.0, target_soc_fraction - 0.56)
    economic_floor_fraction = float(np.clip(0.18 + 0.48 * reliability_buffer, min_soc, target_soc_fraction))
    soc_kwh = initial_soc_fraction * battery_kwh
    records: list[dict[str, Any]] = []
    for window_hour, row in window.reset_index(drop=True).iterrows():
        relative_hour = int(window_hour - outage_start_window_hour)
        clock_hour = int(row["hour"])
        load_kw = float(row["load_kwh"])
        pv_available_kw = float(row["pv_kwh_per_kw"]) * pv_kw
        tou_price = float(row["tou_sgd_per_kwh"])
        is_low_price = tou_price <= low_price_threshold + 1e-12
        is_high_price = tou_price >= high_price_threshold - 1e-12
        tou_period = "low" if is_low_price else "high" if is_high_price else "shoulder"
        in_outage = float(outage_start_window_hour) <= float(window_hour) < float(outage_start_window_hour) + outage_duration_hours
        hours_to_outage = float(outage_start_window_hour - window_hour)
        hours_after_outage = float(window_hour - (float(outage_start_window_hour) + outage_duration_hours))
        in_pre_outage_reserve = 0.0 < hours_to_outage <= 12.0
        in_post_outage_recovery = 0.0 <= hours_after_outage <= 12.0
        reserve_floor_fraction = economic_floor_fraction
        target_fraction = economic_floor_fraction
        if in_pre_outage_reserve:
            reserve_floor_fraction = max(reserve_floor_fraction, target_soc_fraction - 0.04)
            target_fraction = target_soc_fraction
        elif in_post_outage_recovery:
            reserve_floor_fraction = max(reserve_floor_fraction, max(0.50, target_soc_fraction - 0.16))
            target_fraction = max(0.58, target_soc_fraction - 0.08)
        elif is_low_price:
            target_fraction = max(target_fraction, min(max_soc, target_soc_fraction - 0.05))
        elif is_high_price:
            target_fraction = reserve_floor_fraction
        else:
            target_fraction = max(target_fraction, min(0.64, target_soc_fraction - 0.12))
        served_load = 0.0
        pv_to_load = 0.0
        pv_to_battery = 0.0
        grid_to_load = 0.0
        grid_to_battery = 0.0
        battery_discharge = 0.0
        battery_charge = 0.0
        grid_import = 0.0
        unserved = 0.0
        if in_outage:
            pv_to_load = min(load_kw, pv_available_kw)
            deficit = max(0.0, load_kw - pv_to_load)
            available = max(0.0, soc_kwh - min_soc * battery_kwh)
            battery_discharge = min(deficit, available, outage_discharge_cap)
            soc_kwh -= battery_discharge
            unserved = max(0.0, deficit - battery_discharge)
            served_load = load_kw - unserved
            surplus_pv = max(0.0, pv_available_kw - pv_to_load)
            if surplus_pv > 0.0 and soc_kwh < max_soc * battery_kwh:
                pv_to_battery = min(surplus_pv, (max_soc * battery_kwh - soc_kwh) / charge_efficiency, hourly_charge_cap)
                battery_charge = pv_to_battery
                soc_kwh = min(max_soc * battery_kwh, soc_kwh + battery_charge * charge_efficiency)
        else:
            pv_to_load = min(load_kw, pv_available_kw)
            remaining_load = max(0.0, load_kw - pv_to_load)
            served_load = load_kw
            available_above_reserve = max(0.0, soc_kwh - reserve_floor_fraction * battery_kwh)
            if is_high_price and not in_pre_outage_reserve and remaining_load > 0.0:
                battery_discharge = min(remaining_load, available_above_reserve, normal_discharge_cap)
                soc_kwh -= battery_discharge
                remaining_load -= battery_discharge
            grid_to_load = remaining_load

            surplus_pv = max(0.0, pv_available_kw - pv_to_load)
            charge_room = max(0.0, max_soc * battery_kwh - soc_kwh)
            target_gap = max(0.0, target_fraction * battery_kwh - soc_kwh)
            pv_charge_target = charge_room if surplus_pv > 0.0 else 0.0
            pv_to_battery = min(surplus_pv, pv_charge_target / charge_efficiency if charge_efficiency > 0 else 0.0, hourly_charge_cap)
            soc_kwh = min(max_soc * battery_kwh, soc_kwh + pv_to_battery * charge_efficiency)
            charge_room = max(0.0, max_soc * battery_kwh - soc_kwh)
            remaining_charge_cap = max(0.0, hourly_charge_cap - pv_to_battery)
            grid_charge_allowed = bool(is_low_price or in_pre_outage_reserve or in_post_outage_recovery)
            if grid_charge_allowed and target_gap > pv_to_battery * charge_efficiency:
                residual_target_gap = max(0.0, target_fraction * battery_kwh - soc_kwh)
                grid_to_battery = min(remaining_charge_cap, residual_target_gap / charge_efficiency if charge_efficiency > 0 else 0.0, charge_room / charge_efficiency if charge_efficiency > 0 else 0.0)
                soc_kwh = min(max_soc * battery_kwh, soc_kwh + grid_to_battery * charge_efficiency)
            battery_charge = pv_to_battery + grid_to_battery
            grid_import = grid_to_load + grid_to_battery
        pv_supply = pv_to_load + pv_to_battery
        pv_curtailment = max(0.0, pv_available_kw - pv_to_load - pv_to_battery)
        soc_kwh = float(np.clip(soc_kwh, min_soc * battery_kwh, max_soc * battery_kwh))
        records.append(
            {
                "city_key": str(design["city_key"]),
                "city_label": str(design["city_label"]),
                "window_hour": int(window_hour),
                "relative_hour": relative_hour,
                "hour_index": int(row["hour_index"]),
                "clock_hour": clock_hour,
                "representative_month": int(peak_month),
                "outage_start_window_hour": int(outage_start_window_hour),
                "outage_start_relative_hour": 0,
                "outage_end_relative_hour": float(outage_duration_hours),
                "outage_end_window_hour": float(outage_start_window_hour) + float(outage_duration_hours),
                "representative_outage_duration_hours": float(outage_duration_hours),
                "grid_available": 0 if in_outage else 1,
                "tou_sgd_per_kwh": tou_price,
                "tou_period": tou_period,
                "is_low_price": int(is_low_price),
                "is_high_price": int(is_high_price),
                "low_price_threshold_sgd_per_kwh": low_price_threshold,
                "high_price_threshold_sgd_per_kwh": high_price_threshold,
                "tou_spread_sgd_per_kwh": tou_spread,
                "load_kw": load_kw,
                "served_load_kw": served_load,
                "pv_available_kw": pv_available_kw,
                "pv_to_load_kw": pv_to_load,
                "pv_to_battery_kw": pv_to_battery,
                "pv_supply_kw": pv_supply,
                "grid_to_load_kw": grid_to_load,
                "grid_to_battery_kw": grid_to_battery,
                "battery_discharge_kw": battery_discharge,
                "battery_charge_kw": battery_charge,
                "grid_supply_kw": grid_import,
                "grid_import_kw": grid_import,
                "unserved_kw": unserved,
                "pv_curtailment_kw": pv_curtailment,
                "battery_soc_fraction": soc_kwh / max(battery_kwh, 1e-9),
                "reserve_floor_fraction": float(reserve_floor_fraction),
                "target_soc_fraction": float(target_fraction),
                "reliability_target_soc_fraction": float(target_soc_fraction),
                "battery_energy_kwh": battery_kwh,
                "pv_capacity_kw": pv_kw,
                "ue_kwh_in_window": float(unserved),
                "saidi_min_per_year": float(city["saidi_min_per_year"]),
                "saifi_per_year": float(city["saifi_per_year"]),
                "outage_tail_factor": float(city["outage_tail_factor"]),
                "data_status": DATA_STATUS,
            }
        )
    return pd.DataFrame(records)


def _build_sensitivity_sources(
    city_inputs: pd.DataFrame,
    design_results: pd.DataFrame,
    anchors: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    aware = design_results[design_results["design_case"].astype(str) == "resilience_aware"].copy()
    for design in aware.to_dict("records"):
        city = city_inputs[city_inputs["city_key"].astype(str) == str(design["city_key"])].iloc[0]
        base_bess = float(design["battery_energy_kwh"])
        base_ue = float(design["ue_kwh_per_year"])
        for voll in (0.0, 50.0, 100.0, 150.0, 200.0, 300.0):
            rows.append(
                _sensitivity_row(
                    design,
                    "voll_sgd_per_kwh",
                    voll,
                    battery_energy_kwh=max(0.35 * anchors["base_bess_anchor_kwh"], base_bess * (0.35 + 0.65 * (voll / 150.0) ** 0.55) if voll > 0 else 0.35 * anchors["base_bess_anchor_kwh"]),
                    ue_kwh_per_year=base_ue * (1.25 if voll <= 0 else max(0.35, (150.0 / voll) ** 0.18)),
                )
            )
        for capex in (200.0, 400.0, 600.0, 800.0, 1000.0):
            rows.append(
                _sensitivity_row(
                    design,
                    "battery_capex_sgd_per_kwh",
                    capex,
                    battery_energy_kwh=base_bess * (600.0 / capex) ** 0.45,
                    ue_kwh_per_year=base_ue * (capex / 600.0) ** 0.30,
                )
            )
        for multiplier in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            rows.append(
                _sensitivity_row(
                    design,
                    "saidi_multiplier",
                    multiplier,
                    battery_energy_kwh=base_bess * (0.90 + 0.12 * multiplier),
                    ue_kwh_per_year=base_ue * multiplier ** 1.12,
                    saidi_min_per_year=float(city["saidi_min_per_year"]) * multiplier,
                )
            )
    sensitivity = pd.DataFrame(rows)
    regime = _build_regime_map(city_inputs, anchors)
    return sensitivity, regime


def _sensitivity_row(
    design: dict[str, Any],
    sensitivity_type: str,
    x_value: float,
    *,
    battery_energy_kwh: float,
    ue_kwh_per_year: float,
    saidi_min_per_year: float | None = None,
) -> dict[str, Any]:
    return {
        "city_key": design["city_key"],
        "city_label": design["city_label"],
        "sensitivity_type": sensitivity_type,
        "x_value": float(x_value),
        "battery_energy_kwh": float(max(0.0, battery_energy_kwh)),
        "ue_kwh_per_year": float(max(0.0, ue_kwh_per_year)),
        "saidi_min_per_year": saidi_min_per_year,
        "data_status": DATA_STATUS,
    }


def _build_regime_map(city_inputs: pd.DataFrame, anchors: dict[str, float]) -> pd.DataFrame:
    manila = city_inputs[city_inputs["city_key"].astype(str) == "Metro_Manila"].iloc[0]
    data_status = str(manila.get("data_status", DATA_STATUS))
    saidi_norm = math.log1p(float(manila["saidi_min_per_year"])) / math.log1p(120.0)
    rows: list[dict[str, Any]] = []
    for capex in np.linspace(200.0, 1000.0, 9):
        for voll in np.linspace(0.0, 300.0, 11):
            score = (voll / 150.0) * (600.0 / capex) ** 0.7 * (0.4 + saidi_norm)
            if score < 0.35:
                regime = "PV-led / limited storage"
                code = 0
                bess = 0.35 * anchors["base_bess_anchor_kwh"]
            elif score < 0.95:
                regime = "Small PV-BESS"
                code = 1
                bess = anchors["base_bess_anchor_kwh"] * (1.0 + score)
            else:
                regime = "Resilience-oriented PV-BESS"
                code = 2
                bess = anchors["base_bess_anchor_kwh"] * (2.0 + 2.4 * score)
            rows.append(
                {
                    "city_key": "Metro_Manila",
                    "city_label": "Metro Manila",
                    "battery_capex_sgd_per_kwh": float(capex),
                    "voll_sgd_per_kwh": float(voll),
                    "regime": regime,
                    "regime_code": code,
                    "battery_energy_kwh": float(bess),
                    "data_status": data_status,
                }
            )
    return pd.DataFrame(rows)


def _figure_source(sources: dict[str, pd.DataFrame], figure_no: int | str) -> pd.DataFrame:
    if figure_no == FIGURE_7B_ID:
        return _figure_7b_source(sources)
    keys = {
        1: ("city_inputs",),
        2: ("city_inputs", "monthly_profiles", "hourly_profiles"),
        3: ("outage_scenarios", "outage_events"),
        4: ("design_results",),
        5: ("operation_profiles",),
        6: ("sensitivity_curves", "regime_map"),
        7: ("design_results", "scenario_design_results"),
        8: ("city_inputs", "city_operation_profiles"),
        9: ("cost_breakdown",),
    }[figure_no]
    frames = []
    for key in keys:
        frame = sources[key].copy()
        frame.insert(0, "source_table", key)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _figure_7b_source(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    design = sources["design_results"].copy()
    design = design[design["design_case"].astype(str) == "resilience_aware"].copy()
    scenario = sources["scenario_design_results"].copy()
    for frame in (design, scenario):
        frame["pv_capacity_kw"] = frame["pv_capacity_kw"].astype(float)
    design = _apply_figure_7b_battery_adjustments(design)
    scenario = _apply_figure_7b_battery_adjustments(scenario)
    design_columns = [
        "city_key",
        "city_label",
        "design_case",
        "pv_capacity_kw",
        "battery_energy_original_kwh",
        "battery_adjustment_kwh",
        "battery_energy_kwh",
        "ue_kwh_per_year",
        "data_status",
    ]
    scenario_columns = [
        "city_key",
        "city_label",
        "scenario_id",
        "pv_capacity_kw",
        "battery_energy_original_kwh",
        "battery_adjustment_kwh",
        "battery_energy_kwh",
        "ue_kwh_per_year",
        "annual_outage_hours",
        "simulated_saifi_per_year",
        "max_event_duration_hours",
        "plot_jitter",
        "data_status",
    ]
    frames = []
    for source_table, frame, columns in (
        ("design_results", design, design_columns),
        ("scenario_design_results", scenario, scenario_columns),
    ):
        prepared = frame[columns].copy()
        prepared.insert(0, "source_table", source_table)
        frames.append(prepared)
    return pd.concat(frames, ignore_index=True, sort=False)


def _apply_figure_7b_battery_adjustments(frame: pd.DataFrame) -> pd.DataFrame:
    adjusted = frame.copy()
    adjustment = (
        adjusted["city_key"]
        .astype(str)
        .map(FIGURE_7B_BATTERY_ADJUSTMENTS_KWH)
        .fillna(0.0)
        .astype(float)
    )
    adjusted["battery_energy_original_kwh"] = adjusted["battery_energy_kwh"].astype(float)
    adjusted["battery_adjustment_kwh"] = adjustment
    adjusted["battery_energy_kwh"] = np.maximum(0.0, adjusted["battery_energy_original_kwh"] + adjustment)
    return adjusted


def _plot_figure_1(out: Path, case_key: str, sources: dict[str, pd.DataFrame]) -> dict[str, Path]:
    city_inputs = sources["city_inputs"]
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, left=0.08, right=0.98, bottom=0.09, top=0.95, wspace=0.30, hspace=0.35)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_system = fig.add_subplot(grid[0, 1])
    ax_workflow = fig.add_subplot(grid[1, :])

    for row in city_inputs.to_dict("records"):
        color = CITY_COLORS[str(row["city_key"])]
        ax_map.scatter(row["longitude"], row["latitude"], s=38, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax_map.text(row["longitude"] + 1.2, row["latitude"] + 0.8, row["city_label"], fontsize=6.4, color=color)
    ax_map.set_xlim(98, 158)
    ax_map.set_ylim(-32, 19)
    ax_map.set_xlabel("Longitude (deg E)")
    ax_map.set_ylabel("Latitude (deg N)")
    ax_map.set_title("City comparison matrix", loc="left")
    ax_map.grid(color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.5)

    _system_schematic(ax_system)
    _workflow_schematic(ax_workflow)
    _panel_label(ax_map, "a")
    _panel_label(ax_system, "b")
    _panel_label(ax_workflow, "c")
    base = out / f"{case_key}_{FIGURE_NAMES[1]}"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _plot_figure_2(out: Path, case_key: str, sources: dict[str, pd.DataFrame]) -> dict[str, Path]:
    city_inputs = sources["city_inputs"]
    monthly = sources["monthly_profiles"]
    hourly = sources["hourly_profiles"]
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.13, top=0.94, wspace=0.34, hspace=0.36)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    for city_key, group in monthly.groupby("city_key", sort=False):
        color = CITY_COLORS[str(city_key)]
        ax_a.plot(group["month"], group["pv_yield_kwh_per_kw_month"], color=color, label=str(group["city_label"].iloc[0]))
    ax_a.set_xlabel("Month")
    ax_a.set_ylabel("PV yield (kWh kW$^{-1}$ month$^{-1}$)")
    ax_a.set_title("Monthly PV proxy", loc="left")
    ax_a.set_xticks(range(1, 13, 2))

    for city_key, group in hourly.groupby("city_key", sort=False):
        ax_b.plot(group["hour"], group["normalized_load"], color=CITY_COLORS[str(city_key)])
    ax_b.set_xlabel("Hour")
    ax_b.set_ylabel("Normalized load")
    ax_b.set_title("Typical apartment load", loc="left")
    ax_b.set_xticks(range(0, 25, 6))

    for city_key, group in hourly.groupby("city_key", sort=False):
        ax_c.step(group["hour"], group["tou_sgd_per_kwh"], where="mid", color=CITY_COLORS[str(city_key)])
    ax_c.set_xlabel("Hour")
    ax_c.set_ylabel("Tariff (SGD kWh$^{-1}$)")
    ax_c.set_title("TOU proxy", loc="left")
    ax_c.set_xticks(range(0, 25, 6))

    metrics = [
        ("pv_yield_index", "PV\nyield"),
        ("load_index", "Load"),
        ("tariff_spread_sgd_per_kwh", "Tariff\nspread"),
        ("saidi_min_per_year", "SAIDI"),
        ("saifi_per_year", "SAIFI"),
        ("local_voll_proxy_sgd_per_kwh", "VoLL\nproxy"),
    ]
    heat = city_inputs[[metric for metric, _ in metrics]].astype(float)
    normalized = (heat - heat.min(axis=0)) / (heat.max(axis=0) - heat.min(axis=0)).replace(0.0, 1.0)
    image = ax_d.imshow(normalized.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax_d.set_yticks(np.arange(len(city_inputs)))
    ax_d.set_yticklabels(city_inputs["city_label"].astype(str).tolist())
    ax_d.set_xticks(np.arange(len(metrics)))
    ax_d.set_xticklabels([label for _, label in metrics])
    ax_d.set_title("Normalized inputs", loc="left")
    cbar = fig.colorbar(image, ax=ax_d, fraction=0.046, pad=0.03)
    cbar.set_label("0-1 normalized")
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.52, 0.02))
    for label, ax in zip("abcd", axes.flat, strict=True):
        _panel_label(ax, label)
    base = out / f"{case_key}_{FIGURE_NAMES[2]}"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _plot_figure_3(out: Path, case_key: str, sources: dict[str, pd.DataFrame]) -> dict[str, Path]:
    scenarios = sources["outage_scenarios"]
    events = sources["outage_events"]
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.09, top=0.94, wspace=0.34, hspace=0.36)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    _outage_flow(ax_a)
    city_order = _city_order_from(scenarios)
    data = [events.loc[events["city_key"].astype(str) == city, "duration_hours"].astype(float).to_numpy() for city in city_order]
    parts = ax_b.violinplot(data, positions=np.arange(len(city_order)), widths=0.72, showmeans=True, showextrema=False)
    for body, city in zip(parts["bodies"], city_order, strict=True):
        body.set_facecolor(CITY_COLORS[city])
        body.set_edgecolor("white")
        body.set_alpha(0.70)
    parts["cmeans"].set_color(NATURE_PALETTE["neutral_black"])
    parts["cmeans"].set_linewidth(0.8)
    ax_b.set_yscale("log")
    ax_b.set_xticks(np.arange(len(city_order)))
    ax_b.set_xticklabels(_city_labels(scenarios, city_order), rotation=20, ha="right")
    ax_b.set_ylabel("Event duration (h)")
    ax_b.set_title("Outage duration distribution", loc="left")

    validation = scenarios.groupby(["city_key", "city_label"], as_index=False).agg(
        input_saidi=("input_saidi_min_per_year", "mean"),
        simulated_saidi=("simulated_saidi_min_per_year", "mean"),
    )
    max_value = float(max(validation["input_saidi"].max(), validation["simulated_saidi"].max()))
    ax_c.plot([0.0, max_value], [0.0, max_value], color=NATURE_PALETTE["neutral_mid"], linestyle="--", linewidth=0.9)
    for row in validation.to_dict("records"):
        color = CITY_COLORS[str(row["city_key"])]
        ax_c.scatter(row["input_saidi"], row["simulated_saidi"], color=color, s=34, edgecolor="white", linewidth=0.5, zorder=3)
        ax_c.text(row["input_saidi"] + max_value * 0.02, row["simulated_saidi"], row["city_label"], fontsize=6.2, color=color)
    ax_c.set_xlabel("Input SAIDI (min yr$^{-1}$)")
    ax_c.set_ylabel("Simulated mean SAIDI (min yr$^{-1}$)")
    ax_c.set_title("Scenario validation", loc="left")

    annual_data = [
        scenarios.loc[scenarios["city_key"].astype(str) == city, "annual_outage_hours"].astype(float).to_numpy()
        for city in city_order
    ]
    box = ax_d.boxplot(annual_data, patch_artist=True, widths=0.58, showfliers=False)
    for patch, city in zip(box["boxes"], city_order, strict=True):
        patch.set_facecolor(CITY_COLORS[city])
        patch.set_alpha(0.62)
        patch.set_edgecolor("white")
    for median in box["medians"]:
        median.set_color(NATURE_PALETTE["neutral_black"])
        median.set_linewidth(0.9)
    ax_d.set_xticks(np.arange(1, len(city_order) + 1))
    ax_d.set_xticklabels(_city_labels(scenarios, city_order), rotation=20, ha="right")
    ax_d.set_ylabel("Annual outage hours (h yr$^{-1}$)")
    ax_d.set_title("100 synthetic annual scenarios", loc="left")
    for label, ax in zip("abcd", axes.flat, strict=True):
        _panel_label(ax, label)
    base = out / f"{case_key}_{FIGURE_NAMES[3]}"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _plot_figure_4(out: Path, case_key: str, sources: dict[str, pd.DataFrame]) -> dict[str, Path]:
    designs = sources["design_results"].copy()
    city_order = _city_order_from(designs)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.5), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.94, wspace=0.33, hspace=0.38)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    x = np.arange(len(city_order))
    width = 0.36
    for offset, design_case in ((-width / 2, "economic_only"), (width / 2, "resilience_aware")):
        subset = _design_ordered(designs, city_order, design_case)
        ax_a.bar(x + offset, subset["pv_capacity_kw"], width=width, color=DESIGN_COLORS[design_case], label=DESIGN_LABELS[design_case])
        ax_b.bar(x + offset, subset["battery_energy_kwh"], width=width, color=DESIGN_COLORS[design_case])
    for ax in (ax_a, ax_b):
        ax.set_xticks(x)
        ax.set_xticklabels(_city_labels(designs, city_order), rotation=20, ha="right")
    ax_a.set_ylabel("PV capacity (kW)")
    ax_a.set_title("Optimal PV sizing", loc="left")
    ax_b.set_ylabel("BESS energy (kWh)")
    ax_b.set_title("Optimal BESS sizing", loc="left")

    aware = _design_ordered(designs, city_order, "resilience_aware")
    bottom = np.zeros(len(aware))
    components = [
        ("annualized_pv_bess_cost_sgd_per_year", "PV/BESS", NATURE_PALETTE["blue_light"]),
        ("grid_cost_sgd_per_year", "Grid", NATURE_PALETTE["neutral_mid"]),
        ("outage_cost_sgd_per_year", "Outage", NATURE_PALETTE["red_light"]),
    ]
    for column, label, color in components:
        values = aware[column].astype(float).to_numpy() / 1000.0
        ax_c.bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(_city_labels(designs, city_order), rotation=20, ha="right")
    ax_c.set_ylabel("Annualized cost (kSGD yr$^{-1}$)")
    ax_c.set_title("Cost composition", loc="left")

    markers = {"economic_only": "o", "resilience_aware": "s"}
    for design_case, group in designs.groupby("design_case", sort=False):
        for row in group.to_dict("records"):
            ax_d.scatter(
                row["ue_kwh_per_year"],
                row["annualized_total_cost_sgd_per_year"] / 1000.0,
                s=24.0 + 1.8 * float(row["battery_energy_kwh"]),
                color=CITY_COLORS[str(row["city_key"])],
                marker=markers[str(design_case)],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.82,
            )
            if str(design_case) == "resilience_aware":
                ax_d.text(
                    row["ue_kwh_per_year"],
                    row["annualized_total_cost_sgd_per_year"] / 1000.0,
                    "  " + str(row["city_label"]),
                    fontsize=5.8,
                    color=CITY_COLORS[str(row["city_key"])],
                    va="center",
                )
    ax_d.set_xlabel("UE (kWh yr$^{-1}$)")
    ax_d.set_ylabel("Annualized cost (kSGD yr$^{-1}$)")
    ax_d.set_title("Cost-resilience trade-off", loc="left")
    x_min, x_max = ax_d.get_xlim()
    x_span = max(x_max - x_min, 1.0)
    ax_d.set_xlim(max(0.0, x_min - 0.04 * x_span), x_max + 0.22 * x_span)
    handles, labels = ax_a.get_legend_handles_labels()
    cost_handles, cost_labels = ax_c.get_legend_handles_labels()
    fig.legend(handles + cost_handles, labels + cost_labels, loc="lower center", ncol=5, bbox_to_anchor=(0.52, 0.02))
    for label, ax in zip("abcd", axes.flat, strict=True):
        _panel_label(ax, label)
    base = out / f"{case_key}_{FIGURE_NAMES[4]}"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _plot_figure_5(out: Path, case_key: str, sources: dict[str, pd.DataFrame]) -> dict[str, Path]:
    operation = sources["operation_profiles"]
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.2), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.92, wspace=0.24, hspace=0.24)
    city_order = ["Singapore", "Metro_Manila"]
    for col, city_key in enumerate(city_order):
        group = operation[operation["city_key"].astype(str) == city_key].sort_values("relative_hour")
        x = group["relative_hour"].astype(float).to_numpy()
        outage_end = float(group["outage_end_relative_hour"].iloc[0])
        ax_power = axes[0, col]
        ax_soc = axes[1, col]
        _shade_outage(ax_power, 0.0, outage_end)
        _shade_outage(ax_soc, 0.0, outage_end)
        ax_power.stackplot(
            x,
            group["pv_supply_kw"].astype(float),
            group["battery_discharge_kw"].astype(float),
            group["grid_supply_kw"].astype(float),
            group["unserved_kw"].astype(float),
            colors=[NATURE_PALETTE["teal"], NATURE_PALETTE["blue"], NATURE_PALETTE["neutral_mid"], NATURE_PALETTE["red"]],
            labels=["PV", "BESS", "Grid", "Unserved"],
            alpha=0.82,
        )
        ax_power.plot(x, group["load_kw"], color=NATURE_PALETTE["neutral_black"], linewidth=1.1, label="Load")
        ax_power.set_title(str(group["city_label"].iloc[0]), loc="left")
        ax_power.set_ylabel("Power (kW)")
        ax_soc.plot(x, group["battery_soc_fraction"], color=NATURE_PALETTE["blue"], linewidth=1.2)
        ax_soc.axhline(0.10, color=NATURE_PALETTE["red"], linewidth=0.75, linestyle=":")
        ax_soc.set_ylim(0.0, 1.02)
        ax_soc.set_xlabel("Hours from outage start")
        ax_soc.set_ylabel("SOC fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.52, 0.02))
    for label, ax in zip("abcd", axes.flat, strict=True):
        _panel_label(ax, label)
    base = out / f"{case_key}_{FIGURE_NAMES[5]}"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _plot_figure_6(out: Path, case_key: str, sources: dict[str, pd.DataFrame]) -> dict[str, Path]:
    sensitivity = sources["sensitivity_curves"]
    regime = sources["regime_map"]
    mpl, plt = _matplotlib(out)
    _style(mpl)
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.14, top=0.94, wspace=0.40, hspace=0.36)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    _plot_sensitivity_lines(ax_a, sensitivity, "voll_sgd_per_kwh", "VoLL (SGD kWh$^{-1}$)", "BESS energy (kWh)", "VoLL sensitivity", "battery_energy_kwh")
    _plot_sensitivity_lines(ax_b, sensitivity, "battery_capex_sgd_per_kwh", "Battery CAPEX (SGD kWh$^{-1}$)", "BESS energy (kWh)", "Battery-cost sensitivity", "battery_energy_kwh")
    _plot_sensitivity_lines(ax_c, sensitivity, "saidi_multiplier", "SAIDI multiplier", "UE (kWh yr$^{-1}$)", "Outage-intensity sensitivity", "ue_kwh_per_year")

    pivot = regime.pivot(index="voll_sgd_per_kwh", columns="battery_capex_sgd_per_kwh", values="regime_code").sort_index()
    cmap = ListedColormap([NATURE_PALETTE["neutral_light"], NATURE_PALETTE["blue_light"], NATURE_PALETTE["red"]])
    image = ax_d.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
        extent=[
            float(pivot.columns.min()),
            float(pivot.columns.max()),
            float(pivot.index.min()),
            float(pivot.index.max()),
        ],
    )
    ax_d.set_xlabel("Battery CAPEX (SGD kWh$^{-1}$)")
    ax_d.set_ylabel("VoLL (SGD kWh$^{-1}$)")
    ax_d.set_title("Metro Manila design regimes", loc="left")
    cbar = fig.colorbar(image, ax=ax_d, ticks=[0, 1, 2], fraction=0.046, pad=0.03)
    cbar.ax.set_yticklabels(["PV-led", "Small\nPV-BESS", "Resilience\nPV-BESS"])
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.52, 0.02))
    for label, ax in zip("abcd", axes.flat, strict=True):
        _panel_label(ax, label)
    base = out / f"{case_key}_{FIGURE_NAMES[6]}"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _plot_figure_7(out: Path, case_key: str, sources: dict[str, pd.DataFrame]) -> dict[str, Path]:
    scenario_results = sources["scenario_design_results"].copy()
    designs = sources["design_results"].copy()
    city_order = _city_order_from(scenario_results)
    reference = _design_ordered(designs, city_order, "resilience_aware")
    mpl, plt = _matplotlib(out)
    _style(mpl)
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.7), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.30, top=0.83, wspace=0.38)
    metrics = [
        ("pv_capacity_kw", "PV capacity (kW)", "PV size", "pv_capacity_kw"),
        ("battery_energy_kwh", "BESS energy (kWh)", "Battery size", "battery_energy_kwh"),
        ("ue_kwh_per_year", "UE (kWh yr$^{-1}$)", "Unserved energy", "ue_kwh_per_year"),
        ("energy_based_lolp_percent", "Energy-based LOLP (%)", "Energy-based LOLP", "energy_based_lolp"),
    ]
    for label, ax, (scenario_column, ylabel, title, reference_column) in zip("abcd", axes, metrics, strict=True):
        _plot_distribution_metric(ax, scenario_results, reference, city_order, scenario_column, reference_column)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        _panel_label(ax, label)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=NATURE_PALETTE["neutral_mid"],
            markeredgewidth=0.0,
            markersize=4.0,
            alpha=0.70,
            label="100 scenarios",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=NATURE_PALETTE["neutral_black"],
            markeredgewidth=0.8,
            markersize=4.2,
            label="Reference design",
        ),
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center", ncol=2, bbox_to_anchor=(0.55, 0.98))
    base = out / f"{case_key}_{FIGURE_NAMES[7]}"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _plot_figure_7b(out: Path, case_key: str, sources: dict[str, pd.DataFrame]) -> dict[str, Path]:
    scenario_results = sources["scenario_design_results"].copy()
    designs = sources["design_results"].copy()
    scenario_results["pv_capacity_kw"] = scenario_results["pv_capacity_kw"].astype(float)
    scenario_results = _apply_figure_7b_battery_adjustments(scenario_results)
    city_order = _city_order_from(scenario_results)
    reference = _design_ordered(designs, city_order, "resilience_aware")
    reference["pv_capacity_kw"] = reference["pv_capacity_kw"].astype(float)
    reference = _apply_figure_7b_battery_adjustments(reference)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.7), constrained_layout=False)
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.30, top=0.83, wspace=0.40)
    metrics = [
        ("pv_capacity_kw", "PV capacity (kW)", "PV size", "pv_capacity_kw"),
        ("battery_energy_kwh", "BESS energy (kWh)", "Battery size", "battery_energy_kwh"),
        ("ue_kwh_per_year", "UE (kWh yr$^{-1}$)", "Unserved energy", "ue_kwh_per_year"),
    ]
    for label, ax, (scenario_column, ylabel, title, reference_column) in zip("abc", axes, metrics, strict=True):
        _plot_distribution_metric(ax, scenario_results, reference, city_order, scenario_column, reference_column)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        _panel_label(ax, label)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=NATURE_PALETTE["neutral_mid"],
            markeredgewidth=0.0,
            markersize=4.0,
            alpha=0.70,
            label="100 scenarios",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=NATURE_PALETTE["neutral_black"],
            markeredgewidth=0.8,
            markersize=4.2,
            label="Reference design",
        ),
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center", ncol=2, bbox_to_anchor=(0.55, 0.98))
    base = out / f"{case_key}_{FIGURE_7B_NAME}"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _plot_figure_8(out: Path, case_key: str, sources: dict[str, pd.DataFrame]) -> dict[str, Path]:
    operation = sources["city_operation_profiles"].copy()
    city_order = _city_order_from(operation)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    from matplotlib.lines import Line2D

    palette = _outage_balance_palette()
    plot_sources = {
        city_key: _city_operation_to_outage_balance_source(
            operation[operation["city_key"].astype(str) == city_key].sort_values("window_hour")
        )
        for city_key in city_order
    }
    shared_ylim = _shared_city_operation_balance_ylim(pd.concat(plot_sources.values(), ignore_index=True))
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 5.4), sharex=True, sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.070, right=0.925, bottom=0.145, top=0.925, wspace=0.23, hspace=0.33)
    for index, city_key in enumerate(city_order):
        ax = axes.flat[index]
        _plot_city_energy_balance_panel(
            ax,
            plot_sources[city_key],
            palette,
            shared_ylim=shared_ylim,
            show_ylabel=index % 2 == 0,
            show_xlabel=index >= 2,
            show_soc_ylabel=index % 2 == 1,
        )

    legend_handles = _city_energy_balance_legend_handles(plt, palette)
    legend_handles.append(
        Line2D([0], [0], color=palette["soc"], linestyle="-.", linewidth=1.1, label="BESS SOC")
    )
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.52, 0.02),
        handlelength=1.35,
    )
    for label, ax in zip("abcd", axes.flat, strict=True):
        _outage_balance_panel_label(ax, label)
    base = out / f"{case_key}_{FIGURE_NAMES[8]}"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _city_operation_to_outage_balance_source(group: pd.DataFrame) -> pd.DataFrame:
    source = group.sort_values("window_hour").reset_index(drop=True).copy()
    if len(source) != OUTAGE_BALANCE_WINDOW_HOURS:
        raise ValueError(f"Figure 8 requires exactly {OUTAGE_BALANCE_WINDOW_HOURS} hourly records per city.")
    outage_start = float(source["outage_start_window_hour"].iloc[0])
    outage_end = float(source["outage_end_window_hour"].iloc[0])
    adapted = pd.DataFrame(
        {
            "city_key": source["city_key"].astype(str),
            "city_label": source["city_label"].astype(str),
            "plot_hour": source["window_hour"].astype(float),
            "hour_index": source["window_hour"].astype(float),
            "event_start_hour": outage_start,
            "event_end_hour": outage_end,
            "event_duration_hours": outage_end - outage_start,
            "grid_availability": source["grid_available"].astype(float),
            "critical_load_kwh": source["load_kw"].astype(float),
            "pv_available_kwh": source["pv_available_kw"].astype(float),
            "pv_used_kwh": source["pv_supply_kw"].astype(float),
            "grid_import_kwh": source["grid_import_kw"].astype(float),
            "battery_charge_kwh": source["battery_charge_kw"].astype(float),
            "battery_discharge_kwh": source["battery_discharge_kw"].astype(float),
            "served_load_kwh": source["served_load_kw"].astype(float),
            "unmet_load_kwh_equiv": source["unserved_kw"].astype(float),
            "pv_curtailed_kwh": 0.0,
            "conversion_loss_kwh": 0.0,
            "battery_soc_fraction": source["battery_soc_fraction"].astype(float),
        }
    )
    return adapted


def _shared_city_operation_balance_ylim(source: pd.DataFrame) -> tuple[float, float]:
    positive = (
        source["pv_used_kwh"].astype(float)
        + source["grid_import_kwh"].astype(float)
        + source["battery_discharge_kwh"].astype(float)
    )
    negative = (
        source["served_load_kwh"].astype(float)
        + source["battery_charge_kwh"].astype(float)
        + source["unmet_load_kwh_equiv"].astype(float)
    )
    upper = max(float(positive.max()), 1.0)
    lower = -max(float(negative.max()), 1.0)
    pad = 0.14 * max(upper, abs(lower))
    return lower - pad, upper + pad


def _city_energy_balance_legend_handles(plt: Any, palette: dict[str, str]) -> list[Any]:
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette["pv"], edgecolor="#3A3A3A", linewidth=0.25, label="PV used"),
        plt.Rectangle((0, 0), 1, 1, facecolor=palette["grid"], edgecolor="#3A3A3A", linewidth=0.25, label="Grid import"),
        plt.Rectangle((0, 0), 1, 1, facecolor=palette["battery_discharge"], edgecolor="#3A3A3A", linewidth=0.25, label="BESS discharge"),
        plt.Rectangle((0, 0), 1, 1, facecolor=palette["load"], edgecolor="#3A3A3A", linewidth=0.25, label="Served load"),
        plt.Rectangle((0, 0), 1, 1, facecolor=palette["battery_charge"], edgecolor="#3A3A3A", linewidth=0.25, label="BESS charge"),
        plt.Rectangle((0, 0), 1, 1, facecolor=palette["unserved"], edgecolor="#3A3A3A", linewidth=0.25, label="Unserved load"),
    ]
    return handles


def _plot_city_energy_balance_panel(
    ax: Any,
    group: pd.DataFrame,
    palette: dict[str, str],
    *,
    shared_ylim: tuple[float, float],
    show_ylabel: bool,
    show_xlabel: bool,
    show_soc_ylabel: bool,
) -> None:
    x = group["plot_hour"].astype(float).to_numpy()
    x_mid = x + 0.5
    outage_start, outage_end = _plot_outage_balance_bounds(group)
    city_label = str(group["city_label"].iloc[0])
    _shade_outage_balance(ax, outage_start, outage_end)
    _draw_outage_balance_day_boundaries(ax, palette)
    _stack_positive_balance(ax, x, group, palette)
    _stack_negative_balance(ax, x, group, palette)
    ax.axhline(0.0, color=palette["neutral_black"], linewidth=0.75)
    ax.set_xlim(0.0, float(OUTAGE_BALANCE_WINDOW_HOURS))
    ax.set_ylim(shared_ylim)

    ax_soc = ax.twinx()
    ax_soc.plot(x_mid, group["battery_soc_fraction"].astype(float), color=palette["soc"], linestyle="-.", linewidth=1.10, zorder=6)
    ax_soc.set_ylim(_soc_ylim_for_zero_alignment(ax.get_ylim(), soc_upper=1.05))
    ax_soc.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_soc.spines["right"].set_visible(True)
    ax_soc.spines["top"].set_visible(False)
    ax_soc.spines["right"].set_color(palette["neutral_dark"])
    ax_soc.tick_params(axis="y", colors=palette["neutral_dark"], length=2.5, width=0.65)
    if show_soc_ylabel:
        ax_soc.set_ylabel("BESS SOC")
    else:
        ax_soc.set_yticklabels([])

    ax.set_title(city_label, loc="left", pad=4)
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    if show_ylabel:
        ax.set_ylabel("Hourly energy\nbalance (kWh)")
    if show_xlabel:
        ax.set_xlabel("Hour in 72 h window")


def _plot_figure_9(out: Path, case_key: str, sources: dict[str, pd.DataFrame]) -> dict[str, Path]:
    costs = sources["cost_breakdown"].copy()
    city_order = _city_order_from(costs)
    city_labels = _city_labels(costs, city_order)
    annualized = costs[costs["cost_basis"].astype(str) == "annualized"].copy()
    upfront = costs[costs["cost_basis"].astype(str) == "upfront"].copy()
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.7), constrained_layout=False)
    fig.subplots_adjust(left=0.080, right=0.970, bottom=0.175, top=0.930, wspace=0.34, hspace=0.42)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    annual_colors = {
        "pv_capex": "#7EBC67",
        "battery_energy_capex": NATURE_PALETTE["blue_light"],
        "battery_power_capex": NATURE_PALETTE["purple"],
        "pv_om": NATURE_PALETTE["teal_light"],
        "battery_om": NATURE_PALETTE["gold"],
        "electricity_bill": NATURE_PALETTE["neutral_mid"],
        "outage_loss": NATURE_PALETTE["red_light"],
    }
    upfront_colors = {
        "pv_upfront_capex": "#7EBC67",
        "battery_energy_upfront_capex": NATURE_PALETTE["blue_light"],
        "battery_power_upfront_capex": NATURE_PALETTE["purple"],
    }
    grouped_colors = {
        "Initial investment": NATURE_PALETTE["blue_light"],
        "O&M": NATURE_PALETTE["gold"],
        "Electricity": NATURE_PALETTE["neutral_mid"],
        "Reliability": NATURE_PALETTE["red_light"],
    }

    _stack_cost_components(
        ax_a,
        annualized,
        city_order,
        annual_colors,
        value_column="cost_ksgd",
        component_filter=("pv_capex", "battery_energy_capex", "battery_power_capex", "pv_om", "battery_om", "electricity_bill", "outage_loss"),
    )
    ax_a.set_xticks(np.arange(len(city_order)))
    ax_a.set_xticklabels(city_labels, rotation=20, ha="right")
    ax_a.set_ylabel("Annualized cost (kSGD yr$^{-1}$)")
    ax_a.set_title("Annualized cost breakdown", loc="left")

    _stack_cost_components(
        ax_b,
        upfront,
        city_order,
        upfront_colors,
        value_column="cost_ksgd",
        component_filter=("pv_upfront_capex", "battery_energy_upfront_capex", "battery_power_upfront_capex"),
    )
    ax_b.set_xticks(np.arange(len(city_order)))
    ax_b.set_xticklabels(city_labels, rotation=20, ha="right")
    ax_b.set_ylabel("Upfront investment (kSGD)")
    ax_b.set_title("Initial investment breakdown", loc="left")

    share = (
        annualized.groupby(["city_key", "component_group"], as_index=False)["share_of_total_percent"]
        .sum()
        .pivot(index="city_key", columns="component_group", values="share_of_total_percent")
        .reindex(index=city_order, columns=["Initial investment", "O&M", "Electricity", "Reliability"])
        .fillna(0.0)
    )
    image = ax_c.imshow(share.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.0, vmax=max(65.0, float(share.to_numpy(dtype=float).max())))
    ax_c.set_xticks(np.arange(len(share.columns)))
    ax_c.set_xticklabels(list(share.columns), rotation=28, ha="right")
    ax_c.set_yticks(np.arange(len(city_order)))
    ax_c.set_yticklabels(city_labels)
    ax_c.set_title("Annualized cost share", loc="left")
    for row_index in range(len(city_order)):
        for col_index in range(len(share.columns)):
            value = float(share.iloc[row_index, col_index])
            ax_c.text(col_index, row_index, f"{value:.0f}", ha="center", va="center", fontsize=6.1, color=NATURE_PALETTE["neutral_black"])
    cbar = fig.colorbar(image, ax=ax_c, fraction=0.046, pad=0.030)
    cbar.set_label("Share (%)")

    unit = annualized.copy()
    unit["annual_load_kwh"] = unit["annual_load_mwh"].astype(float) * 1000.0
    unit["cost_sgd_per_kwh_load"] = unit["cost_sgd"].astype(float) / unit["annual_load_kwh"].clip(lower=1e-9)
    grouped_unit = (
        unit.groupby(["city_key", "component_group"], as_index=False)["cost_sgd_per_kwh_load"]
        .sum()
        .sort_values("component_group")
    )
    _stack_grouped_cost_components(ax_d, grouped_unit, city_order, grouped_colors)
    ax_d.set_xticks(np.arange(len(city_order)))
    ax_d.set_xticklabels(city_labels, rotation=20, ha="right")
    ax_d.set_ylabel("Cost (SGD kWh$_{load}^{-1}$)")
    ax_d.set_title("Load-normalized annualized cost", loc="left")

    handles, labels = ax_a.get_legend_handles_labels()
    group_handles, group_labels = ax_d.get_legend_handles_labels()
    fig.legend(
        handles + group_handles,
        labels + group_labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.52, 0.025),
        handlelength=1.20,
    )
    for label, ax in zip("abcd", axes.flat, strict=True):
        _panel_label(ax, label)
    base = out / f"{case_key}_{FIGURE_NAMES[9]}"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _stack_cost_components(
    ax: Any,
    frame: pd.DataFrame,
    city_order: Sequence[str],
    colors: dict[str, str],
    *,
    value_column: str,
    component_filter: Sequence[str],
) -> None:
    x = np.arange(len(city_order))
    bottom = np.zeros(len(city_order), dtype=float)
    for component_key in component_filter:
        subset = frame[frame["component_key"].astype(str) == component_key].copy()
        subset = subset.set_index(subset["city_key"].astype(str))
        values = np.array([float(subset.loc[city_key, value_column]) if city_key in subset.index else 0.0 for city_key in city_order])
        label = str(subset["component_label"].iloc[0]) if not subset.empty else component_key
        ax.bar(x, values, bottom=bottom, color=colors[component_key], edgecolor="white", linewidth=0.35, label=label)
        bottom += values
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.55)


def _stack_grouped_cost_components(
    ax: Any,
    frame: pd.DataFrame,
    city_order: Sequence[str],
    colors: dict[str, str],
) -> None:
    x = np.arange(len(city_order))
    bottom = np.zeros(len(city_order), dtype=float)
    for component_group in ("Initial investment", "O&M", "Electricity", "Reliability"):
        subset = frame[frame["component_group"].astype(str) == component_group].copy()
        subset = subset.set_index(subset["city_key"].astype(str))
        values = np.array([float(subset.loc[city_key, "cost_sgd_per_kwh_load"]) if city_key in subset.index else 0.0 for city_key in city_order])
        ax.bar(x, values, bottom=bottom, color=colors[component_group], edgecolor="white", linewidth=0.35, label=component_group)
        bottom += values
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.55)


def _plot_distribution_metric(
    ax: Any,
    scenario_results: pd.DataFrame,
    reference: pd.DataFrame,
    city_order: Sequence[str],
    scenario_column: str,
    reference_column: str,
) -> None:
    positions = np.arange(1, len(city_order) + 1, dtype=float)
    data = [
        scenario_results.loc[scenario_results["city_key"].astype(str) == city, scenario_column].astype(float).to_numpy()
        for city in city_order
    ]
    box = ax.boxplot(
        data,
        positions=positions - 0.12,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": NATURE_PALETTE["neutral_black"], "linewidth": 0.8},
        whiskerprops={"color": NATURE_PALETTE["neutral_mid"], "linewidth": 0.7},
        capprops={"color": NATURE_PALETTE["neutral_mid"], "linewidth": 0.7},
    )
    for patch, city in zip(box["boxes"], city_order, strict=True):
        patch.set_facecolor(CITY_COLORS[city])
        patch.set_edgecolor("white")
        patch.set_alpha(0.62)

    for position, city, values in zip(positions, city_order, data, strict=True):
        color = CITY_COLORS[city]
        group = scenario_results[scenario_results["city_key"].astype(str) == city].sort_values("scenario_id")
        jitter = group["plot_jitter"].astype(float).to_numpy()
        ax.scatter(
            position + jitter,
            group[scenario_column].astype(float).to_numpy(),
            s=7.0,
            color=color,
            alpha=0.55,
            linewidth=0.0,
            zorder=3,
        )
        violin_center = position + 0.22
        parts = ax.violinplot([values], positions=[violin_center], widths=0.42, showmeans=False, showmedians=False, showextrema=False)
        body = parts["bodies"][0]
        vertices = body.get_paths()[0].vertices
        vertices[:, 0] = np.maximum(vertices[:, 0], violin_center)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.34)

    reference_values = []
    for city in city_order:
        row = reference[reference["city_key"].astype(str) == city].iloc[0]
        value = float(row[reference_column])
        if reference_column == "energy_based_lolp":
            value *= 100.0
        reference_values.append(value)
    ax.scatter(
        positions - 0.34,
        reference_values,
        marker="D",
        s=22.0,
        facecolor="white",
        edgecolor=NATURE_PALETTE["neutral_black"],
        linewidth=0.8,
        zorder=5,
    )
    ax.set_xlim(0.45, len(city_order) + 0.72)
    ax.set_xticks(positions)
    ax.set_xticklabels(_city_labels(scenario_results, city_order), rotation=30, ha="right")
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.60)


def _system_schematic(ax: Any) -> None:
    ax.set_axis_off()
    boxes = {
        "PV": (0.18, 0.72),
        "BESS": (0.18, 0.34),
        "Grid": (0.50, 0.86),
        "Apartment\nload": (0.74, 0.58),
        "Unserved\nload": (0.74, 0.24),
    }
    for label, (x, y) in boxes.items():
        color = NATURE_PALETTE["neutral_light"]
        if label == "PV":
            color = NATURE_PALETTE["teal_light"]
        elif label == "BESS":
            color = NATURE_PALETTE["blue_light"]
        elif "Unserved" in label:
            color = NATURE_PALETTE["red_light"]
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": color, "edgecolor": "white", "linewidth": 0.8},
            transform=ax.transAxes,
        )
    _arrow(ax, (0.27, 0.70), (0.65, 0.60))
    _arrow(ax, (0.27, 0.36), (0.65, 0.55))
    _arrow(ax, (0.53, 0.80), (0.67, 0.64))
    _arrow(ax, (0.74, 0.47), (0.74, 0.33), color=NATURE_PALETTE["red"])
    ax.text(0.02, 0.98, "PV-BESS-apartment boundary", transform=ax.transAxes, ha="left", va="top", fontweight="bold")


def _workflow_schematic(ax: Any) -> None:
    ax.set_axis_off()
    steps = [
        "SG Apartment\nload/PV/TOU",
        "City proxy\nmultipliers",
        "SAIDI/SAIFI\nscenarios",
        "Synthetic sizing\nresponses",
        "Figures 1-8\nsource data",
    ]
    x_positions = np.linspace(0.08, 0.92, len(steps))
    for index, (x, text) in enumerate(zip(x_positions, steps, strict=True)):
        ax.text(
            x,
            0.55,
            text,
            ha="center",
            va="center",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": NATURE_PALETTE["neutral_mid"], "linewidth": 0.8},
            transform=ax.transAxes,
        )
        if index < len(steps) - 1:
            _arrow(ax, (x + 0.07, 0.55), (x_positions[index + 1] - 0.07, 0.55), color=NATURE_PALETTE["neutral_dark"])
    ax.text(
        0.5,
        0.17,
        "Temporary synthetic-proxy module: no optimizer rerun and no fake city input folders",
        ha="center",
        va="center",
        fontsize=7,
        color=NATURE_PALETTE["neutral_dark"],
        transform=ax.transAxes,
    )


def _outage_flow(ax: Any) -> None:
    ax.set_axis_off()
    steps = ["SAIDI", "SAIFI", "Duration\nsplit", "Start time", "100 annual\nscenarios"]
    xs = np.linspace(0.08, 0.92, len(steps))
    for index, (x, label) in enumerate(zip(xs, steps, strict=True)):
        ax.text(
            x,
            0.58,
            label,
            ha="center",
            va="center",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": NATURE_PALETTE["neutral_mid"], "linewidth": 0.8},
            transform=ax.transAxes,
        )
        if index < len(steps) - 1:
            _arrow(ax, (x + 0.06, 0.58), (xs[index + 1] - 0.06, 0.58), color=NATURE_PALETTE["neutral_dark"])
    ax.text(0.04, 0.92, "Scenario construction", transform=ax.transAxes, fontweight="bold", va="top")


def _plot_sensitivity_lines(
    ax: Any,
    sensitivity: pd.DataFrame,
    sensitivity_type: str,
    xlabel: str,
    ylabel: str,
    title: str,
    y_column: str,
) -> None:
    selected = sensitivity[sensitivity["sensitivity_type"].astype(str) == sensitivity_type].copy()
    for city_key, group in selected.groupby("city_key", sort=False):
        group = group.sort_values("x_value")
        ax.plot(
            group["x_value"],
            group[y_column],
            color=CITY_COLORS[str(city_key)],
            marker="o",
            linewidth=1.0,
            markersize=2.8,
            label=str(group["city_label"].iloc[0]),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")


def _arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], color: str | None = None) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": color or NATURE_PALETTE["neutral_dark"]},
    )


def _shade_outage(ax: Any, start: float, end: float) -> None:
    ax.axvspan(start, end, color=NATURE_PALETTE["neutral_light"], alpha=0.45, zorder=0)
    ax.axvline(start, color=NATURE_PALETTE["neutral_dark"], linestyle="--", linewidth=0.7)
    ax.axvline(end, color=NATURE_PALETTE["neutral_dark"], linestyle=":", linewidth=0.7)


def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _city_order_from(frame: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(frame["city_key"].astype(str).tolist()))


def _city_labels(frame: pd.DataFrame, city_order: Iterable[str]) -> list[str]:
    labels = {}
    for row in frame[["city_key", "city_label"]].drop_duplicates().to_dict("records"):
        labels[str(row["city_key"])] = str(row["city_label"])
    return [labels[city] for city in city_order]


def _design_ordered(designs: pd.DataFrame, city_order: Sequence[str], design_case: str) -> pd.DataFrame:
    rows = []
    for city in city_order:
        selected = designs[
            (designs["city_key"].astype(str) == city)
            & (designs["design_case"].astype(str) == design_case)
        ]
        if selected.empty:
            raise ValueError(f"Missing design row for {city} / {design_case}")
        rows.append(selected.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _with_city_order(source: pd.DataFrame, city_specs: Sequence[CityComparisonSpec]) -> pd.DataFrame:
    order = {spec.city_key: index for index, spec in enumerate(city_specs)}
    prepared = source.copy()
    prepared["_city_order"] = prepared["city_key"].astype(str).map(order)
    return prepared.sort_values("_city_order").drop(columns="_city_order").reset_index(drop=True)
