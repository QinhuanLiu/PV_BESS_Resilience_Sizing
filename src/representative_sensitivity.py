from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .io.data_loader import CaseParameters, InputCase
from .design_selection import available_design_row
from .optimizer import scenario_config
from .outage_evaluation import OutageEvaluationContext, evaluate_outage_candidate, generate_outage_scenarios
from .resilience_metrics import (
    DEFAULT_LOLP_THRESHOLD,
)


CAPEX_PARAMETERS = [
    "pv_capex_sgd_per_kw",
    "battery_capex_sgd_per_kwh",
    "battery_power_capex_sgd_per_kw",
]
CAPEX_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
OUTAGE_MULTIPLIERS = [0.5, 1.0, 2.0, 5.0, 10.0]


def run_representative_sensitivity(
    case: InputCase,
    config: dict[str, Any],
    parameters: CaseParameters,
    optimal_designs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = _select_minimum_reference(optimal_designs)
    seed = representative_outage_seed(parameters, config)
    capex_results = _run_capex_sensitivity(case, config, parameters, reference, seed)
    outage_results = _run_outage_value_sensitivity(case, config, parameters, reference, seed)
    all_results = pd.concat([capex_results, outage_results], ignore_index=True)
    cost_reliability = all_results[
        [
            "case_key",
            "city_key",
            "building_key",
            "sensitivity_family",
            "sensitivity_parameter",
            "sensitivity_label",
            "sensitivity_multiplier",
            "sensitivity_value",
            "reference_pv_capacity_kw",
            "reference_battery_energy_kwh",
            "representative_seed",
            "saidi_min_per_year",
            "voll_sgd_per_kwh",
            "system_total_cost_sgd_per_year",
            "outage_unserved_energy_cost_sgd_per_year",
            "UE_kwh_per_year",
            "outage_load_served_ratio",
            "mean_scenario_LOLP",
            "chance_constraint_probability",
        ]
    ].copy()
    return all_results, cost_reliability


def representative_outage_seed(parameters: CaseParameters, config: dict[str, Any]) -> int:
    base = parameters.base
    value = base.get("frontier_seed_start")
    if value is None:
        value = base.get("outage_seed")
    if value is None:
        value = config.get("outage", {}).get("seed", 42)
    result = int(float(value))
    if result < 0:
        raise ValueError("representative outage seed must be non-negative")
    return result


def _run_capex_sensitivity(
    case: InputCase,
    config: dict[str, Any],
    parameters: CaseParameters,
    reference: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_values = {
        "pv_capex_sgd_per_kw": float(config["pv"]["capex_sgd_per_kw"]),
        "battery_capex_sgd_per_kwh": float(config["economics"]["battery_capex_sgd_per_kwh"]),
        "battery_power_capex_sgd_per_kw": float(config["economics"]["battery_power_capex_sgd_per_kw"]),
    }
    for parameter in CAPEX_PARAMETERS:
        for multiplier in CAPEX_MULTIPLIERS:
            cfg = deepcopy(config)
            value = base_values[parameter] * multiplier
            _apply_parameter_value(cfg, parameter, value)
            row = _evaluate_fixed_design(
                case,
                cfg,
                parameters,
                reference,
                seed,
                sensitivity_family="capex",
                sensitivity_parameter=parameter,
                sensitivity_label=_parameter_label(parameter),
                sensitivity_multiplier=multiplier,
                sensitivity_value=value,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _run_outage_value_sensitivity(
    case: InputCase,
    config: dict[str, Any],
    parameters: CaseParameters,
    reference: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenario = config["scenarios"][case.case_key]
    base_saidi = float(scenario["saidi_min_per_year"])
    base_voll = float(scenario["voll_sgd_per_kwh"])
    for saidi_multiplier in OUTAGE_MULTIPLIERS:
        for voll_multiplier in OUTAGE_MULTIPLIERS:
            cfg = deepcopy(config)
            cfg["scenarios"][case.case_key]["saidi_min_per_year"] = base_saidi * saidi_multiplier
            cfg["scenarios"][case.case_key]["voll_sgd_per_kwh"] = base_voll * voll_multiplier
            row = _evaluate_fixed_design(
                case,
                cfg,
                parameters,
                reference,
                seed,
                sensitivity_family="saidi_voll",
                sensitivity_parameter="saidi_min_per_year__voll_sgd_per_kwh",
                sensitivity_label=f"{saidi_multiplier:g}x SAIDI, {voll_multiplier:g}x VoLL",
                sensitivity_multiplier=np.nan,
                sensitivity_value=np.nan,
            )
            row["saidi_multiplier"] = saidi_multiplier
            row["voll_multiplier"] = voll_multiplier
            rows.append(row)
    return pd.DataFrame(rows)


def _evaluate_fixed_design(
    case: InputCase,
    config: dict[str, Any],
    parameters: CaseParameters,
    reference: dict[str, Any],
    seed: int,
    sensitivity_family: str,
    sensitivity_parameter: str,
    sensitivity_label: str,
    sensitivity_multiplier: float,
    sensitivity_value: float,
    include_fixed_costs: bool = True,
) -> dict[str, Any]:
    cfg = scenario_config(config, case.case_key)
    scenario = cfg["active_scenario"]
    outage_row = generate_outage_scenarios(
        case,
        cfg,
        scenario_count=1,
        seed_start=seed,
        scenario_id_column="representative_scenario_id",
    ).iloc[0]
    pv_capacity = float(reference["pv_capacity_kw"])
    battery_energy = float(reference["battery_energy_kwh"])
    context = OutageEvaluationContext(case, cfg, pv_capacity, retain_dispatch=True)
    evaluation = evaluate_outage_candidate(
        context,
        outage_row["_outage_profile"],
        battery_energy,
        require_dispatch=True,
    )
    result = evaluation.result
    scenario_ue = evaluation.metrics["UE_kwh_per_year"]
    scenario_lolp_value = evaluation.metrics["scenario_LOLP"]
    outage_cost = evaluation.metrics["unserved_energy_cost_sgd_per_year"]
    bill_cost = float(result["pv_bess_grid_cost_sgd"])
    net_cost = bill_cost + outage_cost
    system_total = float(result["pv_bess_system_annual_cost_sgd"]) if include_fixed_costs else net_cost
    threshold = float(parameters.base.get("lolp_threshold") or DEFAULT_LOLP_THRESHOLD)
    return {
        "case_key": case.case_key,
        "city_key": case.city_key,
        "building_key": case.building_key,
        "sensitivity_family": sensitivity_family,
        "sensitivity_parameter": sensitivity_parameter,
        "sensitivity_label": sensitivity_label,
        "sensitivity_multiplier": sensitivity_multiplier,
        "sensitivity_value": sensitivity_value,
        "reference_design_type": str(reference["design_type"]),
        "reference_pv_capacity_kw": pv_capacity,
        "reference_pv_capacity_mw": pv_capacity / 1000.0,
        "reference_battery_energy_kwh": battery_energy,
        "reference_battery_energy_mwh": battery_energy / 1000.0,
        "reference_battery_power_kw": float(result["battery_power_kw"]),
        "representative_seed": seed,
        "saifi_per_year": float(scenario["saifi_per_year"]),
        "saidi_min_per_year": float(scenario["saidi_min_per_year"]),
        "voll_sgd_per_kwh": float(scenario["voll_sgd_per_kwh"]),
        "pv_capex_sgd_per_kw": float(cfg["pv"]["capex_sgd_per_kw"]),
        "battery_capex_sgd_per_kwh": float(cfg["economics"]["battery_capex_sgd_per_kwh"]),
        "battery_power_capex_sgd_per_kw": float(cfg["economics"]["battery_power_capex_sgd_per_kw"]),
        "normal_operation_bill_cost_sgd_per_year": bill_cost,
        "outage_unserved_energy_cost_sgd_per_year": outage_cost,
        "net_cost_sgd_per_year": net_cost,
        "system_total_cost_sgd_per_year": system_total,
        "fixed_pv_bess_cost_sgd_per_year": system_total - net_cost if include_fixed_costs else 0.0,
        "UE_kwh_per_year": scenario_ue,
        "outage_load_served_ratio": evaluation.metrics["outage_load_served_ratio"],
        "mean_scenario_LOLP": scenario_lolp_value,
        "chance_constraint_probability": 1.0 if scenario_lolp_value <= threshold else 0.0,
        "lolp_threshold": threshold,
        "simulated_saidi_min_per_year": float(outage_row["simulated_saidi_min_per_year"]),
        "simulated_saifi_per_year": float(outage_row["simulated_saifi_per_year"]),
    }


def _select_minimum_reference(optimal_designs: pd.DataFrame) -> dict[str, Any]:
    return available_design_row(
        optimal_designs,
        "minimum_total_cost_design",
        missing_message="Representative sensitivity requires minimum_total_cost_design from frontier outputs",
        unavailable_message="minimum_total_cost_design is unavailable in frontier outputs",
        required_columns=("pv_capacity_kw", "battery_energy_kwh"),
    )


def _apply_parameter_value(config: dict[str, Any], parameter: str, value: float) -> None:
    if parameter == "pv_capex_sgd_per_kw":
        config["pv"]["capex_sgd_per_kw"] = float(value)
    elif parameter == "battery_capex_sgd_per_kwh":
        config["economics"]["battery_capex_sgd_per_kwh"] = float(value)
    elif parameter == "battery_power_capex_sgd_per_kw":
        config["economics"]["battery_power_capex_sgd_per_kw"] = float(value)
    else:
        raise ValueError(f"Unsupported representative sensitivity parameter: {parameter}")


def _parameter_label(parameter: str) -> str:
    labels = {
        "pv_capex_sgd_per_kw": "PV CAPEX",
        "battery_capex_sgd_per_kwh": "Battery energy CAPEX",
        "battery_power_capex_sgd_per_kw": "Battery power CAPEX",
    }
    return labels.get(parameter, parameter)
