from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..io.reporting import write_csv


DispatchKey = tuple[str, int, float, float]


def save_candidate_dispatch_workbooks(config: dict[str, Any]) -> bool:
    return bool(config.get("outputs", {}).get("save_candidate_dispatch_workbooks", False))


def clear_candidate_dispatch_dir(parent_dir: Path) -> None:
    candidate_dispatch_dir = parent_dir / "Candidate_Dispatch"
    if candidate_dispatch_dir.exists():
        shutil.rmtree(candidate_dispatch_dir)


def write_case_outputs(
    case_dir: Path,
    case_key: str,
    config: dict[str, Any],
    design_rows: pd.DataFrame,
    outage_profile: pd.DataFrame,
    outage_events: pd.DataFrame,
    outage_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    dispatch: pd.DataFrame,
    summary: pd.DataFrame,
    candidate_dispatches: dict[DispatchKey, pd.DataFrame],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths["optimal_design"] = write_csv(case_dir / f"{case_key}_Baseline_Optimal_Design.csv", design_rows)
    paths["outage_profile"] = write_csv(case_dir / f"{case_key}_Baseline_Outage_Profile.csv", outage_profile)
    paths["outage_events"] = write_csv(case_dir / f"{case_key}_Baseline_Outage_Events.csv", outage_events)
    paths["outage_summary"] = write_csv(case_dir / f"{case_key}_Baseline_Outage_Summary.csv", outage_summary)
    paths["yearly_summary"] = write_csv(case_dir / f"{case_key}_Baseline_Yearly_Summary.csv", yearly_summary)
    paths["dispatch"] = write_csv(case_dir / f"{case_key}_Baseline_Dispatch.csv", dispatch)
    paths["candidate_summary"] = write_csv(case_dir / f"{case_key}_Baseline_Candidate_Summary.csv", summary)
    clear_candidate_dispatch_dir(case_dir)
    if save_candidate_dispatch_workbooks(config):
        paths.update(write_candidate_dispatch_workbooks(case_dir, case_key, config, candidate_dispatches, summary))
    return paths


def _safe_label(value: float | str) -> str:
    try:
        text = f"{float(value):.6f}"
    except (TypeError, ValueError):
        text = str(value)
    return text.rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _candidate_summary_row(
    summary: pd.DataFrame,
    search_stage: str,
    search_iteration: int,
    pv_capacity: float,
    battery_energy: float,
) -> dict[str, Any]:
    mask = (
        (summary["search_stage"].astype(str) == str(search_stage))
        & (summary["search_iteration"].astype(int) == int(search_iteration))
        & (summary["pv_capacity_kw"].astype(float).round(6) == round(float(pv_capacity), 6))
        & (summary["battery_energy_kwh"].astype(float).round(6) == round(float(battery_energy), 6))
    )
    if not mask.any():
        raise ValueError(
            "Could not find candidate summary row for "
            f"{search_stage} iteration {search_iteration}, PV {pv_capacity}, BESS {battery_energy}."
        )
    return dict(summary.loc[mask].iloc[0])


def _parameter_row(category: str, parameter: str, value: Any, unit: str = "-") -> dict[str, Any]:
    return {
        "category": category,
        "parameter": parameter,
        "value": value,
        "unit": unit,
    }


def _candidate_parameter_table(
    case_key: str,
    config: dict[str, Any],
    search_stage: str,
    search_iteration: int,
    pv_capacity: float,
    battery_energy: float,
    candidate: dict[str, Any],
) -> pd.DataFrame:
    scenario = config["scenarios"][case_key]
    pv = config["pv"]
    battery = config["battery"]
    economics = config["economics"]
    rows = [
        _parameter_row("case", "case_key", case_key),
        _parameter_row("case", "analysis_objective", "core_resilience_sizing"),
        _parameter_row("case", "analysis_years", config.get("analysis", {}).get("years", 1), "years"),
        _parameter_row("search", "search_stage", search_stage),
        _parameter_row("search", "search_iteration", int(search_iteration)),
        _parameter_row("dispatch", "dispatch_backend", candidate.get("dispatch_backend", config.get("dispatch", {}).get("backend", "auto"))),
        _parameter_row("dispatch", "solver_status", candidate.get("solver_status", "")),
        _parameter_row("sizing", "pv_capacity_kw", float(pv_capacity), "kW"),
        _parameter_row("sizing", "battery_energy_kwh", float(battery_energy), "kWh"),
        _parameter_row("sizing", "battery_power_kw", candidate.get("battery_power_kw"), "kW"),
        _parameter_row("sizing", "pv_rooftop_area_limit_enabled", pv.get("rooftop_area_limit_enabled", True), "boolean"),
        _parameter_row("sizing", "rooftop_area_m2", pv.get("rooftop_area_m2"), "m2"),
        _parameter_row("sizing", "pv_rooftop_area_utilization_fraction", pv.get("rooftop_area_utilization_fraction"), "fraction"),
        _parameter_row("sizing", "pv_rooftop_capacity_limit_kw", pv.get("rooftop_capacity_limit_kw"), "kW"),
        _parameter_row("sizing", "pv_installed_area_m2", candidate.get("pv_installed_area_m2"), "m2"),
        _parameter_row("sizing", "pv_rooftop_utilization_fraction", candidate.get("pv_rooftop_utilization_fraction"), "fraction"),
        _parameter_row("economics", "pv_capex_sgd_per_kw", pv.get("capex_sgd_per_kw"), "SGD/kW"),
        _parameter_row("economics", "discount_rate", economics.get("discount_rate", 0.0), "fraction/year"),
        _parameter_row("policy", "pv_export_enabled", pv.get("export_enabled", False), "boolean"),
        _parameter_row("economics", "battery_capex_sgd_per_kwh", economics.get("battery_capex_sgd_per_kwh"), "SGD/kWh"),
        _parameter_row("economics", "battery_power_capex_sgd_per_kw", economics.get("battery_power_capex_sgd_per_kw"), "SGD/kW"),
        _parameter_row("economics", "battery_om_fraction_per_year", economics.get("battery_om_fraction_per_year"), "fraction/year"),
        _parameter_row("reliability", "saidi_min_per_year", scenario.get("saidi_min_per_year"), "min/year"),
        _parameter_row("reliability", "saifi_per_year", scenario.get("saifi_per_year"), "events/year"),
        _parameter_row("reliability", "voll_sgd_per_kwh", scenario.get("voll_sgd_per_kwh"), "SGD/kWh"),
        _parameter_row("technical", "cooling_cop", config["technical"].get("cooling_cop"), "-"),
        _parameter_row("technical", "heating_cop", config["technical"].get("heating_cop"), "-"),
        _parameter_row("technical", "battery_roundtrip_efficiency", battery.get("roundtrip_efficiency"), "fraction"),
        _parameter_row("technical", "battery_soc_min_fraction", battery.get("soc_min_fraction"), "fraction"),
        _parameter_row("technical", "battery_soc_max_fraction", battery.get("soc_max_fraction"), "fraction"),
        _parameter_row("technical", "battery_initial_soc_fraction", battery.get("initial_soc_fraction"), "fraction"),
        _parameter_row("result", "pv_capital_recovery_factor", candidate.get("pv_capital_recovery_factor"), "1/year"),
        _parameter_row("result", "battery_capital_recovery_factor", candidate.get("battery_capital_recovery_factor"), "1/year"),
        _parameter_row("result", "pv_bess_private_annual_cost_sgd", candidate.get("pv_bess_private_annual_cost_sgd"), "SGD/year"),
        _parameter_row("result", "pv_export_energy_kwh_per_year", candidate.get("pv_export_energy_kwh_per_year"), "kWh/year"),
        _parameter_row("result", "pv_export_revenue_sgd_per_year", candidate.get("pv_export_revenue_sgd_per_year"), "SGD/year"),
        _parameter_row("result", "pv_bess_net_grid_cost_sgd", candidate.get("pv_bess_net_grid_cost_sgd"), "SGD/year"),
        _parameter_row("result", "unserved_energy_kwh_per_year", candidate.get("unserved_energy_kwh_per_year"), "kWh/year"),
        _parameter_row("result", "unserved_energy_cost_sgd_per_year", candidate.get("unserved_energy_cost_sgd_per_year"), "SGD/year"),
        _parameter_row("result", "pv_bess_system_annual_cost_sgd", candidate.get("pv_bess_system_annual_cost_sgd"), "SGD/year"),
    ]
    return pd.DataFrame(rows)


def _dispatch_workbook_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "case_key",
        "analysis_objective",
        "search_stage",
        "search_iteration",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "hour_index",
        "grid_availability",
        "outage_fraction",
        "tou_sgd_per_kwh",
        "pv_export_sgd_per_kwh",
        "total_load_e_equiv_kwh",
        "pv_available_kwh",
        "pv_used_kwh",
        "pv_export_kwh",
        "pv_curtailed_kwh",
        "grid_import_kwh",
        "battery_charge_kwh",
        "battery_discharge_kwh",
        "battery_soc_kwh",
        "battery_capacity_kwh",
        "unmet_load_kwh_equiv",
    ]
    return [column for column in preferred if column in frame.columns] + [
        column for column in frame.columns if column not in preferred
    ]


def write_candidate_dispatch_workbooks(
    case_dir: Path,
    case_key: str,
    config: dict[str, Any],
    candidate_dispatches: dict[DispatchKey, pd.DataFrame],
    summary: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    dispatch_dir = case_dir / "Candidate_Dispatch"
    for (search_stage, search_iteration, pv_capacity, battery_energy), dispatch in sorted(candidate_dispatches.items()):
        frame = dispatch.copy()
        frame["case_key"] = case_key
        frame["analysis_objective"] = "core_resilience_sizing"
        frame["search_stage"] = str(search_stage)
        frame["search_iteration"] = int(search_iteration)
        frame["pv_capacity_kw"] = float(pv_capacity)
        frame["battery_energy_kwh"] = float(battery_energy)
        frame = frame[_dispatch_workbook_columns(frame)]
        candidate = _candidate_summary_row(summary, search_stage, int(search_iteration), float(pv_capacity), float(battery_energy))
        parameter_table = _candidate_parameter_table(
            case_key,
            config,
            search_stage,
            int(search_iteration),
            float(pv_capacity),
            float(battery_energy),
            candidate,
        )
        pv_label = _safe_label(pv_capacity)
        battery_label = _safe_label(battery_energy)
        filename = (
            f"{case_key}_Dispatch_{search_stage}_i{int(search_iteration)}_"
            f"PV_{pv_label}kW_BESS_{battery_label}kWh.xlsx"
        )
        key = f"candidate_dispatch_{search_stage}_i{int(search_iteration)}_pv_{pv_label}_bess_{battery_label}"
        path = dispatch_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="Dispatch_8760", index=False)
            parameter_table.to_excel(writer, sheet_name="Parameters", index=False)
        paths[key] = path
    return paths
