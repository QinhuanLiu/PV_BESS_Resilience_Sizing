from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.figures.uncertainty import make_uncertainty_baseline_figures
from src.figures.lolp_zero_uncertainty import (
    make_lolp_zero_uncertainty_figures,
    shared_lolp_zero_distribution_x_limits,
)
from src.figures.lolp_zero_three_year_uncertainty import make_lolp_zero_three_year_uncertainty_figures
from src.figures.variable_uncertainty import make_variable_uncertainty_distribution_figures
from ..io.reporting import write_csv
from src.resilience_frontier import frontier_outage_scenarios_for_csv
from src.variable_uncertainty import variable_uncertainty_outage_scenarios_for_csv


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
    dispatch: pd.DataFrame,
    summary: pd.DataFrame,
    candidate_dispatches: dict[tuple[str, int, float, float], pd.DataFrame],
    sensitivity_parameter: str = "baseline",
    sensitivity_value: float | str = "",
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths["optimal_design"] = write_csv(case_dir / f"{case_key}_Baseline_Optimal_Design.csv", design_rows)
    paths["outage_profile"] = write_csv(case_dir / f"{case_key}_Baseline_Outage_Profile.csv", outage_profile)
    paths["outage_events"] = write_csv(case_dir / f"{case_key}_Baseline_Outage_Events.csv", outage_events)
    paths["dispatch"] = write_csv(case_dir / f"{case_key}_Baseline_Dispatch.csv", dispatch)
    paths["candidate_summary"] = write_csv(
        case_dir / f"{case_key}_Baseline_Candidate_Summary.csv",
        summary,
    )
    clear_candidate_dispatch_dir(case_dir)
    if save_candidate_dispatch_workbooks(config):
        for name, path in write_candidate_dispatch_workbooks(
            case_dir,
            case_key,
            config,
            candidate_dispatches,
            summary,
            sensitivity_parameter=sensitivity_parameter,
            sensitivity_value=sensitivity_value,
        ).items():
            paths[name] = path
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
    keyed = summary.copy()
    mask = (
        (keyed["search_stage"].astype(str) == str(search_stage))
        & (keyed["search_iteration"].astype(int) == int(search_iteration))
        & (keyed["pv_capacity_kw"].astype(float).round(6) == round(float(pv_capacity), 6))
        & (keyed["battery_energy_kwh"].astype(float).round(6) == round(float(battery_energy), 6))
    )
    if not mask.any():
        raise ValueError(
            "Could not find candidate summary row for "
            f"{search_stage} iteration {search_iteration}, PV {pv_capacity}, BESS {battery_energy}."
        )
    return dict(keyed.loc[mask].iloc[0])


def _parameter_row(category: str, parameter: str, value: Any, unit: str = "-", description: str = "") -> dict[str, Any]:
    return {
        "category": category,
        "parameter": parameter,
        "value": value,
        "unit": unit,
        "description": description,
    }


def _candidate_parameter_table(
    case_key: str,
    config: dict[str, Any],
    search_stage: str,
    search_iteration: int,
    pv_capacity: float,
    battery_energy: float,
    candidate: dict[str, Any],
    sensitivity_parameter: str = "baseline",
    sensitivity_value: float | str = "",
) -> pd.DataFrame:
    scenario = config["scenarios"][case_key]
    pv = config["pv"]
    battery = config["battery"]
    economics = config["economics"]
    rows = [
        _parameter_row("case", "case_key", case_key),
        _parameter_row("case", "analysis_objective", "resilience_sizing"),
        _parameter_row("sensitivity", "sensitivity_parameter", sensitivity_parameter),
        _parameter_row("sensitivity", "sensitivity_value", sensitivity_value),
        _parameter_row("search", "search_stage", search_stage),
        _parameter_row("search", "search_iteration", int(search_iteration)),
        _parameter_row("dispatch", "dispatch_backend", candidate.get("dispatch_backend", config.get("dispatch", {}).get("backend", "auto"))),
        _parameter_row("dispatch", "solver_status", candidate.get("solver_status", "")),
        _parameter_row("dispatch", "milp_objective_sgd", candidate.get("milp_objective_sgd"), "SGD"),
        _parameter_row("dispatch", "degradation_penalty_sgd_per_kwh", candidate.get("degradation_penalty_sgd_per_kwh"), "SGD/kWh"),
        _parameter_row("dispatch", "simultaneous_charge_discharge_kwh", candidate.get("simultaneous_charge_discharge_kwh"), "kWh/year"),
        _parameter_row("sizing", "pv_capacity_kw", float(pv_capacity), "kW"),
        _parameter_row("sizing", "battery_energy_kwh", float(battery_energy), "kWh"),
        _parameter_row("sizing", "battery_power_kw", candidate.get("battery_power_kw"), "kW"),
        _parameter_row("sizing", "battery_c_rate", candidate.get("battery_c_rate"), "1/h"),
        _parameter_row("economics", "pv_capex_sgd_per_kw", pv.get("capex_sgd_per_kw"), "SGD/kW"),
        _parameter_row("economics", "battery_capex_sgd_per_kwh", economics.get("battery_capex_sgd_per_kwh"), "SGD/kWh"),
        _parameter_row("economics", "battery_power_capex_sgd_per_kw", economics.get("battery_power_capex_sgd_per_kw"), "SGD/kW"),
        _parameter_row("economics", "battery_om_fraction_per_year", economics.get("battery_om_fraction_per_year"), "fraction/year"),
        _parameter_row("economics", "pv_om_fraction_per_year", pv.get("om_fraction_per_year"), "fraction/year"),
        _parameter_row("economics", "pv_project_years", pv.get("project_years"), "years"),
        _parameter_row("reliability", "saidi_min_per_year", scenario.get("saidi_min_per_year"), "min/year"),
        _parameter_row("reliability", "saifi_per_year", scenario.get("saifi_per_year"), "events/year"),
        _parameter_row("reliability", "voll_sgd_per_kwh", scenario.get("voll_sgd_per_kwh"), "SGD/kWh"),
        _parameter_row("costing", "costing_voll_sgd_per_kwh", scenario.get("voll_sgd_per_kwh"), "SGD/kWh"),
        _parameter_row("technical", "cooling_cop", config["technical"].get("cooling_cop"), "-"),
        _parameter_row("technical", "heating_cop", config["technical"].get("heating_cop"), "-"),
        _parameter_row("technical", "battery_roundtrip_efficiency", battery.get("roundtrip_efficiency"), "fraction"),
        _parameter_row("technical", "battery_soc_min_fraction", battery.get("soc_min_fraction"), "fraction"),
        _parameter_row("technical", "battery_soc_max_fraction", battery.get("soc_max_fraction"), "fraction"),
        _parameter_row("technical", "battery_initial_soc_fraction", battery.get("initial_soc_fraction"), "fraction"),
        _parameter_row("result", "pv_bess_private_annual_cost_sgd", candidate.get("pv_bess_private_annual_cost_sgd"), "SGD/year"),
        _parameter_row("result", "unserved_energy_kwh_per_year", candidate.get("unserved_energy_kwh_per_year"), "kWh/year"),
        _parameter_row("result", "unserved_energy_cost_sgd_per_year", candidate.get("unserved_energy_cost_sgd_per_year"), "SGD/year"),
        _parameter_row("result", "pv_bess_system_annual_cost_sgd", candidate.get("pv_bess_system_annual_cost_sgd"), "SGD/year"),
        _parameter_row("result", "pv_bess_grid_cost_sgd", candidate.get("pv_bess_grid_cost_sgd"), "SGD/year"),
        _parameter_row("result", "annualized_pv_cost_sgd", candidate.get("annualized_pv_cost_sgd"), "SGD/year"),
        _parameter_row("result", "annualized_battery_cost_sgd", candidate.get("annualized_battery_cost_sgd"), "SGD/year"),
        _parameter_row("result", "pv_om_cost_sgd", candidate.get("pv_om_cost_sgd"), "SGD/year"),
        _parameter_row("result", "battery_om_cost_sgd", candidate.get("battery_om_cost_sgd"), "SGD/year"),
    ]
    return pd.DataFrame(rows)


def _dispatch_workbook_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "case_key",
        "analysis_objective",
        "sensitivity_parameter",
        "sensitivity_value",
        "search_stage",
        "search_iteration",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "hour_index",
        "grid_availability",
        "outage_fraction",
        "tou_sgd_per_kwh",
        "total_load_e_equiv_kwh",
        "pv_available_kwh",
        "pv_used_kwh",
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
    candidate_dispatches: dict[tuple[str, int, float, float], pd.DataFrame],
    summary: pd.DataFrame,
    sensitivity_parameter: str = "baseline",
    sensitivity_value: float | str = "",
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    dispatch_dir = case_dir / "Candidate_Dispatch"
    for (search_stage, search_iteration, pv_capacity, battery_energy), dispatch in sorted(candidate_dispatches.items()):
        frame = dispatch.copy()
        frame["case_key"] = case_key
        frame["analysis_objective"] = "resilience_sizing"
        frame["search_stage"] = str(search_stage)
        frame["search_iteration"] = int(search_iteration)
        frame["pv_capacity_kw"] = float(pv_capacity)
        frame["battery_energy_kwh"] = float(battery_energy)
        if sensitivity_parameter != "baseline":
            frame["sensitivity_parameter"] = sensitivity_parameter
            frame["sensitivity_value"] = sensitivity_value
        frame = frame[_dispatch_workbook_columns(frame)]
        candidate = _candidate_summary_row(
            summary,
            search_stage,
            int(search_iteration),
            float(pv_capacity),
            float(battery_energy),
        )
        parameter_table = _candidate_parameter_table(
            case_key,
            config,
            search_stage,
            int(search_iteration),
            float(pv_capacity),
            float(battery_energy),
            candidate,
            sensitivity_parameter=sensitivity_parameter,
            sensitivity_value=sensitivity_value,
        )
        prefix = ""
        key_prefix = f"candidate_dispatch_{search_stage}_i{int(search_iteration)}"
        if sensitivity_parameter != "baseline":
            value_label = _safe_label(sensitivity_value)
            prefix = f"Sensitivity_{sensitivity_parameter}_value_{value_label}_"
            key_prefix = (
                f"candidate_dispatch_{sensitivity_parameter}_{value_label}_"
                f"{search_stage}_i{int(search_iteration)}"
            )
        pv_label = _safe_label(pv_capacity)
        battery_label = _safe_label(battery_energy)
        filename = (
            f"{case_key}_Dispatch_{prefix}{search_stage}_i{int(search_iteration)}_"
            f"PV_{pv_label}kW_BESS_{battery_label}kWh.xlsx"
        )
        key = f"{key_prefix}_pv_{pv_label}_bess_{battery_label}"
        path = dispatch_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="Dispatch_8760", index=False)
            parameter_table.to_excel(writer, sheet_name="Parameters", index=False)
        paths[key] = path
    return paths


def write_uncertainty_baseline_outputs(
    uncertainty_dir: Path,
    case_key: str,
    all_results: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths["uncertainty_baseline_all_results"] = write_csv(
        uncertainty_dir / f"{case_key}_Uncertainty_Baseline_All_Results.csv",
        all_results,
    )
    paths["uncertainty_baseline_outage_scenarios"] = write_csv(
        uncertainty_dir / f"{case_key}_Uncertainty_Baseline_Outage_Scenarios.csv",
        outage_scenarios,
    )
    paths["uncertainty_baseline_design_summary"] = write_csv(
        uncertainty_dir / f"{case_key}_Uncertainty_Baseline_Design_Summary.csv",
        summary,
    )
    paths.update(make_uncertainty_baseline_figures(uncertainty_dir, case_key, all_results))
    return paths


def write_uncertainty_sensitivity_outputs(
    uncertainty_dir: Path,
    case_key: str,
    all_results: pd.DataFrame,
    selected_parameters: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for parameter in selected_parameters:
        (uncertainty_dir / f"{case_key}_Uncertainty_Sensitive_Analysis_{parameter}").mkdir(
            parents=True,
            exist_ok=True,
        )
    paths["uncertainty_sensitivity_all_results"] = write_csv(
        uncertainty_dir / f"{case_key}_Uncertainty_Sensitive_Analysis_All_Results.csv",
        all_results,
    )
    return paths


def write_lolp_zero_uncertainty_outputs(
    uncertainty_dir: Path,
    case_key: str,
    all_results: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
    cv_summary: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths["lolp_zero_uncertainty_all_results"] = write_csv(
        uncertainty_dir / f"{case_key}_LOLP_Zero_Uncertainty_All_Results.csv",
        all_results,
    )
    paths["lolp_zero_uncertainty_outage_scenarios"] = write_csv(
        uncertainty_dir / f"{case_key}_LOLP_Zero_Uncertainty_Outage_Scenarios.csv",
        outage_scenarios,
    )
    paths["lolp_zero_uncertainty_cv_summary"] = write_csv(
        uncertainty_dir / f"{case_key}_LOLP_Zero_Uncertainty_CV_Summary.csv",
        cv_summary,
    )
    figure2_dir = uncertainty_dir.parent / f"{case_key}_Figure2_LOLP_Zero_Three_Year_Uncertainty"
    figure2_stem = f"{case_key}_Figure2_LOLP_Zero_Three_Year_Uncertainty"
    figure2_results_path = figure2_dir / f"{figure2_stem}_All_Results.csv"
    if figure2_results_path.exists():
        figure2_results = pd.read_csv(figure2_results_path)
        x_limits = shared_lolp_zero_distribution_x_limits(all_results, figure2_results)
        paths.update(make_lolp_zero_uncertainty_figures(uncertainty_dir, case_key, all_results, x_limits_by_metric=x_limits))
        make_lolp_zero_three_year_uncertainty_figures(
            figure2_dir,
            case_key,
            figure2_results,
            x_limits_by_metric=x_limits,
        )
    else:
        paths.update(make_lolp_zero_uncertainty_figures(uncertainty_dir, case_key, all_results))
    return paths


def write_lolp_zero_three_year_uncertainty_outputs(
    uncertainty_dir: Path,
    case_key: str,
    all_results: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
    cv_summary: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    stem = f"{case_key}_Figure2_LOLP_Zero_Three_Year_Uncertainty"
    paths["lolp_zero_three_year_uncertainty_all_results"] = write_csv(
        uncertainty_dir / f"{stem}_All_Results.csv",
        all_results,
    )
    paths["lolp_zero_three_year_uncertainty_outage_scenarios"] = write_csv(
        uncertainty_dir / f"{stem}_Outage_Scenarios.csv",
        outage_scenarios,
    )
    paths["lolp_zero_three_year_uncertainty_cv_summary"] = write_csv(
        uncertainty_dir / f"{stem}_CV_Summary.csv",
        cv_summary,
    )
    figure1_dir = uncertainty_dir.parent / f"{case_key}_LOLP_Zero_Uncertainty"
    figure1_results_path = figure1_dir / f"{case_key}_LOLP_Zero_Uncertainty_All_Results.csv"
    if figure1_results_path.exists():
        figure1_results = pd.read_csv(figure1_results_path)
        x_limits = shared_lolp_zero_distribution_x_limits(figure1_results, all_results)
        paths.update(
            make_lolp_zero_three_year_uncertainty_figures(
                uncertainty_dir,
                case_key,
                all_results,
                x_limits_by_metric=x_limits,
            )
        )
        make_lolp_zero_uncertainty_figures(
            figure1_dir,
            case_key,
            figure1_results,
            x_limits_by_metric=x_limits,
        )
    else:
        paths.update(make_lolp_zero_three_year_uncertainty_figures(uncertainty_dir, case_key, all_results))
    return paths


def write_variable_uncertainty_outputs(
    uncertainty_dir: Path,
    case_key: str,
    scenario_optima: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    candidate_scenario_results: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    selected_designs: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths["variable_uncertainty_scenario_optima"] = write_csv(
        uncertainty_dir / f"{case_key}_Variable_Uncertainty_Scenario_Optima.csv",
        scenario_optima,
    )
    paths["variable_uncertainty_outage_scenarios"] = write_csv(
        uncertainty_dir / f"{case_key}_Variable_Uncertainty_Outage_Scenarios.csv",
        variable_uncertainty_outage_scenarios_for_csv(outage_scenarios),
    )
    paths["variable_uncertainty_distribution_summary"] = write_csv(
        uncertainty_dir / f"{case_key}_Variable_Uncertainty_Distribution_Summary.csv",
        distribution_summary,
    )
    paths["variable_uncertainty_candidate_scenario_results"] = write_csv(
        uncertainty_dir / f"{case_key}_Variable_Uncertainty_Candidate_Scenario_Results.csv",
        candidate_scenario_results,
    )
    paths["variable_uncertainty_candidate_summary"] = write_csv(
        uncertainty_dir / f"{case_key}_Variable_Uncertainty_Candidate_Summary.csv",
        candidate_summary,
    )
    paths["variable_uncertainty_selected_designs"] = write_csv(
        uncertainty_dir / f"{case_key}_Variable_Uncertainty_Selected_Designs.csv",
        selected_designs,
    )
    paths.update(
        write_variable_uncertainty_distribution_figure_outputs(
            uncertainty_dir,
            case_key,
            scenario_optima,
            selected_designs,
        )
    )
    return paths


def write_variable_uncertainty_distribution_figure_outputs(
    uncertainty_dir: Path,
    case_key: str,
    scenario_optima: pd.DataFrame,
    selected_designs: pd.DataFrame | None = None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths.update(
        make_variable_uncertainty_distribution_figures(
            uncertainty_dir,
            case_key,
            scenario_optima,
            selected_designs=selected_designs,
        )
    )
    return paths


def write_frontier_outputs(
    frontier_dir: Path,
    case_key: str,
    all_candidates: pd.DataFrame,
    scenario_results: pd.DataFrame,
    optimal_designs: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths["frontier_all_candidates"] = write_csv(
        frontier_dir / f"{case_key}_Frontier_All_Candidates.csv",
        all_candidates,
    )
    paths["frontier_scenario_results"] = write_csv(
        frontier_dir / f"{case_key}_Frontier_Scenario_Results.csv",
        scenario_results,
    )
    paths["frontier_optimal_designs"] = write_csv(
        frontier_dir / f"{case_key}_Frontier_Optimal_Designs.csv",
        optimal_designs,
    )
    paths["frontier_outage_scenarios"] = write_csv(
        frontier_dir / f"{case_key}_Frontier_Outage_Scenarios.csv",
        frontier_outage_scenarios_for_csv(outage_scenarios),
    )
    return paths


def write_representative_sensitivity_outputs(
    sensitivity_dir: Path,
    case_key: str,
    all_results: pd.DataFrame,
    cost_reliability: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths["representative_sensitivity_all_results"] = write_csv(
        sensitivity_dir / f"{case_key}_Representative_Sensitivity_All_Results.csv",
        all_results,
    )
    paths["representative_sensitivity_cost_reliability"] = write_csv(
        sensitivity_dir / f"{case_key}_Representative_Sensitivity_Cost_Reliability.csv",
        cost_reliability,
    )
    return paths
