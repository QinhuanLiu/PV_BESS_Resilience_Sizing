from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from ..io.data_loader import SENSITIVITY_PARAMETERS, build_case_frame, discover_input_cases, load_input_case, sensitivity_runs
from src.figures.baseline_design_space import build_baseline_design_space_source, make_baseline_design_space_figures
from src.figures.baseline_cost_comparison import make_baseline_cost_comparison_figure
from src.figures.baseline_lolp import (
    build_baseline_lolp_bess_slice_source,
    make_baseline_lolp_bess_slice_figure,
)
from src.figures.baseline_outage_balance import make_baseline_outage_balance_comparison_figure
from src.figures.names import RESULT_FIGURE_NUMBERS
from src.figures.registry import make_result_figures
from src.figures.sources import (
    build_result_sources,
    missing_result_sensitivity_parameters,
    read_result_sources,
)
from src.figures.versioning import archive_case_figures
from src.optimizer import resilience_design
from src.outage import simulate_outages, summarize_outages
from ..io.reporting import write_csv
from src.representative_sensitivity import (
    run_representative_sensitivity as run_representative_sensitivity_analysis,
)
from src.resilience_frontier import load_frontier_outputs, run_resilience_frontier
from src.resilience_metrics import electric_equivalent_building_load, outage_weighted_load_kwh
from src.lolp_zero_uncertainty import (
    DEFAULT_LOLP_ZERO_SCENARIO_COUNT,
    MANILA_LOLP_ZERO_CASE_KEY,
    run_lolp_zero_uncertainty_analysis,
)
from src.lolp_zero_three_year_uncertainty import (
    DEFAULT_LOLP_ZERO_THREE_YEAR_SCENARIO_COUNT,
    DEFAULT_LOLP_ZERO_THREE_YEAR_SIMULATION_YEARS,
    run_lolp_zero_three_year_uncertainty_analysis,
)
from src.uncertainty import run_uncertainty_analysis
from src.variable_uncertainty import run_variable_uncertainty_analysis
from . import outputs as workflow_outputs
from .paths import case_output_paths


ORDINARY_SENSITIVITY_PARAMETERS = [
    "battery_capex_sgd_per_kwh",
    "pv_capex_sgd_per_kw",
    "saidi_min_per_year",
    "saifi_per_year",
    "voll_sgd_per_kwh",
]


def _add_outage(
    frame: pd.DataFrame,
    scenario_key: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    sc = config["scenarios"][scenario_key]
    outage_profile, outage_events = simulate_outages(
        saifi_per_year=float(sc["saifi_per_year"]),
        saidi_min_per_year=float(sc["saidi_min_per_year"]),
        hours=len(frame),
        seed=int(config["outage"].get("seed", 42)),
        forced_event_mode=str(config["outage"].get("forced_event_mode", "match_saidi")),
    )
    merged = frame.merge(outage_profile, on="hour_index", how="left")
    return merged, outage_profile, outage_events, summarize_outages(outage_profile, outage_events)


def _annotate_design_rows(
    rows: pd.DataFrame,
    case_key: str,
    config: dict[str, Any],
    outage_summary: dict[str, float],
    outage_weighted_load_total_kwh: float | None = None,
    sensitivity_parameter: str = "baseline",
    sensitivity_value: float | str = "",
) -> pd.DataFrame:
    scenario = config["scenarios"][case_key]
    annotated = rows.copy()
    annotated.insert(0, "sensitivity_value", sensitivity_value)
    annotated.insert(0, "sensitivity_parameter", sensitivity_parameter)
    annotated.insert(0, "building_key", case_key.split("_", 1)[1] if "_" in case_key else "")
    annotated.insert(0, "city_key", case_key.split("_", 1)[0])
    annotated.insert(0, "case_key", case_key)
    annotated["scenario_name"] = scenario["name"]
    annotated["city_or_service_area"] = scenario["city_or_service_area"]
    annotated["saidi_min_per_year"] = scenario["saidi_min_per_year"]
    annotated["saifi_per_year"] = scenario["saifi_per_year"]
    annotated["voll_sgd_per_kwh"] = scenario["voll_sgd_per_kwh"]
    annotated["battery_capex_sgd_per_kwh"] = config["economics"]["battery_capex_sgd_per_kwh"]
    annotated["battery_power_capex_sgd_per_kw"] = config["economics"]["battery_power_capex_sgd_per_kw"]
    annotated["battery_om_fraction_per_year"] = config["economics"]["battery_om_fraction_per_year"]
    annotated["pv_capex_sgd_per_kw"] = config["pv"]["capex_sgd_per_kw"]
    for key, value in outage_summary.items():
        annotated[key] = value
    if "unserved_energy_kwh_per_year" in annotated.columns:
        annotated["UE_kwh_per_year"] = annotated["unserved_energy_kwh_per_year"].astype(float)
        annotated["UE_mwh_per_year"] = annotated["UE_kwh_per_year"] / 1000.0
    if "pv_bess_system_annual_cost_sgd" in annotated.columns:
        annotated["system_total_cost_sgd_per_year"] = annotated["pv_bess_system_annual_cost_sgd"].astype(float)
        annotated["system_total_cost_million_sgd_per_year"] = annotated["system_total_cost_sgd_per_year"] / 1e6
    if outage_weighted_load_total_kwh is not None:
        annotated["outage_weighted_load_kwh"] = float(outage_weighted_load_total_kwh)
        if "UE_kwh_per_year" in annotated.columns:
            if float(outage_weighted_load_total_kwh) <= 0.0:
                annotated["energy_based_LOLP"] = 0.0
            else:
                annotated["energy_based_LOLP"] = annotated["UE_kwh_per_year"] / float(outage_weighted_load_total_kwh)
    return annotated


def _parameter_table(case_key: str, city_key: str, building_key: str, raw: pd.DataFrame) -> pd.DataFrame:
    table = raw.copy()
    table.insert(0, "building_key", building_key)
    table.insert(0, "city_key", city_key)
    table.insert(0, "case_key", case_key)
    return table

def _candidate_summary_uses_greedy(summary: pd.DataFrame) -> bool:
    if summary.empty or "dispatch_backend" not in summary.columns:
        return False
    return set(summary["dispatch_backend"].dropna().astype(str).str.lower().unique().tolist()) == {"greedy"}


def _baseline_lolp_scan_candidate_summary(
    case,
    config: dict[str, Any],
    case_key: str,
    existing_summary: pd.DataFrame,
) -> pd.DataFrame:
    if _candidate_summary_uses_greedy(existing_summary):
        return existing_summary
    greedy_config = deepcopy(config)
    greedy_config.setdefault("dispatch", {})["backend"] = "greedy"
    _, _, _, _, greedy_summary, _ = _run_design_case(case, greedy_config, case_key)
    return greedy_summary


def _filter_cases(cases, city_filters: list[str] | None, case_filters: list[str] | None):
    city_set = set(city_filters or [])
    case_set = set(case_filters or [])
    all_cities = {case.city_key for case in cases}
    all_case_keys = {case.case_key for case in cases}
    missing_cities = sorted(city_set - all_cities)
    missing_cases = sorted(case_set - all_case_keys)
    if missing_cities:
        raise ValueError(f"Unknown city filter(s): {missing_cities}. Available cities: {sorted(all_cities)}")
    if missing_cases:
        raise ValueError(f"Unknown case filter(s): {missing_cases}. Available cases: {sorted(all_case_keys)}")
    selected = [
        case
        for case in cases
        if (not city_set or case.city_key in city_set) and (not case_set or case.case_key in case_set)
    ]
    if not selected:
        raise ValueError("No input cases matched the requested city/case filters.")
    return selected


def _filter_lolp_zero_uncertainty_cases(cases):
    selected = [case for case in cases if case.case_key == MANILA_LOLP_ZERO_CASE_KEY]
    if not selected:
        raise ValueError(f"LOLP=0 uncertainty analysis requires the {MANILA_LOLP_ZERO_CASE_KEY} input case.")
    return selected


def _validate_sensitivity_parameters(
    parameters: list[str] | None,
    allowed_parameters: list[str] | None = None,
) -> list[str] | None:
    if not parameters:
        return None
    allowed_list = allowed_parameters or SENSITIVITY_PARAMETERS
    allowed = set(allowed_list)
    requested = list(dict.fromkeys(parameters))
    invalid = sorted(set(requested) - allowed)
    if invalid:
        raise ValueError(f"Unknown sensitivity parameter(s): {invalid}. Allowed parameters: {allowed_list}")
    return requested


def _sensitivity_runs_for_parameters(parameters, selected_parameters: list[str] | None) -> list[dict[str, Any]]:
    runs = sensitivity_runs(parameters)
    if selected_parameters is None:
        return runs
    selected = set(selected_parameters)
    return [run for run in runs if str(run["sensitivity_parameter"]) in selected]


def _ordinary_sensitivity_runs_for_parameters(parameters, selected_parameters: list[str] | None) -> list[dict[str, Any]]:
    selected = selected_parameters or ORDINARY_SENSITIVITY_PARAMETERS
    return _sensitivity_runs_for_parameters(parameters, selected)


def _merge_sensitivity_results(
    existing_path: Path,
    new_rows: pd.DataFrame,
    selected_parameters: list[str] | None,
) -> pd.DataFrame:
    if selected_parameters is None or not existing_path.exists():
        return new_rows.copy()
    existing = pd.read_csv(existing_path)
    keep = existing[~existing["sensitivity_parameter"].astype(str).isin(set(selected_parameters))].copy()
    if new_rows.empty:
        return keep.reset_index(drop=True)
    return pd.concat([keep, new_rows], ignore_index=True)


def _clear_selected_sensitivity_dirs(sens_dir: Path, case_key: str, selected_parameters: list[str] | None) -> None:
    if selected_parameters is None:
        return
    for parameter in selected_parameters:
        path = sens_dir / f"{case_key}_Sensitive_Analysis_{parameter}"
        if path.exists():
            shutil.rmtree(path)


def _outage_weighted_load_total(frame: pd.DataFrame, outage_profile: pd.DataFrame, config: dict[str, Any]) -> float:
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
    return float(outage_weighted_load_kwh(total_load, outage).sum())


def _run_design_case(
    case,
    config: dict[str, Any],
    case_key: str,
    sensitivity_parameter: str = "baseline",
    sensitivity_value: float | str = "",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, int, float, float], pd.DataFrame],
]:
    preview_frame, _ = build_case_frame(case, config, max(config["pv"]["capacity_kw_candidates"]))
    _, outage_profile, outage_events, outage_summary = _add_outage(preview_frame, case_key, config)
    outage_weighted_total = _outage_weighted_load_total(preview_frame, outage_profile, config)

    def pv_capacity_frame_builder(pv_capacity_kw: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, config, pv_capacity_kw)
        return frame.merge(outage_profile, on="hour_index", how="left"), pv_meta

    rows, dispatch, summary, candidate_dispatches = resilience_design(pv_capacity_frame_builder, config, case_key)
    annotated = _annotate_design_rows(
        rows,
        case_key,
        config,
        outage_summary,
        outage_weighted_load_total_kwh=outage_weighted_total,
        sensitivity_parameter=sensitivity_parameter,
        sensitivity_value=sensitivity_value,
    )
    return annotated, outage_profile, outage_events, dispatch, summary, candidate_dispatches


def run_all(config: dict[str, Any], input_dir: Path, output_dir: Path, run_sensitivity: bool = True) -> dict[str, Path]:
    return run_selected(
        config,
        input_dir,
        output_dir,
        run_baseline=True,
        run_sensitivity=run_sensitivity,
        run_uncertainty_baseline=True,
        run_uncertainty_sensitivity=run_sensitivity,
    )


def run_selected(
    config: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    run_baseline: bool = True,
    run_sensitivity: bool = True,
    run_uncertainty_baseline: bool = False,
    run_uncertainty_sensitivity: bool = False,
    run_lolp_zero_uncertainty: bool = False,
    run_lolp_zero_three_year_uncertainty: bool = False,
    run_variable_uncertainty: bool = False,
    run_variable_uncertainty_figures: bool = False,
    run_non_sg_extrapolated_cost_distribution_figures: bool = False,
    run_frontier: bool = False,
    run_representative_sensitivity: bool = False,
    run_result_figures: bool = False,
    run_city_comparison: bool = False,
    run_single_outage_city_baseline: bool = False,
    run_single_family_preliminary_result: bool = False,
    city_filters: list[str] | None = None,
    case_filters: list[str] | None = None,
    sensitivity_parameters: list[str] | None = None,
    uncertainty_sensitivity_parameters: list[str] | None = None,
    result_figures: list[int] | None = None,
    city_comparison_figures: list[int | str] | None = None,
    single_outage_city_figures: list[int | str] | None = None,
    single_family_preliminary_figures: list[int | str] | None = None,
    city_comparison_source: str = "apartment-synthetic",
    reuse_source_data: bool = False,
    uncertainty_scenario_count: int | None = None,
    uncertainty_seed_start: int | None = None,
    lolp_threshold: float | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if run_non_sg_extrapolated_cost_distribution_figures:
        from src.figures.non_sg_extrapolated_cost_distribution import (
            OUTPUT_DIR_NAME,
            make_non_sg_extrapolated_cost_distribution_figures,
        )

        figure_dir = output_dir / "City_Comparison" / OUTPUT_DIR_NAME
        for name, path in make_non_sg_extrapolated_cost_distribution_figures(
            figure_dir,
            input_root=input_dir,
            output_root=output_dir,
        ).items():
            paths[f"SingleFamily_{name}"] = path
        return paths
    if run_single_outage_city_baseline:
        from src.figures.single_outage_city_baseline import make_single_outage_city_baseline_figures

        single_outage_dir = output_dir / "City_Comparison" / "SingleFamily_Single_Outage_Baseline"
        for name, path in make_single_outage_city_baseline_figures(
            single_outage_dir,
            input_root=input_dir,
            config=config,
            figures=single_outage_city_figures,
        ).items():
            paths[f"SingleFamily_single_outage_city_baseline_{name}"] = path
        return paths
    if run_single_family_preliminary_result:
        from src.figures.singlefamily_preliminary_result import make_singlefamily_preliminary_result_figures

        preliminary_dir = output_dir / "City_Comparison" / "Single Family_Preliminary_Result"
        for name, path in make_singlefamily_preliminary_result_figures(
            preliminary_dir,
            input_root=input_dir,
            config=config,
            figures=single_family_preliminary_figures,
        ).items():
            paths[f"SingleFamily_preliminary_result_{name}"] = path
        return paths
    if run_city_comparison and city_comparison_source == "singlefamily-real":
        from src.figures.city_comparison import (
            SINGLEFAMILY_CASE_KEY,
            build_singlefamily_city_comparison_sources,
            make_city_comparison_figures,
        )

        city_comparison_dir = output_dir / "City_Comparison" / "SingleFamily_City_Comparison"
        sources = build_singlefamily_city_comparison_sources(
            input_dir,
            config=config,
            trace_output_root=output_dir,
        )
        for name, path in make_city_comparison_figures(
            city_comparison_dir,
            input_root=input_dir,
            case_key=SINGLEFAMILY_CASE_KEY,
            figures=city_comparison_figures,
            sources=sources,
        ).items():
            paths[f"{SINGLEFAMILY_CASE_KEY}_city_comparison_{name}"] = path
        return paths
    if run_city_comparison and city_comparison_source != "apartment-synthetic":
        raise ValueError("city_comparison_source must be one of: apartment-synthetic, singlefamily-real")
    cases = _filter_cases(discover_input_cases(input_dir), city_filters, case_filters)
    if run_lolp_zero_uncertainty or run_lolp_zero_three_year_uncertainty:
        cases = _filter_lolp_zero_uncertainty_cases(cases)
    if run_city_comparison:
        apartment_cases = [case for case in cases if case.case_key == "SG_Apartment_Apartment"]
        if not apartment_cases:
            raise ValueError("City comparison requires the SG_Apartment_Apartment input case.")
        cases = apartment_cases
    selected_sensitivity_parameters = (
        _validate_sensitivity_parameters(sensitivity_parameters)
        if run_sensitivity
        else None
    )
    selected_uncertainty_sensitivity_parameters = _validate_sensitivity_parameters(uncertainty_sensitivity_parameters)
    baseline_rows: list[pd.DataFrame] = []
    sensitivity_rows: list[pd.DataFrame] = []

    for case in cases:
        frame, _, case_config, parameters = load_input_case(case, config)
        case_key = case.case_key
        output_paths = case_output_paths(output_dir, case.city_key, case_key)
        base_dir = output_paths.baseline
        sens_dir = output_paths.sensitivity
        uncertainty_baseline_dir = output_paths.uncertainty_baseline
        uncertainty_sensitivity_dir = output_paths.uncertainty_sensitivity
        lolp_zero_uncertainty_dir = output_paths.lolp_zero_uncertainty
        lolp_zero_three_year_uncertainty_dir = output_paths.lolp_zero_three_year_uncertainty
        variable_uncertainty_dir = output_paths.variable_uncertainty
        frontier_dir = output_paths.frontier
        representative_sensitivity_dir = output_paths.representative_sensitivity
        result_dir = output_paths.result_figures
        parameter_table = _parameter_table(case_key, case.city_key, case.building_key, parameters.raw)
        frontier_results: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
        uncertainty_results = pd.DataFrame()
        uncertainty_sensitivity = pd.DataFrame()

        rows = pd.DataFrame()
        case_sensitivity = pd.DataFrame()
        if run_baseline:
            paths[f"{case_key}_baseline_timeseries"] = write_csv(
                base_dir / f"{case_key}_Baseline_Timeseries_Max_PV_Capacity.csv",
                frame,
            )
            paths[f"{case_key}_baseline_parameters"] = write_csv(
                base_dir / f"{case_key}_Baseline_Parameter_Table.csv",
                parameter_table,
            )

            rows, outage_profile, outage_events, dispatch, summary, candidate_dispatches = _run_design_case(
                case,
                case_config,
                case_key,
            )
            baseline_rows.append(rows)
            for name, path in workflow_outputs.write_case_outputs(
                base_dir,
                case_key,
                case_config,
                rows,
                outage_profile,
                outage_events,
                dispatch,
                summary,
                candidate_dispatches,
            ).items():
                paths[f"{case_key}_baseline_{name}"] = path
            baseline_figure_source = build_baseline_design_space_source(
                summary,
                frame,
                outage_profile,
                case_config,
            )
            baseline_figure_dir = output_paths.baseline_result_figures
            for name, path in make_baseline_design_space_figures(
                baseline_figure_dir,
                case_key,
                baseline_figure_source,
            ).items():
                paths[f"{case_key}_baseline_figure3_{name}"] = path
            for name, path in make_baseline_cost_comparison_figure(
                baseline_figure_dir,
                case_key,
                summary,
                case_config,
            ).items():
                paths[f"{case_key}_baseline_figure4_{name}"] = path
            for name, path in make_baseline_outage_balance_comparison_figure(
                baseline_figure_dir,
                case_key,
                case,
                case_config,
                summary,
                outage_profile,
                outage_events,
                optimal_dispatch=dispatch,
            ).items():
                paths[f"{case_key}_baseline_figure5_{name}"] = path
            lolp_scan_summary = _baseline_lolp_scan_candidate_summary(
                case,
                case_config,
                case_key,
                summary,
            )
            lolp_bess_slice_source = build_baseline_lolp_bess_slice_source(
                lolp_scan_summary,
                frame,
                outage_profile,
                case_config,
            )
            for name, path in make_baseline_lolp_bess_slice_figure(
                baseline_figure_dir,
                case_key,
                lolp_bess_slice_source,
            ).items():
                paths[f"{case_key}_baseline_lolp_bess_slices_{name}"] = path

        if run_sensitivity:
            case_sensitivity_rows: list[pd.DataFrame] = []
            cleared_candidate_dispatch_parameters: set[str] = set()
            effective_sensitivity_parameters = selected_sensitivity_parameters or ORDINARY_SENSITIVITY_PARAMETERS
            _clear_selected_sensitivity_dirs(sens_dir, case_key, effective_sensitivity_parameters)
            for run in _ordinary_sensitivity_runs_for_parameters(parameters, selected_sensitivity_parameters):
                parameter = str(run["sensitivity_parameter"])
                value = float(run["sensitivity_value"])
                overrides = {parameter: value}
                _, _, s_config, _ = load_input_case(case, config, parameter_overrides=overrides)
                s_config.setdefault("dispatch", {})["backend"] = "greedy"
                s_rows, _, _, _, s_summary, s_candidate_dispatches = _run_design_case(
                    case,
                    s_config,
                    case_key,
                    sensitivity_parameter=parameter,
                    sensitivity_value=value,
                )
                case_sensitivity_rows.append(s_rows)
                sensitivity_parameter_dir = sens_dir / f"{case_key}_Sensitive_Analysis_{parameter}"
                sensitivity_parameter_dir.mkdir(parents=True, exist_ok=True)
                if parameter not in cleared_candidate_dispatch_parameters:
                    workflow_outputs.clear_candidate_dispatch_dir(sensitivity_parameter_dir)
                    cleared_candidate_dispatch_parameters.add(parameter)
                if workflow_outputs.save_candidate_dispatch_workbooks(s_config):
                    for name, path in workflow_outputs.write_candidate_dispatch_workbooks(
                        sensitivity_parameter_dir,
                        case_key,
                        s_config,
                        s_candidate_dispatches,
                        s_summary,
                        sensitivity_parameter=parameter,
                        sensitivity_value=value,
                    ).items():
                        paths[f"{case_key}_sensitivity_{name}"] = path
            results_path = sens_dir / f"{case_key}_Sensitive_Analysis_All_Results.csv"
            if case_sensitivity_rows:
                new_case_sensitivity = pd.concat(case_sensitivity_rows, ignore_index=True)
            else:
                new_case_sensitivity = pd.DataFrame()

            case_sensitivity = _merge_sensitivity_results(
                results_path,
                new_case_sensitivity,
                selected_sensitivity_parameters,
            )
            if not case_sensitivity.empty:
                sensitivity_rows.append(case_sensitivity)
                paths[f"{case_key}_sensitivity_results"] = write_csv(results_path, case_sensitivity)

        if run_uncertainty_baseline:
            uncertainty_cfg = case_config["uncertainty"]
            uncertainty_results, outage_scenarios, uncertainty_summary = run_uncertainty_analysis(
                case,
                case_config,
                case_key,
                scenario_count=int(uncertainty_cfg["baseline_scenario_count"]),
                seed_start=int(uncertainty_cfg["seed_start"]),
            )
            for name, path in workflow_outputs.write_uncertainty_baseline_outputs(
                uncertainty_baseline_dir,
                case_key,
                uncertainty_results,
                outage_scenarios,
                uncertainty_summary,
            ).items():
                paths[f"{case_key}_{name}"] = path

        if run_uncertainty_sensitivity:
            uncertainty_sensitivity_rows: list[pd.DataFrame] = []
            for run in _sensitivity_runs_for_parameters(parameters, selected_uncertainty_sensitivity_parameters):
                parameter = str(run["sensitivity_parameter"])
                value = float(run["sensitivity_value"])
                _, _, s_config, _ = load_input_case(case, config, parameter_overrides={parameter: value})
                uncertainty_cfg = s_config["uncertainty"]
                s_results, _, _ = run_uncertainty_analysis(
                    case,
                    s_config,
                    case_key,
                    scenario_count=int(uncertainty_cfg["sensitivity_scenario_count"]),
                    seed_start=int(uncertainty_cfg["seed_start"]),
                )
                s_results.insert(0, "sensitivity_value", value)
                s_results.insert(0, "sensitivity_parameter", parameter)
                uncertainty_sensitivity_rows.append(s_results)

            if uncertainty_sensitivity_rows:
                uncertainty_sensitivity = pd.concat(uncertainty_sensitivity_rows, ignore_index=True)
            else:
                uncertainty_sensitivity = pd.DataFrame()
            if not uncertainty_sensitivity.empty:
                for name, path in workflow_outputs.write_uncertainty_sensitivity_outputs(
                    uncertainty_sensitivity_dir,
                    case_key,
                    uncertainty_sensitivity,
                    selected_uncertainty_sensitivity_parameters or list(SENSITIVITY_PARAMETERS),
                ).items():
                    paths[f"{case_key}_{name}"] = path

        if run_lolp_zero_uncertainty:
            lolp_zero_results, lolp_zero_outage_scenarios, lolp_zero_cv_summary = run_lolp_zero_uncertainty_analysis(
                case,
                case_config,
                case_key,
                scenario_count=DEFAULT_LOLP_ZERO_SCENARIO_COUNT,
                seed_start=int(case_config["uncertainty"]["seed_start"]),
            )
            for name, path in workflow_outputs.write_lolp_zero_uncertainty_outputs(
                lolp_zero_uncertainty_dir,
                case_key,
                lolp_zero_results,
                lolp_zero_outage_scenarios,
                lolp_zero_cv_summary,
            ).items():
                paths[f"{case_key}_{name}"] = path

        if run_lolp_zero_three_year_uncertainty:
            (
                lolp_zero_three_year_results,
                lolp_zero_three_year_outage_scenarios,
                lolp_zero_three_year_cv_summary,
            ) = run_lolp_zero_three_year_uncertainty_analysis(
                case,
                case_config,
                case_key,
                scenario_count=DEFAULT_LOLP_ZERO_THREE_YEAR_SCENARIO_COUNT,
                simulation_years=DEFAULT_LOLP_ZERO_THREE_YEAR_SIMULATION_YEARS,
                seed_start=int(case_config["uncertainty"]["seed_start"]),
            )
            for name, path in workflow_outputs.write_lolp_zero_three_year_uncertainty_outputs(
                lolp_zero_three_year_uncertainty_dir,
                case_key,
                lolp_zero_three_year_results,
                lolp_zero_three_year_outage_scenarios,
                lolp_zero_three_year_cv_summary,
            ).items():
                paths[f"{case_key}_{name}"] = path

        if run_variable_uncertainty:
            uncertainty_cfg = case_config["uncertainty"]
            (
                variable_scenario_optima,
                variable_outage_scenarios,
                variable_distribution_summary,
                variable_candidate_scenario_results,
                variable_candidate_summary,
                variable_selected_designs,
            ) = run_variable_uncertainty_analysis(
                case,
                case_config,
                case_key,
                scenario_count=int(
                    uncertainty_scenario_count
                    if uncertainty_scenario_count is not None
                    else uncertainty_cfg["baseline_scenario_count"]
                ),
                seed_start=int(
                    uncertainty_seed_start
                    if uncertainty_seed_start is not None
                    else uncertainty_cfg["seed_start"]
                ),
                lolp_threshold=lolp_threshold,
            )
            for name, path in workflow_outputs.write_variable_uncertainty_outputs(
                variable_uncertainty_dir,
                case_key,
                variable_scenario_optima,
                variable_outage_scenarios,
                variable_distribution_summary,
                variable_candidate_scenario_results,
                variable_candidate_summary,
                variable_selected_designs,
            ).items():
                paths[f"{case_key}_{name}"] = path

        if run_variable_uncertainty_figures:
            scenario_optima_path = (
                variable_uncertainty_dir / f"{case_key}_Variable_Uncertainty_Scenario_Optima.csv"
            )
            selected_designs_path = (
                variable_uncertainty_dir / f"{case_key}_Variable_Uncertainty_Selected_Designs.csv"
            )
            if not scenario_optima_path.exists():
                raise FileNotFoundError(f"Variable uncertainty scenario optima not found: {scenario_optima_path}")
            scenario_optima = pd.read_csv(scenario_optima_path)
            selected_designs = pd.read_csv(selected_designs_path) if selected_designs_path.exists() else pd.DataFrame()
            for name, path in workflow_outputs.write_variable_uncertainty_distribution_figure_outputs(
                variable_uncertainty_dir,
                case_key,
                scenario_optima,
                selected_designs,
            ).items():
                paths[f"{case_key}_{name}"] = path

        if run_frontier:
            frontier_results = run_resilience_frontier(case, case_config, parameters)
            all_candidates, scenario_results, optimal_designs, outage_scenarios = frontier_results
            for name, path in workflow_outputs.write_frontier_outputs(
                frontier_dir,
                case_key,
                all_candidates,
                scenario_results,
                optimal_designs,
                outage_scenarios,
            ).items():
                paths[f"{case_key}_{name}"] = path

        if run_representative_sensitivity:
            if frontier_results is None:
                loaded_frontier = load_frontier_outputs(case, case_config, frontier_dir, case_key)
                if loaded_frontier is None:
                    frontier_results = run_resilience_frontier(case, case_config, parameters)
                    all_candidates, scenario_results, optimal_designs, outage_scenarios = frontier_results
                    for name, path in workflow_outputs.write_frontier_outputs(
                        frontier_dir,
                        case_key,
                        all_candidates,
                        scenario_results,
                        optimal_designs,
                        outage_scenarios,
                    ).items():
                        paths[f"{case_key}_{name}"] = path
                else:
                    optimal_designs, _ = loaded_frontier
            else:
                _, _, optimal_designs, _ = frontier_results

            representative_results = run_representative_sensitivity_analysis(
                case,
                case_config,
                parameters,
                optimal_designs,
            )
            all_results, cost_reliability = representative_results
            for name, path in workflow_outputs.write_representative_sensitivity_outputs(
                representative_sensitivity_dir,
                case_key,
                all_results,
                cost_reliability,
            ).items():
                paths[f"{case_key}_{name}"] = path

        if run_result_figures:
            if reuse_source_data:
                sources = read_result_sources(result_dir, case_key, result_figures)
            else:
                selected_result_figures = set(result_figures or RESULT_FIGURE_NUMBERS)
                needs_frontier_sources = bool(selected_result_figures.intersection({4, 5, 12}))
                if needs_frontier_sources and frontier_results is None:
                    loaded_frontier_candidates = _read_frontier_candidate_results(frontier_dir, case_key)
                    if loaded_frontier_candidates is not None:
                        all_candidates, optimal_designs = loaded_frontier_candidates
                        scenario_results = pd.DataFrame()
                        outage_scenarios = pd.DataFrame()
                    else:
                        frontier_results = run_resilience_frontier(case, case_config, parameters)
                        all_candidates, scenario_results, optimal_designs, outage_scenarios = frontier_results
                        for name, path in workflow_outputs.write_frontier_outputs(
                            frontier_dir,
                            case_key,
                            all_candidates,
                            scenario_results,
                            optimal_designs,
                            outage_scenarios,
                        ).items():
                            paths[f"{case_key}_{name}"] = path
                elif frontier_results is not None:
                    all_candidates, scenario_results, optimal_designs, outage_scenarios = frontier_results
                else:
                    all_candidates = pd.DataFrame()
                    scenario_results = pd.DataFrame()
                    optimal_designs = pd.DataFrame()
                    outage_scenarios = pd.DataFrame()
                if selected_result_figures.intersection({8, 11}) and uncertainty_results.empty:
                    uncertainty_results = _read_uncertainty_baseline_results(uncertainty_baseline_dir, case_key)
                if selected_result_figures.intersection({9, 10, 11}) and uncertainty_sensitivity.empty:
                    uncertainty_sensitivity = _read_uncertainty_sensitivity_results(uncertainty_sensitivity_dir, case_key)
                sources = build_result_sources(
                    case,
                    case_config,
                    parameters,
                    all_candidates,
                    scenario_results,
                    optimal_designs,
                    outage_scenarios,
                    case_sensitivity,
                    uncertainty_results,
                    uncertainty_sensitivity,
                    result_figures,
                    compute_missing_sensitivity_rows=lambda missing: _compute_result_sensitivity_rows(
                        case,
                        config,
                        missing,
                    ),
                    compute_saifi_saidi_grid=lambda: _compute_saifi_saidi_grid_rows(
                        case,
                        config,
                        parameters,
                    ),
                )
            for name, path in make_result_figures(result_dir, case_key, sources, figures=result_figures).items():
                paths[f"{case_key}_result_{name}"] = path

        if run_city_comparison:
            from src.figures.city_comparison import make_city_comparison_figures

            city_comparison_dir = output_dir / "SG_Apartment" / "SG_Apartment_Apartment_City_Comparison"
            for name, path in make_city_comparison_figures(
                city_comparison_dir,
                input_root=input_dir,
                case_key=case_key,
                figures=city_comparison_figures,
            ).items():
                paths[f"{case_key}_city_comparison_{name}"] = path

        if not run_city_comparison:
            for name, path in archive_case_figures(
                output_paths.baseline.parent,
                case_key,
                output_paths.figure_versions,
            ).items():
                paths[f"{case_key}_{name}"] = path

    if baseline_rows:
        _ = pd.concat(baseline_rows, ignore_index=True)
    if sensitivity_rows:
        _ = pd.concat(sensitivity_rows, ignore_index=True)
    return paths


def _build_result_sources(
    case,
    global_config: dict[str, Any],
    case_config: dict[str, Any],
    parameters,
    all_candidates: pd.DataFrame,
    scenario_results: pd.DataFrame,
    optimal_designs: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
    case_sensitivity: pd.DataFrame,
    figures: list[int] | None,
) -> dict[str, pd.DataFrame]:
    return build_result_sources(
        case,
        case_config,
        parameters,
        all_candidates,
        scenario_results,
        optimal_designs,
        outage_scenarios,
        case_sensitivity,
        uncertainty_baseline=pd.DataFrame(),
        uncertainty_sensitivity=pd.DataFrame(),
        figures=figures,
        compute_missing_sensitivity_rows=lambda missing: _compute_result_sensitivity_rows(
            case,
            global_config,
            missing,
        ),
        compute_saifi_saidi_grid=lambda: _compute_saifi_saidi_grid_rows(
            case,
            global_config,
            parameters,
        ),
    )


def _missing_result_sensitivity_parameters(sensitivity: pd.DataFrame) -> list[str]:
    return missing_result_sensitivity_parameters(sensitivity)


def _compute_result_sensitivity_rows(case, global_config: dict[str, Any], parameters_to_run: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    _, _, _, parameters = load_input_case(case, global_config)
    for run in _sensitivity_runs_for_parameters(parameters, parameters_to_run):
        parameter = str(run["sensitivity_parameter"])
        value = float(run["sensitivity_value"])
        _, _, s_config, _ = load_input_case(case, global_config, parameter_overrides={parameter: value})
        s_config.setdefault("dispatch", {})["backend"] = "greedy"
        s_rows, _, _, _, _, _ = _run_design_case(
            case,
            s_config,
            case.case_key,
            sensitivity_parameter=parameter,
            sensitivity_value=value,
        )
        rows.append(s_rows.copy())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _compute_saifi_saidi_grid_rows(case, global_config: dict[str, Any], parameters) -> pd.DataFrame:
    saidi_values = parameters.sensitivity.get("saidi_min_per_year", [])
    saifi_values = parameters.sensitivity.get("saifi_per_year", [])
    if not saidi_values or not saifi_values:
        raise ValueError("Figure 7 requires sensitivity ranges for both saidi_min_per_year and saifi_per_year.")
    scenario_count = 1
    seed_start = int(parameters.base.get("uncertainty_seed_start", 1000))
    workers = int(parameters.base.get("parallel_workers", global_config.get("uncertainty", {}).get("parallel_workers", 1)))
    tasks: list[tuple[Any, dict[str, Any], float, float, int, int, int, int]] = []
    for saidi in saidi_values:
        for saifi in saifi_values:
            for scenario_index, seed in enumerate(range(seed_start, seed_start + scenario_count), start=1):
                tasks.append(
                    (
                        case,
                        global_config,
                        float(saidi),
                        float(saifi),
                        int(seed),
                        int(scenario_index),
                        int(scenario_count),
                        int(seed_start),
                    )
                )
    if workers <= 1 or len(tasks) <= 1:
        rows = [_compute_saifi_saidi_seed_row(task) for task in tasks]
    else:
        max_workers = min(workers, len(tasks))
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
        except OSError as exc:
            warnings.warn(
                f"Could not start Figure 7 SAIFI-SAIDI ProcessPoolExecutor ({exc}); falling back to serial execution.",
                RuntimeWarning,
                stacklevel=2,
            )
            rows = [_compute_saifi_saidi_seed_row(task) for task in tasks]
        else:
            with executor:
                rows = list(executor.map(_compute_saifi_saidi_seed_row, tasks))
    return pd.concat(rows, ignore_index=True)


def _compute_saifi_saidi_seed_row(args: tuple[Any, dict[str, Any], float, float, int, int, int, int]) -> pd.DataFrame:
    case, global_config, saidi, saifi, seed, scenario_index, scenario_count, seed_start = args
    overrides = {
        "saidi_min_per_year": float(saidi),
        "saifi_per_year": float(saifi),
    }
    _, _, grid_config, _ = load_input_case(case, global_config, parameter_overrides=overrides)
    grid_config.setdefault("dispatch", {})["backend"] = "greedy"
    grid_config.setdefault("outage", {})["seed"] = int(seed)
    grid_rows, _, _, _, _, _ = _run_design_case(
        case,
        grid_config,
        case.case_key,
        sensitivity_parameter="saifi_saidi_grid",
        sensitivity_value=f"SAIDI={float(saidi):g};SAIFI={float(saifi):g}",
    )
    grid_rows = grid_rows.copy()
    grid_rows["saifi_saidi_scenario_id"] = int(scenario_index)
    grid_rows["outage_seed"] = int(seed)
    grid_rows["scenario_count"] = int(scenario_count)
    grid_rows["outage_seed_start"] = int(seed_start)
    grid_rows["outage_seed_end"] = int(seed_start + scenario_count - 1)
    return grid_rows


def _read_result_sources(result_dir: Path, case_key: str, figures: list[int] | None) -> dict[str, pd.DataFrame]:
    return read_result_sources(result_dir, case_key, figures)


def _read_frontier_candidate_results(frontier_dir: Path, case_key: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    candidates_path = frontier_dir / f"{case_key}_Frontier_All_Candidates.csv"
    optimal_path = frontier_dir / f"{case_key}_Frontier_Optimal_Designs.csv"
    if not candidates_path.exists():
        return None
    candidates = pd.read_csv(candidates_path)
    optimal_designs = pd.read_csv(optimal_path) if optimal_path.exists() else pd.DataFrame()
    return candidates, optimal_designs


def _read_uncertainty_baseline_results(uncertainty_baseline_dir: Path, case_key: str) -> pd.DataFrame:
    path = uncertainty_baseline_dir / f"{case_key}_Uncertainty_Baseline_All_Results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Figure 8/11 source generation requires uncertainty baseline results: {path}"
        )
    return pd.read_csv(path)


def _read_uncertainty_sensitivity_results(uncertainty_sensitivity_dir: Path, case_key: str) -> pd.DataFrame:
    path = uncertainty_sensitivity_dir / f"{case_key}_Uncertainty_Sensitive_Analysis_All_Results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Figure 9/10/11 source generation requires uncertainty sensitivity results: {path}"
        )
    return pd.read_csv(path)
