from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.data_loader import SENSITIVITY_PARAMETERS, build_case_frame, discover_input_cases, load_input_case, sensitivity_runs
from src.optimizer import resilience_design_pair
from src.outage import simulate_outages, summarize_outages
from src.reporting import write_csv
from src.uncertainty import run_uncertainty_analysis
from src.visualization import (
    make_baseline_figures,
    make_sensitivity_figures,
    make_uncertainty_figures,
    make_uncertainty_sensitivity_figures,
    normalize_design_case_names,
)


def _input_dir(config: dict[str, Any], override: str | None) -> Path:
    return Path(override or config.get("input", {}).get("directory", "Input")).resolve()


def _output_dir(config: dict[str, Any], override: str | None) -> Path:
    return Path(override or config.get("outputs", {}).get("directory", "outputs")).resolve()


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
    annotated["pv_capex_sgd_per_kwp"] = config["pv"]["capex_sgd_per_kwp"]
    for key, value in outage_summary.items():
        annotated[key] = value
    return annotated


def _parameter_table(case_key: str, city_key: str, building_key: str, raw: pd.DataFrame) -> pd.DataFrame:
    table = raw.copy()
    table.insert(0, "building_key", building_key)
    table.insert(0, "city_key", city_key)
    table.insert(0, "case_key", case_key)
    return table


def _baseline_dir(output_dir: Path, city_key: str, case_key: str) -> Path:
    return output_dir / city_key / f"{case_key}_Baseline"


def _sensitivity_dir(output_dir: Path, city_key: str, case_key: str) -> Path:
    return output_dir / city_key / f"{case_key}_Sensitive_Analysis"


def _uncertainty_dir(output_dir: Path, city_key: str, case_key: str) -> Path:
    return output_dir / city_key / f"{case_key}_Uncertainty_Analysis"


def _uncertainty_baseline_dir(output_dir: Path, city_key: str, case_key: str) -> Path:
    return _uncertainty_dir(output_dir, city_key, case_key) / f"{case_key}_Uncertainty_Baseline"


def _uncertainty_sensitivity_dir(output_dir: Path, city_key: str, case_key: str) -> Path:
    return _uncertainty_dir(output_dir, city_key, case_key) / f"{case_key}_Uncertainty_Sensitive_Analysis"


def _write_case_outputs(
    case_dir: Path,
    case_key: str,
    design_rows: pd.DataFrame,
    outage_profile: pd.DataFrame,
    outage_events: pd.DataFrame,
    dispatches: dict[str, pd.DataFrame],
    summaries: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths["design_comparison"] = write_csv(case_dir / f"{case_key}_Baseline_Design_Comparison.csv", design_rows)
    paths["outage_profile"] = write_csv(case_dir / f"{case_key}_Baseline_Outage_Profile.csv", outage_profile)
    paths["outage_events"] = write_csv(case_dir / f"{case_key}_Baseline_Outage_Events.csv", outage_events)
    for design_case, dispatch in dispatches.items():
        label = design_case.title().replace("_", "_")
        paths[f"{design_case}_dispatch"] = write_csv(case_dir / f"{case_key}_Baseline_Dispatch_{label}.csv", dispatch)
        paths[f"{design_case}_candidate_summary"] = write_csv(
            case_dir / f"{case_key}_Baseline_Candidate_Summary_{label}.csv",
            summaries[design_case],
        )
    return paths


def _write_uncertainty_baseline_outputs(
    uncertainty_dir: Path,
    case_key: str,
    all_results: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    all_results = normalize_design_case_names(all_results)
    summary = normalize_design_case_names(summary)
    paths["uncertainty_baseline_all_results"] = write_csv(
        uncertainty_dir / f"{case_key}_Uncertainty_Baseline_All_Results.csv",
        all_results,
    )
    paths["uncertainty_baseline_outage_scenarios"] = write_csv(
        uncertainty_dir / f"{case_key}_Uncertainty_Baseline_Outage_Scenarios.csv",
        outage_scenarios,
    )
    paths["uncertainty_baseline_design_summary"] = write_csv(
        uncertainty_dir / f"{case_key}_Uncertainty_Baseline_Design_Comparison_Summary.csv",
        summary,
    )
    for name, path in make_uncertainty_figures(
        uncertainty_dir,
        case_key,
        all_results,
        outage_scenarios,
        file_prefix="Uncertainty_Baseline",
    ).items():
        paths[f"uncertainty_{name}"] = path
    return paths


def _write_uncertainty_sensitivity_outputs(
    uncertainty_dir: Path,
    case_key: str,
    all_results: pd.DataFrame,
    selected_parameters: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    all_results = normalize_design_case_names(all_results)
    paths["uncertainty_sensitivity_all_results"] = write_csv(
        uncertainty_dir / f"{case_key}_Uncertainty_Sensitive_Analysis_All_Results.csv",
        all_results,
    )
    for name, path in make_uncertainty_sensitivity_figures(
        uncertainty_dir,
        case_key,
        all_results,
        parameters_to_plot=selected_parameters,
    ).items():
        paths[f"uncertainty_sensitivity_{name}"] = path
    return paths


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


def _validate_sensitivity_parameters(parameters: list[str] | None) -> list[str] | None:
    if not parameters:
        return None
    allowed = set(SENSITIVITY_PARAMETERS)
    requested = list(dict.fromkeys(parameters))
    invalid = sorted(set(requested) - allowed)
    if invalid:
        raise ValueError(f"Unknown sensitivity parameter(s): {invalid}. Allowed parameters: {SENSITIVITY_PARAMETERS}")
    return requested


def _sensitivity_runs_for_parameters(parameters, selected_parameters: list[str] | None) -> list[dict[str, Any]]:
    runs = sensitivity_runs(parameters)
    if selected_parameters is None:
        return runs
    selected = set(selected_parameters)
    return [run for run in runs if str(run["sensitivity_parameter"]) in selected]


def _merge_sensitivity_results(
    existing_path: Path,
    new_rows: pd.DataFrame,
    selected_parameters: list[str] | None,
) -> pd.DataFrame:
    new_rows = normalize_design_case_names(new_rows)
    if selected_parameters is None or not existing_path.exists():
        return new_rows.copy()
    existing = normalize_design_case_names(pd.read_csv(existing_path))
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


def _run_design_case(
    case,
    config: dict[str, Any],
    case_key: str,
    sensitivity_parameter: str = "baseline",
    sensitivity_value: float | str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    preview_frame, _ = build_case_frame(case, config, max(config["pv"]["capacity_kwp_candidates"]))
    _, outage_profile, outage_events, outage_summary = _add_outage(preview_frame, case_key, config)

    def pv_capacity_frame_builder(pv_capacity_kwp: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, config, pv_capacity_kwp)
        return frame.merge(outage_profile, on="hour_index", how="left"), pv_meta

    rows, dispatches, summaries = resilience_design_pair(pv_capacity_frame_builder, config, case_key)
    annotated = _annotate_design_rows(
        rows,
        case_key,
        config,
        outage_summary,
        sensitivity_parameter=sensitivity_parameter,
        sensitivity_value=sensitivity_value,
    )
    return annotated, outage_profile, outage_events, dispatches, summaries


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
    city_filters: list[str] | None = None,
    case_filters: list[str] | None = None,
    sensitivity_parameters: list[str] | None = None,
    uncertainty_sensitivity_parameters: list[str] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    cases = _filter_cases(discover_input_cases(input_dir), city_filters, case_filters)
    selected_sensitivity_parameters = _validate_sensitivity_parameters(sensitivity_parameters)
    selected_uncertainty_sensitivity_parameters = _validate_sensitivity_parameters(uncertainty_sensitivity_parameters)
    baseline_rows: list[pd.DataFrame] = []
    sensitivity_rows: list[pd.DataFrame] = []

    for case in cases:
        frame, _, case_config, parameters = load_input_case(case, config)
        case_key = case.case_key
        base_dir = _baseline_dir(output_dir, case.city_key, case_key)
        sens_dir = _sensitivity_dir(output_dir, case.city_key, case_key)
        uncertainty_baseline_dir = _uncertainty_baseline_dir(output_dir, case.city_key, case_key)
        uncertainty_sensitivity_dir = _uncertainty_sensitivity_dir(output_dir, case.city_key, case_key)
        parameter_table = _parameter_table(case_key, case.city_key, case.building_key, parameters.raw)

        rows = pd.DataFrame()
        if run_baseline:
            paths[f"{case_key}_baseline_timeseries"] = write_csv(
                base_dir / f"{case_key}_Baseline_Timeseries_Max_PV_Capacity.csv",
                frame,
            )
            paths[f"{case_key}_baseline_parameters"] = write_csv(
                base_dir / f"{case_key}_Baseline_Parameter_Table.csv",
                parameter_table,
            )

            rows, outage_profile, outage_events, dispatches, summaries = _run_design_case(case, case_config, case_key)
            baseline_rows.append(rows)
            for name, path in _write_case_outputs(base_dir, case_key, rows, outage_profile, outage_events, dispatches, summaries).items():
                paths[f"{case_key}_baseline_{name}"] = path
            for name, path in make_baseline_figures(base_dir, case_key, rows, summaries, case_config).items():
                paths[f"{case_key}_baseline_{name}"] = path

        if run_sensitivity:
            case_sensitivity_rows: list[pd.DataFrame] = []
            for run in _sensitivity_runs_for_parameters(parameters, selected_sensitivity_parameters):
                parameter = str(run["sensitivity_parameter"])
                value = float(run["sensitivity_value"])
                overrides = {parameter: value}
                _, _, s_config, _ = load_input_case(case, config, parameter_overrides=overrides)
                s_rows, _, _, _, _ = _run_design_case(
                    case,
                    s_config,
                    case_key,
                    sensitivity_parameter=parameter,
                    sensitivity_value=value,
                )
                case_sensitivity_rows.append(s_rows)

            results_path = sens_dir / f"{case_key}_Sensitive_Analysis_All_Results.csv"
            if case_sensitivity_rows:
                new_case_sensitivity = pd.concat(case_sensitivity_rows, ignore_index=True)
            else:
                new_case_sensitivity = pd.DataFrame()

            case_sensitivity = _merge_sensitivity_results(results_path, new_case_sensitivity, selected_sensitivity_parameters)
            if not case_sensitivity.empty:
                sensitivity_rows.append(case_sensitivity)
                paths[f"{case_key}_sensitivity_results"] = write_csv(results_path, case_sensitivity)
            _clear_selected_sensitivity_dirs(sens_dir, case_key, selected_sensitivity_parameters)
            for name, path in make_sensitivity_figures(
                sens_dir,
                case_key,
                rows,
                case_sensitivity,
                parameter_table,
                parameters_to_plot=selected_sensitivity_parameters,
            ).items():
                paths[f"{case_key}_sensitivity_{name}"] = path

        if run_uncertainty_baseline:
            uncertainty_cfg = case_config["uncertainty"]
            uncertainty_results, outage_scenarios, uncertainty_summary = run_uncertainty_analysis(
                case,
                case_config,
                case_key,
                scenario_count=int(uncertainty_cfg["baseline_scenario_count"]),
                seed_start=int(uncertainty_cfg["seed_start"]),
            )
            for name, path in _write_uncertainty_baseline_outputs(
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
                for name, path in _write_uncertainty_sensitivity_outputs(
                    uncertainty_sensitivity_dir,
                    case_key,
                    uncertainty_sensitivity,
                    selected_uncertainty_sensitivity_parameters or list(SENSITIVITY_PARAMETERS),
                ).items():
                    paths[f"{case_key}_{name}"] = path

    if baseline_rows:
        _ = pd.concat(baseline_rows, ignore_index=True)
    if sensitivity_rows:
        _ = pd.concat(sensitivity_rows, ignore_index=True)
    return paths


def _resolve_analysis_selection(args: argparse.Namespace) -> tuple[str, str, list[str] | None, list[str] | None]:
    analysis = args.analysis
    uncertainty_analysis = args.uncertainty_analysis
    sensitivity_parameters = args.sensitivity_parameter
    uncertainty_sensitivity_parameters = args.uncertainty_sensitivity_parameter

    if args.run_mode:
        if args.run_mode == "all":
            analysis = "all"
        elif args.run_mode == "baseline":
            analysis = "baseline"
        elif args.run_mode == "sensitivity":
            analysis = "sensitivity"
        elif args.run_mode == "uncertainty":
            analysis = "uncertainty"
            uncertainty_analysis = "baseline"
        elif args.run_mode == "uncertainty-sensitivity":
            analysis = "uncertainty"
            uncertainty_analysis = "sensitivity"
            if uncertainty_sensitivity_parameters is None:
                uncertainty_sensitivity_parameters = sensitivity_parameters

    if args.skip_sensitivity:
        analysis = "baseline"

    return analysis, uncertainty_analysis, sensitivity_parameters, uncertainty_sensitivity_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Input-driven PV-BESS resilience-informed optimal sizing.")
    parser.add_argument("--config", default="config.yaml", help="Optional YAML config override.")
    parser.add_argument("--input-dir", default=None, help="Input directory; default comes from config.")
    parser.add_argument("--output-dir", default=None, help="Output directory; default comes from config.")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Deprecated alias for --analysis baseline.")
    parser.add_argument("--city", action="append", default=None, help="City key to run; can be repeated.")
    parser.add_argument("--case", action="append", default=None, help="Case key to run, such as LA_UTown; can be repeated.")
    parser.add_argument(
        "--sensitivity-parameter",
        action="append",
        default=None,
        help="Ordinary sensitivity parameter to run; can be repeated. Defaults to all five for --analysis sensitivity.",
    )
    parser.add_argument(
        "--uncertainty-sensitivity-parameter",
        action="append",
        default=None,
        help="Uncertainty sensitivity parameter to run; can be repeated. Defaults to all five for uncertainty sensitivity.",
    )
    parser.add_argument(
        "--analysis",
        choices=["all", "baseline", "sensitivity", "uncertainty"],
        default="all",
        help="Top-level analysis selection. Default: all.",
    )
    parser.add_argument(
        "--uncertainty-analysis",
        choices=["all", "baseline", "sensitivity"],
        default="all",
        help="Sub-analysis to run when --analysis uncertainty is selected. Default: all.",
    )
    parser.add_argument(
        "--run-mode",
        choices=["all", "baseline", "sensitivity", "uncertainty", "uncertainty-sensitivity"],
        default=None,
        help="Deprecated alias for --analysis / --uncertainty-analysis.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    input_root = _input_dir(config, args.input_dir)
    out = _output_dir(config, args.output_dir)
    analysis, uncertainty_analysis, sensitivity_parameters, uncertainty_sensitivity_parameters = _resolve_analysis_selection(args)
    run_baseline = analysis in {"all", "baseline"}
    run_sensitivity = analysis in {"all", "sensitivity"}
    run_uncertainty_baseline = analysis == "all" or (
        analysis == "uncertainty" and uncertainty_analysis in {"all", "baseline"}
    )
    run_uncertainty_sensitivity = analysis == "all" or (
        analysis == "uncertainty" and uncertainty_analysis in {"all", "sensitivity"}
    )
    paths = run_selected(
        config,
        input_root,
        out,
        run_baseline=run_baseline,
        run_sensitivity=run_sensitivity,
        run_uncertainty_baseline=run_uncertainty_baseline,
        run_uncertainty_sensitivity=run_uncertainty_sensitivity,
        city_filters=args.city,
        case_filters=args.case,
        sensitivity_parameters=sensitivity_parameters,
        uncertainty_sensitivity_parameters=uncertainty_sensitivity_parameters,
    )
    print("Input-driven PV-BESS resilience sizing completed.")
    print(f"Input directory: {input_root}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
