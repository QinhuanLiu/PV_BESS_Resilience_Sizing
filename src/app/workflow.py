from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..design_evaluation import evaluate_design_case

from ..io.data_loader import discover_input_cases, load_input_case
from ..io.reporting import write_csv
from ..tou_analysis import (
    apply_tou_profile,
    build_tou_spread_runs,
    order_tou_analysis_summary_columns,
    tou_analysis_result_row,
)
from . import outputs as workflow_outputs
from .paths import case_output_paths


def filter_cases(cases, city_filters: list[str] | None, case_filters: list[str] | None):
    """Return requested input cases and reject unknown city or case filters."""
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


def _parameter_table(case_key: str, city_key: str, building_key: str, raw: pd.DataFrame) -> pd.DataFrame:
    table = raw.copy()
    table.insert(0, "building_key", building_key)
    table.insert(0, "city_key", city_key)
    table.insert(0, "case_key", case_key)
    return table


def run_selected(
    config: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    city_filters: list[str] | None = None,
    case_filters: list[str] | None = None,
    tou_analysis: bool = False,
    tou_spread_k_values: list[float] | None = None,
) -> dict[str, Path]:
    """Run baseline sizing and the separate TOU analysis workflow for selected cases."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    cases = filter_cases(discover_input_cases(input_dir), city_filters, case_filters)

    for case in cases:
        frame, _, case_config, parameters = load_input_case(case, config)
        case_key = case.case_key
        output_paths = case_output_paths(output_dir, case.city_key, case_key)
        base_dir = output_paths.baseline
        parameter_table = _parameter_table(case_key, case.city_key, case.building_key, parameters.raw)

        paths[f"{case_key}_baseline_timeseries"] = write_csv(
            base_dir / f"{case_key}_Baseline_Timeseries_Max_PV_Capacity.csv",
            frame,
        )
        paths[f"{case_key}_baseline_parameters"] = write_csv(
            base_dir / f"{case_key}_Baseline_Parameter_Table.csv",
            parameter_table,
        )

        rows, outage_profile, outage_events, dispatch, summary, candidate_dispatches, outage_summary, yearly_summary = evaluate_design_case(
            case,
            case_config,
            case_key,
        )
        outage_summary_frame = pd.DataFrame([outage_summary])
        for name, path in workflow_outputs.write_case_outputs(
            base_dir,
            case_key,
            case_config,
            rows,
            outage_profile,
            outage_events,
            outage_summary_frame,
            yearly_summary,
            dispatch,
            summary,
            candidate_dispatches,
        ).items():
            paths[f"{case_key}_baseline_{name}"] = path

        if tou_analysis or tou_spread_k_values is not None:
            tou_rows: list[dict[str, Any]] = []
            for tou_run in build_tou_spread_runs(case.tou_csv, tou_spread_k_values):
                tou_design, *_ = evaluate_design_case(
                    case,
                    case_config,
                    case_key,
                    frame_transform=lambda case_frame, profile=tou_run.profile: apply_tou_profile(case_frame, profile),
                )
                tou_rows.append(tou_analysis_result_row(tou_run, tou_design))
            tou_summary = order_tou_analysis_summary_columns(pd.DataFrame(tou_rows))
            paths[f"{case_key}_tou_analysis_summary"] = write_csv(
                output_paths.tou_analysis / f"{case_key}_TOU_Analysis_Summary.csv",
                tou_summary,
            )
    return paths


def run_all(config: dict[str, Any], input_dir: Path, output_dir: Path) -> dict[str, Path]:
    return run_selected(config, input_dir, output_dir)
