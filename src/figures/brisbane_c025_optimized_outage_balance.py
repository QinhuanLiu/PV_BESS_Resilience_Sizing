"""Plot a representative outage from the Brisbane 0.25C, 15-year optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..brisbane_c025_15y_optimization import CASE_KEY, RESULTS_DIR, ROOT, _sha256
from ..io.data_loader import build_case_frame, discover_input_cases
from ..optimizer import evaluate_candidate, scenario_config
from .core import write_source_data
from .typical_outage_72h_energy_balance import (
    ENERGY_TOLERANCE_KWH,
    WINDOW_HOURS,
    _balance_columns,
    _window_bounds,
    plot_figure,
)


BASE_NAME = "Brisbane_15Y_C0p25_Optimized_Typical_Outage_72h_Energy_Balance"
MEDIAN_TIE_TOLERANCE_KWH = 1e-9


def _validated_run(results_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((results_dir / "manifest.json").read_text())
    config_path = results_dir / "effective_config.json"
    config = json.loads(config_path.read_text())
    if (
        manifest["case_key"] != CASE_KEY
        or manifest["changed_effective_config_paths"] != ["analysis.years", "battery.max_c_rate"]
        or int(config["analysis"]["years"]) != 15
        or float(config["battery"]["max_c_rate"]) != 0.25
        or int(config["outage"]["seed"]) != 42
        or manifest["dispatch_backend"] != "greedy"
    ):
        raise ValueError("The saved run is not the Brisbane 15-year, 0.25C optimization")
    for name, digest in manifest["output_sha256"].items():
        filename = {
            "optimal_design": f"{CASE_KEY}_Baseline_Optimal_Design.csv",
            "outage_profile": f"{CASE_KEY}_Baseline_Outage_Profile.csv",
            "outage_events": f"{CASE_KEY}_Baseline_Outage_Events.csv",
            "outage_summary": f"{CASE_KEY}_Baseline_Outage_Summary.csv",
            "yearly_summary": f"{CASE_KEY}_Baseline_Yearly_Summary.csv",
            "dispatch": f"{CASE_KEY}_Baseline_Dispatch.csv",
            "candidate_summary": f"{CASE_KEY}_Baseline_Candidate_Summary.csv",
            "effective_config": "effective_config.json",
        }[name]
        if _sha256(results_dir / filename) != digest:
            raise ValueError(f"Optimization output checksum mismatch: {filename}")
    case = next(
        (item for item in discover_input_cases(ROOT / "Input") if item.case_key == CASE_KEY), None,
    )
    if case is None:
        raise FileNotFoundError(f"Input case does not exist: {CASE_KEY}")
    for path in (case.load_csv, case.pv_csv, case.tou_csv, case.parameter_csv):
        if _sha256(path) != manifest["source_input_sha256"][path.name]:
            raise ValueError(f"Source input changed since optimization: {path.name}")
    return config, manifest


def _read_csv(results_dir: Path, suffix: str) -> pd.DataFrame:
    return pd.read_csv(results_dir / f"{CASE_KEY}_Baseline_{suffix}.csv", float_precision="round_trip")


def _plotted_design(optimal: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.Series, bool]:
    if len(optimal) != 1 or candidates.empty:
        raise ValueError("Saved optimization lacks a unique optimum or candidate records")
    best = optimal.iloc[0]
    if float(best["battery_energy_kwh"]) > 0:
        return best, True
    positive = candidates.loc[candidates["battery_energy_kwh"].astype(float) > 0].copy()
    if positive.empty:
        raise ValueError("No positive-BESS candidate exists for the requested fallback")
    positive = positive.sort_values(
        ["pv_bess_system_annual_cost_sgd", "battery_energy_kwh", "pv_capacity_kw",
         "search_stage", "search_iteration"],
        kind="stable",
    )
    return positive.iloc[0], False


def _event_scores(
    events: pd.DataFrame, load: np.ndarray, pv: np.ndarray,
) -> tuple[pd.Series, float, pd.DataFrame]:
    if events.empty:
        raise ValueError("The optimization has no outage events to plot")
    deficit = np.maximum(load - pv, 0.0)
    scored = events.copy()
    values: list[float] = []
    eligible: list[bool] = []
    for event in scored.itertuples(index=False):
        start, end = float(event.start_hour), float(event.end_hour)
        hours = np.arange(int(np.floor(start)), int(np.ceil(end)))
        overlap = np.maximum(0.0, np.minimum(end, hours + 1) - np.maximum(start, hours))
        values.append(float(np.dot(deficit[hours], overlap)))
        try:
            window_start, window_end = _window_bounds(start, end, len(load))
            others = scored.loc[
                (scored["end_hour"] > window_start) & (scored["start_hour"] < window_end)
            ]
            eligible.append(len(others) == 1)
        except ValueError:
            eligible.append(False)
    scored["pre_storage_net_deficit_kwh"] = values
    scored["eligible_72h"] = eligible
    valid = scored.loc[scored["eligible_72h"]].copy()
    if valid.empty:
        raise ValueError("No outage event supports an isolated three-day window")
    median = float(scored["pre_storage_net_deficit_kwh"].median())
    valid["distance_to_catalog_median_kwh"] = (
        valid["pre_storage_net_deficit_kwh"] - median
    ).abs()
    minimum_distance = float(valid["distance_to_catalog_median_kwh"].min())
    tied = valid.loc[np.isclose(
        valid["distance_to_catalog_median_kwh"], minimum_distance,
        rtol=0, atol=MEDIAN_TIE_TOLERANCE_KWH,
    )]
    selected = tied.sort_values("event_id", kind="stable").iloc[0]
    return selected, median, scored


def build_source_data(results_dir: Path = RESULTS_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    results_dir = Path(results_dir)
    config, manifest = _validated_run(results_dir)
    optimal = _read_csv(results_dir, "Optimal_Design")
    candidates = _read_csv(results_dir, "Candidate_Summary")
    events = _read_csv(results_dir, "Outage_Events")
    profile = _read_csv(results_dir, "Outage_Profile")
    chosen, is_global_optimum = _plotted_design(optimal, candidates)
    pv_kw = float(chosen["pv_capacity_kw"])
    battery_kwh = float(chosen["battery_energy_kwh"])
    case = next(item for item in discover_input_cases(ROOT / "Input") if item.case_key == CASE_KEY)
    frame, pv_meta = build_case_frame(case, config, pv_kw)
    if len(frame) != 15 * 8760 or len(profile) != len(frame):
        raise ValueError("Saved outage profile does not cover the 15-year input horizon")
    frame = frame.merge(
        profile[["hour_index", "grid_availability", "outage_fraction"]],
        on="hour_index", how="left", validate="one_to_one",
    )
    if frame[["grid_availability", "outage_fraction"]].isna().any().any():
        raise ValueError("Saved outage profile does not align with the case frame")
    if is_global_optimum:
        dispatch = _read_csv(results_dir, "Dispatch")
    else:
        result, dispatch = evaluate_candidate(
            frame, battery_kwh, pv_meta, scenario_config(config, CASE_KEY),
        )
        for field in ("pv_bess_system_annual_cost_sgd", "unserved_energy_kwh_per_year"):
            if not np.isclose(float(result[field]), float(chosen[field]), rtol=1e-8, atol=1e-8):
                raise ValueError(f"Positive-BESS replay differs from saved candidate: {field}")
    if len(dispatch) != len(frame) or not dispatch["dispatch_backend"].eq("greedy").all():
        raise ValueError("Plotted design dispatch does not match the 15-year greedy run")
    if not np.array_equal(
        dispatch["hour_index"].to_numpy(dtype=int), frame["hour_index"].to_numpy(dtype=int),
    ):
        raise ValueError("Plotted design dispatch is not aligned to the saved outage profile")
    if not np.allclose(dispatch["grid_availability"], frame["grid_availability"], atol=1e-9, rtol=0):
        raise ValueError("Plotted design dispatch uses a different outage profile")
    annual_unserved = float(dispatch["unmet_load_kwh_equiv"].sum()) / 15.0
    if not np.isclose(
        annual_unserved, float(chosen["unserved_energy_kwh_per_year"]), rtol=1e-8, atol=1e-8,
    ):
        raise ValueError("Plotted design dispatch does not reproduce saved unserved energy")

    selected, median, scored = _event_scores(
        events,
        dispatch["total_load_e_equiv_kwh"].to_numpy(dtype=float),
        dispatch["pv_available_kwh"].to_numpy(dtype=float),
    )
    start, end = float(selected["start_hour"]), float(selected["end_hour"])
    window_start, window_end = _window_bounds(start, end, len(dispatch))
    source = dispatch.iloc[window_start:window_end].copy().reset_index(drop=True)
    source.insert(0, "plot_hour", np.arange(WINDOW_HOURS))
    source["analysis_year"] = frame["analysis_year"].iloc[window_start:window_end].to_numpy()
    source["hour_of_year"] = frame["hour_of_year"].iloc[window_start:window_end].to_numpy()
    source["day_label"] = np.repeat(["Previous day", "Outage day", "Recovery day"], 24)
    source["event_outage_fraction"] = np.maximum(
        0.0,
        np.minimum(end, source["hour_index"].to_numpy(dtype=float) + 1.0)
        - np.maximum(start, source["hour_index"].to_numpy(dtype=float)),
    )
    if not np.allclose(source["event_outage_fraction"], source["outage_fraction"], atol=1e-9, rtol=0):
        raise ValueError("Selected event differs from the saved outage profile in the plot window")
    source["is_event_hour"] = source["event_outage_fraction"] > 0
    source = _balance_columns(source, config, battery_kwh)
    source["outage_load_kwh"] = source["critical_load_kwh"] * source["event_outage_fraction"]
    source["pv_to_outage_kwh"] = np.minimum(source["pv_available_kwh"], source["outage_load_kwh"])
    remaining = np.maximum(0.0, source["outage_load_kwh"] - source["pv_to_outage_kwh"])
    source["battery_to_outage_kwh"] = np.minimum(source["battery_discharge_kwh"], remaining)
    source["outage_balance_residual_kwh"] = (
        source["outage_load_kwh"] - source["pv_to_outage_kwh"]
        - source["battery_to_outage_kwh"] - source["unmet_load_kwh_equiv"]
    )
    affected = source.loc[source["is_event_hour"]]
    if affected.empty or float(affected["outage_balance_residual_kwh"].abs().max()) > ENERGY_TOLERANCE_KWH:
        raise ValueError("Selected outage energy balance does not close")
    source["event_id"] = int(selected["event_id"])
    source["event_start_hour"] = start
    source["event_end_hour"] = end
    columns = [
        "event_id", "analysis_year", "hour_index", "hour_of_year", "plot_hour", "day_label",
        "event_start_hour", "event_end_hour", "grid_availability", "outage_fraction",
        "event_outage_fraction", "is_event_hour", "critical_load_kwh", "served_load_kwh",
        "unmet_load_kwh_equiv", "outage_load_kwh", "pv_available_kwh", "pv_to_outage_kwh",
        "pv_to_load_kwh", "pv_to_battery_kwh", "pv_curtailed_kwh", "pv_export_kwh",
        "grid_import_kwh", "grid_to_load_kwh", "grid_to_battery_kwh", "battery_charge_kwh",
        "battery_discharge_kwh", "battery_to_outage_kwh", "battery_soc_kwh",
        "battery_soc_fraction", "minimum_soc_fraction", "battery_conversion_loss_kwh",
        "ac_balance_residual_kwh", "outage_balance_residual_kwh",
    ]
    source = source[columns]
    summary = {
        "case_key": CASE_KEY,
        "analysis_years": 15,
        "battery_max_c_rate": 0.25,
        "outage_seed": 42,
        "event_catalog_count": len(events),
        "eligible_72h_event_count": int(scored["eligible_72h"].sum()),
        "selection_rule": "minimum_absolute_distance_to_all_event_pre_storage_net_deficit_median_ties_within_1e-9_kwh_then_event_id",
        "median_tie_tolerance_kwh": MEDIAN_TIE_TOLERANCE_KWH,
        "median_pre_storage_net_deficit_kwh": median,
        "selected_event_pre_storage_net_deficit_kwh": float(selected["pre_storage_net_deficit_kwh"]),
        "selected_event_median_distance_kwh": float(selected["distance_to_catalog_median_kwh"]),
        "event_id": int(selected["event_id"]),
        "analysis_year": int(selected["start_analysis_year"]),
        "event_start_hour": start,
        "event_end_hour": end,
        "event_duration_hours": end - start,
        "window_start_hour": window_start,
        "window_end_exclusive_hour": window_end,
        "window_hours": WINDOW_HOURS,
        "pv_capacity_kw": pv_kw,
        "battery_energy_kwh": battery_kwh,
        "battery_power_kw": 0.25 * battery_kwh,
        "plotted_design_is_global_optimum": is_global_optimum,
        "plotted_design_label": (
            "Global minimum-cost optimum" if is_global_optimum
            else "Lowest-cost positive-BESS candidate (not global optimum)"
        ),
        "plotted_design_annual_cost_sgd": float(chosen["pv_bess_system_annual_cost_sgd"]),
        "global_optimum_pv_capacity_kw": float(optimal.iloc[0]["pv_capacity_kw"]),
        "global_optimum_battery_energy_kwh": float(optimal.iloc[0]["battery_energy_kwh"]),
        "global_optimum_annual_cost_sgd": float(optimal.iloc[0]["pv_bess_system_annual_cost_sgd"]),
        "outage_load_kwh": float(affected["outage_load_kwh"].sum()),
        "pv_to_outage_kwh": float(affected["pv_to_outage_kwh"].sum()),
        "battery_to_outage_kwh": float(affected["battery_to_outage_kwh"].sum()),
        "unserved_outage_kwh": float(affected["unmet_load_kwh_equiv"].sum()),
        "maximum_hourly_ac_balance_residual_kwh": float(source["ac_balance_residual_kwh"].abs().max()),
        "maximum_outage_balance_residual_kwh": float(affected["outage_balance_residual_kwh"].abs().max()),
        "input_sha256": manifest["source_input_sha256"],
        "ac_balance_note": "AC inputs equal served load, battery charge, PV curtailment and export. Battery conversion loss affects SOC and is not an additional AC sink.",
    }
    return source, summary


def run(results_dir: Path = RESULTS_DIR) -> dict[str, Path]:
    results_dir = Path(results_dir)
    source, summary = build_source_data(results_dir)
    output_dir = results_dir / "figures"
    base_name = f"{BASE_NAME}_Event{summary['event_id']}"
    paths = plot_figure(
        source, summary, output_dir, base_name=base_name,
        pv_capacity_kw=summary["pv_capacity_kw"],
        battery_energy_kwh=summary["battery_energy_kwh"],
        design_status=(
            f"15-year optimization | 0.25C | seed 42, event {summary['event_id']} | "
            f"{summary['plotted_design_label']}"
        ),
    )
    base = output_dir / base_name
    paths["source_data"] = write_source_data(base, source)
    summary["source_data_sha256"] = _sha256(paths["source_data"])
    summary["figure_sha256"] = {key: _sha256(path) for key, path in paths.items() if key != "source_data"}
    summary_path = base.with_name(base.name + "_Event_Summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    paths["event_summary"] = summary_path
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    for label, path in run(args.results_dir).items():
        print(f"{label}: {path}")
