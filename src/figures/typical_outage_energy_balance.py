"""Replay and plot the saved Brisbane median-deficit outage event.

Run with ``python -m src.figures.typical_outage_energy_balance``. The figure
uses the paired run's frozen inputs and sparse 15-year outage profile. It never
draws a new outage scenario or changes the dispatch model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..dispatch import solve_dispatch_greedy
from ..io.data_loader import InputCase, build_case_frame
from ..optimizer import battery_sizing
from ..resilience_metrics import DEFAULT_TOLERANCE, electric_equivalent_building_load
from .core import NATURE_PALETTE, apply_nature_style, matplotlib_context, save_publication_figure, write_source_data


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = (
    ROOT
    / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV3p25_Battery8p5_Battery11p5_Battery14p5_15Y_1000Scenarios_PairedReplay"
)
BASE_NAME = "Brisbane_Typical_Outage_Energy_Balance_Seed999_Event6"
CASE_KEY = "Brisbane_SingleFamily"
SCENARIO_ID = 999
OUTAGE_SEED = 999
EVENT_ID = 6
PV_CAPACITY_KW = 3.25
BATTERY_ENERGY_KWH = 11.5
PRE_HOURS = 2
POST_HOURS = 3
ENERGY_TOLERANCE_KWH = 1e-8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _saved_case(results_dir: Path, manifest: dict[str, Any]) -> InputCase:
    input_dir = results_dir / "inputs"
    for filename, expected_hash in manifest["input_hashes"].items():
        path = input_dir / filename
        if _sha256(path) != expected_hash:
            raise ValueError(f"Saved input checksum mismatch: {filename}")
    prefix = CASE_KEY
    return InputCase(
        city_key="Brisbane",
        building_key="SingleFamily",
        case_key=CASE_KEY,
        city_dir=input_dir,
        load_csv=input_dir / f"{prefix}_Load.csv",
        pv_csv=input_dir / f"{prefix}_PV.csv",
        tou_csv=input_dir / f"{prefix}_TOU.csv",
        parameter_csv=input_dir / f"{prefix}_Parameter.csv",
    )


def _profile(results_dir: Path, hours: int) -> tuple[np.ndarray, dict[str, Any]]:
    payloads = json.loads((results_dir / "outage_profiles_sparse.json").read_text())
    matches = [
        item for item in payloads
        if int(item["scenario_id"]) == SCENARIO_ID and int(item["outage_seed"]) == OUTAGE_SEED
    ]
    if len(matches) != 1:
        raise ValueError("The saved outage profile is missing or duplicated")
    payload = matches[0]
    indices = np.asarray(payload["hour_indices"], dtype=int)
    values = np.asarray(payload["availability"], dtype=float)
    if (
        len(indices) != len(values)
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= hours)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values >= 1)
    ):
        raise ValueError("Invalid saved sparse outage profile")
    availability = np.ones(hours, dtype=float)
    availability[indices] = values
    digest = hashlib.sha256(np.asarray(availability, dtype="<f8").tobytes()).hexdigest()
    if digest != payload["outage_profile_sha256"]:
        raise ValueError("Saved outage profile checksum mismatch")
    return availability, payload


def _event(results_dir: Path) -> tuple[pd.Series, pd.DataFrame]:
    events = pd.read_csv(results_dir / "outage_events.csv", float_precision="round_trip")
    selected = events.loc[
        (events["scenario_id"] == SCENARIO_ID)
        & (events["outage_seed"] == OUTAGE_SEED)
        & (events["event_id"] == EVENT_ID)
    ]
    if len(selected) != 1 or len(events) != 9805:
        raise ValueError("Saved event catalog differs from the selected 9,805-event run")
    return selected.iloc[0], events


def _median_deficit(events: pd.DataFrame, load: np.ndarray, pv: np.ndarray) -> float:
    """Check the frozen event's pre-storage deficit against the full catalog."""
    deficit = np.maximum(load - pv, 0.0)
    scores = np.empty(len(events), dtype=float)
    for index, (start, end) in enumerate(zip(events.start_hour, events.end_hour, strict=True)):
        hours = np.arange(int(np.floor(start)), int(np.ceil(end)))
        overlap = np.maximum(0.0, np.minimum(float(end), hours + 1) - np.maximum(float(start), hours))
        scores[index] = float(np.dot(deficit[hours], overlap))
    selected_mask = (
        (events.scenario_id == SCENARIO_ID)
        & (events.outage_seed == OUTAGE_SEED)
        & (events.event_id == EVENT_ID)
    )
    selected_score = float(scores[np.flatnonzero(selected_mask.to_numpy())[0]])
    median = float(np.median(scores))
    if not np.isclose(selected_score, median, atol=ENERGY_TOLERANCE_KWH, rtol=0):
        raise ValueError("Selected event is no longer the median-deficit event")
    return median


def build_source_data(results_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct the exact profile, replay continuous dispatch, and verify it."""
    results_dir = Path(results_dir)
    validation = json.loads((results_dir / "validation.json").read_text())
    if not validation.get("passed"):
        raise ValueError("The saved source run has not passed validation")
    for filename in ("scenario_results.csv",):
        if _sha256(results_dir / filename) != validation["file_sha256"][filename]:
            raise ValueError(f"Saved results checksum mismatch: {filename}")
    manifest = json.loads((results_dir / "manifest.json").read_text())
    config = json.loads((results_dir / "effective_config.json").read_text())
    if config != manifest["effective_config"]:
        raise ValueError("Effective configuration differs from the saved run manifest")
    if int(config["analysis"]["years"]) != 15 or config["dispatch"]["backend"] != "greedy":
        raise ValueError("The selected figure requires the saved 15-year greedy run")

    case = _saved_case(results_dir, manifest)
    frame, _ = build_case_frame(case, config, PV_CAPACITY_KW)
    availability, payload = _profile(results_dir, len(frame))
    frame["grid_availability"] = availability
    frame["outage_fraction"] = 1.0 - availability
    event, events = _event(results_dir)
    if str(event["outage_profile_sha256"]) != payload["outage_profile_sha256"]:
        raise ValueError("Event and outage profile identities differ")
    technical = config["technical"]
    critical_load = electric_equivalent_building_load(
        frame, float(technical["cooling_cop"]), float(technical["heating_cop"])
    ).to_numpy(dtype=float)
    median_deficit = _median_deficit(
        events, critical_load, frame["pv_kwh"].to_numpy(dtype=float)
    )
    dispatch = solve_dispatch_greedy(frame, battery_sizing(BATTERY_ENERGY_KWH, config), config)
    saved_results = pd.read_csv(results_dir / "scenario_results.csv", float_precision="round_trip")
    saved = saved_results.loc[
        (saved_results["scenario_id"] == SCENARIO_ID)
        & (saved_results["outage_seed"] == OUTAGE_SEED)
        & np.isclose(saved_results["pv_capacity_kw"], PV_CAPACITY_KW)
        & np.isclose(saved_results["battery_energy_kwh"], BATTERY_ENERGY_KWH)
    ]
    if len(saved) != 1 or str(saved.iloc[0]["outage_profile_sha256"]) != payload["outage_profile_sha256"]:
        raise ValueError("Saved design/scenario result is missing or has a different profile")
    replay_eue = float(
        dispatch["unmet_load_kwh_equiv"]
        .where(dispatch["unmet_load_kwh_equiv"] > DEFAULT_TOLERANCE, 0.0)
        .sum()
    )
    saved_eue = float(saved.iloc[0]["EUE_kwh_total"])
    if not np.isclose(replay_eue, saved_eue, atol=ENERGY_TOLERANCE_KWH, rtol=0):
        raise ValueError("Full-horizon replay does not reproduce saved EUE")

    start, end = float(event["start_hour"]), float(event["end_hour"])
    first_hour, last_hour = int(np.floor(start)), int(np.ceil(end))
    window_start, window_end = first_hour - PRE_HOURS, last_hour + POST_HOURS
    if window_start < 0 or window_end > len(frame):
        raise ValueError("Selected event cannot support the complete SOC context window")
    source = dispatch.iloc[window_start:window_end].copy().reset_index(drop=True)
    source["analysis_year"] = frame["analysis_year"].iloc[window_start:window_end].to_numpy()
    source["hour_of_year"] = frame["hour_of_year"].iloc[window_start:window_end].to_numpy()
    source["critical_load_kwh"] = critical_load[window_start:window_end]
    source["event_outage_fraction"] = np.maximum(
        0.0,
        np.minimum(end, source["hour_index"].to_numpy(dtype=float) + 1.0)
        - np.maximum(start, source["hour_index"].to_numpy(dtype=float)),
    )
    if not np.allclose(source["event_outage_fraction"], source["outage_fraction"], atol=1e-9, rtol=0):
        raise ValueError("Selected event does not match the saved outage profile in the figure window")
    source["outage_load_kwh"] = source["critical_load_kwh"] * source["event_outage_fraction"]
    source["pv_to_outage_kwh"] = np.minimum(source["pv_available_kwh"], source["outage_load_kwh"])
    remaining = np.maximum(0.0, source["outage_load_kwh"] - source["pv_to_outage_kwh"])
    source["battery_to_outage_kwh"] = np.minimum(source["battery_discharge_kwh"], remaining)
    source["unserved_outage_kwh"] = source["unmet_load_kwh_equiv"]
    source["balance_residual_kwh"] = (
        source["outage_load_kwh"] - source["pv_to_outage_kwh"]
        - source["battery_to_outage_kwh"] - source["unserved_outage_kwh"]
    )
    if float(source["balance_residual_kwh"].abs().max()) > ENERGY_TOLERANCE_KWH:
        raise ValueError("Outage-hour energy balance does not close")
    source["is_event_hour"] = source["event_outage_fraction"] > 0
    if int(source["is_event_hour"].sum()) != 3:
        raise ValueError("The selected outage should intersect exactly three model hours")
    source["hours_from_outage_start_at_hour_end"] = source["hour_index"] + 1.0 - start
    source["battery_soc_percent"] = 100.0 * source["battery_soc_kwh"] / BATTERY_ENERGY_KWH
    source["minimum_soc_percent"] = 100.0 * float(config["battery"]["soc_min_fraction"])
    source["event_start_hour"] = start
    source["event_end_hour"] = end
    source["scenario_id"] = SCENARIO_ID
    source["outage_seed"] = OUTAGE_SEED
    source["event_id"] = EVENT_ID

    affected = source.loc[source["is_event_hour"]]
    summary = {
        "case_key": CASE_KEY,
        "selection_rule": "median_pre_storage_event_net_deficit",
        "event_catalog_count": int(len(events)),
        "median_pre_storage_net_deficit_kwh": median_deficit,
        "scenario_id": SCENARIO_ID,
        "outage_seed": OUTAGE_SEED,
        "event_id": EVENT_ID,
        "analysis_year": int(event["start_analysis_year"]),
        "event_start_hour": start,
        "event_end_hour": end,
        "event_duration_hours": end - start,
        "affected_model_hours": 3,
        "pv_capacity_kw": PV_CAPACITY_KW,
        "battery_energy_kwh": BATTERY_ENERGY_KWH,
        "outage_load_kwh": float(affected["outage_load_kwh"].sum()),
        "pv_to_outage_kwh": float(affected["pv_to_outage_kwh"].sum()),
        "battery_to_outage_kwh": float(affected["battery_to_outage_kwh"].sum()),
        "unserved_outage_kwh": float(affected["unserved_outage_kwh"].sum()),
        "maximum_hourly_balance_residual_kwh": float(affected["balance_residual_kwh"].abs().max()),
        "saved_scenario_EUE_kwh_total": saved_eue,
        "replayed_scenario_EUE_kwh_total": replay_eue,
        "outage_profile_sha256": str(payload["outage_profile_sha256"]),
        "soc_sampling_note": "SOC is recorded at model-hour end; boundary hours may include grid-available time.",
        "outage_allocation_note": "Outage-only PV and BESS supply follows the greedy dispatch priority; grid does not serve interrupted load.",
    }
    columns = [
        "scenario_id", "outage_seed", "event_id", "analysis_year", "hour_index", "hour_of_year",
        "hours_from_outage_start_at_hour_end", "event_start_hour", "event_end_hour",
        "grid_availability", "event_outage_fraction", "is_event_hour", "critical_load_kwh",
        "outage_load_kwh", "pv_available_kwh", "pv_to_outage_kwh", "battery_to_outage_kwh",
        "unserved_outage_kwh", "balance_residual_kwh", "battery_soc_kwh", "battery_soc_percent",
        "minimum_soc_percent", "battery_charge_kwh", "grid_to_battery_kwh",
    ]
    return source[columns], summary


def plot_figure(source: pd.DataFrame, summary: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Draw a double-column two-panel figure from validated replay source data."""
    output_dir = Path(output_dir)
    mpl, plt = matplotlib_context(output_dir / BASE_NAME)
    apply_nature_style(mpl)
    affected = source.loc[source["is_event_hour"]].sort_values("hour_index").reset_index(drop=True)
    x = np.arange(1, len(affected) + 1, dtype=float)
    pv = affected["pv_to_outage_kwh"].to_numpy(dtype=float)
    battery = affected["battery_to_outage_kwh"].to_numpy(dtype=float)
    unserved = affected["unserved_outage_kwh"].to_numpy(dtype=float)
    demand = affected["outage_load_kwh"].to_numpy(dtype=float)

    fig, (balance_ax, soc_ax) = plt.subplots(
        2, 1, figsize=(7.2, 4.8), gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    fig.subplots_adjust(left=0.11, right=0.95, top=0.72, bottom=0.20, hspace=0.62)
    fig.text(0.11, 0.96, "Typical outage: PV and BESS serve all interrupted load",
             fontsize=9.0, fontweight="bold", ha="left", va="top")
    fig.text(
        0.11, 0.91,
        f"Brisbane single-family  |  PV {PV_CAPACITY_KW:g} kW  |  BESS {BATTERY_ENERGY_KWH:g} kWh  |  "
        f"Outage {summary['event_duration_hours']:.2f} h",
        fontsize=7, color=NATURE_PALETTE["neutral_dark"], ha="left", va="top",
    )
    fig.text(
        0.11, 0.865,
        f"Interrupted load {summary['outage_load_kwh']:.3f} kWh  =  "
        f"PV {summary['pv_to_outage_kwh']:.3f}  +  BESS {summary['battery_to_outage_kwh']:.3f} kWh"
        f"     |     Unserved {summary['unserved_outage_kwh']:.3f} kWh",
        fontsize=7, color=NATURE_PALETTE["neutral_black"], ha="left", va="top",
    )

    pv_bars = balance_ax.bar(x, pv, width=0.62, color=NATURE_PALETTE["teal"], label="PV to interrupted load")
    battery_bars = balance_ax.bar(x, battery, bottom=pv, width=0.62,
                                  color=NATURE_PALETTE["purple"], label="BESS to interrupted load")
    if np.any(unserved > ENERGY_TOLERANCE_KWH):
        balance_ax.bar(x, unserved, bottom=pv + battery, width=0.62, facecolor="none",
                       edgecolor=NATURE_PALETTE["red"], hatch="////", linewidth=0.9,
                       label="Unserved gap")
    load_line, = balance_ax.plot(x, demand, color=NATURE_PALETTE["neutral_black"],
                                 marker="o", markersize=3.6, linewidth=1.15,
                                 label="Interrupted load")
    balance_ax.set_xlim(0.45, 3.55)
    balance_ax.set_ylim(0, max(float(demand.max()) * 1.22, 0.1))
    balance_ax.set_ylabel("Outage energy (kWh / model hour)")
    balance_ax.set_xticks(x, [f"Hour {i}\n{fraction:.0%} interrupted"
                                for i, fraction in enumerate(affected["event_outage_fraction"], start=1)])
    balance_ax.set_title("a   Interrupted-load energy balance", loc="left", pad=9)
    balance_ax.spines["top"].set_visible(False)
    balance_ax.spines["right"].set_visible(False)
    handles = [pv_bars, battery_bars, load_line]
    labels = [handle.get_label() for handle in handles]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.10, 0.825),
               ncol=3, columnspacing=1.5, handlelength=1.5)

    relative = source["hours_from_outage_start_at_hour_end"].to_numpy(dtype=float)
    soc = source["battery_soc_percent"].to_numpy(dtype=float)
    soc_ax.axvspan(0, float(summary["event_duration_hours"]), color=NATURE_PALETTE["neutral_light"],
                   alpha=0.55, linewidth=0)
    soc_ax.plot(relative, soc, color=NATURE_PALETTE["blue"], marker="o",
                markersize=3.2, linewidth=1.15)
    minimum_soc = float(source["minimum_soc_percent"].iloc[0])
    soc_ax.axhline(minimum_soc, color=NATURE_PALETTE["red"], linestyle="--", linewidth=0.8)
    soc_ax.text(relative.max(), minimum_soc + 2.5, f"Minimum {minimum_soc:.0f}%",
                ha="right", va="bottom", fontsize=6.5, color=NATURE_PALETTE["red"])
    soc_ax.set_xlim(float(relative.min()) - 0.2, float(relative.max()) + 0.2)
    soc_ax.set_ylim(0, 103)
    soc_ax.set_yticks([0, 25, 50, 75, 100])
    soc_ax.set_ylabel("BESS SOC (%)")
    soc_ax.set_xlabel("Hours relative to outage start")
    soc_ax.set_title("b   Battery state of charge before and after the outage", loc="left", pad=8)
    soc_ax.text(float(summary["event_duration_hours"]) / 2.0, 100.0, "Outage",
                ha="center", va="top", fontsize=6.5, color=NATURE_PALETTE["neutral_dark"])
    fig.text(0.11, 0.065,
             "SOC is sampled at each model-hour end; the first and last affected hours include grid-available time.",
             fontsize=6.4, color=NATURE_PALETTE["neutral_dark"], ha="left", va="bottom")

    base = output_dir / BASE_NAME
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def run(results_dir: Path = DEFAULT_RESULTS_DIR, output_dir: Path | None = None) -> dict[str, Path]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir is not None else results_dir / "figures"
    source, summary = build_source_data(results_dir)
    paths = plot_figure(source, summary, output_dir)
    base = output_dir / BASE_NAME
    paths["source_data"] = write_source_data(base, source)
    summary_path = base.with_name(base.name + "_Event_Summary.json")
    summary["source_data_sha256"] = _sha256(paths["source_data"])
    summary["figure_sha256"] = {key: _sha256(path) for key, path in paths.items() if key != "source_data"}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    paths["event_summary"] = summary_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for label, path in run(args.results_dir, args.output_dir).items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
