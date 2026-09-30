"""Figure 5-style, 72-hour energy balance for the saved median-deficit outage.

Run with ``python -m src.figures.typical_outage_72h_energy_balance``. The
figure replays the validated paired run; it does not sample another outage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..dispatch import solve_dispatch_greedy
from ..io.data_loader import build_case_frame
from ..optimizer import battery_sizing
from ..resilience_metrics import DEFAULT_TOLERANCE
from .core import apply_nature_style, matplotlib_context, save_publication_figure, write_source_data
from .typical_outage_energy_balance import (
    BATTERY_ENERGY_KWH,
    CASE_KEY,
    DEFAULT_RESULTS_DIR,
    ENERGY_TOLERANCE_KWH,
    EVENT_ID,
    OUTAGE_SEED,
    PV_CAPACITY_KW,
    SCENARIO_ID,
    _event,
    _profile,
    _saved_case,
    _sha256,
    build_source_data as build_short_source_data,
)


BASE_NAME = "Brisbane_Typical_Outage_72h_Energy_Balance_Seed999_Event6"
WINDOW_HOURS = 72
PALETTE = {
    "pv": "#7EBC67",
    "grid": "#767676",
    "battery_discharge": "#D6B4D4",
    "served_load": "#272727",
    "battery_charge": "#F28E8C",
    "curtailment": "#B9B7D9",
    "export": "#9A9A9A",
    "unserved": "#B64342",
    "soc": "#7E3FA1",
    "boundary": "#767676",
}


def _window_bounds(event_start: float, event_end: float, total_hours: int) -> tuple[int, int]:
    outage_day_start = int(np.floor(event_start / 24.0)) * 24
    window_start = outage_day_start - 24
    window_end = window_start + WINDOW_HOURS
    if (
        window_start < 0
        or window_end > total_hours
        or event_end > outage_day_start + 24
        or event_end <= event_start
    ):
        raise ValueError("Selected event does not fit a complete three-day window")
    return window_start, window_end


def _balance_columns(
    source: pd.DataFrame, config: dict[str, Any], battery_energy_kwh: float = BATTERY_ENERGY_KWH,
) -> pd.DataFrame:
    if battery_energy_kwh <= 0:
        raise ValueError("Figure requires a positive battery capacity for the SOC axis")
    source = source.copy()
    source["critical_load_kwh"] = source["total_load_e_equiv_kwh"].astype(float)
    source["served_load_kwh"] = source["critical_load_kwh"] - source["unmet_load_kwh_equiv"]
    source["ac_balance_residual_kwh"] = (
        source["pv_available_kwh"] + source["grid_import_kwh"]
        + source["battery_discharge_kwh"] - source["served_load_kwh"]
        - source["battery_charge_kwh"] - source["pv_curtailed_kwh"]
        - source["pv_export_kwh"]
    )
    if float(source["ac_balance_residual_kwh"].abs().max()) > ENERGY_TOLERANCE_KWH:
        raise ValueError("Hourly AC energy balance does not close")
    eta = float(config["battery"]["roundtrip_efficiency"]) ** 0.5
    source["battery_conversion_loss_kwh"] = (
        source["battery_charge_kwh"] * (1.0 - eta)
        + source["battery_discharge_kwh"] * (1.0 / eta - 1.0)
    )
    source["battery_soc_fraction"] = source["battery_soc_kwh"] / battery_energy_kwh
    source["minimum_soc_fraction"] = float(config["battery"]["soc_min_fraction"])
    if (source["served_load_kwh"] < -ENERGY_TOLERANCE_KWH).any():
        raise ValueError("Served load is negative")
    return source


def build_source_data(results_dir: Path = DEFAULT_RESULTS_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the frozen event, replay dispatch, and return all 72 model hours."""
    results_dir = Path(results_dir)
    short_source, short_summary = build_short_source_data(results_dir)
    manifest = json.loads((results_dir / "manifest.json").read_text())
    config = json.loads((results_dir / "effective_config.json").read_text())
    case = _saved_case(results_dir, manifest)
    frame, _ = build_case_frame(case, config, PV_CAPACITY_KW)
    availability, profile = _profile(results_dir, len(frame))
    event, events = _event(results_dir)
    if str(event["outage_profile_sha256"]) != profile["outage_profile_sha256"]:
        raise ValueError("Selected event and outage profile differ")
    frame["grid_availability"] = availability
    frame["outage_fraction"] = 1.0 - availability
    dispatch = solve_dispatch_greedy(frame, battery_sizing(BATTERY_ENERGY_KWH, config), config)
    replay_eue = float(
        dispatch["unmet_load_kwh_equiv"]
        .where(dispatch["unmet_load_kwh_equiv"] > DEFAULT_TOLERANCE, 0.0).sum()
    )
    if not np.isclose(
        replay_eue, float(short_summary["saved_scenario_EUE_kwh_total"]),
        atol=ENERGY_TOLERANCE_KWH, rtol=0,
    ):
        raise ValueError("Full-horizon replay does not match the saved scenario")

    event_start, event_end = float(event["start_hour"]), float(event["end_hour"])
    window_start, window_end = _window_bounds(event_start, event_end, len(frame))
    overlapping = events.loc[
        (events["scenario_id"] == SCENARIO_ID)
        & (events["end_hour"] > window_start)
        & (events["start_hour"] < window_end)
    ]
    if len(overlapping) != 1 or int(overlapping.iloc[0]["event_id"]) != EVENT_ID:
        raise ValueError("Three-day window contains another outage event")

    source = dispatch.iloc[window_start:window_end].copy().reset_index(drop=True)
    source.insert(0, "plot_hour", np.arange(WINDOW_HOURS, dtype=int))
    source["analysis_year"] = frame["analysis_year"].iloc[window_start:window_end].to_numpy()
    source["hour_of_year"] = frame["hour_of_year"].iloc[window_start:window_end].to_numpy()
    source["day_label"] = np.repeat(
        ["Previous day", "Outage day", "Recovery day"], 24
    )
    source["event_outage_fraction"] = np.maximum(
        0.0,
        np.minimum(event_end, source["hour_index"].to_numpy(dtype=float) + 1.0)
        - np.maximum(event_start, source["hour_index"].to_numpy(dtype=float)),
    )
    if not np.allclose(
        source["event_outage_fraction"], source["outage_fraction"], atol=1e-9, rtol=0,
    ):
        raise ValueError("Saved outage profile differs from the selected event in the window")
    source["is_event_hour"] = source["event_outage_fraction"] > 0
    source = _balance_columns(source, config)
    source["outage_load_kwh"] = source["critical_load_kwh"] * source["event_outage_fraction"]
    source["pv_to_outage_kwh"] = np.minimum(source["pv_available_kwh"], source["outage_load_kwh"])
    remaining = np.maximum(0.0, source["outage_load_kwh"] - source["pv_to_outage_kwh"])
    source["battery_to_outage_kwh"] = np.minimum(source["battery_discharge_kwh"], remaining)
    source["outage_balance_residual_kwh"] = (
        source["outage_load_kwh"] - source["pv_to_outage_kwh"]
        - source["battery_to_outage_kwh"] - source["unmet_load_kwh_equiv"]
    )
    affected = source.loc[source["is_event_hour"]]
    if len(affected) != int(short_summary["affected_model_hours"]):
        raise ValueError("Affected hour count differs from the saved typical-outage figure")
    if float(affected["outage_balance_residual_kwh"].abs().max()) > ENERGY_TOLERANCE_KWH:
        raise ValueError("Selected outage energy balance does not close")
    short_hours = short_source.set_index("hour_index")
    common = source.set_index("hour_index").loc[short_hours.index]
    for column in ("battery_soc_kwh", "pv_available_kwh"):
        if not np.allclose(common[column], short_hours[column], atol=ENERGY_TOLERANCE_KWH, rtol=0):
            raise ValueError(f"Three-day replay differs from the existing figure: {column}")

    source["scenario_id"] = SCENARIO_ID
    source["outage_seed"] = OUTAGE_SEED
    source["event_id"] = EVENT_ID
    source["event_start_hour"] = event_start
    source["event_end_hour"] = event_end
    columns = [
        "scenario_id", "outage_seed", "event_id", "analysis_year", "hour_index", "hour_of_year",
        "plot_hour", "day_label", "event_start_hour", "event_end_hour", "grid_availability",
        "outage_fraction", "event_outage_fraction", "is_event_hour", "critical_load_kwh",
        "served_load_kwh", "unmet_load_kwh_equiv", "outage_load_kwh", "pv_available_kwh",
        "pv_to_outage_kwh", "pv_to_load_kwh", "pv_to_battery_kwh", "pv_curtailed_kwh",
        "pv_export_kwh", "grid_import_kwh", "grid_to_load_kwh", "grid_to_battery_kwh",
        "battery_charge_kwh", "battery_discharge_kwh", "battery_to_outage_kwh",
        "battery_soc_kwh", "battery_soc_fraction", "minimum_soc_fraction",
        "battery_conversion_loss_kwh", "ac_balance_residual_kwh", "outage_balance_residual_kwh",
    ]
    source = source[columns]
    summary = dict(short_summary)
    summary.update({
        "window_start_hour": window_start,
        "window_end_exclusive_hour": window_end,
        "window_hours": WINDOW_HOURS,
        "window_outage_event_count": len(overlapping),
        "maximum_hourly_ac_balance_residual_kwh": float(source["ac_balance_residual_kwh"].abs().max()),
        "window_unserved_kwh": float(source["unmet_load_kwh_equiv"].sum()),
        "window_battery_conversion_loss_kwh": float(source["battery_conversion_loss_kwh"].sum()),
        "ac_balance_note": "AC inputs equal served load, battery charge, PV curtailment and export. Battery conversion loss changes SOC and is not an additional AC sink.",
        "hourly_plot_note": "Each column is one model hour; a boundary hour can include both grid-available and interrupted minutes.",
    })
    return source, summary


def _soc_limits(energy_limits: tuple[float, float]) -> tuple[float, float]:
    lower, upper = energy_limits
    zero_fraction = -lower / (upper - lower)
    return -zero_fraction * 1.05 / (1.0 - zero_fraction), 1.05


def plot_figure(
    source: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    *,
    base_name: str = BASE_NAME,
    pv_capacity_kw: float = PV_CAPACITY_KW,
    battery_energy_kwh: float = BATTERY_ENERGY_KWH,
    design_status: str | None = None,
) -> dict[str, Path]:
    """Draw a single-design panel using the former Figure 5 visual grammar."""
    output_dir = Path(output_dir)
    mpl, plt = matplotlib_context(output_dir / base_name)
    apply_nature_style(mpl)
    if len(source) != WINDOW_HOURS or source["plot_hour"].tolist() != list(range(WINDOW_HOURS)):
        raise ValueError("Figure requires 72 ordered hourly records")
    x = source["plot_hour"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(left=0.105, right=0.875, bottom=0.18, top=0.72)
    outage_start = float(summary["event_start_hour"]) - float(summary["window_start_hour"])
    outage_end = float(summary["event_end_hour"]) - float(summary["window_start_hour"])
    ax.axvspan(outage_start, outage_end, color="#E6E6E6", alpha=0.70, linewidth=0, zorder=0)
    for hour in (24, 48):
        ax.axvline(hour, color=PALETTE["boundary"], linewidth=0.7, linestyle=":")

    handles: list[Any] = []
    positive = np.zeros(WINDOW_HOURS)
    for column, label, color in (
        ("pv_available_kwh", "PV available", PALETTE["pv"]),
        ("grid_import_kwh", "Grid import", PALETTE["grid"]),
        ("battery_discharge_kwh", "BESS discharge", PALETTE["battery_discharge"]),
    ):
        values = source[column].to_numpy(dtype=float)
        handles.append(ax.bar(x, values, bottom=positive, width=0.92, align="edge",
                              color=color, edgecolor="#3A3A3A", linewidth=0.25, label=label))
        positive += values

    negative = np.zeros(WINDOW_HOURS)
    for column, label, color in (
        ("served_load_kwh", "Served load", PALETTE["served_load"]),
        ("battery_charge_kwh", "BESS charge", PALETTE["battery_charge"]),
        ("pv_curtailed_kwh", "PV curtailed", PALETTE["curtailment"]),
        ("pv_export_kwh", "PV export", PALETTE["export"]),
    ):
        values = source[column].to_numpy(dtype=float)
        if column == "pv_export_kwh" and not np.any(values > ENERGY_TOLERANCE_KWH):
            continue
        handles.append(ax.bar(x, -values, bottom=negative, width=0.92, align="edge",
                              color=color, edgecolor="#3A3A3A", linewidth=0.25, label=label))
        negative -= values
    ax.axhline(0.0, color=PALETTE["served_load"], linewidth=0.75)
    magnitude = max(float(positive.max()), float((-negative).max()), 1.0)
    ax.set_ylim(float(negative.min()) - 0.14 * magnitude, float(positive.max()) + 0.14 * magnitude)
    ax.set_xlim(0, WINDOW_HOURS)
    ax.set_xticks(np.arange(0, WINDOW_HOURS + 1, 12))
    ax.set_xlabel("Hour in 72 h window")
    ax.set_ylabel("Hourly energy balance (kWh)")
    for label, location in (("Previous day", 12), ("Outage day", 36), ("Recovery day", 60)):
        ax.text(location, 1.035, label, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=6.8, color=PALETTE["served_load"])

    unmet = source["unmet_load_kwh_equiv"].to_numpy(dtype=float)
    if np.any(unmet > ENERGY_TOLERANCE_KWH):
        handles.append(ax.bar(x, -unmet, bottom=-source["served_load_kwh"].to_numpy(dtype=float),
                              width=0.92, align="edge", facecolor="none", edgecolor=PALETTE["unserved"],
                              hatch="////", linewidth=0.7, label="Unserved load (demand gap)"))

    soc_ax = ax.twinx()
    soc_line = soc_ax.plot(x + 0.5, source["battery_soc_fraction"], color=PALETTE["soc"],
                           linewidth=1.25, linestyle="-.", label="BESS SOC", zorder=6)[0]
    minimum_line = soc_ax.plot(x + 0.5, source["minimum_soc_fraction"],
                                color=PALETTE["unserved"], linewidth=0.8, linestyle=":",
                                label="Minimum SOC", zorder=5)[0]
    soc_ax.set_ylim(_soc_limits(ax.get_ylim()))
    soc_ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    soc_ax.set_ylabel("BESS SOC")
    soc_ax.spines["top"].set_visible(False)
    handles.extend([soc_line, minimum_line])
    fig.legend(handles, [handle.get_label() for handle in handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=4, handlelength=1.35)
    fig.text(0.105, 0.865, f"PV {pv_capacity_kw:g} kW / BESS {battery_energy_kwh:g} kWh  |  "
             f"Outage {summary['event_duration_hours']:.2f} h  |  "
             f"Interrupted load {summary['outage_load_kwh']:.2f} kWh; "
             f"unserved {summary['unserved_outage_kwh']:.2f} kWh",
             ha="left", va="center", fontsize=7.0, color=PALETTE["served_load"])
    if design_status:
        fig.text(0.105, 0.815, design_status, ha="left", va="center", fontsize=6.7,
                 color=PALETTE["served_load"])
    fig.text(0.105, 0.045,
             "Hourly AC flows close; battery loss affects SOC. Boundary-hour bars include available and outage minutes.",
             ha="left", va="center", fontsize=6.3, color=PALETTE["boundary"])

    base = output_dir / base_name
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
    summary["source_data_sha256"] = _sha256(paths["source_data"])
    summary["figure_sha256"] = {key: _sha256(path) for key, path in paths.items() if key != "source_data"}
    summary_path = base.with_name(base.name + "_Event_Summary.json")
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
