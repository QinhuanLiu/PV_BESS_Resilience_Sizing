"""Draw source-backed Brisbane building and stochastic outage profile figures.

Run with ``python -m src.figures.building_and_outage_profiles``. The figures use
the frozen inputs and outage realization from the completed paired run; no new
outage scenario or dispatch result is generated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..io.data_loader import build_case_frame
from ..resilience_metrics import electric_equivalent_building_load
from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    matplotlib_context,
    save_publication_figure,
    write_source_data,
)
from .typical_outage_energy_balance import (
    CASE_KEY,
    DEFAULT_RESULTS_DIR,
    EVENT_ID,
    OUTAGE_SEED,
    PV_CAPACITY_KW,
    SCENARIO_ID,
    _event,
    _profile,
    _saved_case,
)


ENERGY_BASE_NAME = "Brisbane_Typical_Daily_Load_PV_Profiles"
OUTAGE_BASE_PREFIX = "Brisbane_Stochastic_Outage_Profile_Seed999_Event6"
PAPER_WIDTH_IN = 180.0 / 25.4
YEAR_HOURS = 8760
DAY_HOURS = 24


def _frozen_sources(results_dir: Path) -> tuple[pd.DataFrame, dict, np.ndarray, pd.Series, pd.DataFrame]:
    """Load the validated source run, including the exact saved outage profile."""
    results_dir = Path(results_dir)
    validation = json.loads((results_dir / "validation.json").read_text())
    if not validation.get("passed"):
        raise ValueError("The saved source run has not passed validation")
    manifest = json.loads((results_dir / "manifest.json").read_text())
    config = json.loads((results_dir / "effective_config.json").read_text())
    if config != manifest["effective_config"] or int(config["analysis"]["years"]) != 15:
        raise ValueError("The selected figure requires the frozen 15-year configuration")
    case = _saved_case(results_dir, manifest)
    if case.case_key != CASE_KEY:
        raise ValueError("The source run is not the Brisbane single-family case")
    frame, _ = build_case_frame(case, config, PV_CAPACITY_KW)
    if len(frame) != 15 * YEAR_HOURS:
        raise ValueError("The saved case does not contain 15 complete model years")
    availability, profile = _profile(results_dir, len(frame))
    selected, all_events = _event(results_dir)
    if str(selected["outage_profile_sha256"]) != profile["outage_profile_sha256"]:
        raise ValueError("The selected event and availability profile differ")
    events = all_events.loc[
        (all_events["scenario_id"] == SCENARIO_ID)
        & (all_events["outage_seed"] == OUTAGE_SEED)
    ].sort_values("event_id").copy()
    if len(events) != 11 or events["event_id"].tolist() != list(range(1, 12)):
        raise ValueError("The saved seed 999 realization no longer has 11 events")
    if not (events["outage_profile_sha256"] == profile["outage_profile_sha256"]).all():
        raise ValueError("Event catalog and availability profile identities differ")
    return frame, config, availability, selected, events


def build_daily_source(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Average the 365 source days into one 24-hour profile."""
    source = frame.iloc[:YEAR_HOURS]
    if len(source) != YEAR_HOURS or not np.array_equal(
        source["hour_of_year"].to_numpy(dtype=int), np.arange(YEAR_HOURS)
    ):
        raise ValueError("The frozen load/PV source is not a complete 8760-hour year")
    technical = config["technical"]
    load = electric_equivalent_building_load(
        source, float(technical["cooling_cop"]), float(technical["heating_cop"])
    ).to_numpy(dtype=float)
    pv = source["pv_kwh"].to_numpy(dtype=float)
    if not np.isfinite(load).all() or not np.isfinite(pv).all() or np.any(load < 0) or np.any(pv < 0):
        raise ValueError("Load and PV input must be finite and nonnegative")
    daily = pd.DataFrame({
        "hour_start": np.arange(DAY_HOURS, dtype=int),
        "hour_end": np.arange(1, DAY_HOURS + 1, dtype=int),
        "hour_midpoint": np.arange(DAY_HOURS, dtype=float) + 0.5,
        "mean_building_load_kwh": load.reshape(-1, DAY_HOURS).mean(axis=0),
        "mean_pv_generation_kwh": pv.reshape(-1, DAY_HOURS).mean(axis=0),
        "source_days": np.full(DAY_HOURS, YEAR_HOURS // DAY_HOURS, dtype=int),
        "pv_capacity_kw": np.full(DAY_HOURS, PV_CAPACITY_KW),
    })
    if len(daily) != DAY_HOURS or not np.isclose(
        daily["mean_building_load_kwh"].sum(), load.sum() / 365.0, atol=1e-9
    ) or not np.isclose(daily["mean_pv_generation_kwh"].sum(), pv.sum() / 365.0, atol=1e-9):
        raise ValueError("The typical-day means do not reproduce annual daily energy")
    return daily


def _window_bounds(event: pd.Series, zoom_hours: int, total_hours: int) -> tuple[int, int]:
    if zoom_hours not in (24, 72):
        raise ValueError("zoom_hours must be 24 or 72")
    start, end = float(event["start_hour"]), float(event["end_hour"])
    outage_day_start = int(np.floor(start / DAY_HOURS)) * DAY_HOURS
    window_start = outage_day_start if zoom_hours == 24 else outage_day_start - DAY_HOURS
    window_end = window_start + zoom_hours
    if window_start < 0 or window_end > total_hours or end > outage_day_start + DAY_HOURS or end <= start:
        raise ValueError("The selected outage does not fit the complete requested window")
    return window_start, window_end


def build_outage_sources(
    availability: np.ndarray, event: pd.Series, events: pd.DataFrame, zoom_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return 15-year event marks and the exact hourly grid availability zoom."""
    window_start, window_end = _window_bounds(event, zoom_hours, len(availability))
    overlapping = events.loc[
        (events["end_hour"] > window_start) & (events["start_hour"] < window_end)
    ]
    if len(overlapping) != 1 or int(overlapping.iloc[0]["event_id"]) != EVENT_ID:
        raise ValueError("The zoom window contains another outage event")
    yearly_counts = events.groupby("start_analysis_year").size()
    timeline = events[[
        "event_id", "start_analysis_year", "start_hour_of_year", "start_hour",
        "end_hour", "duration_minutes",
    ]].copy()
    timeline = timeline.rename(columns={
        "start_analysis_year": "model_year", "start_hour_of_year": "hour_of_model_year",
    })
    timeline["events_in_model_year"] = timeline["model_year"].map(yearly_counts).astype(int)
    timeline["is_selected_event"] = timeline["event_id"].eq(EVENT_ID)

    hours = np.arange(window_start, window_end, dtype=int)
    start, end = float(event["start_hour"]), float(event["end_hour"])
    overlap = np.maximum(0.0, np.minimum(end, hours + 1.0) - np.maximum(start, hours))
    zoom = pd.DataFrame({
        "plot_hour": np.arange(zoom_hours, dtype=int),
        "hour_index": hours,
        "model_year": hours // YEAR_HOURS + 1,
        "hour_of_model_year": hours % YEAR_HOURS,
        "grid_availability": availability[hours],
        "event_outage_fraction": overlap,
        "event_start_hour": np.full(zoom_hours, start),
        "event_end_hour": np.full(zoom_hours, end),
    })
    if (
        not np.isfinite(zoom["grid_availability"]).all()
        or not zoom["grid_availability"].between(0, 1).all()
        or not np.allclose(zoom["grid_availability"].to_numpy(), 1.0 - overlap, atol=1e-9)
        or not np.isclose(overlap.sum(), float(event["duration_minutes"]) / 60.0, atol=1e-9)
    ):
        raise ValueError("Saved hourly availability does not reproduce the selected event")
    return timeline, zoom


def plot_daily_profile(daily: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    mpl, plt = matplotlib_context(output_dir / ENERGY_BASE_NAME)
    apply_nature_style(mpl)
    fig = plt.figure(figsize=(PAPER_WIDTH_IN, 3.55))
    ax = fig.add_axes([0.11, 0.19, 0.83, 0.56])
    fig.text(0.055, 0.975, "Brisbane · typical daily building energy profiles",
             ha="left", va="top", fontsize=10.5, fontweight="bold", color=NATURE_PALETTE["neutral_black"])
    fig.text(0.055, 0.902, "Annual mean of 365 days · full-building electric-equivalent load · PV 3.25 kW",
             ha="left", va="top", fontsize=7.2, color=NATURE_PALETTE["neutral_dark"])
    x = daily["hour_midpoint"].to_numpy(dtype=float)
    ax.plot(x, daily["mean_building_load_kwh"], color=NATURE_PALETTE["neutral_black"],
            linewidth=1.55, marker="o", markersize=2.1, label="Building load")
    ax.plot(x, daily["mean_pv_generation_kwh"], color=NATURE_PALETTE["teal"],
            linewidth=1.55, marker="o", markersize=2.1, label="PV generation")
    ax.set_xlim(0, DAY_HOURS)
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, DAY_HOURS + 1, 4))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Hourly energy (kWh)")
    ax.tick_params(direction="out", length=3)
    fig.legend(loc="upper center", bbox_to_anchor=(0.50, 0.835), ncol=2,
               handlelength=2.2, columnspacing=2.0)
    paths = save_publication_figure(fig, output_dir / ENERGY_BASE_NAME)
    plt.close(fig)
    return paths


def plot_outage_profile(
    timeline: pd.DataFrame, zoom: pd.DataFrame, output_dir: Path, zoom_hours: int,
) -> dict[str, Path]:
    mpl, plt = matplotlib_context(output_dir / OUTAGE_BASE_PREFIX)
    apply_nature_style(mpl)
    fig = plt.figure(figsize=(PAPER_WIDTH_IN, 6.85))
    top = fig.add_axes([0.11, 0.49, 0.81, 0.37])
    bottom = fig.add_axes([0.11, 0.15, 0.81, 0.22])
    fig.text(0.055, 0.98, "Brisbane · stochastic grid outage profile",
             ha="left", va="top", fontsize=10.5, fontweight="bold", color=NATURE_PALETTE["neutral_black"])
    fig.text(0.055, 0.94, "One saved 15-year realization · seed 999 · 11 interruptions",
             ha="left", va="top", fontsize=7.2, color=NATURE_PALETTE["neutral_dark"])
    fig.text(0.11, 0.895, "a   Event timing and duration", ha="left", va="top",
             fontsize=8, fontweight="bold", color=NATURE_PALETTE["neutral_black"])
    for year in range(1, 16):
        top.hlines(year, 0, YEAR_HOURS, color="#E5E5E5", linewidth=0.55, zorder=0)
        count = int((timeline["model_year"] == year).sum())
        top.text(9160, year, str(count), ha="center", va="center", fontsize=6.2,
                 color=NATURE_PALETTE["neutral_dark"])
    for row in timeline.itertuples(index=False):
        selected = bool(row.is_selected_event)
        color = NATURE_PALETTE["red"]
        top.vlines(row.hour_of_model_year, row.model_year - 0.30, row.model_year + 0.30,
                   color=color, linewidth=2.0 if selected else 1.35, zorder=3)
        label_to_left = row.hour_of_model_year > 8000
        label_x = row.hour_of_model_year - 80 if label_to_left else row.hour_of_model_year + 80
        top.text(label_x, row.model_year,
                 f"{row.duration_minutes / 60.0:.1f} h" + ("  [6]" if selected else ""),
                 ha="right" if label_to_left else "left", va="center", fontsize=6.1,
                 color=NATURE_PALETTE["neutral_black"] if selected else NATURE_PALETTE["neutral_mid"],
                 fontweight="bold" if selected else "normal")
    top.set_xlim(0, 9400)
    top.set_ylim(15.6, 0.4)
    top.set_yticks(np.arange(1, 16))
    top.set_xticks([0, 2000, 4000, 6000, 8000])
    top.set_xlabel("Hour of model year")
    top.set_ylabel("Model year")
    top.tick_params(direction="out", length=2.6, labelsize=6.1)
    top.text(9160, 0.55, "Count", ha="center", va="center", fontsize=6.1,
             color=NATURE_PALETTE["neutral_dark"])

    fig.text(0.11, 0.405, f"b   Grid availability around event 6 ({zoom_hours} h)",
             ha="left", va="top", fontsize=8, fontweight="bold", color=NATURE_PALETTE["neutral_black"])
    hours = zoom["plot_hour"].to_numpy(dtype=int)
    values = zoom["grid_availability"].to_numpy(dtype=float)
    event_start = float(zoom["event_start_hour"].iloc[0] - zoom["hour_index"].iloc[0])
    event_end = float(zoom["event_end_hour"].iloc[0] - zoom["hour_index"].iloc[0])
    bottom.axvspan(event_start, event_end, color=NATURE_PALETTE["red_light"], alpha=0.24, zorder=0)
    bottom.bar(hours, 1.0 - values, bottom=values, width=1.0, align="edge",
               color=NATURE_PALETTE["red_light"], alpha=0.45, edgecolor="none", zorder=1)
    bottom.stairs(values, np.arange(zoom_hours + 1), color=NATURE_PALETTE["neutral_black"],
                  linewidth=1.35, zorder=3)
    bottom.axhline(1.0, color=NATURE_PALETTE["neutral_mid"], linewidth=0.65, linestyle=":", zorder=0)
    bottom.set_xlim(0, zoom_hours)
    bottom.set_ylim(-0.04, 1.12)
    bottom.set_xticks(np.arange(0, zoom_hours + 1, 4 if zoom_hours == 24 else 12))
    bottom.set_yticks([0, 0.5, 1.0])
    bottom.set_ylabel("Grid availability")
    bottom.set_xlabel(f"Hour in {zoom_hours} h window")
    bottom.tick_params(direction="out", length=3)
    bottom.text((event_start + event_end) / 2.0, 0.18,
                f"{event_end - event_start:.2f} h", ha="center", va="center",
                fontsize=6.7, fontweight="bold", color=NATURE_PALETTE["red"])
    fig.text(0.11, 0.035,
             "Availability is the fraction of each model hour with grid service; boundary hours may be partial.",
             ha="left", va="bottom", fontsize=6.2, color=NATURE_PALETTE["neutral_mid"])
    base = output_dir / f"{OUTAGE_BASE_PREFIX}_{zoom_hours}h"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def run(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    output_dir: Path | None = None,
    zoom_hours: int = 24,
) -> dict[str, Path]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir is not None else results_dir / "figures"
    frame, config, availability, event, events = _frozen_sources(results_dir)
    daily = build_daily_source(frame, config)
    timeline, zoom = build_outage_sources(availability, event, events, zoom_hours)
    paths = plot_daily_profile(daily, output_dir)
    paths["daily_source_data"] = write_source_data(output_dir / ENERGY_BASE_NAME, daily)
    paths.update(plot_outage_profile(timeline, zoom, output_dir, zoom_hours))
    outage_base = output_dir / f"{OUTAGE_BASE_PREFIX}_{zoom_hours}h"
    paths["outage_events_source_data"] = write_source_data(
        outage_base.with_name(outage_base.name + "_Events"), timeline
    )
    yearly_counts = pd.DataFrame({"model_year": np.arange(1, 16, dtype=int)})
    yearly_counts["events_in_model_year"] = (
        yearly_counts["model_year"].map(timeline["model_year"].value_counts()).fillna(0).astype(int)
    )
    paths["outage_yearly_counts_source_data"] = write_source_data(
        outage_base.with_name(outage_base.name + "_Yearly_Counts"), yearly_counts
    )
    paths["outage_availability_source_data"] = write_source_data(
        outage_base.with_name(outage_base.name + "_Availability"), zoom
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--zoom-hours", type=int, choices=(24, 72), default=24)
    args = parser.parse_args()
    for label, path in run(args.results_dir, args.output_dir, args.zoom_hours).items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
