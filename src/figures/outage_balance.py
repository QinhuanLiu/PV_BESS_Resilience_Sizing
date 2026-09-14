from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io.data_loader import InputCase, build_case_frame
from ..dispatch import solve_dispatch
from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    matplotlib_context,
    save_publication_figure,
    write_source_data as _write_source_data,
)
from ..optimizer import battery_sizing, scenario_config
from ..outage import simulate_outages
from .outage_operation import (
    _critical_load,
    _event_catalog,
    _select_operation_design,
)


FIGURE_OUTAGE_CONTEXT_NO = 9
FIGURE_OUTAGE_CONTEXT_NAME = "Figure_9_Outage_Context_72h"
SELECTION_RULE = "max_event_net_deficit"
WINDOW_HOURS = 72
OUTAGE_DAY_START_PLOT_HOUR = 24
OUTAGE_DAY_END_PLOT_HOUR = 48


def build_outage_energy_balance_sources(
    case: InputCase,
    config: dict[str, Any],
    optimal_designs: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build source data for the result Figure 9 72 h outage context figure."""
    if outage_scenarios.empty:
        raise ValueError("Outage energy-balance figures require at least one outage scenario")

    aware_cfg = _baseline_energy_balance_config(config, case.case_key)
    design = _select_operation_design(optimal_designs)
    pv_capacity = float(design["pv_capacity_kw"])
    battery_energy = float(design["battery_energy_kwh"])
    case_frame, _ = build_case_frame(case, aware_cfg, pv_capacity)
    critical_load = _critical_load(case_frame, aware_cfg)
    event_catalog = _event_catalog(case, aware_cfg, outage_scenarios, case_frame, critical_load)
    selected = _select_max_impact_event(event_catalog, total_hours=len(case_frame))

    profile, _ = simulate_outages(
        saifi_per_year=float(aware_cfg["active_scenario"]["saifi_per_year"]),
        saidi_min_per_year=float(aware_cfg["active_scenario"]["saidi_min_per_year"]),
        hours=len(case_frame),
        seed=int(selected["outage_seed"]),
        forced_event_mode=str(aware_cfg.get("outage", {}).get("forced_event_mode", "match_saidi")),
    )
    dispatch_frame = case_frame.merge(profile, on="hour_index", how="left")
    sizing = battery_sizing(battery_energy, aware_cfg)
    dispatch = solve_dispatch(dispatch_frame, sizing, aware_cfg)

    summary = _event_summary(case, selected, design, sizing, aware_cfg, len(case_frame))
    source = _dispatch_source(dispatch_frame, dispatch, critical_load, selected, summary.iloc[0], aware_cfg)
    summary["window_UE_kwh"] = float(source["unmet_load_kwh_equiv"].astype(float).sum())
    summary["window_conversion_loss_kwh"] = float(source["conversion_loss_kwh"].astype(float).sum())
    summary["window_start_hour"] = int(source["hour_index"].astype(int).min())
    summary["window_end_hour"] = int(source["hour_index"].astype(int).max())
    return summary, source


def plot_outage_context_figure(
    out: Path,
    case_key: str,
    event_summary: pd.DataFrame,
    dispatch_source: pd.DataFrame,
) -> dict[str, Path]:
    """Draw Figure 9: the 72 h outage interval and operating context."""
    summary, source = _validated_sources(event_summary, dispatch_source)
    mpl, plt = matplotlib_context(out / FIGURE_OUTAGE_CONTEXT_NAME)
    apply_nature_style(mpl)
    palette = _palette()

    x = source["plot_hour"].astype(float).to_numpy()
    x_mid = x + 0.5
    outage_start, outage_end = _plot_outage_bounds(source)
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.905, bottom=0.18, top=0.73)
    _shade_outage(ax, outage_start, outage_end)
    _draw_day_boundaries(ax, palette)

    load_line = ax.plot(
        x_mid,
        source["critical_load_kwh"].astype(float),
        color=palette["neutral_black"],
        lw=1.15,
        label="Critical load",
    )[0]
    pv_line = ax.plot(
        x_mid,
        source["pv_available_kwh"].astype(float),
        color=palette["pv"],
        lw=1.05,
        label="PV availability",
    )[0]
    outage_load = source["outage_weighted_load_kwh"].astype(float).to_numpy()
    outage_patch = ax.fill_between(
        x_mid,
        0,
        outage_load,
        where=outage_load > 0,
        color=palette["unserved"],
        alpha=0.18,
        linewidth=0,
        label="Outage-weighted load",
    )
    ax.set_xlim(0.0, float(WINDOW_HOURS))
    ax.set_ylabel("Hourly energy (kWh)")
    ax.set_xlabel("Hour in 72 h window")
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    _add_day_labels(ax, palette)

    availability_ax = ax.twinx()
    availability_line = availability_ax.step(
        x_mid,
        source["grid_availability"].astype(float),
        where="mid",
        color=palette["grid"],
        lw=1.0,
        label="Grid availability",
    )[0]
    availability_ax.set_ylim(-0.05, 1.05)
    availability_ax.set_yticks([0.0, 0.5, 1.0])
    availability_ax.set_ylabel("Grid availability")
    availability_ax.spines["top"].set_visible(False)

    _annotate_context(ax, summary, outage_start, outage_end, palette)
    _panel_label(ax, "a")
    handles = [load_line, pv_line, outage_patch, availability_line]
    labels = [handle.get_label() for handle in handles]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4, handlelength=1.8)

    base = out / f"{case_key}_{FIGURE_OUTAGE_CONTEXT_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    paths[f"{base.name}_source_data"] = _write_source_data(base, source)
    paths[f"{base.name}_event_summary_source_data"] = _write_event_summary(base, summary)
    return paths


def _baseline_energy_balance_config(config: dict[str, Any], case_key: str) -> dict[str, Any]:
    """Resolve the baseline scenario config without adding the legacy outage reserve."""
    resolved = scenario_config(config, case_key)
    resolved["battery"] = dict(resolved["battery"])
    resolved["battery"].pop("normal_operation_reserve_fraction", None)
    return resolved


def _select_max_impact_event(catalog: pd.DataFrame, total_hours: int) -> dict[str, Any]:
    if catalog.empty:
        raise ValueError("No outage events were available for energy-balance event selection")
    ranked = catalog.copy()
    ranked["window_start_hour"] = ranked["event_start_hour"].astype(float).map(_window_start_hour)
    ranked["window_end_exclusive_hour"] = ranked["window_start_hour"].astype(int) + WINDOW_HOURS
    valid = ranked[
        (ranked["window_start_hour"].astype(int) >= 0)
        & (ranked["window_end_exclusive_hour"].astype(int) <= int(total_hours))
    ].copy()
    if valid.empty:
        raise ValueError("No outage event can support a complete 72 h window")
    return dict(
        valid.sort_values(
            ["event_net_deficit_kwh", "event_duration_hours", "frontier_scenario_id", "event_id"],
            ascending=[False, False, True, True],
        ).iloc[0]
    )


def _event_summary(
    case: InputCase,
    selected: dict[str, Any],
    design: dict[str, Any],
    sizing: dict[str, float],
    config: dict[str, Any],
    total_hours: int,
) -> pd.DataFrame:
    window_start = _window_start_hour(float(selected["event_start_hour"]))
    window_end_exclusive = window_start + WINDOW_HOURS
    if window_start < 0 or window_end_exclusive > total_hours:
        raise ValueError("Selected outage event cannot support a complete 72 h window")
    battery_capacity = float(sizing["battery_energy_kwh"])
    reserve_fraction = float(config["battery"].get("normal_operation_reserve_fraction", config["battery"]["soc_min_fraction"]))
    return pd.DataFrame(
        [
            {
                "case_key": case.case_key,
                "city_key": case.city_key,
                "building_key": case.building_key,
                "selection_rule": SELECTION_RULE,
                "selected_design_type": design["selected_design_type"],
                "frontier_scenario_id": int(selected["frontier_scenario_id"]),
                "outage_seed": int(selected["outage_seed"]),
                "event_id": int(selected["event_id"]),
                "event_start_hour": float(selected["event_start_hour"]),
                "event_end_hour": float(selected["event_end_hour"]),
                "event_duration_hours": float(selected["event_duration_hours"]),
                "event_pv_available_kwh": float(selected["event_pv_available_kwh"]),
                "event_outage_weighted_load_kwh": float(selected["event_outage_weighted_load_kwh"]),
                "event_net_deficit_kwh": float(selected["event_net_deficit_kwh"]),
                "event_day_start_hour": int(np.floor(float(selected["event_start_hour"]) / 24.0) * 24),
                "window_start_hour": int(window_start),
                "window_end_exclusive_hour": int(window_end_exclusive),
                "window_hours": WINDOW_HOURS,
                "pv_capacity_kw": float(design["pv_capacity_kw"]),
                "battery_energy_kwh": battery_capacity,
                "battery_power_kw": float(sizing["battery_power_kw"]),
                "reserve_fraction": reserve_fraction,
                "reserve_soc_kwh": battery_capacity * reserve_fraction,
                "minimum_soc_fraction": float(config["battery"]["soc_min_fraction"]),
                "minimum_soc_kwh": battery_capacity * float(config["battery"]["soc_min_fraction"]),
                "roundtrip_efficiency": float(config["battery"]["roundtrip_efficiency"]),
            }
        ]
    )


def _dispatch_source(
    frame: pd.DataFrame,
    dispatch: pd.DataFrame,
    critical_load: pd.Series,
    selected: dict[str, Any],
    summary: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    merged = dispatch.copy()
    if "timestamp" in frame.columns and "timestamp" not in merged.columns:
        merged["timestamp"] = frame["timestamp"].values
    merged["critical_load_kwh"] = critical_load.astype(float).values
    merged["outage_weighted_load_kwh"] = merged["critical_load_kwh"] * frame["outage_fraction"].astype(float).values
    merged["served_load_kwh"] = np.maximum(
        0.0,
        merged["critical_load_kwh"].astype(float) - merged["unmet_load_kwh_equiv"].astype(float),
    )

    start_hour = int(summary["window_start_hour"])
    end_exclusive = int(summary["window_end_exclusive_hour"])
    window = merged[
        (merged["hour_index"].astype(int) >= start_hour)
        & (merged["hour_index"].astype(int) < end_exclusive)
    ].copy()
    if len(window) != WINDOW_HOURS:
        raise ValueError(f"Outage energy-balance source expected {WINDOW_HOURS} rows, got {len(window)}")

    eta = float(config["battery"]["roundtrip_efficiency"]) ** 0.5
    window["conversion_loss_kwh"] = _conversion_loss(
        window["battery_charge_kwh"],
        window["battery_discharge_kwh"],
        eta,
    )
    event_start = float(selected["event_start_hour"])
    event_end = float(selected["event_end_hour"])
    window["plot_hour"] = window["hour_index"].astype(int) - start_hour
    window["hour_of_day"] = window["plot_hour"].astype(int) % 24
    window["day_label"] = np.select(
        [
            window["plot_hour"].astype(int) < OUTAGE_DAY_START_PLOT_HOUR,
            window["plot_hour"].astype(int) < OUTAGE_DAY_END_PLOT_HOUR,
        ],
        ["Previous day", "Outage day"],
        default="Recovery day",
    )
    window["relative_outage_hour"] = window["hour_index"].astype(float) - event_start
    window["relative_outage_start_hour"] = 0.0
    window["relative_outage_end_hour"] = event_end - event_start
    window["is_outage_interval"] = window["outage_fraction"].astype(float) > 1e-12
    window["selection_rule"] = str(summary["selection_rule"])
    window["selected_design_type"] = str(summary["selected_design_type"])
    window["frontier_scenario_id"] = int(selected["frontier_scenario_id"])
    window["outage_seed"] = int(selected["outage_seed"])
    window["event_id"] = int(selected["event_id"])
    window["event_start_hour"] = event_start
    window["event_end_hour"] = event_end
    window["event_duration_hours"] = float(selected["event_duration_hours"])
    window["event_net_deficit_kwh"] = float(selected["event_net_deficit_kwh"])
    window["pv_capacity_kw"] = float(summary["pv_capacity_kw"])
    window["battery_energy_kwh"] = float(summary["battery_energy_kwh"])
    window["reserve_fraction"] = float(summary["reserve_fraction"])
    window["reserve_soc_kwh"] = float(summary["reserve_soc_kwh"])
    window["minimum_soc_kwh"] = float(summary["minimum_soc_kwh"])
    window["battery_soc_fraction"] = window["battery_soc_kwh"].astype(float) / max(float(summary["battery_energy_kwh"]), 1e-9)
    window["reserve_soc_fraction"] = float(summary["reserve_soc_kwh"]) / max(float(summary["battery_energy_kwh"]), 1e-9)
    window["minimum_soc_fraction"] = float(summary["minimum_soc_fraction"])
    window["cumulative_UE_kwh"] = window["unmet_load_kwh_equiv"].astype(float).cumsum()

    columns = [
        "selection_rule",
        "selected_design_type",
        "frontier_scenario_id",
        "outage_seed",
        "event_id",
        "event_start_hour",
        "event_end_hour",
        "event_duration_hours",
        "event_net_deficit_kwh",
        "hour_index",
        "timestamp",
        "plot_hour",
        "day_label",
        "hour_of_day",
        "relative_outage_hour",
        "relative_outage_start_hour",
        "relative_outage_end_hour",
        "grid_availability",
        "outage_fraction",
        "is_outage_interval",
        "critical_load_kwh",
        "outage_weighted_load_kwh",
        "served_load_kwh",
        "pv_available_kwh",
        "pv_used_kwh",
        "grid_import_kwh",
        "battery_charge_kwh",
        "battery_discharge_kwh",
        "pv_curtailed_kwh",
        "conversion_loss_kwh",
        "unmet_load_kwh_equiv",
        "cumulative_UE_kwh",
        "battery_soc_kwh",
        "battery_soc_fraction",
        "reserve_soc_kwh",
        "reserve_soc_fraction",
        "minimum_soc_kwh",
        "minimum_soc_fraction",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "reserve_fraction",
    ]
    return window[[column for column in columns if column in window.columns]].reset_index(drop=True)


def _conversion_loss(
    battery_charge_kwh: pd.Series | np.ndarray,
    battery_discharge_kwh: pd.Series | np.ndarray,
    eta: float,
) -> np.ndarray:
    clipped_eta = max(float(eta), 1e-9)
    charge = np.asarray(battery_charge_kwh, dtype=float)
    discharge = np.asarray(battery_discharge_kwh, dtype=float)
    charge_loss = charge * max(0.0, 1.0 - clipped_eta)
    discharge_loss = discharge * max(0.0, (1.0 / clipped_eta) - 1.0)
    return np.maximum(0.0, charge_loss + discharge_loss)


def _soc_ylim_for_zero_alignment(power_ylim: tuple[float, float], soc_upper: float = 1.05) -> tuple[float, float]:
    lower, upper = (float(power_ylim[0]), float(power_ylim[1]))
    if lower >= 0.0 or upper <= 0.0:
        return 0.0, float(soc_upper)
    zero_fraction = (0.0 - lower) / (upper - lower)
    zero_fraction = min(max(zero_fraction, 1e-6), 1.0 - 1e-6)
    soc_lower = -zero_fraction * float(soc_upper) / (1.0 - zero_fraction)
    return soc_lower, float(soc_upper)


def _validated_sources(event_summary: pd.DataFrame, dispatch_source: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    if event_summary.empty or dispatch_source.empty:
        raise ValueError("Outage energy-balance figures require non-empty source data")
    missing = {
        "plot_hour",
        "grid_availability",
        "critical_load_kwh",
        "pv_available_kwh",
        "pv_used_kwh",
        "grid_import_kwh",
        "battery_charge_kwh",
        "battery_discharge_kwh",
        "pv_curtailed_kwh",
        "conversion_loss_kwh",
        "battery_soc_fraction",
    } - set(dispatch_source.columns)
    if missing:
        raise ValueError("Outage energy-balance source is missing required column(s): " + ", ".join(sorted(missing)))
    source = dispatch_source.copy().sort_values("plot_hour").reset_index(drop=True)
    if len(source) != WINDOW_HOURS:
        raise ValueError(f"Outage energy-balance figures require exactly {WINDOW_HOURS} hourly records")
    return event_summary.iloc[0], source


def _window_start_hour(event_start_hour: float) -> int:
    event_day_start = int(np.floor(float(event_start_hour) / 24.0) * 24)
    return event_day_start - OUTAGE_DAY_START_PLOT_HOUR


def _plot_outage_bounds(source: pd.DataFrame) -> tuple[float, float]:
    event_start = float(source["event_start_hour"].iloc[0])
    window_start = float(source["hour_index"].astype(float).min())
    duration = float(source["event_duration_hours"].iloc[0])
    outage_start = event_start - window_start
    outage_end = outage_start + duration
    return outage_start, outage_end


def _palette() -> dict[str, str]:
    return {
        **NATURE_PALETTE,
        "pv": "#7EBC67",
        "grid": "#767676",
        "battery_discharge": "#D6B4D4",
        "battery_charge": "#F28E8C",
        "curtailment": "#B9B7D9",
        "loss": "#9A9A9A",
        "load": "#272727",
        "unserved": "#B64342",
        "soc": "#7E3FA1",
        "reserve": "#8C6BB1",
    }


def _stack_positive_balance(ax: Any, x: np.ndarray, source: pd.DataFrame, palette: dict[str, str]) -> list[Any]:
    series = [
        ("PV used", source["pv_used_kwh"].astype(float), palette["pv"]),
        ("BESS discharge", source["battery_discharge_kwh"].astype(float), palette["battery_discharge"]),
    ]
    bottom = np.zeros(len(source))
    handles: list[Any] = []
    for label, values, color in series:
        handle = ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.92,
            align="edge",
            color=color,
            edgecolor="#3A3A3A",
            linewidth=0.25,
            label=label,
        )
        handles.append(handle)
        bottom += values.to_numpy()
    grid_handle = _plot_grid_import_available_segments(ax, source, bottom, palette)
    handles.insert(1, grid_handle)
    return handles


def _stack_negative_balance(ax: Any, x: np.ndarray, source: pd.DataFrame, palette: dict[str, str]) -> list[Any]:
    series = [
        ("Served load", source["served_load_kwh"].astype(float), palette["load"]),
        ("BESS charge", source["battery_charge_kwh"].astype(float), palette["battery_charge"]),
        ("PV curtailed", source["pv_curtailed_kwh"].astype(float), palette["curtailment"]),
        ("Conversion loss", source["conversion_loss_kwh"].astype(float), palette["loss"]),
        ("Unserved load", source["unmet_load_kwh_equiv"].astype(float), palette["unserved"]),
    ]
    bottom = np.zeros(len(source))
    handles: list[Any] = []
    for label, values, color in series:
        plotted = -values.to_numpy()
        handle = ax.bar(
            x,
            plotted,
            bottom=bottom,
            width=0.92,
            align="edge",
            color=color,
            edgecolor="#3A3A3A",
            linewidth=0.25,
            label=label,
        )
        handles.append(handle)
        bottom += plotted
    return handles


def _plot_grid_import_available_segments(
    ax: Any,
    source: pd.DataFrame,
    bottom: np.ndarray,
    palette: dict[str, str],
) -> Any:
    handle = None
    for row_index, row in source.reset_index(drop=True).iterrows():
        value = float(row["grid_import_kwh"])
        if value <= 0.0:
            continue
        segments = _available_segments_for_hour(row)
        if not segments:
            continue
        total_width = sum(width for _, width in segments)
        for start, width in segments:
            segment_value = value * width / max(total_width, 1e-9)
            plotted = ax.bar(
                start,
                segment_value,
                bottom=float(bottom[row_index]),
                width=width,
                align="edge",
                color=palette["grid"],
                edgecolor="#3A3A3A",
                linewidth=0.25,
                label="Grid import" if handle is None else "_nolegend_",
            )
            if handle is None:
                handle = plotted
    if handle is None:
        handle = ax.bar([], [], color=palette["grid"], label="Grid import")
    return handle


def _available_segments_for_hour(row: pd.Series) -> list[tuple[float, float]]:
    plot_hour = float(row["plot_hour"])
    hour_start = plot_hour
    hour_end = plot_hour + 1.0
    outage_start = float(row["event_start_hour"]) - float(row["hour_index"]) + plot_hour
    outage_end = float(row["event_end_hour"]) - float(row["hour_index"]) + plot_hour
    outage_start = max(hour_start, min(hour_end, outage_start))
    outage_end = max(hour_start, min(hour_end, outage_end))
    if outage_end <= hour_start or outage_start >= hour_end:
        return [(hour_start + 0.04, 0.92)]
    segments: list[tuple[float, float]] = []
    if outage_start > hour_start:
        segments.append((hour_start, outage_start - hour_start))
    if outage_end < hour_end:
        segments.append((outage_end, hour_end - outage_end))
    return [(start, width) for start, width in segments if width > 1e-9]


def _set_symmetric_balance_limits(ax: Any, source: pd.DataFrame) -> None:
    positive = (
        source["pv_used_kwh"].astype(float)
        + source["grid_import_kwh"].astype(float)
        + source["battery_discharge_kwh"].astype(float)
    )
    negative = (
        source["served_load_kwh"].astype(float)
        + source["battery_charge_kwh"].astype(float)
        + source["pv_curtailed_kwh"].astype(float)
        + source["conversion_loss_kwh"].astype(float)
        + source["unmet_load_kwh_equiv"].astype(float)
    )
    upper = max(float(positive.max()), 1.0)
    lower = -max(float(negative.max()), 1.0)
    pad = 0.14 * max(upper, abs(lower))
    ax.set_ylim(lower - pad, upper + pad)


def _draw_day_boundaries(ax: Any, palette: dict[str, str]) -> None:
    for hour in (OUTAGE_DAY_START_PLOT_HOUR, OUTAGE_DAY_END_PLOT_HOUR):
        ax.axvline(hour, color=palette["neutral_mid"], lw=0.7, ls=":")


def _shade_outage(ax: Any, start: float, end: float) -> None:
    ax.axvspan(start, end, color="#E6E6E6", alpha=0.70, lw=0, zorder=0)


def _add_day_labels(ax: Any, palette: dict[str, str]) -> None:
    labels = [("Previous day", 12.0), ("Outage day", 36.0), ("Recovery day", 60.0)]
    y = 1.025
    for label, x in labels:
        ax.text(x, y, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.4, color=palette["neutral_dark"])


def _annotate_context(ax: Any, summary: pd.Series, outage_start: float, outage_end: float, palette: dict[str, str]) -> None:
    y_top = ax.get_ylim()[1]
    ax.annotate(
        f"Outage {float(summary['event_duration_hours']):.2f} h\nNet deficit {float(summary['event_net_deficit_kwh']) / 1000.0:.2f} MWh",
        xy=((outage_start + outage_end) / 2.0, y_top * 0.78),
        xytext=(42.0, y_top * 0.94),
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": palette["neutral_dark"]},
        ha="left",
        va="top",
        fontsize=6.4,
        color=palette["neutral_black"],
    )


def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.075,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def _write_event_summary(base: Path, summary: pd.Series) -> Path:
    path = base.with_name(base.name + "_Event_Summary_Source_Data.csv")
    pd.DataFrame([summary.to_dict()]).to_csv(path, index=False)
    return path
