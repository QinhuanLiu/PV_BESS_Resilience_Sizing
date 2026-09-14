from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..io.data_loader import InputCase, build_case_frame
from ..dispatch import solve_dispatch
from ..optimizer import battery_sizing, scenario_config
from ..outage import OutageEvent, events_to_availability, simulate_outages
from ..resilience_metrics import electric_equivalent_building_load


OperationEventKind = Literal["typical", "severe"]

DEFAULT_PRE_HOURS = 24
DEFAULT_POST_HOURS = 24
DEFAULT_MECHANISM_RESERVE_FRACTION = 0.50
FIGURE_EVENT_KIND = {
    9: "typical",
    10: "severe",
}
FIGURE_BASE_NAMES = {
    9: "Figure_9_Typical_Outage_Prepare_Outage_Recovery",
    10: "Figure_10_Severe_Outage_Prepare_Outage_Recovery",
}


def build_outage_operation_sources(
    case: InputCase,
    config: dict[str, Any],
    optimal_designs: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
    event_kind: OperationEventKind,
    pre_hours: int = DEFAULT_PRE_HOURS,
    post_hours: int = DEFAULT_POST_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build source data for one prepare-outage-recovery operation figure."""
    if event_kind not in {"typical", "severe"}:
        raise ValueError("event_kind must be either 'typical' or 'severe'")
    if outage_scenarios.empty:
        raise ValueError("Outage operation figures require at least one outage scenario")

    aware_cfg = _operation_config(config, case.case_key)
    design = _select_operation_design(optimal_designs)
    pv_capacity = float(design["pv_capacity_kw"])
    battery_energy = float(design["battery_energy_kwh"])
    case_frame, pv_meta = build_case_frame(case, aware_cfg, pv_capacity)
    critical_load = _critical_load(case_frame, aware_cfg)
    event_catalog = _event_catalog(case, aware_cfg, outage_scenarios, case_frame, critical_load)
    selected = _select_event(event_catalog, event_kind)

    profile, _ = simulate_outages(
        saifi_per_year=float(aware_cfg["active_scenario"]["saifi_per_year"]),
        saidi_min_per_year=float(aware_cfg["active_scenario"]["saidi_min_per_year"]),
        hours=8760,
        seed=int(selected["outage_seed"]),
        forced_event_mode=str(aware_cfg.get("outage", {}).get("forced_event_mode", "match_saidi")),
    )
    dispatch_frame = case_frame.merge(profile, on="hour_index", how="left")
    sizing = battery_sizing(battery_energy, aware_cfg)
    dispatch = solve_dispatch(dispatch_frame, sizing, aware_cfg)

    summary = _event_summary(
        case,
        selected,
        event_kind,
        design,
        aware_cfg,
        sizing,
        pre_hours=pre_hours,
        post_hours=post_hours,
    )
    source = _dispatch_source(
        dispatch_frame,
        dispatch,
        critical_load,
        selected,
        summary.iloc[0],
        aware_cfg,
    )
    summary["window_UE_kwh"] = float(source["unmet_load_kwh_equiv"].astype(float).sum())
    summary["window_start_relative_hour"] = float(source["relative_hour"].min())
    summary["window_end_relative_hour"] = float(source["relative_hour"].max())
    return summary, source


def plot_outage_operation_figure(
    out: Path,
    case_key: str,
    event_summary: pd.DataFrame,
    dispatch_source: pd.DataFrame,
    figure_no: int,
) -> dict[str, Path]:
    """Draw a Nature-style prepare-outage-recovery operation figure."""
    if figure_no not in FIGURE_BASE_NAMES:
        raise ValueError("outage operation figure_no must be 9 or 10")
    if event_summary.empty or dispatch_source.empty:
        raise ValueError("outage operation figures require non-empty source data")

    from .core import NATURE_PALETTE, _matplotlib, _save_all, _style, _write_source_data

    mpl, plt = _matplotlib(out)
    _style(mpl)
    palette = {
        **NATURE_PALETTE,
        "pv": "#42949E",
        "battery": "#0F4D92",
        "grid": "#767676",
        "load": "#272727",
        "curtailment": "#CFCECE",
        "unserved": "#B64342",
        "reserve": "#9A4D8E",
    }
    source = dispatch_source.sort_values("relative_hour").reset_index(drop=True)
    summary = event_summary.iloc[0]
    x = source["relative_hour"].astype(float).to_numpy()
    outage_start = float(source["relative_outage_start_hour"].iloc[0])
    outage_end = float(source["relative_outage_end_hour"].iloc[0])
    title = _figure_title(int(figure_no), summary)

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(7.2, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [0.65, 1.35, 0.9, 1.0, 1.0], "hspace": 0.28},
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=8.5, fontweight="bold")
    for ax in axes:
        _shade_outage(ax, outage_start, outage_end)
        ax.axvline(0, color=palette["neutral_dark"], lw=0.7, ls="--")
        ax.axvline(outage_end, color=palette["neutral_dark"], lw=0.7, ls=":")

    axes[0].plot(x, source["grid_availability"], color=palette["grid"], lw=1.1)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_yticks([0, 1])
    axes[0].set_ylabel("Grid")
    _stage_labels(axes[0], x, outage_end, palette)

    _stacked_power_balance(axes[1], x, source, palette)
    axes[1].set_ylabel("Balance\n(kWh)")

    axes[2].bar(x, source["battery_discharge_kwh"], width=0.82, color=palette["battery"], label="Discharge")
    axes[2].bar(x, -source["battery_charge_kwh"], width=0.82, color=palette["teal"], alpha=0.65, label="Charge")
    axes[2].axhline(0, color=palette["neutral_black"], lw=0.7)
    axes[2].set_ylabel("BESS\n(kWh)")
    _add_top_legend_space(axes[2], top_fraction=0.22)
    _legend_in_top_space(axes[2], ncol=2, handlelength=1.3)

    axes[3].plot(x, source["battery_soc_fraction"], color=palette["battery"], lw=1.25, label="SOC")
    axes[3].plot(x, source["reserve_soc_fraction"], color=palette["reserve"], lw=0.95, ls="--", label="Reserve")
    axes[3].plot(x, source["minimum_soc_fraction"], color=palette["unserved"], lw=0.85, ls=":", label="Minimum")
    axes[3].set_ylim(0, max(1.0, float(source["battery_soc_fraction"].max()) * 1.28))
    axes[3].set_ylabel("SOC\n(fraction)")
    _legend_in_top_space(axes[3], ncol=3, handlelength=1.5, columnspacing=0.8)

    axes[4].bar(x, source["unmet_load_kwh_equiv"], width=0.82, color=palette["unserved"], alpha=0.78)
    axes[4].set_ylabel("Unserved\n(kWh)")
    ue_ax = axes[4].twinx()
    ue_ax.plot(x, source["cumulative_UE_kwh"], color=palette["neutral_black"], lw=1.2)
    ue_ax.set_ylabel("Cumulative\nUE (kWh)")
    ue_ax.spines["top"].set_visible(False)
    _outcome_note(axes[4], summary, palette)

    axes[-1].set_xlabel("Hours from outage start")
    for label, ax in zip("abcde", axes, strict=True):
        _panel_label(ax, label)

    base = out / f"{case_key}_{FIGURE_BASE_NAMES[figure_no]}"
    paths = _save_all(fig, base)
    plt.close(fig)
    source_path = _write_source_data(base, source)
    summary_path = base.with_name(base.name + "_Event_Summary_Source_Data.csv")
    event_summary.to_csv(summary_path, index=False)
    paths[f"{base.name}_source_data"] = source_path
    paths[f"{base.name}_event_summary_source_data"] = summary_path
    return paths


def _operation_config(config: dict[str, Any], case_key: str) -> dict[str, Any]:
    aware_cfg = scenario_config(config, case_key)
    aware_cfg["battery"] = deepcopy(aware_cfg["battery"])
    if aware_cfg["battery"].get("normal_operation_reserve_fraction") is None:
        aware_cfg["battery"]["normal_operation_reserve_fraction"] = min(
            DEFAULT_MECHANISM_RESERVE_FRACTION,
            float(aware_cfg["battery"]["soc_max_fraction"]),
        )
    return aware_cfg


def _select_operation_design(optimal_designs: pd.DataFrame) -> dict[str, Any]:
    for design_type in ("reliability_constrained_design", "minimum_total_cost_design"):
        matches = optimal_designs[
            (optimal_designs["design_type"].astype(str) == design_type)
            & optimal_designs.get("available", pd.Series(True, index=optimal_designs.index)).map(_available)
        ]
        if not matches.empty:
            row = dict(matches.iloc[0])
            row["selected_design_type"] = design_type
            return row
    raise ValueError("No available operation-design row found in optimal_designs")


def _critical_load(case_frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    if "critical_load_e_equiv_kwh" in case_frame.columns:
        return case_frame["critical_load_e_equiv_kwh"].astype(float)
    technical = config["technical"]
    return electric_equivalent_building_load(
        case_frame,
        float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0))),
        float(technical.get("heating_cop", 3.5)),
    )


def _event_catalog(
    case: InputCase,
    config: dict[str, Any],
    outage_scenarios: pd.DataFrame,
    case_frame: pd.DataFrame,
    critical_load: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenario = config["active_scenario"]
    for scenario_row in outage_scenarios.sort_values("frontier_scenario_id").to_dict("records"):
        seed = int(scenario_row["outage_seed"])
        _, events = simulate_outages(
            saifi_per_year=float(scenario["saifi_per_year"]),
            saidi_min_per_year=float(scenario["saidi_min_per_year"]),
            hours=8760,
            seed=seed,
            forced_event_mode=str(config.get("outage", {}).get("forced_event_mode", "match_saidi")),
        )
        for event in events.to_dict("records"):
            profile = _single_event_profile(event, len(case_frame))
            outage_fraction = profile["outage_fraction"].astype(float)
            net_deficit = np.maximum(
                critical_load.astype(float).to_numpy() - case_frame["pv_kwh"].astype(float).to_numpy(),
                0.0,
            ) * outage_fraction.to_numpy()
            rows.append(
                {
                    "case_key": case.case_key,
                    "city_key": case.city_key,
                    "building_key": case.building_key,
                    "frontier_scenario_id": int(scenario_row["frontier_scenario_id"]),
                    "outage_seed": seed,
                    "event_id": int(event["event_id"]),
                    "event_start_hour": float(event["start_hour"]),
                    "event_end_hour": float(event["end_hour"]),
                    "event_duration_hours": float(event["duration_minutes"]) / 60.0,
                    "event_pv_available_kwh": float((case_frame["pv_kwh"].astype(float) * outage_fraction).sum()),
                    "event_outage_weighted_load_kwh": float((critical_load.astype(float) * outage_fraction).sum()),
                    "event_net_deficit_kwh": float(net_deficit.sum()),
                }
            )
    catalog = pd.DataFrame(rows)
    if catalog.empty:
        raise ValueError("No outage events were available for operation-figure event selection")
    return catalog.sort_values(["frontier_scenario_id", "event_id"]).reset_index(drop=True)


def _single_event_profile(event: dict[str, Any], hours: int) -> pd.DataFrame:
    outage_event = OutageEvent(
        event_id=int(event["event_id"]),
        start_minute=float(event["start_hour"]) * 60.0,
        end_minute=float(event["end_hour"]) * 60.0,
    )
    availability = events_to_availability([outage_event], hours=hours)
    return pd.DataFrame(
        {
            "hour_index": np.arange(hours, dtype=int),
            "grid_availability": availability,
            "outage_fraction": 1.0 - availability,
        }
    )


def _select_event(catalog: pd.DataFrame, event_kind: OperationEventKind) -> dict[str, Any]:
    ranked = catalog.copy()
    if event_kind == "severe":
        return dict(ranked.sort_values(["event_net_deficit_kwh", "event_duration_hours"], ascending=[False, False]).iloc[0])
    median = float(ranked["event_net_deficit_kwh"].astype(float).median())
    ranked["distance_to_median_net_deficit_kwh"] = (ranked["event_net_deficit_kwh"].astype(float) - median).abs()
    return dict(
        ranked.sort_values(
            ["distance_to_median_net_deficit_kwh", "event_net_deficit_kwh", "frontier_scenario_id", "event_id"],
            ascending=[True, True, True, True],
        ).iloc[0]
    )


def _event_summary(
    case: InputCase,
    selected: dict[str, Any],
    event_kind: OperationEventKind,
    design: dict[str, Any],
    config: dict[str, Any],
    sizing: dict[str, float],
    pre_hours: int,
    post_hours: int,
) -> pd.DataFrame:
    event_start = float(selected["event_start_hour"])
    event_end = float(selected["event_end_hour"])
    start_hour = max(0, int(np.floor(event_start)) - int(pre_hours))
    end_hour = min(8759, int(np.ceil(event_end)) + int(post_hours))
    reserve_fraction = float(config["battery"].get("normal_operation_reserve_fraction", 0.0))
    battery_capacity = float(sizing["battery_energy_kwh"])
    return pd.DataFrame(
        [
            {
                "case_key": case.case_key,
                "city_key": case.city_key,
                "building_key": case.building_key,
                "event_kind": event_kind,
                "selected_design_type": design["selected_design_type"],
                "frontier_scenario_id": int(selected["frontier_scenario_id"]),
                "outage_seed": int(selected["outage_seed"]),
                "event_id": int(selected["event_id"]),
                "event_start_hour": event_start,
                "event_end_hour": event_end,
                "event_duration_hours": float(selected["event_duration_hours"]),
                "event_pv_available_kwh": float(selected["event_pv_available_kwh"]),
                "event_outage_weighted_load_kwh": float(selected["event_outage_weighted_load_kwh"]),
                "event_net_deficit_kwh": float(selected["event_net_deficit_kwh"]),
                "plot_window_start_hour": int(start_hour),
                "plot_window_end_hour": int(end_hour),
                "pre_hours": int(pre_hours),
                "post_hours": int(post_hours),
                "pv_capacity_kw": float(design["pv_capacity_kw"]),
                "battery_energy_kwh": battery_capacity,
                "battery_power_kw": float(sizing["battery_power_kw"]),
                "reserve_fraction": reserve_fraction,
                "reserve_soc_kwh": battery_capacity * reserve_fraction,
                "minimum_soc_fraction": float(config["battery"]["soc_min_fraction"]),
                "minimum_soc_kwh": battery_capacity * float(config["battery"]["soc_min_fraction"]),
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
    for column in ("timestamp",):
        if column in frame.columns and column not in merged.columns:
            merged[column] = frame[column].values
    merged["critical_load_kwh"] = critical_load.astype(float).values
    merged["outage_weighted_load_kwh"] = merged["critical_load_kwh"] * frame["outage_fraction"].astype(float).values
    merged["served_load_kwh"] = np.maximum(
        0.0,
        merged["critical_load_kwh"].astype(float) - merged["unmet_load_kwh_equiv"].astype(float),
    )
    merged["net_load_without_bess_kwh"] = np.maximum(
        0.0,
        merged["critical_load_kwh"].astype(float) - merged["pv_available_kwh"].astype(float),
    )
    window = merged[
        (merged["hour_index"].astype(int) >= int(summary["plot_window_start_hour"]))
        & (merged["hour_index"].astype(int) <= int(summary["plot_window_end_hour"]))
    ].copy()
    event_start = float(selected["event_start_hour"])
    event_end = float(selected["event_end_hour"])
    window["relative_hour"] = window["hour_index"].astype(float) - event_start
    window["relative_outage_start_hour"] = 0.0
    window["relative_outage_end_hour"] = event_end - event_start
    window["event_kind"] = str(summary["event_kind"])
    window["selected_design_type"] = str(summary["selected_design_type"])
    window["frontier_scenario_id"] = int(selected["frontier_scenario_id"])
    window["outage_seed"] = int(selected["outage_seed"])
    window["event_id"] = int(selected["event_id"])
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
        "event_kind",
        "selected_design_type",
        "frontier_scenario_id",
        "outage_seed",
        "event_id",
        "hour_index",
        "timestamp",
        "relative_hour",
        "relative_outage_start_hour",
        "relative_outage_end_hour",
        "grid_availability",
        "outage_fraction",
        "critical_load_kwh",
        "outage_weighted_load_kwh",
        "served_load_kwh",
        "pv_available_kwh",
        "pv_used_kwh",
        "pv_curtailed_kwh",
        "grid_import_kwh",
        "battery_charge_kwh",
        "battery_discharge_kwh",
        "battery_soc_kwh",
        "battery_soc_fraction",
        "reserve_soc_kwh",
        "reserve_soc_fraction",
        "minimum_soc_kwh",
        "minimum_soc_fraction",
        "unmet_load_kwh_equiv",
        "cumulative_UE_kwh",
        "net_load_without_bess_kwh",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "reserve_fraction",
    ]
    return window[[column for column in columns if column in window.columns]].reset_index(drop=True)


def _figure_title(figure_no: int, summary: pd.Series) -> str:
    label = "Typical outage" if figure_no == 9 else "Severe outage"
    return (
        f"{label}: duration {float(summary['event_duration_hours']):.1f} h, "
        f"net deficit {float(summary['event_net_deficit_kwh']) / 1000.0:.2f} MWh, "
        f"BESS {float(summary['battery_energy_kwh']) / 1000.0:.2f} MWh"
    )


def _shade_outage(ax: Any, start: float, end: float) -> None:
    ax.axvspan(start, end, color="#E6E6E6", alpha=0.55, lw=0)


def _stage_labels(ax: Any, x: np.ndarray, outage_end: float, palette: dict[str, str]) -> None:
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    y = 1.08
    ax.text((xmin + 0.0) / 2.0, y, "Preparation", ha="center", va="bottom", fontsize=6.3, color=palette["neutral_dark"])
    ax.text(outage_end / 2.0, y, "Outage", ha="center", va="bottom", fontsize=6.3, color=palette["neutral_dark"])
    ax.text((outage_end + xmax) / 2.0, y, "Recovery", ha="center", va="bottom", fontsize=6.3, color=palette["neutral_dark"])


def _stacked_power_balance(ax: Any, x: np.ndarray, source: pd.DataFrame, palette: dict[str, str]) -> None:
    positive_series = [
        ("PV used", source["pv_used_kwh"].astype(float), palette["pv"]),
        ("BESS discharge", source["battery_discharge_kwh"].astype(float), palette["battery"]),
        ("Grid import", source["grid_import_kwh"].astype(float), palette["grid"]),
    ]
    negative_series = [
        ("Served load", source["served_load_kwh"].astype(float), palette["neutral_dark"]),
        ("BESS charge", source["battery_charge_kwh"].astype(float), palette["teal"]),
        ("PV curtailed", source["pv_curtailed_kwh"].astype(float), palette["curtailment"]),
        ("Unserved", source["unmet_load_kwh_equiv"].astype(float), palette["unserved"]),
    ]
    bottom = np.zeros(len(source))
    for label, values, color in positive_series:
        ax.bar(x, values, bottom=bottom, width=0.82, color=color, label=label, alpha=0.82)
        bottom += values.to_numpy()
    bottom = np.zeros(len(source))
    for label, values, color in negative_series:
        plotted = -values.to_numpy()
        ax.bar(x, plotted, bottom=bottom, width=0.82, color=color, label=label, alpha=0.72)
        bottom += plotted
    ax.axhline(0, color=palette["neutral_black"], lw=0.7)
    _add_top_legend_space(ax, top_fraction=0.24)
    _legend_in_top_space(ax, ncol=4, handlelength=1.1, columnspacing=0.7)


def _add_top_legend_space(ax: Any, *, top_fraction: float) -> None:
    ymin, ymax = ax.get_ylim()
    span = max(float(ymax - ymin), 1.0)
    ax.set_ylim(float(ymin), float(ymax) + span * top_fraction)


def _legend_in_top_space(ax: Any, *, ncol: int, handlelength: float, columnspacing: float = 0.8) -> None:
    ax.legend(
        loc="upper right",
        borderaxespad=0.0,
        ncol=ncol,
        handlelength=handlelength,
        columnspacing=columnspacing,
        handletextpad=0.35,
    )


def _outcome_note(ax: Any, summary: pd.Series, palette: dict[str, str]) -> None:
    note = (
        f"UE {float(summary.get('window_UE_kwh', 0.0)) / 1000.0:.2f} MWh\n"
        f"Deficit {float(summary['event_net_deficit_kwh']) / 1000.0:.2f} MWh"
    )
    ax.text(
        0.01,
        0.95,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=palette["neutral_black"],
    )


def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.075,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def _available(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "available"}
    return bool(value)
