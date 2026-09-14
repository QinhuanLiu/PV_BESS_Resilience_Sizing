from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..dispatch import solve_dispatch
from ..io.data_loader import InputCase, build_case_frame
from ..optimizer import battery_sizing, scenario_config
from .core import apply_nature_style, matplotlib_context, save_publication_figure
from .core import write_source_data as _write_source_data
from .display_units import (
    figure_display_units,
    format_design_label as format_display_design_label,
    format_energy_from_kwh,
)
from .outage_balance import (
    WINDOW_HOURS,
    _add_day_labels,
    _dispatch_source,
    _draw_day_boundaries,
    _palette,
    _panel_label,
    _plot_outage_bounds,
    _shade_outage,
    _soc_ylim_for_zero_alignment,
    _stack_negative_balance,
    _stack_positive_balance,
    _window_start_hour,
)
from .outage_operation import _critical_load


FIGURE_5_NAME = "Figure_5_Baseline_Outage_Energy_Balance_Comparison"
SELECTION_RULE_AVOIDED_UE = "max_avoided_UE"
SELECTION_RULE_FALLBACK_REFERENCE_UE = "fallback_max_reference_UE"
DESIGN_CASES = ("no_pv_bess", "pv20_no_bess", "pv20_bess10")
COMPARISON_DESIGN_CASE = "pv20_bess10"
DESIGN_LABELS = {
    "no_pv_bess": "No PV/BESS",
    "pv20_no_bess": "20 MW PV / 0 MWh Battery",
    "pv20_bess10": "20 MW PV / 10 MWh Battery",
}


def make_baseline_outage_balance_comparison_figure(
    out: Path,
    case_key: str,
    case: InputCase,
    config: dict[str, Any],
    candidate_summary: pd.DataFrame,
    outage_profile: pd.DataFrame,
    outage_events: pd.DataFrame,
    optimal_dispatch: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Build and draw baseline Figure 5 using the active baseline outage profile."""
    event_summary, dispatch_source = build_baseline_outage_balance_comparison_sources(
        case,
        config,
        candidate_summary,
        outage_profile,
        outage_events,
        optimal_dispatch=optimal_dispatch,
    )
    base = out / f"{case_key}_{FIGURE_5_NAME}"
    paths: dict[str, Path] = {
        "source_data": _write_source_data(base, dispatch_source),
        "event_summary_source_data": _write_event_summary(base, event_summary),
    }
    paths.update(plot_baseline_outage_balance_comparison(out, case_key, event_summary, dispatch_source))
    return paths


def build_baseline_outage_balance_comparison_sources(
    case: InputCase,
    config: dict[str, Any],
    candidate_summary: pd.DataFrame,
    outage_profile: pd.DataFrame,
    outage_events: pd.DataFrame,
    optimal_dispatch: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Figure 5 source data for the fixed Figure 4B comparison designs."""
    _ = optimal_dispatch
    if outage_profile.empty:
        raise ValueError("Baseline outage balance requires a non-empty outage profile.")
    if outage_events.empty:
        raise ValueError("Baseline outage balance requires at least one outage event.")

    aware_cfg = scenario_config(config, case.case_key)
    design_rows = _select_baseline_design_rows(candidate_summary)
    no_system_row = design_rows["no_pv_bess"]
    comparison_row = design_rows[COMPARISON_DESIGN_CASE]
    dispatch_by_design: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for design_case in DESIGN_CASES:
        row = design_rows[design_case]
        dispatch_by_design[design_case] = _dispatch_for_design(
            case,
            aware_cfg,
            outage_profile,
            pv_capacity_kw=float(row["pv_capacity_kw"]),
            battery_energy_kwh=float(row["battery_energy_kwh"]),
        )

    reference_frame, reference_dispatch = dispatch_by_design["no_pv_bess"]
    comparison_frame, comparison_dispatch = dispatch_by_design[COMPARISON_DESIGN_CASE]
    critical_load = _critical_load(comparison_frame, aware_cfg)
    event_catalog = build_baseline_outage_event_catalog(
        outage_events,
        reference_dispatch,
        comparison_dispatch,
        comparison_frame,
        critical_load,
        aware_cfg,
    )
    selected = _select_baseline_outage_event(event_catalog, total_hours=len(comparison_frame))

    summary = _comparison_summary(
        case,
        selected,
        no_system_row,
        comparison_row,
        aware_cfg,
    )
    sources = []
    for design_case in DESIGN_CASES:
        frame, dispatch = dispatch_by_design[design_case]
        row = design_rows[design_case]
        design_summary = _design_summary(summary.iloc[0], design_case, row, aware_cfg)
        source = _dispatch_source(
            frame,
            dispatch,
            critical_load,
            selected,
            design_summary,
            aware_cfg,
        )
        source.insert(0, "design_case_label", _design_label(design_case, row))
        source.insert(0, "design_case", design_case)
        source["event_reference_UE_kwh"] = float(selected["reference_UE_kwh"])
        source["event_optimal_UE_kwh"] = float(selected["optimal_UE_kwh"])
        source["event_comparison_UE_kwh"] = float(selected["optimal_UE_kwh"])
        source["avoided_UE_kwh"] = float(selected["avoided_UE_kwh"])
        source["avoided_lost_load_cost_sgd"] = float(selected["avoided_lost_load_cost_sgd"])
        source["event_pv_bess_support_kwh"] = float(selected["pv_bess_support_kwh"])
        sources.append(source)

    dispatch_source = validate_baseline_outage_balance_source(pd.concat(sources, ignore_index=True))
    summary["window_reference_UE_kwh"] = _window_ue(dispatch_source, "no_pv_bess")
    summary["window_optimal_UE_kwh"] = _window_ue(dispatch_source, COMPARISON_DESIGN_CASE)
    summary["window_comparison_UE_kwh"] = summary["window_optimal_UE_kwh"]
    summary["window_avoided_UE_kwh"] = summary["window_reference_UE_kwh"] - summary["window_comparison_UE_kwh"]
    summary["window_start_hour"] = int(dispatch_source["hour_index"].astype(int).min())
    summary["window_end_hour"] = int(dispatch_source["hour_index"].astype(int).max())
    return summary, dispatch_source


def build_baseline_outage_event_catalog(
    outage_events: pd.DataFrame,
    reference_dispatch: pd.DataFrame,
    optimal_dispatch: pd.DataFrame,
    optimal_frame: pd.DataFrame,
    critical_load: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Rankable event table for baseline Figure 5."""
    required = {"event_id", "start_hour", "end_hour", "duration_minutes"}
    missing = required - set(outage_events.columns)
    if missing:
        raise ValueError("Baseline outage events are missing required column(s): " + ", ".join(sorted(missing)))

    rows: list[dict[str, Any]] = []
    voll = float(config["active_scenario"]["voll_sgd_per_kwh"])
    seed = int(config.get("outage", {}).get("seed", 42))
    for event in outage_events.sort_values("event_id").itertuples(index=False):
        start = float(getattr(event, "start_hour"))
        end = float(getattr(event, "end_hour"))
        hour_mask = _event_hour_mask(optimal_frame["hour_index"], start, end)
        if not bool(hour_mask.any()):
            continue
        ref_event = reference_dispatch.loc[hour_mask].copy()
        opt_event = optimal_dispatch.loc[hour_mask].copy()
        frame_event = optimal_frame.loc[hour_mask].copy()
        critical_event = critical_load.loc[hour_mask].astype(float)
        outage_fraction = frame_event["outage_fraction"].astype(float)
        reference_ue = float(ref_event["unmet_load_kwh_equiv"].astype(float).sum())
        optimal_ue = float(opt_event["unmet_load_kwh_equiv"].astype(float).sum())
        pv_bess_support = float(
            opt_event["pv_used_kwh"].astype(float).mul(outage_fraction).sum()
            + opt_event["battery_discharge_kwh"].astype(float).sum()
        )
        rows.append(
            {
                "frontier_scenario_id": 0,
                "outage_seed": seed,
                "event_id": int(getattr(event, "event_id")),
                "event_start_hour": start,
                "event_end_hour": end,
                "event_duration_hours": float(getattr(event, "duration_minutes")) / 60.0,
                "event_pv_available_kwh": float(frame_event["pv_kwh"].astype(float).mul(outage_fraction).sum()),
                "event_outage_weighted_load_kwh": float(critical_event.mul(outage_fraction).sum()),
                "event_net_deficit_kwh": float(
                    np.maximum(
                        critical_event.to_numpy() - frame_event["pv_kwh"].astype(float).to_numpy(),
                        0.0,
                    ).dot(outage_fraction.to_numpy())
                ),
                "reference_UE_kwh": reference_ue,
                "optimal_UE_kwh": optimal_ue,
                "avoided_UE_kwh": reference_ue - optimal_ue,
                "avoided_lost_load_cost_sgd": (reference_ue - optimal_ue) * voll,
                "pv_bess_support_kwh": pv_bess_support,
            }
        )
    if not rows:
        raise ValueError("No baseline outage events were available for Figure 5 selection.")
    return pd.DataFrame(rows)


def validate_baseline_outage_balance_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "design_case",
        "design_case_label",
        "plot_hour",
        "hour_index",
        "grid_availability",
        "critical_load_kwh",
        "pv_available_kwh",
        "pv_used_kwh",
        "grid_import_kwh",
        "battery_charge_kwh",
        "battery_discharge_kwh",
        "pv_curtailed_kwh",
        "conversion_loss_kwh",
        "unmet_load_kwh_equiv",
        "battery_soc_fraction",
        "event_reference_UE_kwh",
        "event_optimal_UE_kwh",
        "event_comparison_UE_kwh",
        "avoided_UE_kwh",
        "event_pv_bess_support_kwh",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Baseline outage balance source is missing required column(s): " + ", ".join(sorted(missing)))
    if source.empty:
        raise ValueError("Baseline outage balance source data is empty.")
    validated = source.copy()
    validated["design_case"] = validated["design_case"].astype(str)
    if set(validated["design_case"].unique()) != set(DESIGN_CASES):
        expected = ", ".join(DESIGN_CASES)
        raise ValueError(f"Baseline outage balance source must contain exactly: {expected}.")
    counts = validated.groupby("design_case").size().to_dict()
    if any(counts.get(design_case, 0) != WINDOW_HOURS for design_case in DESIGN_CASES):
        raise ValueError(f"Baseline outage balance requires {WINDOW_HOURS} hourly records per design case.")
    validated["design_case"] = pd.Categorical(validated["design_case"], categories=list(DESIGN_CASES), ordered=True)
    validated = validated.sort_values(["design_case", "plot_hour"]).reset_index(drop=True)
    validated["design_case"] = validated["design_case"].astype(str)
    return validated


def plot_baseline_outage_balance_comparison(
    out: Path,
    case_key: str,
    event_summary: pd.DataFrame,
    dispatch_source: pd.DataFrame,
) -> dict[str, Path]:
    """Draw baseline Figure 5 as a three-panel outage energy-balance comparison."""
    if event_summary.empty:
        raise ValueError("Baseline outage balance requires non-empty event summary source data.")
    source = validate_baseline_outage_balance_source(dispatch_source)
    summary = event_summary.iloc[0]
    units = figure_display_units(source)
    mpl, plt = matplotlib_context(out / FIGURE_5_NAME)
    apply_nature_style(mpl)
    palette = _palette()

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.75), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.895, bottom=0.10, top=0.79, hspace=0.30)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    shared_ylim = _shared_balance_ylim(source)
    soc_handles: list[Any] = []

    for index, design_case in enumerate(DESIGN_CASES):
        ax = axes[index]
        panel_source = source[source["design_case"] == design_case].sort_values("plot_hour").reset_index(drop=True)
        x = panel_source["plot_hour"].astype(float).to_numpy()
        x_mid = x + 0.5
        outage_start, outage_end = _plot_outage_bounds(panel_source)
        _shade_outage(ax, outage_start, outage_end)
        _draw_day_boundaries(ax, palette)
        positive_handles = _stack_positive_balance(ax, x, panel_source, palette)
        negative_handles = _stack_negative_balance(ax, x, panel_source, palette)
        ax.axhline(0.0, color=palette["neutral_black"], lw=0.75)
        ax.set_xlim(0.0, float(WINDOW_HOURS))
        ax.set_ylim(shared_ylim)
        ax.set_ylabel("Hourly energy\nbalance (kWh)")
        ax.set_title(_plot_design_case_label(panel_source.iloc[0], units), loc="left", pad=4)
        _panel_label(ax, chr(ord("a") + index))
        _annotate_panel_ue(ax, panel_source, design_case, palette, units.use_battery_kwh)
        if index == 0:
            legend_handles = [*positive_handles, *negative_handles]
            legend_labels = [handle.get_label() for handle in legend_handles]
        if design_case == COMPARISON_DESIGN_CASE:
            soc_ax = ax.twinx()
            soc_line = soc_ax.plot(
                x_mid,
                panel_source["battery_soc_fraction"].astype(float),
                color=palette["soc"],
                lw=1.25,
                ls="-.",
                label="SOC",
                zorder=6,
            )[0]
            min_line = soc_ax.plot(
                x_mid,
                panel_source["minimum_soc_fraction"].astype(float),
                color=palette["unserved"],
                lw=0.8,
                ls=":",
                label="Minimum SOC",
                zorder=5,
            )[0]
            soc_ax.set_ylim(_soc_ylim_for_zero_alignment(ax.get_ylim(), soc_upper=1.05))
            soc_ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            soc_ax.set_ylabel("BESS SOC")
            soc_ax.spines["top"].set_visible(False)
            soc_handles = [soc_line, min_line]

    axes[-1].set_xticks([0, 12, 24, 36, 48, 60, 72])
    axes[-1].set_xlabel("Hour in 72 h window")
    _add_day_labels(axes[0], palette)
    _annotate_event_summary(fig, summary, palette, units.use_battery_kwh)
    handles = [*legend_handles, *soc_handles]
    labels = [*legend_labels, *[handle.get_label() for handle in soc_handles]]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.972), ncol=5, handlelength=1.35)

    base = out / f"{case_key}_{FIGURE_5_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _select_baseline_design_rows(candidate_summary: pd.DataFrame) -> dict[str, pd.Series]:
    required = {"pv_capacity_kw", "battery_energy_kwh", "pv_bess_system_annual_cost_sgd"}
    missing = required - set(candidate_summary.columns)
    if missing:
        raise ValueError("Baseline outage balance candidate summary is missing column(s): " + ", ".join(sorted(missing)))
    if candidate_summary.empty:
        raise ValueError("Baseline outage balance candidate summary is empty.")
    prepared = candidate_summary.copy()
    for column in required:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return {
        "no_pv_bess": _select_exact_design_row(prepared, 0.0, 0.0, "No PV/BESS"),
        "pv20_no_bess": _select_reference_pv_only_row(prepared),
        "pv20_bess10": _select_reference_pv_bess_row(prepared),
    }


def _select_exact_design_row(
    candidate_summary: pd.DataFrame,
    pv_capacity_kw: float,
    battery_energy_kwh: float,
    design_label: str,
) -> pd.Series:
    matches = candidate_summary[
        np.isclose(candidate_summary["pv_capacity_kw"].astype(float), float(pv_capacity_kw))
        & np.isclose(candidate_summary["battery_energy_kwh"].astype(float), float(battery_energy_kwh))
    ].copy()
    if matches.empty:
        raise ValueError(
            "Baseline outage balance requires an existing candidate summary row for "
            f"{design_label} ({pv_capacity_kw:.0f} kW PV, {battery_energy_kwh:.0f} kWh BESS)."
        )
    return matches.sort_values("pv_bess_system_annual_cost_sgd").iloc[0]


def _select_reference_pv_only_row(candidate_summary: pd.DataFrame) -> pd.Series:
    try:
        return _select_exact_design_row(candidate_summary, 20_000.0, 0.0, "20 MW PV / 0 MWh Battery")
    except ValueError:
        no_bess = candidate_summary[np.isclose(candidate_summary["battery_energy_kwh"].astype(float), 0.0)].copy()
        no_bess = no_bess[no_bess["pv_capacity_kw"].astype(float) > 0.0]
        if no_bess.empty:
            raise ValueError("Baseline outage balance requires at least one PV-only candidate row.") from None
        max_pv = float(no_bess["pv_capacity_kw"].astype(float).max())
        matches = no_bess[np.isclose(no_bess["pv_capacity_kw"].astype(float), max_pv)].copy()
        return matches.sort_values("pv_bess_system_annual_cost_sgd").iloc[0]


def _select_reference_pv_bess_row(candidate_summary: pd.DataFrame) -> pd.Series:
    try:
        return _select_exact_design_row(candidate_summary, 20_000.0, 10_000.0, "20 MW PV / 10 MWh Battery")
    except ValueError:
        pv_bess = candidate_summary[
            (candidate_summary["pv_capacity_kw"].astype(float) > 0.0)
            & (candidate_summary["battery_energy_kwh"].astype(float) > 0.0)
        ].copy()
        if pv_bess.empty:
            raise ValueError("Baseline outage balance requires at least one PV-BESS candidate row.") from None
        return pv_bess.sort_values("pv_bess_system_annual_cost_sgd").iloc[0]


def _design_label(design_case: str, row: pd.Series) -> str:
    if design_case == "no_pv_bess":
        return DESIGN_LABELS[design_case]
    return f"{float(row['pv_capacity_kw']) / 1000.0:g} MW PV / {float(row['battery_energy_kwh']) / 1000.0:g} MWh Battery"


def _plot_design_case_label(row: pd.Series, units: Any) -> str:
    return format_display_design_label(row, units)


def _dispatch_for_design(
    case: InputCase,
    config: dict[str, Any],
    outage_profile: pd.DataFrame,
    pv_capacity_kw: float,
    battery_energy_kwh: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, _ = build_case_frame(case, config, float(pv_capacity_kw))
    dispatch_frame = frame.merge(outage_profile, on="hour_index", how="left")
    if dispatch_frame[["grid_availability", "outage_fraction"]].isna().any().any():
        raise ValueError("Baseline outage profile did not align with the case frame.")
    dispatch = solve_dispatch(dispatch_frame, battery_sizing(float(battery_energy_kwh), config), config)
    return dispatch_frame, dispatch


def _select_baseline_outage_event(catalog: pd.DataFrame, total_hours: int) -> dict[str, Any]:
    ranked = catalog.copy()
    ranked["window_start_hour"] = ranked["event_start_hour"].astype(float).map(_window_start_hour)
    ranked["window_end_exclusive_hour"] = ranked["window_start_hour"].astype(int) + WINDOW_HOURS
    valid = ranked[
        (ranked["window_start_hour"].astype(int) >= 0)
        & (ranked["window_end_exclusive_hour"].astype(int) <= int(total_hours))
    ].copy()
    if valid.empty:
        raise ValueError("No baseline outage event can support a complete 72 h Figure 5 window.")
    positive = valid[valid["avoided_UE_kwh"].astype(float) > 1e-9].copy()
    if positive.empty:
        selected = valid.sort_values(
            ["reference_UE_kwh", "event_duration_hours", "event_id"],
            ascending=[False, False, True],
        ).iloc[0].to_dict()
        selected["selection_rule"] = SELECTION_RULE_FALLBACK_REFERENCE_UE
        return selected
    selected = positive.sort_values(
        [
            "avoided_UE_kwh",
            "avoided_lost_load_cost_sgd",
            "pv_bess_support_kwh",
            "event_duration_hours",
            "event_id",
        ],
        ascending=[False, False, False, False, True],
    ).iloc[0].to_dict()
    selected["selection_rule"] = SELECTION_RULE_AVOIDED_UE
    return selected


def _comparison_summary(
    case: InputCase,
    selected: dict[str, Any],
    no_system_row: pd.Series,
    comparison_row: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    window_start = int(selected["window_start_hour"])
    window_end_exclusive = int(selected["window_end_exclusive_hour"])
    return pd.DataFrame(
        [
            {
                "case_key": case.case_key,
                "city_key": case.city_key,
                "building_key": case.building_key,
                "selection_rule": str(selected["selection_rule"]),
                "outage_seed": int(selected["outage_seed"]),
                "event_id": int(selected["event_id"]),
                "event_start_hour": float(selected["event_start_hour"]),
                "event_end_hour": float(selected["event_end_hour"]),
                "event_duration_hours": float(selected["event_duration_hours"]),
                "event_pv_available_kwh": float(selected["event_pv_available_kwh"]),
                "event_outage_weighted_load_kwh": float(selected["event_outage_weighted_load_kwh"]),
                "event_net_deficit_kwh": float(selected["event_net_deficit_kwh"]),
                "event_reference_UE_kwh": float(selected["reference_UE_kwh"]),
                "event_optimal_UE_kwh": float(selected["optimal_UE_kwh"]),
                "event_comparison_UE_kwh": float(selected["optimal_UE_kwh"]),
                "avoided_UE_kwh": float(selected["avoided_UE_kwh"]),
                "avoided_lost_load_cost_sgd": float(selected["avoided_lost_load_cost_sgd"]),
                "event_pv_bess_support_kwh": float(selected["pv_bess_support_kwh"]),
                "event_day_start_hour": int(np.floor(float(selected["event_start_hour"]) / 24.0) * 24),
                "window_start_hour": window_start,
                "window_end_exclusive_hour": window_end_exclusive,
                "window_hours": WINDOW_HOURS,
                "reference_pv_capacity_kw": float(no_system_row["pv_capacity_kw"]),
                "reference_battery_energy_kwh": float(no_system_row["battery_energy_kwh"]),
                "optimal_pv_capacity_kw": float(comparison_row["pv_capacity_kw"]),
                "optimal_battery_energy_kwh": float(comparison_row["battery_energy_kwh"]),
                "comparison_design_case": COMPARISON_DESIGN_CASE,
                "comparison_pv_capacity_kw": float(comparison_row["pv_capacity_kw"]),
                "comparison_battery_energy_kwh": float(comparison_row["battery_energy_kwh"]),
                "voll_sgd_per_kwh": float(config["active_scenario"]["voll_sgd_per_kwh"]),
            }
        ]
    )


def _design_summary(
    summary: pd.Series,
    design_case: str,
    row: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    battery_energy = float(row["battery_energy_kwh"])
    sizing = battery_sizing(battery_energy, config)
    reserve_fraction = float(config["battery"].get("normal_operation_reserve_fraction", config["battery"]["soc_min_fraction"]))
    minimum_soc_fraction = float(config["battery"]["soc_min_fraction"])
    if battery_energy <= 0.0:
        reserve_fraction = 0.0
        minimum_soc_fraction = 0.0
    data = summary.to_dict()
    data.update(
        {
            "selected_design_type": design_case,
            "pv_capacity_kw": float(row["pv_capacity_kw"]),
            "battery_energy_kwh": battery_energy,
            "battery_power_kw": float(sizing["battery_power_kw"]),
            "reserve_fraction": reserve_fraction,
            "reserve_soc_kwh": battery_energy * reserve_fraction,
            "minimum_soc_fraction": minimum_soc_fraction,
            "minimum_soc_kwh": battery_energy * minimum_soc_fraction,
            "roundtrip_efficiency": float(config["battery"]["roundtrip_efficiency"]),
        }
    )
    return pd.Series(data)


def _event_hour_mask(hour_index: pd.Series, start_hour: float, end_hour: float) -> pd.Series:
    hours = hour_index.astype(float)
    return (hours + 1.0 > float(start_hour)) & (hours < float(end_hour))


def _window_ue(source: pd.DataFrame, design_case: str) -> float:
    return float(
        source.loc[source["design_case"].astype(str) == design_case, "unmet_load_kwh_equiv"].astype(float).sum()
    )


def _shared_balance_ylim(source: pd.DataFrame) -> tuple[float, float]:
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
    return lower - pad, upper + pad


def _annotate_panel_ue(
    ax: Any,
    source: pd.DataFrame,
    design_case: str,
    palette: dict[str, str],
    use_kwh: bool,
) -> None:
    y_min, y_max = ax.get_ylim()
    ue = float(source["unmet_load_kwh_equiv"].astype(float).sum())
    label = "Reference UE" if design_case == "no_pv_bess" else "Comparison UE"
    ax.text(
        50.0,
        y_max - 0.08 * (y_max - y_min),
        f"{label}\n{format_energy_from_kwh(ue, use_kwh)}",
        ha="left",
        va="top",
        fontsize=6.3,
        color=palette["neutral_black"],
    )


def _annotate_event_summary(fig: Any, summary: pd.Series, palette: dict[str, str], use_kwh: bool) -> None:
    fig.text(
        0.085,
        0.825,
        (
            f"Selected outage: {float(summary['event_duration_hours']):.2f} h, "
            f"avoided UE {format_energy_from_kwh(float(summary['avoided_UE_kwh']), use_kwh)}, "
            f"avoided lost-load cost {float(summary['avoided_lost_load_cost_sgd']) / 1000.0:.1f} kSGD"
        ),
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=palette["neutral_black"],
    )


def _write_event_summary(base: Path, summary: pd.DataFrame) -> Path:
    path = base.with_name(base.name + "_Event_Summary.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path
