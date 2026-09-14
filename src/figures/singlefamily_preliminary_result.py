from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..io.config import DEFAULT_CONFIG, load_config
from ..io.data_loader import InputCase, build_case_frame, load_case_parameters, load_input_case
from ..optimizer import resilience_design, scenario_config
from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    matplotlib_context,
    plot_smooth_line,
    save_publication_figure,
    write_source_data,
)
from .display_units import DISPLAY_BATTERY_ENERGY, DISPLAY_PV_CAPACITY, DISPLAY_SYSTEM_COST, add_display_unit_columns
from .single_outage_city_baseline import (
    AFFORDABLE_COST_DESIGN_LABELS,
    AFFORDABLE_LOLP_ZERO_TOLERANCE_KWH,
    AFFORDABLE_PLOT_GRID_MAX_EXPANSIONS,
    CITY_COLORS,
    CITY_LABELS,
    _affordable_boundary_edge_touch_flags,
    _affordable_design_space_plot_grid_source,
    _affordable_lolp_cost_distribution_source,
    _affordable_lolp_design_space_source,
    _apply_single_outage_readable_font_style,
    _baseline_outage,
    _candidate_dispatch,
    _city_energy_balance_source,
    _city_order,
    _contour_metric_specs,
    _cost_design_row,
    _draw_affordable_cost_boundary,
    _draw_contour_panel,
    _draw_energy_balance_panel,
    _energy_balance_legend_handles,
    _format_metric_colorbar,
    _input_case_for_exact_case_key,
    _mark_lolp_zero_design,
    _metric_levels,
    _outer_ring_capacity_points,
    _outage_weighted_load_total,
    _refresh_affordable_selection_metadata,
    _select_lolp_zero_min_cost_design,
    _shared_energy_balance_ylim,
    _single_outage_energy_balance_palette,
    _single_outage_panel_label,
    _validate_affordable_cost_distribution_source,
    _validate_affordable_design_space_source,
    _validate_affordable_plot_grid_source,
    _validate_energy_balance_source,
)


PRELIMINARY_CITY_ORDER = ("Singapore", "Jakarta", "Metro_Manila", "Brisbane")
PRELIMINARY_CASE_BY_CITY = {
    "Singapore": "SG_SingleFamily",
    "Jakarta": "Jakarta_SingleFamily",
    "Metro_Manila": "Manila_SingleFamily",
    "Brisbane": "Brisbane_SingleFamily",
}
PRELIMINARY_FIGURE_NUMBERS = (1, 2, 3, 4)
PRELIMINARY_FIGURE_NAMES = {
    1: "Figure_1_Four_City_TOU_Profile",
    2: "Figure_2_Four_City_Night_Affordable_LOLP_Design_Space_Contours",
    3: "Figure_3_Four_City_Night_Affordable_LOLP_Outage_Energy_Balance",
    4: "Figure_4_Four_City_LOLP_Zero_Sensitivity",
}
PRELIMINARY_SOURCE_KEYS = {
    1: ("tou_profile",),
    2: ("affordable_design_space", "affordable_design_space_plot_grid"),
    3: ("affordable_energy_balance", "affordable_event_summary"),
    4: ("sensitivity_results",),
}
PRELIMINARY_SENSITIVITY_PARAMETERS = (
    "battery_capex_sgd_per_kwh",
    "pv_capex_sgd_per_kw",
    "saidi_min_per_year",
    "voll_sgd_per_kwh",
)
PRELIMINARY_SENSITIVITY_LABELS = {
    "battery_capex_sgd_per_kwh": "Battery cost",
    "pv_capex_sgd_per_kw": "PV cost",
    "saidi_min_per_year": "SAIDI",
    "voll_sgd_per_kwh": "VoLL",
}
PRELIMINARY_SENSITIVITY_COLORS = {
    "battery_capex_sgd_per_kwh": NATURE_PALETTE["blue"],
    "pv_capex_sgd_per_kw": "#5F8A70",
    "saidi_min_per_year": NATURE_PALETTE["red"],
    "voll_sgd_per_kwh": NATURE_PALETTE["gold"],
}
PRELIMINARY_SENSITIVITY_LINESTYLES = {
    "battery_capex_sgd_per_kwh": "-",
    "pv_capex_sgd_per_kw": "--",
    "saidi_min_per_year": "-.",
    "voll_sgd_per_kwh": ":",
}
PRELIMINARY_DATA_STATUS = "real_singlefamily_night_preliminary"
NIGHT_OUTAGE_MODE = "night_window_random"


def build_singlefamily_preliminary_result_sources(
    input_root: Path | str = "Input",
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all source tables for the Single Family preliminary QE-report figure set."""
    runtime_config = _preliminary_runtime_config(config)
    sources = build_singlefamily_preliminary_night_affordable_sources(input_root, config=runtime_config)
    sources["tou_profile"] = build_singlefamily_preliminary_tou_profile_source(input_root)
    sources["sensitivity_results"] = build_singlefamily_preliminary_lolp_zero_sensitivity_source(
        input_root,
        config=runtime_config,
    )
    return sources


def make_singlefamily_preliminary_result_figures(
    out: Path | str,
    *,
    input_root: Path | str = "Input",
    config: dict[str, Any] | None = None,
    figures: Sequence[int | str] | None = None,
    sources: dict[str, pd.DataFrame] | None = None,
    case_key: str = "SingleFamily",
) -> dict[str, Path]:
    """Create preliminary Single Family city-comparison figures and source-data CSV files."""
    selected = _expand_preliminary_figures(figures)
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_preliminary_sources(
        input_root,
        config=config,
        figures=selected,
        sources=sources,
    )
    plotters = {
        1: plot_singlefamily_preliminary_tou_profile,
        2: plot_singlefamily_preliminary_night_affordable_lolp_contours,
        3: plot_singlefamily_preliminary_night_energy_balance,
        4: plot_singlefamily_preliminary_lolp_zero_sensitivity,
    }
    paths: dict[str, Path] = {}
    for figure_no in selected:
        base = output_dir / f"{case_key}_{PRELIMINARY_FIGURE_NAMES[figure_no]}"
        paths[f"figure_{figure_no}_source_data"] = write_source_data(base, _figure_source(prepared, figure_no))
        paths.update(plotters[figure_no](output_dir, case_key, prepared))
    return paths


def build_singlefamily_preliminary_tou_profile_source(input_root: Path | str = "Input") -> pd.DataFrame:
    """Build 24 h TOU tariff data for Singapore, Jakarta, Metro Manila, and Brisbane."""
    cases_by_city = _preliminary_cases_by_city(Path(input_root))
    rows: list[dict[str, Any]] = []
    for city_key in PRELIMINARY_CITY_ORDER:
        case = cases_by_city[city_key]
        tou = pd.read_csv(case.tou_csv)
        required = {"hour", "tou_sgd_per_kwh"}
        missing = required - set(tou.columns)
        if missing:
            raise ValueError(f"{case.tou_csv} missing TOU column(s): " + ", ".join(sorted(missing)))
        tou = tou.copy()
        tou["hour"] = pd.to_numeric(tou["hour"], errors="coerce").astype("Int64")
        tou["tou_sgd_per_kwh"] = pd.to_numeric(tou["tou_sgd_per_kwh"], errors="coerce")
        if tou[["hour", "tou_sgd_per_kwh"]].isna().any().any():
            raise ValueError(f"{case.tou_csv} contains non-numeric TOU values.")
        tou = tou.sort_values("hour").reset_index(drop=True)
        if tou["hour"].astype(int).tolist() != list(range(24)):
            raise ValueError(f"{case.tou_csv} must cover hours 0-23 exactly.")
        for row in tou.to_dict("records"):
            hour = int(row["hour"])
            rows.append(
                {
                    "city_key": city_key,
                    "city_label": CITY_LABELS[city_key],
                    "case_key": case.case_key,
                    "source_case_key": case.case_key,
                    "hour": hour,
                    "interval_start_hour": float(hour),
                    "interval_end_hour": float(hour + 1),
                    "tou_sgd_per_kwh": float(row["tou_sgd_per_kwh"]),
                    "data_status": PRELIMINARY_DATA_STATUS,
                }
            )
    return validate_preliminary_tou_profile_source(pd.DataFrame(rows))


def build_singlefamily_preliminary_night_affordable_sources(
    input_root: Path | str = "Input",
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build Figure 2/3 source data with night-window outages for all four cities."""
    runtime_config = _preliminary_runtime_config(config)
    cases_by_city = _preliminary_cases_by_city(Path(input_root))
    tasks = [
        {
            "city_key": city_key,
            "city_order": index,
            "case": cases_by_city[city_key],
            "config": runtime_config,
        }
        for index, city_key in enumerate(PRELIMINARY_CITY_ORDER)
    ]
    results = _build_preliminary_night_affordable_sources_parallel(tasks)
    ordered = sorted(results, key=lambda item: int(item["city_order"]))
    design_space = _validate_affordable_design_space_source(
        pd.concat([item["design_space"] for item in ordered], ignore_index=True),
        city_order=PRELIMINARY_CITY_ORDER,
    )
    cost_distribution = _validate_affordable_cost_distribution_source(
        pd.concat([item["cost_distribution"] for item in ordered], ignore_index=True),
        city_order=PRELIMINARY_CITY_ORDER,
    )
    energy_balance = _validate_energy_balance_source(
        pd.concat([item["energy_balance"] for item in ordered], ignore_index=True),
        city_order=PRELIMINARY_CITY_ORDER,
    )
    event_summary = _with_preliminary_city_order(
        pd.concat([item["event_summary"] for item in ordered], ignore_index=True)
    )
    plot_grid = _preliminary_affordable_plot_grid(input_root, runtime_config, design_space)
    return {
        "affordable_design_space": design_space,
        "affordable_design_space_plot_grid": plot_grid,
        "affordable_cost_distribution": cost_distribution,
        "affordable_energy_balance": energy_balance,
        "affordable_event_summary": event_summary,
    }


def build_singlefamily_preliminary_lolp_zero_sensitivity_source(
    input_root: Path | str = "Input",
    *,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate input-range sensitivity using the LOLP=0 minimum-cost design criterion."""
    runtime_config = _preliminary_runtime_config(config)
    tasks = _preliminary_sensitivity_tasks(Path(input_root), runtime_config)
    rows = _evaluate_preliminary_sensitivity_tasks_parallel(tasks)
    return validate_preliminary_sensitivity_source(pd.DataFrame(rows))


def plot_singlefamily_preliminary_tou_profile(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    source = validate_preliminary_tou_profile_source(sources["tou_profile"])
    city_order = _preliminary_city_order(source)
    mpl, plt = matplotlib_context(out / PRELIMINARY_FIGURE_NAMES[1])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    fig, ax = plt.subplots(figsize=(7.4, 3.80), constrained_layout=False)
    fig.subplots_adjust(left=0.135, right=0.975, bottom=0.230, top=0.745)

    for city_key in city_order:
        group = source[source["city_key"].astype(str) == city_key].sort_values("hour")
        x, y = _preliminary_tou_step_xy(group)
        ax.step(
            x,
            y,
            where="post",
            color=CITY_COLORS[city_key],
            linewidth=1.35,
            label=CITY_LABELS[city_key],
        )

    ax.set_xlim(0.0, 24.0)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("TOU tariff (SGD kWh$^{-1}$)")
    ax.set_ylim(0.0, max(float(source["tou_sgd_per_kwh"].max()) * 1.12, 0.05))
    ax.grid(False)
    _single_outage_panel_label(ax, "a")
    fig.legend(
        loc="upper center",
        ncol=len(city_order),
        bbox_to_anchor=(0.55, 0.950),
        handlelength=1.8,
    )
    base = out / f"{case_key}_{PRELIMINARY_FIGURE_NAMES[1]}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_singlefamily_preliminary_night_affordable_lolp_contours(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    source = _validate_affordable_design_space_source(
        sources["affordable_design_space"],
        city_order=PRELIMINARY_CITY_ORDER,
    )
    plot_source = validate_preliminary_affordable_plot_grid_source(
        sources.get("affordable_design_space_plot_grid", _affordable_design_space_plot_grid_source(source))
    )
    _, units = add_display_unit_columns(pd.concat([source, plot_source], ignore_index=True, sort=False))
    display_source, _ = add_display_unit_columns(source, units=units)
    display_plot_source, _ = add_display_unit_columns(plot_source, units=units)
    metric_specs = tuple(
        spec
        for spec in _contour_metric_specs(units)
        if spec["key"] in {"system_total_cost", "lolp"}
    )
    city_order = _city_order(display_source, order=PRELIMINARY_CITY_ORDER)
    mpl, plt = matplotlib_context(out / PRELIMINARY_FIGURE_NAMES[2])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(14.2, 6.7), constrained_layout=False)
    grid = fig.add_gridspec(
        len(metric_specs),
        len(city_order) + 1,
        left=0.070,
        right=0.945,
        bottom=0.135,
        top=0.895,
        wspace=0.28,
        hspace=0.34,
        width_ratios=[1.0] * len(city_order) + [0.060],
    )
    for metric_index, spec in enumerate(metric_specs):
        metric_values = display_plot_source[spec["column"]].astype(float).to_numpy()
        levels = _metric_levels(metric_values, spec["unit_type"])
        cbar_artist = None
        for city_index, city_key in enumerate(city_order):
            ax = fig.add_subplot(grid[metric_index, city_index])
            city_source = display_source[display_source["city_key"].astype(str) == city_key].copy()
            city_plot_source = display_plot_source[display_plot_source["city_key"].astype(str) == city_key].copy()
            cbar_artist = _draw_contour_panel(mpl, ax, city_plot_source, spec, levels)
            boundary = float(city_plot_source["affordable_boundary_cost_sgd_per_year"].iloc[0]) / max(float(units.cost_divisor), 1e-9)
            _draw_affordable_cost_boundary(ax, city_plot_source, boundary, DISPLAY_SYSTEM_COST)
            _mark_lolp_zero_design(ax, city_source)
            if metric_index == 0:
                ax.set_title(CITY_LABELS[city_key], loc="left", pad=5)
            if city_index == 0:
                ax.set_ylabel(f"{spec['title']}\n{units.pv_axis_label}")
            else:
                ax.set_yticklabels([])
            if metric_index == len(metric_specs) - 1:
                ax.set_xlabel(units.battery_axis_label)
            else:
                ax.set_xticklabels([])
            ax.set_box_aspect(1.0)
        cax = fig.add_subplot(grid[metric_index, -1])
        if cbar_artist is not None:
            cbar = fig.colorbar(cbar_artist, cax=cax)
            _format_metric_colorbar(cbar, levels, spec)

    legend_handles = [
        Line2D([0], [0], color=NATURE_PALETTE["neutral_black"], linewidth=1.8, label="0 PV / 0 BESS cost"),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markerfacecolor=NATURE_PALETTE["neutral_black"],
            markeredgecolor="white",
            markeredgewidth=0.50,
            markersize=8.5,
            label="LOLP=0 min-cost design",
        ),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.52, 0.018),
        handlelength=1.65,
    )
    base = out / f"{case_key}_{PRELIMINARY_FIGURE_NAMES[2]}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_singlefamily_preliminary_night_energy_balance(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    source = _validate_energy_balance_source(
        sources["affordable_energy_balance"],
        city_order=PRELIMINARY_CITY_ORDER,
    )
    city_order = _city_order(source, order=PRELIMINARY_CITY_ORDER)
    mpl, plt = matplotlib_context(out / PRELIMINARY_FIGURE_NAMES[3])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    palette = _single_outage_energy_balance_palette()
    shared_ylim = _shared_energy_balance_ylim(source)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.35), sharex=True, sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.900, bottom=0.290, top=0.850, wspace=0.30, hspace=0.46)
    for index, city_key in enumerate(city_order):
        ax = axes.flat[index]
        group = source[source["city_key"].astype(str) == city_key].sort_values("plot_hour")
        _draw_energy_balance_panel(
            ax,
            group,
            palette,
            shared_ylim,
            show_ylabel=index % 2 == 0,
            show_xlabel=index >= 2,
            show_soc_ylabel=index % 2 == 1,
        )
        _single_outage_panel_label(ax, chr(ord("a") + index))

    legend_handles = _energy_balance_legend_handles(Patch, Line2D, palette)
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.52, 0.035),
        handlelength=1.35,
    )
    base = out / f"{case_key}_{PRELIMINARY_FIGURE_NAMES[3]}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_singlefamily_preliminary_lolp_zero_sensitivity(
    out: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    source = validate_preliminary_sensitivity_source(sources["sensitivity_results"])
    plot_source = source.copy()
    plot_source["system_total_cost_ksgd_per_year"] = plot_source["system_total_cost_sgd_per_year"].astype(float) / 1000.0
    plot_source["energy_based_lolp_percent"] = plot_source["energy_based_LOLP"].astype(float) * 100.0
    metric_specs = (
        ("system_total_cost_ksgd_per_year", "Total cost\n(kSGD yr$^{-1}$)"),
        ("energy_based_lolp_percent", "LOLP\n(%)"),
        ("pv_capacity_kw", "Optimal PV\n(kW)"),
        ("battery_energy_kwh", "Optimal BESS\n(kWh)"),
    )
    city_order = _preliminary_city_order(plot_source)
    y_limits = {
        column: _metric_ylim(plot_source[column].astype(float).to_numpy(), is_lolp=column == "energy_based_lolp_percent")
        for column, _ in metric_specs
    }

    mpl, plt = matplotlib_context(out / PRELIMINARY_FIGURE_NAMES[4])
    apply_nature_style(mpl)
    _apply_single_outage_readable_font_style(mpl)
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(
        len(city_order),
        len(metric_specs),
        figsize=(14.4, 9.8),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.100, right=0.985, bottom=0.135, top=0.890, wspace=0.32, hspace=0.34)
    axes_array = np.asarray(axes)
    for row_index, city_key in enumerate(city_order):
        city_source = plot_source[plot_source["city_key"].astype(str) == city_key].copy()
        for col_index, (column, title) in enumerate(metric_specs):
            ax = axes_array[row_index, col_index]
            for parameter in PRELIMINARY_SENSITIVITY_PARAMETERS:
                group = city_source[city_source["sensitivity_parameter"].astype(str) == parameter].sort_values("sensitivity_x_normalized")
                if group.empty:
                    continue
                x = group["sensitivity_x_normalized"].astype(float).to_numpy()
                y = group[column].astype(float).to_numpy()
                line_kwargs = {
                    "color": PRELIMINARY_SENSITIVITY_COLORS[parameter],
                    "linestyle": PRELIMINARY_SENSITIVITY_LINESTYLES[parameter],
                    "linewidth": 1.30,
                    "alpha": 0.95,
                }
                if len(np.unique(x)) >= 3:
                    plot_smooth_line(ax, x, y, show_points=False, **line_kwargs)
                else:
                    ax.plot(x, y, marker=None, **line_kwargs)
            ax.set_ylim(*y_limits[column])
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.55)
            if row_index == 0:
                ax.set_title(title, loc="center", pad=5)
            if row_index == len(city_order) - 1:
                ax.set_xlabel("Input range position")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if col_index == 0:
                ax.set_ylabel(CITY_LABELS[city_key], fontweight="bold")
            _single_outage_panel_label(ax, chr(ord("a") + row_index * len(metric_specs) + col_index))

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=PRELIMINARY_SENSITIVITY_COLORS[parameter],
            linestyle=PRELIMINARY_SENSITIVITY_LINESTYLES[parameter],
            linewidth=1.30,
            marker=None,
            label=PRELIMINARY_SENSITIVITY_LABELS[parameter],
        )
        for parameter in PRELIMINARY_SENSITIVITY_PARAMETERS
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.55, 0.975),
        handlelength=2.1,
    )
    base = out / f"{case_key}_{PRELIMINARY_FIGURE_NAMES[4]}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def validate_preliminary_tou_profile_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "case_key",
        "hour",
        "interval_start_hour",
        "interval_end_hour",
        "tou_sgd_per_kwh",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Preliminary TOU profile source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    text_columns = {"city_key", "city_label", "case_key", "source_case_key", "data_status"}
    for column in required - text_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(required - text_columns)].isna().any().any():
        raise ValueError("Preliminary TOU profile source contains non-numeric values.")
    present = set(validated["city_key"].astype(str))
    if present != set(PRELIMINARY_CITY_ORDER):
        raise ValueError("Preliminary TOU profile source must contain Singapore, Jakarta, Metro Manila, and Brisbane.")
    for city_key, group in validated.groupby(validated["city_key"].astype(str), sort=False):
        if sorted(group["hour"].astype(int).tolist()) != list(range(24)):
            raise ValueError(f"Preliminary TOU profile requires hours 0-23 exactly for {city_key}.")
    return _with_preliminary_city_order(validated, secondary_sort=["hour"])


def validate_preliminary_affordable_plot_grid_source(source: pd.DataFrame) -> pd.DataFrame:
    validated = _validate_affordable_plot_grid_source(source)
    if set(validated["city_key"].astype(str)) != set(PRELIMINARY_CITY_ORDER):
        raise ValueError("Preliminary Figure 2 plot grid must contain exactly the four preliminary cities.")
    return _with_preliminary_city_order(validated, secondary_sort=["pv_capacity_kw", "battery_energy_kwh"])


def validate_preliminary_sensitivity_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_key",
        "city_label",
        "city_order",
        "case_key",
        "source_case_key",
        "sensitivity_parameter",
        "sensitivity_parameter_order",
        "sensitivity_label",
        "sensitivity_value",
        "sensitivity_unit",
        "sensitivity_min",
        "sensitivity_max",
        "sensitivity_x_normalized",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "system_total_cost_sgd_per_year",
        "system_total_cost_million_sgd_per_year",
        "UE_kwh_per_year",
        "UE_mwh_per_year",
        "energy_based_LOLP",
        "is_lolp_zero_min_cost_design",
        "outage_mode",
        "outage_seed",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError("Preliminary sensitivity source missing column(s): " + ", ".join(sorted(missing)))
    validated = source.copy()
    text_columns = {
        "city_key",
        "city_label",
        "case_key",
        "source_case_key",
        "sensitivity_parameter",
        "sensitivity_label",
        "sensitivity_unit",
        "outage_mode",
        "data_status",
        "is_lolp_zero_min_cost_design",
    }
    for column in required - text_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(required - text_columns)].isna().any().any():
        raise ValueError("Preliminary sensitivity source contains non-numeric values.")
    if set(validated["city_key"].astype(str)) != set(PRELIMINARY_CITY_ORDER):
        raise ValueError("Preliminary sensitivity source must contain exactly the four preliminary cities.")
    if set(validated["sensitivity_parameter"].astype(str)) != set(PRELIMINARY_SENSITIVITY_PARAMETERS):
        raise ValueError("Preliminary sensitivity source must contain battery cost, PV cost, SAIDI, and VoLL.")
    if not validated["sensitivity_x_normalized"].between(0.0, 1.0).all():
        raise ValueError("Preliminary sensitivity normalized x values must be between 0 and 1.")
    if not validated["energy_based_LOLP"].between(0.0, 1.0).all():
        raise ValueError("Preliminary sensitivity LOLP values must be between 0 and 1.")
    if (validated[["pv_capacity_kw", "battery_energy_kwh", "system_total_cost_sgd_per_year", "UE_kwh_per_year"]] < 0.0).any().any():
        raise ValueError("Preliminary sensitivity sizing, cost, and UE values must be non-negative.")
    if (validated["UE_kwh_per_year"].astype(float) > AFFORDABLE_LOLP_ZERO_TOLERANCE_KWH).any():
        raise ValueError("Preliminary sensitivity source contains a non-zero-LOLP selected design.")
    if set(validated["outage_mode"].astype(str)) != {NIGHT_OUTAGE_MODE}:
        raise ValueError("Preliminary sensitivity source must use night-window outages.")
    validated["is_lolp_zero_min_cost_design"] = validated["is_lolp_zero_min_cost_design"].astype(bool)
    validated["city_key"] = pd.Categorical(validated["city_key"].astype(str), categories=list(PRELIMINARY_CITY_ORDER), ordered=True)
    validated["sensitivity_parameter"] = pd.Categorical(
        validated["sensitivity_parameter"].astype(str),
        categories=list(PRELIMINARY_SENSITIVITY_PARAMETERS),
        ordered=True,
    )
    validated = validated.sort_values(["city_key", "sensitivity_parameter", "sensitivity_value"]).reset_index(drop=True)
    validated["city_key"] = validated["city_key"].astype(str)
    validated["sensitivity_parameter"] = validated["sensitivity_parameter"].astype(str)
    return validated


def _prepare_preliminary_sources(
    input_root: Path | str,
    *,
    config: dict[str, Any] | None,
    figures: Sequence[int],
    sources: dict[str, pd.DataFrame] | None,
) -> dict[str, pd.DataFrame]:
    prepared = dict(sources or {})
    runtime_config = _preliminary_runtime_config(config)
    selected = set(figures)
    if 1 in selected and "tou_profile" not in prepared:
        prepared["tou_profile"] = build_singlefamily_preliminary_tou_profile_source(input_root)
    if selected.intersection({2, 3}) and not {"affordable_design_space", "affordable_energy_balance", "affordable_event_summary"}.issubset(prepared):
        prepared.update(build_singlefamily_preliminary_night_affordable_sources(input_root, config=runtime_config))
    if 2 in selected and "affordable_design_space_plot_grid" not in prepared:
        prepared["affordable_design_space_plot_grid"] = _preliminary_affordable_plot_grid(
            input_root,
            runtime_config,
            prepared["affordable_design_space"],
        )
    if 4 in selected and "sensitivity_results" not in prepared:
        prepared["sensitivity_results"] = build_singlefamily_preliminary_lolp_zero_sensitivity_source(
            input_root,
            config=runtime_config,
        )
    missing = {
        key
        for figure in selected
        for key in PRELIMINARY_SOURCE_KEYS[figure]
        if key not in prepared
    }
    if missing:
        raise ValueError("Preliminary source preparation missing table(s): " + ", ".join(sorted(missing)))
    return prepared


def _build_preliminary_night_affordable_sources_parallel(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tasks:
        raise ValueError("Preliminary affordable source building requires at least one city task.")
    workers = _preliminary_worker_count(tasks)
    if workers <= 1:
        return [_build_preliminary_night_affordable_city_sources_task(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_build_preliminary_night_affordable_city_sources_task, tasks, chunksize=1))
    except OSError:
        return [_build_preliminary_night_affordable_city_sources_task(task) for task in tasks]


def _build_preliminary_night_affordable_city_sources_task(payload: dict[str, Any]) -> dict[str, Any]:
    city_key = str(payload["city_key"])
    case = payload["case"]
    run = _run_preliminary_night_lolp_zero_context(
        city_key,
        case,
        payload["config"],
        parameter_overrides=None,
    )
    selected = run["selected"]
    selected_result = run["selected_result"]
    selected_dispatch = run["selected_dispatch"]
    selected_frame, _ = build_case_frame(case, run["case_config"], float(selected_result["pv_capacity_kw"]))
    selected_frame = selected_frame.merge(run["outage_profile"], on="hour_index", how="left")
    energy_balance, event_summary = _city_energy_balance_source(
        city_key,
        case,
        run["case_config"],
        selected_frame,
        selected_dispatch,
        run["outage_events"],
        selected_result,
        city_order=PRELIMINARY_CITY_ORDER,
    )
    for frame in (energy_balance, event_summary):
        frame["design_case"] = "lolp_zero_min_cost"
        frame["design_case_label"] = AFFORDABLE_COST_DESIGN_LABELS["lolp_zero_min_cost"]
        frame["outage_mode"] = NIGHT_OUTAGE_MODE
        frame["outage_seed"] = int(run["case_config"]["outage"].get("seed", 42))
        frame["zero_pv_zero_bess_cost_sgd_per_year"] = float(selected["zero_pv_zero_bess_cost_sgd_per_year"])
        frame["affordable_boundary_cost_sgd_per_year"] = float(selected["affordable_boundary_cost_sgd_per_year"])
        frame["selected_lolp_zero_cost_sgd_per_year"] = float(selected["system_total_cost_sgd_per_year"])
        frame["cost_delta_vs_no_pv_no_bess_sgd_per_year"] = float(selected["cost_delta_vs_no_pv_no_bess_sgd_per_year"])
        frame["data_status"] = PRELIMINARY_DATA_STATUS
    return {
        "city_order": int(payload["city_order"]),
        "design_space": run["design_space"],
        "cost_distribution": run["cost_distribution"],
        "energy_balance": energy_balance,
        "event_summary": event_summary,
    }


def _run_preliminary_night_lolp_zero_context(
    city_key: str,
    case: InputCase,
    global_config: dict[str, Any],
    *,
    parameter_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    _, _, case_config, _ = load_input_case(case, deepcopy(global_config), parameter_overrides=parameter_overrides)
    case_config = deepcopy(case_config)
    case_config.setdefault("outage", {})["forced_event_mode"] = NIGHT_OUTAGE_MODE
    preview_frame, _ = build_case_frame(case, case_config, max(case_config["pv"]["capacity_kw_candidates"]))
    outage_profile, outage_events = _baseline_outage(case_config, case.case_key, len(preview_frame))
    outage_weighted_total = _outage_weighted_load_total(preview_frame, outage_profile, case_config)
    outage_summary = {
        "simulated_events": float(len(outage_events)),
        "simulated_saifi_per_year": float(len(outage_events)),
        "simulated_saidi_min_per_year": float(outage_profile["outage_fraction"].sum() * 60.0),
        "min_grid_availability": float(outage_profile["grid_availability"].min()),
    }

    def pv_capacity_frame_builder(pv_capacity_kw: float) -> tuple[pd.DataFrame, dict[str, float | str]]:
        frame, pv_meta = build_case_frame(case, case_config, float(pv_capacity_kw))
        return frame.merge(outage_profile, on="hour_index", how="left"), pv_meta

    _, _, candidate_summary, candidate_dispatches = resilience_design(
        pv_capacity_frame_builder,
        case_config,
        case.case_key,
    )
    annual_load = float(case_config["capacity_search"]["annual_load_e_equiv_kwh"])
    design_space = _affordable_lolp_design_space_source(
        city_key,
        case,
        candidate_summary,
        candidate_dispatches,
        annual_load,
        outage_weighted_total,
        outage_summary,
        NIGHT_OUTAGE_MODE,
        int(case_config["outage"].get("seed", 42)),
    )
    design_space["data_status"] = PRELIMINARY_DATA_STATUS
    selected = _select_lolp_zero_min_cost_design(design_space, city_order=(city_key,))
    selected_result = _cost_design_row(
        candidate_summary,
        pv_capacity_frame_builder,
        scenario_config(case_config, case.case_key),
        float(selected["pv_capacity_kw"]),
        float(selected["battery_energy_kwh"]),
    )
    selected_dispatch = _candidate_dispatch(selected.to_dict(), candidate_dispatches)
    cost_distribution = _affordable_lolp_cost_distribution_source(
        city_key,
        case,
        scenario_config(case_config, case.case_key),
        candidate_summary,
        pv_capacity_frame_builder,
        selected_result,
        design_space,
    )
    cost_distribution["data_status"] = PRELIMINARY_DATA_STATUS
    return {
        "case_config": case_config,
        "outage_profile": outage_profile,
        "outage_events": outage_events,
        "outage_weighted_total": outage_weighted_total,
        "design_space": design_space,
        "cost_distribution": cost_distribution,
        "selected": selected,
        "selected_result": selected_result,
        "selected_dispatch": selected_dispatch,
    }


def _preliminary_affordable_plot_grid(
    input_root: Path | str,
    config: dict[str, Any],
    design_space: pd.DataFrame,
) -> pd.DataFrame:
    base_design = _validate_affordable_design_space_source(design_space, city_order=PRELIMINARY_CITY_ORDER)
    extended = base_design.copy()
    for city_key in PRELIMINARY_CITY_ORDER:
        for _ in range(AFFORDABLE_PLOT_GRID_MAX_EXPANSIONS):
            city_source = extended[extended["city_key"].astype(str) == city_key].copy()
            plot_grid = _affordable_design_space_plot_grid_source(city_source)
            flags = _affordable_boundary_edge_touch_flags(plot_grid)
            if not flags["top"] and not flags["right"]:
                break
            new_points = _outer_ring_capacity_points(city_source, expand_pv=flags["top"], expand_battery=flags["right"])
            if not new_points:
                break
            new_rows = _evaluate_preliminary_affordable_points_for_city(
                Path(input_root),
                config,
                city_key,
                new_points,
                zero_zero_cost=float(city_source["zero_pv_zero_bess_cost_sgd_per_year"].iloc[0]),
            )
            extended = pd.concat([extended, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
            extended = _refresh_affordable_selection_metadata(extended, city_order=PRELIMINARY_CITY_ORDER)
    plot_grid = _affordable_design_space_plot_grid_source(extended)
    return validate_preliminary_affordable_plot_grid_source(plot_grid)


def _evaluate_preliminary_affordable_points_for_city(
    input_root: Path,
    config: dict[str, Any],
    city_key: str,
    points: Sequence[tuple[float, float]],
    *,
    zero_zero_cost: float,
) -> list[dict[str, Any]]:
    from .single_outage_city_baseline import _evaluate_affordable_points_for_city

    rows = _evaluate_affordable_points_for_city(
        input_root,
        config,
        city_key,
        NIGHT_OUTAGE_MODE,
        points,
        point_source="new_evaluated",
        zero_zero_cost=zero_zero_cost,
    )
    for row in rows:
        row["data_status"] = PRELIMINARY_DATA_STATUS
    return rows


def _preliminary_sensitivity_tasks(input_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    cases_by_city = _preliminary_cases_by_city(input_root)
    tasks: list[dict[str, Any]] = []
    for city_order, city_key in enumerate(PRELIMINARY_CITY_ORDER):
        case = cases_by_city[city_key]
        parameters = load_case_parameters(case.parameter_csv)
        raw = parameters.raw.set_index("parameter")
        configured_workers = int(float(parameters.base.get("parallel_workers", 1) or 1))
        for parameter_order, parameter in enumerate(PRELIMINARY_SENSITIVITY_PARAMETERS):
            values = parameters.sensitivity.get(parameter, [])
            if not values:
                raise ValueError(f"{case.parameter_csv} has no sensitivity range for {parameter}.")
            value_min = float(min(values))
            value_max = float(max(values))
            unit = str(raw.loc[parameter, "unit"])
            base_value = float(parameters.base[parameter])
            for value_index, value in enumerate(values):
                denominator = max(value_max - value_min, 1e-9)
                tasks.append(
                    {
                        "city_key": city_key,
                        "city_order": int(city_order),
                        "case": case,
                        "config": config,
                        "parallel_workers": configured_workers,
                        "sensitivity_parameter": parameter,
                        "sensitivity_parameter_order": int(parameter_order),
                        "sensitivity_label": PRELIMINARY_SENSITIVITY_LABELS[parameter],
                        "sensitivity_value": float(value),
                        "sensitivity_value_index": int(value_index),
                        "sensitivity_unit": unit,
                        "sensitivity_base_value": base_value,
                        "sensitivity_min": value_min,
                        "sensitivity_max": value_max,
                        "sensitivity_x_normalized": float((float(value) - value_min) / denominator),
                    }
                )
    return tasks


def _evaluate_preliminary_sensitivity_tasks_parallel(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tasks:
        raise ValueError("Preliminary sensitivity source building requires at least one task.")
    workers = _preliminary_worker_count(tasks)
    if workers <= 1:
        return [_evaluate_preliminary_sensitivity_task(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_evaluate_preliminary_sensitivity_task, tasks, chunksize=1))
    except OSError:
        return [_evaluate_preliminary_sensitivity_task(task) for task in tasks]


def _evaluate_preliminary_sensitivity_task(payload: dict[str, Any]) -> dict[str, Any]:
    parameter = str(payload["sensitivity_parameter"])
    value = float(payload["sensitivity_value"])
    city_key = str(payload["city_key"])
    case = payload["case"]
    run = _run_preliminary_night_lolp_zero_context(
        city_key,
        case,
        payload["config"],
        parameter_overrides={parameter: value},
    )
    selected = run["selected"]
    ue = float(selected["UE_kwh_per_year"])
    lolp = float(selected["energy_based_LOLP"])
    system_cost = float(selected["system_total_cost_sgd_per_year"])
    return {
        "city_key": city_key,
        "city_label": CITY_LABELS[city_key],
        "city_order": int(payload["city_order"]),
        "case_key": case.case_key,
        "source_case_key": case.case_key,
        "sensitivity_parameter": parameter,
        "sensitivity_parameter_order": int(payload["sensitivity_parameter_order"]),
        "sensitivity_label": str(payload["sensitivity_label"]),
        "sensitivity_value": value,
        "sensitivity_value_index": int(payload["sensitivity_value_index"]),
        "sensitivity_unit": str(payload["sensitivity_unit"]),
        "sensitivity_base_value": float(payload["sensitivity_base_value"]),
        "sensitivity_min": float(payload["sensitivity_min"]),
        "sensitivity_max": float(payload["sensitivity_max"]),
        "sensitivity_x_normalized": float(payload["sensitivity_x_normalized"]),
        "pv_capacity_kw": float(selected["pv_capacity_kw"]),
        "pv_capacity_mw": float(selected["pv_capacity_mw"]),
        "battery_energy_kwh": float(selected["battery_energy_kwh"]),
        "battery_energy_mwh": float(selected["battery_energy_mwh"]),
        "system_total_cost_sgd_per_year": system_cost,
        "system_total_cost_million_sgd_per_year": system_cost / 1e6,
        "UE_kwh_per_year": ue,
        "UE_mwh_per_year": ue / 1000.0,
        "energy_based_LOLP": lolp,
        "energy_based_lolp": lolp,
        "energy_based_lolp_percent": 100.0 * lolp,
        "outage_weighted_load_kwh": float(selected["outage_weighted_load_kwh"]),
        "is_lolp_zero_min_cost_design": True,
        "outage_mode": NIGHT_OUTAGE_MODE,
        "outage_seed": int(run["case_config"]["outage"].get("seed", 42)),
        "night_window_start_hour": float(run["case_config"]["outage"].get("night_window_start_hour", 19.0)),
        "night_window_end_hour": float(run["case_config"]["outage"].get("night_window_end_hour", 23.0)),
        "data_status": PRELIMINARY_DATA_STATUS,
    }


def _preliminary_cases_by_city(input_root: Path) -> dict[str, InputCase]:
    resolved: dict[str, InputCase] = {}
    missing: list[str] = []
    for city_key in PRELIMINARY_CITY_ORDER:
        case_key = PRELIMINARY_CASE_BY_CITY[city_key]
        try:
            resolved[city_key] = _input_case_for_exact_case_key(input_root, case_key)
        except (FileNotFoundError, ValueError) as exc:
            missing.append(f"{case_key} ({exc})")
    if missing:
        raise ValueError("Missing preliminary SingleFamily input case(s): " + ", ".join(missing))
    return resolved


def _preliminary_runtime_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        try:
            runtime = load_config("config.yaml")
        except Exception:
            runtime = deepcopy(DEFAULT_CONFIG)
    else:
        runtime = deepcopy(config)
    if str(runtime.get("dispatch", {}).get("backend", "auto")).lower() == "auto":
        runtime.setdefault("dispatch", {})["backend"] = "greedy"
    return runtime


def _preliminary_worker_count(tasks: Sequence[dict[str, Any]]) -> int:
    configured = [int(task.get("parallel_workers", 1) or 1) for task in tasks]
    base_workers = max(configured) if configured else (os.cpu_count() or 1)
    return min(max(1, int(base_workers)), max(1, len(tasks)))


def _figure_source(sources: dict[str, pd.DataFrame], figure_no: int) -> pd.DataFrame:
    frames = []
    for key in PRELIMINARY_SOURCE_KEYS[figure_no]:
        frame = sources[key].copy()
        if "city_key" in frame.columns:
            frame = _with_preliminary_city_order(frame)
        frame.insert(0, "source_table", key)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _expand_preliminary_figures(figures: Sequence[int | str] | None) -> tuple[int, ...]:
    if figures is None:
        return PRELIMINARY_FIGURE_NUMBERS
    selected: list[int] = []
    for figure in figures:
        if isinstance(figure, str) and figure.strip().lower() == "all":
            selected.extend(PRELIMINARY_FIGURE_NUMBERS)
            continue
        normalized = str(figure).strip().lower()
        if not normalized.isdigit():
            raise ValueError(f"Unsupported preliminary figure: {figure}")
        value = int(normalized)
        if value not in PRELIMINARY_FIGURE_NUMBERS:
            raise ValueError(f"Unsupported preliminary figure: {figure}")
        selected.append(value)
    return tuple(dict.fromkeys(selected))


def _with_preliminary_city_order(frame: pd.DataFrame, secondary_sort: Sequence[str] | None = None) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["city_key"] = pd.Categorical(ordered["city_key"].astype(str), categories=list(PRELIMINARY_CITY_ORDER), ordered=True)
    sort_columns = ["city_key", *(secondary_sort or [])]
    ordered = ordered.sort_values(sort_columns).reset_index(drop=True)
    ordered["city_key"] = ordered["city_key"].astype(str)
    return ordered


def _preliminary_city_order(source: pd.DataFrame) -> list[str]:
    present = set(source["city_key"].astype(str))
    return [city for city in PRELIMINARY_CITY_ORDER if city in present]


def _preliminary_tou_step_xy(tou_profile: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = tou_profile.copy()
    ordered["hour"] = pd.to_numeric(ordered["hour"], errors="coerce")
    ordered["interval_start_hour"] = pd.to_numeric(ordered["interval_start_hour"], errors="coerce")
    ordered["interval_end_hour"] = pd.to_numeric(ordered["interval_end_hour"], errors="coerce")
    ordered["tou_sgd_per_kwh"] = pd.to_numeric(ordered["tou_sgd_per_kwh"], errors="coerce")
    ordered = ordered.sort_values("hour").copy()
    if len(ordered) != 24 or ordered[["hour", "interval_start_hour", "interval_end_hour", "tou_sgd_per_kwh"]].isna().any().any():
        raise ValueError("Preliminary TOU step source requires one complete 24-hour city profile.")
    x = ordered["interval_start_hour"].astype(float).to_numpy()
    y = ordered["tou_sgd_per_kwh"].astype(float).to_numpy()
    return np.append(x, float(ordered["interval_end_hour"].astype(float).iloc[-1])), np.append(y, float(y[-1]))


def _metric_ylim(values: np.ndarray, *, is_lolp: bool = False) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)
    lower = max(0.0, float(finite.min()))
    upper = max(float(finite.max()), lower)
    if is_lolp and upper <= 0.0:
        return (0.0, 1.0)
    if np.isclose(lower, upper):
        pad = max(abs(upper) * 0.08, 1.0)
        return (max(0.0, lower - pad), upper + pad)
    pad = 0.10 * (upper - lower)
    return (max(0.0, lower - pad), upper + pad)
