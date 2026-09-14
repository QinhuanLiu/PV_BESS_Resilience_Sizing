from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    matplotlib_context,
    nice_display_ticks,
    plot_smooth_line,
    save_publication_figure,
    write_source_data as _write_source_data,
)
from .display_units import (
    DISPLAY_BATTERY_ENERGY,
    DISPLAY_UE,
    DISPLAY_PV_CAPACITY,
    DISPLAY_SYSTEM_COST,
    FigureDisplayUnits,
    add_display_unit_columns,
)

FIGURE_6_NAME = "Figure_6_CAPEX_SAIDI_SAIFI_VoLL_Sensitivity"
SENSITIVITY_PARAMETERS = (
    "battery_capex_sgd_per_kwh",
    "pv_capex_sgd_per_kw",
    "saidi_min_per_year",
    "saifi_per_year",
    "voll_sgd_per_kwh",
)
PARAMETER_LABELS = {
    "battery_capex_sgd_per_kwh": "Battery CAPEX",
    "pv_capex_sgd_per_kw": "PV CAPEX",
    "saidi_min_per_year": "SAIDI",
    "saifi_per_year": "SAIFI",
    "voll_sgd_per_kwh": "VoLL",
}
PARAMETER_AXIS_LABELS = {
    "battery_capex_sgd_per_kwh": "Battery CAPEX (SGD kWh$^{-1}$)",
    "pv_capex_sgd_per_kw": "PV CAPEX (SGD kW$^{-1}$)",
    "saidi_min_per_year": "SAIDI (min yr$^{-1}$)",
    "saifi_per_year": "SAIFI (events yr$^{-1}$)",
    "voll_sgd_per_kwh": "VoLL (SGD kWh$^{-1}$)",
}
PARAMETER_UNIT_TYPES = {
    "battery_capex_sgd_per_kwh": "cost",
    "pv_capex_sgd_per_kw": "cost",
    "saidi_min_per_year": "duration",
    "saifi_per_year": "count",
    "voll_sgd_per_kwh": "money",
}
REQUIRED_COLUMNS = (
    "sensitivity_parameter",
    "sensitivity_value",
    "pv_bess_system_annual_cost_sgd",
    "unserved_energy_kwh_per_year",
)
PLOT_COLUMNS = (
    "sensitivity_value",
    "system_total_cost_million_sgd_per_year",
    "pv_capacity_mw",
    "battery_energy_mwh",
    "UE_mwh_per_year",
    "energy_based_LOLP",
)


@dataclass(frozen=True)
class SensitivityMetricSpec:
    key: str
    label: str
    unit_type: str
    color: str


METRICS = (
    SensitivityMetricSpec(
        key="system_total_cost_million_sgd_per_year",
        label="System total cost\n(million SGD yr$^{-1}$)",
        unit_type="cost",
        color=NATURE_PALETTE["blue"],
    ),
    SensitivityMetricSpec(
        key="pv_capacity_mw",
        label="PV capacity (MW)",
        unit_type="capacity",
        color=NATURE_PALETTE["gold"],
    ),
    SensitivityMetricSpec(
        key="battery_energy_mwh",
        label="Battery capacity (MWh)",
        unit_type="energy",
        color=NATURE_PALETTE["purple"],
    ),
    SensitivityMetricSpec(
        key="UE_mwh_per_year",
        label="UE (MWh yr$^{-1}$)",
        unit_type="ue",
        color=NATURE_PALETTE["red"],
    ),
    SensitivityMetricSpec(
        key="energy_based_LOLP",
        label="Energy-based LOLP",
        unit_type="lolp",
        color=NATURE_PALETTE["teal"],
    ),
)


def _display_metrics(units: FigureDisplayUnits) -> tuple[SensitivityMetricSpec, ...]:
    return (
        SensitivityMetricSpec(
            key=DISPLAY_SYSTEM_COST,
            label=f"System total cost\n({units.cost_axis_label.removeprefix('Cost (').removesuffix(')')})",
            unit_type="cost",
            color=NATURE_PALETTE["blue"],
        ),
        SensitivityMetricSpec(
            key=DISPLAY_PV_CAPACITY,
            label=units.pv_axis_label,
            unit_type="capacity",
            color=NATURE_PALETTE["gold"],
        ),
        SensitivityMetricSpec(
            key=DISPLAY_BATTERY_ENERGY,
            label=units.battery_capacity_label,
            unit_type="energy",
            color=NATURE_PALETTE["purple"],
        ),
        SensitivityMetricSpec(
            key=DISPLAY_UE,
            label=units.ue_axis_label,
            unit_type="ue",
            color=NATURE_PALETTE["red"],
        ),
        SensitivityMetricSpec(
            key="energy_based_LOLP",
            label="Energy-based LOLP",
            unit_type="lolp",
            color=NATURE_PALETTE["teal"],
        ),
    )


def build_sensitivity_comparison_source(sensitivity_rows: pd.DataFrame) -> pd.DataFrame:
    """Build Figure 6 source data from ordinary resilience-informed sensitivity outputs."""
    if sensitivity_rows.empty:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS) + ["parameter_label", *PLOT_COLUMNS])
    source = sensitivity_rows.copy()
    source = source[source["sensitivity_parameter"].astype(str).isin(SENSITIVITY_PARAMETERS)].copy()
    if source.empty:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS) + ["parameter_label", *PLOT_COLUMNS])

    missing = [column for column in REQUIRED_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Sensitivity comparison source is missing required column(s): " + ", ".join(missing))

    source["sensitivity_value"] = pd.to_numeric(source["sensitivity_value"], errors="coerce")
    source["unserved_energy_kwh_per_year"] = pd.to_numeric(source["unserved_energy_kwh_per_year"], errors="coerce")
    source["pv_bess_system_annual_cost_sgd"] = pd.to_numeric(
        source["pv_bess_system_annual_cost_sgd"],
        errors="coerce",
    )
    if source[["sensitivity_value", "unserved_energy_kwh_per_year", "pv_bess_system_annual_cost_sgd"]].isna().any().any():
        raise ValueError("Sensitivity comparison source contains missing or non-numeric required values.")

    source["parameter_label"] = source["sensitivity_parameter"].astype(str).map(PARAMETER_LABELS)
    if "pv_capacity_kw" in source.columns:
        source["pv_capacity_mw"] = pd.to_numeric(source["pv_capacity_kw"], errors="coerce") / 1000.0
    if "battery_energy_kwh" in source.columns:
        source["battery_energy_mwh"] = pd.to_numeric(source["battery_energy_kwh"], errors="coerce") / 1000.0
    for source_column, target_column in (
        ("annualized_pv_cost_sgd", "annualized_pv_cost_million_sgd_per_year"),
        ("annualized_battery_cost_sgd", "annualized_battery_cost_million_sgd_per_year"),
        ("pv_bess_grid_cost_sgd", "grid_cost_million_sgd_per_year"),
        ("unserved_energy_cost_sgd_per_year", "unserved_energy_cost_million_sgd_per_year"),
    ):
        if source_column in source.columns:
            source[target_column] = pd.to_numeric(source[source_column], errors="coerce") / 1e6
    source["UE_kwh_per_year"] = source["unserved_energy_kwh_per_year"].astype(float)
    source["UE_mwh_per_year"] = source["UE_kwh_per_year"] / 1000.0
    source["system_total_cost_sgd_per_year"] = source["pv_bess_system_annual_cost_sgd"].astype(float)
    source["system_total_cost_million_sgd_per_year"] = source["system_total_cost_sgd_per_year"] / 1e6
    if "outage_weighted_load_kwh" in source.columns:
        source["outage_weighted_load_kwh"] = pd.to_numeric(source["outage_weighted_load_kwh"], errors="coerce")
    elif "energy_based_LOLP" not in source.columns:
        raise ValueError("Sensitivity comparison source requires outage_weighted_load_kwh or energy_based_LOLP.")

    if "energy_based_LOLP" not in source.columns:
        source["energy_based_LOLP"] = source.apply(_energy_based_lolp, axis=1)
    else:
        source["energy_based_LOLP"] = pd.to_numeric(source["energy_based_LOLP"], errors="coerce")

    return validate_sensitivity_comparison_source(source)


def validate_sensitivity_comparison_source(source: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize Figure 6 source data."""
    if source.empty:
        raise ValueError("Sensitivity comparison source data is empty.")
    prepared = source.copy()
    missing = [column for column in PLOT_COLUMNS + ("sensitivity_parameter", "parameter_label") if column not in prepared.columns]
    if missing:
        raise ValueError("Sensitivity comparison source data is missing required column(s): " + ", ".join(missing))
    for column in PLOT_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if prepared[list(PLOT_COLUMNS)].isna().any().any():
        raise ValueError("Sensitivity comparison source data contains missing or non-numeric plot values.")
    if (prepared["UE_mwh_per_year"] < 0).any():
        raise ValueError("Sensitivity comparison UE values must be non-negative.")
    if (prepared["system_total_cost_million_sgd_per_year"] < 0).any():
        raise ValueError("Sensitivity comparison costs must be non-negative.")
    if not prepared["energy_based_LOLP"].between(0.0, 1.0).all():
        raise ValueError("Sensitivity comparison LOLP values must be between 0 and 1.")
    order = {parameter: index for index, parameter in enumerate(SENSITIVITY_PARAMETERS)}
    prepared["_parameter_order"] = prepared["sensitivity_parameter"].astype(str).map(order)
    if prepared["_parameter_order"].isna().any():
        raise ValueError("Sensitivity comparison source contains unsupported sensitivity parameter values.")
    return prepared.sort_values(["_parameter_order", "sensitivity_value"]).drop(columns=["_parameter_order"]).reset_index(drop=True)


def make_sensitivity_comparison_publication_figure(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
) -> dict[str, Path]:
    """Create the 5x5 publication-style sensitivity comparison figure."""
    prepared = build_sensitivity_comparison_source(source)
    base = out / f"{case_key}_{FIGURE_6_NAME}"
    paths: dict[str, Path] = {"source_data": _write_source_data(base, prepared)}
    paths.update(plot_sensitivity_comparison(out, case_key, prepared))
    return paths


def plot_sensitivity_comparison(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    prepared = validate_sensitivity_comparison_source(source)
    prepared, units = add_display_unit_columns(prepared)
    metrics = _display_metrics(units)
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)

    y_axes = {metric.key: _shared_y_axis(prepared, metric) for metric in metrics}
    fig, axes = plt.subplots(len(metrics), len(SENSITIVITY_PARAMETERS), figsize=(11.3, 8.9), constrained_layout=True)
    for col_index, parameter in enumerate(SENSITIVITY_PARAMETERS):
        parameter_rows = prepared[prepared["sensitivity_parameter"].astype(str) == parameter].copy()
        if parameter_rows.empty:
            raise ValueError(f"Figure 6 requires sensitivity rows for {parameter}.")
        parameter_rows = parameter_rows.sort_values("sensitivity_value")
        for row_index, metric in enumerate(metrics):
            ax = axes[row_index, col_index]
            _plot_metric_panel(ax, parameter_rows, parameter, metric, y_axes[metric.key])
            if row_index == 0:
                ax.set_title(f"{chr(ord('a') + col_index)}  {PARAMETER_LABELS[parameter]}", loc="left")
            if row_index == len(metrics) - 1:
                ax.set_xlabel(PARAMETER_AXIS_LABELS[parameter])
            if col_index == 0:
                ax.set_ylabel(metric.label)
    base = out / f"{case_key}_{FIGURE_6_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _plot_metric_panel(
    ax: Any,
    rows: pd.DataFrame,
    parameter: str,
    metric: SensitivityMetricSpec,
    y_axis: tuple[list[float], tuple[float, float]],
) -> None:
    x = rows["sensitivity_value"].astype(float).to_numpy()
    y = rows[metric.key].astype(float).to_numpy()
    plot_smooth_line(
        ax,
        x,
        y,
        color=metric.color,
        linewidth=1.25,
    )
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.55)
    _apply_axis_ticks(ax, x, axis="x", unit_type=PARAMETER_UNIT_TYPES[parameter])
    _apply_axis_ticks(ax, y_axis[0], axis="y", unit_type=metric.unit_type)
    ax.set_ylim(*y_axis[1])


def _shared_y_axis(source: pd.DataFrame, metric: SensitivityMetricSpec) -> tuple[list[float], tuple[float, float]]:
    values = source[metric.key].astype(float).to_numpy()
    ticks = nice_display_ticks(values, unit_type=metric.unit_type, target_count=4)
    lower = min(min(ticks), float(values.min()))
    upper = max(max(ticks), float(values.max()))
    if lower == upper:
        pad = max(abs(lower) * 0.05, 1.0 if metric.unit_type != "lolp" else 0.01)
    else:
        pad = (upper - lower) * 0.06
    if metric.unit_type == "lolp":
        return ticks, (max(0.0, lower - pad), min(1.0, upper + pad))
    return ticks, (lower - pad, upper + pad)


def _apply_axis_ticks(ax: Any, values: Any, axis: str, unit_type: str) -> None:
    ticks = nice_display_ticks(values, unit_type=unit_type, target_count=4)
    if axis == "x":
        ax.set_xticks(ticks)
        labels = _tick_labels(ticks, unit_type)
        ax.set_xticklabels(labels, rotation=25 if len(labels) > 3 else 0, ha="right" if len(labels) > 3 else "center")
        return
    ax.set_yticks(ticks)
    ax.set_yticklabels(_tick_labels(ticks, unit_type))


def _tick_labels(ticks: list[float], unit_type: str) -> list[str]:
    if unit_type == "lolp":
        return [f"{tick * 100:.0f}%" for tick in ticks]
    return [f"{tick:g}" for tick in ticks]


def _energy_based_lolp(row: pd.Series) -> float:
    outage_load = float(row["outage_weighted_load_kwh"])
    if outage_load <= 0.0:
        return 0.0
    return float(row["UE_kwh_per_year"]) / outage_load
