from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    matplotlib_context,
    nice_display_ticks,
    save_publication_figure,
    write_source_data,
)
from .display_units import (
    DISPLAY_BATTERY_ENERGY,
    DISPLAY_UE,
    DISPLAY_PV_CAPACITY,
    DISPLAY_SYSTEM_COST,
    add_display_unit_columns,
)
from .sensitivity import (
    METRICS as FIGURE_6_METRICS,
    PARAMETER_AXIS_LABELS,
    PARAMETER_LABELS,
    PARAMETER_UNIT_TYPES,
    SENSITIVITY_PARAMETERS,
)
from .uncertainty import build_uncertainty_baseline_figure_source


FIGURE_8A_NAME = "Figure_8A_Uncertainty_Baseline_Distributions"
FIGURE_8B_NAME = "Figure_8B_Uncertainty_Design_Cost_Bubble"
FIGURE_8_SOURCE_NAME = "Figure_8_Uncertainty_Baseline"
FIGURE_9_NAME = "Figure_9_Robust_Sensitivity_Bands"
FIGURE_10A_NAME = "Figure_10A_Robust_Design_Trajectories"
FIGURE_10B_NAME = "Figure_10B_Robust_Cost_Composition"
FIGURE_10_SOURCE_NAME = "Figure_10_Robust_Sensitivity_Mechanism"
FIGURE_11_NAME = "Figure_11_Tail_Robustness_Resilience"


ROBUST_REQUIRED_COLUMNS = (
    "sensitivity_parameter",
    "sensitivity_value",
    "uncertainty_scenario_id",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "pv_bess_system_annual_cost_sgd",
    "unserved_energy_kwh_per_year",
)
MECHANISM_COST_COLUMNS = (
    "pv_bess_grid_cost_sgd",
    "unserved_energy_cost_sgd_per_year",
)
PV_COST_OPTIONS = (
    ("pv_total_annual_cost_sgd",),
    ("annualized_pv_cost_sgd", "pv_om_cost_sgd"),
)
BATTERY_COST_OPTIONS = (
    ("annualized_battery_cost_sgd", "battery_om_cost_sgd"),
)


@dataclass(frozen=True)
class RobustMetricSpec:
    key: str
    label: str
    unit_type: str
    color: str


ROBUST_METRICS = tuple(
    RobustMetricSpec(
        key=metric.key,
        label=metric.label,
        unit_type=metric.unit_type,
        color=metric.color,
    )
    for metric in FIGURE_6_METRICS
)


def build_figure_8_uncertainty_baseline_source(all_results: pd.DataFrame) -> pd.DataFrame:
    return build_uncertainty_baseline_figure_source(all_results)


def make_figure_8_uncertainty_baseline(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
) -> dict[str, Path]:
    prepared = build_figure_8_uncertainty_baseline_source(source)
    base = out / f"{case_key}_{FIGURE_8_SOURCE_NAME}"
    paths: dict[str, Path] = {"source_data": write_source_data(base, prepared)}
    paths.update(plot_figure_8a_uncertainty_distributions(out, case_key, prepared))
    paths.update(plot_figure_8b_uncertainty_design_bubble(out, case_key, prepared))
    return paths


def plot_figure_8a_uncertainty_distributions(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    prepared = build_figure_8_uncertainty_baseline_source(source)
    prepared, units = add_display_unit_columns(prepared)
    mpl, plt = matplotlib_context(out / f"{case_key}_{FIGURE_8A_NAME}")
    apply_nature_style(mpl)
    fig, axes = plt.subplots(1, 5, figsize=(7.8, 2.55), constrained_layout=False)
    panels = [
        ("Cost", DISPLAY_SYSTEM_COST, units.cost_axis_label.removeprefix("Cost ").strip("()"), NATURE_PALETTE["blue"]),
        ("PV", DISPLAY_PV_CAPACITY, units.pv_unit, NATURE_PALETTE["teal"]),
        ("BESS", DISPLAY_BATTERY_ENERGY, units.battery_unit, NATURE_PALETTE["purple"]),
        ("UE", DISPLAY_UE, units.ue_axis_label.removeprefix("UE ").strip("()"), NATURE_PALETTE["red"]),
        ("LOLP", "scenario_LOLP", "Fraction", NATURE_PALETTE["gold"]),
    ]
    rng = np.random.default_rng(20260727)
    for index, (ax, (title, column, ylabel, color)) in enumerate(zip(axes, panels, strict=True), start=1):
        _draw_violin_box(ax, prepared[column].to_numpy(dtype=float), color, rng)
        ax.set_title(f"{chr(96 + index)}  {title}", loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.45)
        ax.set_xticks([])
        if column in {DISPLAY_BATTERY_ENERGY, DISPLAY_UE}:
            current_top = float(ax.get_ylim()[1])
            value_top = float(prepared[column].max()) * 1.15
            ax.set_ylim(0.0, max(current_top, value_top, 0.05))
        if column == "scenario_LOLP":
            ax.set_ylim(0.0, min(1.0, max(0.05, float(prepared[column].max()) * 1.15)))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.24, top=0.84, wspace=0.52)
    base = out / f"{case_key}_{FIGURE_8A_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_figure_8b_uncertainty_design_bubble(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    prepared = build_figure_8_uncertainty_baseline_source(source)
    prepared, units = add_display_unit_columns(prepared)
    mpl, plt = matplotlib_context(out / f"{case_key}_{FIGURE_8B_NAME}")
    apply_nature_style(mpl)
    fig, ax = plt.subplots(figsize=(4.9, 3.65), constrained_layout=False)
    sizes = _size_from_values(prepared[DISPLAY_UE].to_numpy(dtype=float), 34.0, 260.0)
    cost_values = prepared[DISPLAY_SYSTEM_COST].to_numpy(dtype=float)
    cost_ticks = _figure_8b_cost_colorbar_ticks(cost_values)
    cost_norm = mpl.colors.Normalize(vmin=float(cost_ticks[0]), vmax=float(cost_ticks[-1]))
    sc = ax.scatter(
        prepared[DISPLAY_PV_CAPACITY],
        prepared[DISPLAY_BATTERY_ENERGY],
        c=cost_values,
        s=sizes,
        cmap="viridis",
        norm=cost_norm,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.78,
        zorder=3,
    )
    ax.grid(color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.45, zorder=0)
    ax.set_title("Robust design cloud across outage realizations", loc="left")
    ax.set_xlabel(units.pv_axis_label)
    ax.set_ylabel(units.battery_axis_label)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.025)
    _format_figure_8b_cost_colorbar(cbar, cost_ticks, units.cost_axis_label)
    _add_size_key(fig, sizes, prepared[DISPLAY_UE].to_numpy(dtype=float), units.ue_axis_label)
    fig.subplots_adjust(left=0.14, right=0.80, bottom=0.16, top=0.90)
    base = out / f"{case_key}_{FIGURE_8B_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _figure_8b_cost_colorbar_ticks(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Figure 8B cost colorbar requires at least one finite cost value.")
    lower = float(finite.min())
    upper = float(finite.max())
    span = max(upper - lower, 1e-9)
    if span <= 3.0:
        step = 0.5
    elif span <= 8.0:
        step = 1.0
    else:
        step = 2.0
    tick_lower = np.floor(lower / step) * step
    tick_upper = np.ceil(upper / step) * step
    if np.isclose(tick_lower, tick_upper):
        tick_upper = tick_lower + step
    return np.arange(tick_lower, tick_upper + step * 0.5, step, dtype=float)


def _format_figure_8b_cost_colorbar(colorbar: Any, ticks: np.ndarray, label: str) -> Any:
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{tick:g}" for tick in ticks])
    colorbar.set_label(label.replace(" (", "\n("), fontsize=6.2)
    colorbar.ax.tick_params(labelsize=6.0, length=2.2, width=0.6)
    return colorbar


def build_robust_sensitivity_band_source(uncertainty_sensitivity: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_robust_sensitivity_rows(uncertainty_sensitivity, require_cost_components=False)
    rows: list[dict[str, Any]] = []
    order = {parameter: index for index, parameter in enumerate(SENSITIVITY_PARAMETERS)}
    metric_order = {metric.key: index for index, metric in enumerate(ROBUST_METRICS)}
    for (parameter, value), group in normalized.groupby(["sensitivity_parameter", "sensitivity_value"], sort=False):
        for metric in ROBUST_METRICS:
            values = group[metric.key].astype(float).to_numpy()
            rows.append(
                {
                    "sensitivity_parameter": str(parameter),
                    "parameter_label": PARAMETER_LABELS[str(parameter)],
                    "parameter_order": order[str(parameter)],
                    "sensitivity_value": float(value),
                    "metric_key": metric.key,
                    "metric_label": metric.label.replace("\n", " "),
                    "metric_order": metric_order[metric.key],
                    "unit_type": metric.unit_type,
                    "p10": float(np.quantile(values, 0.10)),
                    "median": float(np.quantile(values, 0.50)),
                    "p90": float(np.quantile(values, 0.90)),
                    "scenario_count": int(np.isfinite(values).sum()),
                }
            )
    source = pd.DataFrame(rows)
    return validate_robust_sensitivity_band_source(source)


def make_figure_9_robust_sensitivity_bands(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
) -> dict[str, Path]:
    prepared = (
        validate_robust_sensitivity_band_source(source)
        if {"metric_key", "p10", "median", "p90"}.issubset(source.columns)
        else build_robust_sensitivity_band_source(source)
    )
    base = out / f"{case_key}_{FIGURE_9_NAME}"
    paths: dict[str, Path] = {"source_data": write_source_data(base, prepared)}
    paths.update(plot_figure_9_robust_sensitivity_bands(out, case_key, prepared))
    return paths


def plot_figure_9_robust_sensitivity_bands(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    prepared = validate_robust_sensitivity_band_source(source)
    mpl, plt = matplotlib_context(out / f"{case_key}_{FIGURE_9_NAME}")
    apply_nature_style(mpl)
    fig, axes = plt.subplots(len(ROBUST_METRICS), len(SENSITIVITY_PARAMETERS), figsize=(11.4, 8.9), constrained_layout=True)
    y_axes = {metric.key: _band_y_axis(prepared, metric) for metric in ROBUST_METRICS}
    for col_index, parameter in enumerate(SENSITIVITY_PARAMETERS):
        for row_index, metric in enumerate(ROBUST_METRICS):
            ax = axes[row_index, col_index]
            rows = prepared[
                (prepared["sensitivity_parameter"] == parameter)
                & (prepared["metric_key"] == metric.key)
            ].sort_values("sensitivity_value")
            x = rows["sensitivity_value"].to_numpy(dtype=float)
            p10 = rows["p10"].to_numpy(dtype=float)
            median = rows["median"].to_numpy(dtype=float)
            p90 = rows["p90"].to_numpy(dtype=float)
            ax.fill_between(x, p10, p90, color=metric.color, alpha=0.20, linewidth=0.0)
            ax.plot(x, median, marker="o", color=metric.color, markerfacecolor="white", markeredgewidth=0.75)
            ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.55)
            _apply_ticks(ax, x, "x", PARAMETER_UNIT_TYPES[parameter])
            _apply_ticks(ax, y_axes[metric.key][0], "y", metric.unit_type)
            ax.set_ylim(*y_axes[metric.key][1])
            if row_index == 0:
                ax.set_title(f"{chr(ord('a') + col_index)}  {PARAMETER_LABELS[parameter]}", loc="left")
            if row_index == len(ROBUST_METRICS) - 1:
                ax.set_xlabel(PARAMETER_AXIS_LABELS[parameter])
            if col_index == 0:
                ax.set_ylabel(metric.label)
    base = out / f"{case_key}_{FIGURE_9_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def build_robust_sensitivity_mechanism_source(uncertainty_sensitivity: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_robust_sensitivity_rows(uncertainty_sensitivity, require_cost_components=True)
    rows: list[dict[str, Any]] = []
    order = {parameter: index for index, parameter in enumerate(SENSITIVITY_PARAMETERS)}
    for (parameter, value), group in normalized.groupby(["sensitivity_parameter", "sensitivity_value"], sort=False):
        row: dict[str, Any] = {
            "sensitivity_parameter": str(parameter),
            "parameter_label": PARAMETER_LABELS[str(parameter)],
            "parameter_order": order[str(parameter)],
            "sensitivity_value": float(value),
            "scenario_count": int(len(group)),
        }
        for column in (
            "pv_capacity_mw",
            "battery_energy_mwh",
            "grid_cost_million_sgd_per_year",
            "pv_annual_cost_million_sgd_per_year",
            "battery_annual_cost_million_sgd_per_year",
            "lost_load_cost_million_sgd_per_year",
        ):
            values = group[column].astype(float).to_numpy()
            row[f"{column}_p10"] = float(np.quantile(values, 0.10))
            row[f"{column}_median"] = float(np.quantile(values, 0.50))
            row[f"{column}_p90"] = float(np.quantile(values, 0.90))
        rows.append(row)
    return validate_robust_sensitivity_mechanism_source(pd.DataFrame(rows))


def make_figure_10_robust_sensitivity_mechanism(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
) -> dict[str, Path]:
    prepared = (
        validate_robust_sensitivity_mechanism_source(source)
        if {"pv_capacity_mw_median", "grid_cost_million_sgd_per_year_median"}.issubset(source.columns)
        else build_robust_sensitivity_mechanism_source(source)
    )
    base = out / f"{case_key}_{FIGURE_10_SOURCE_NAME}"
    paths: dict[str, Path] = {"source_data": write_source_data(base, prepared)}
    paths.update(plot_figure_10a_robust_design_trajectories(out, case_key, prepared))
    paths.update(plot_figure_10b_robust_cost_composition(out, case_key, prepared))
    return paths


def plot_figure_10a_robust_design_trajectories(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    prepared = validate_robust_sensitivity_mechanism_source(source)
    mpl, plt = matplotlib_context(out / f"{case_key}_{FIGURE_10A_NAME}")
    apply_nature_style(mpl)
    fig, axes = plt.subplots(1, len(SENSITIVITY_PARAMETERS), figsize=(11.4, 2.85), constrained_layout=True)
    for index, parameter in enumerate(SENSITIVITY_PARAMETERS):
        ax = axes[index]
        rows = prepared[prepared["sensitivity_parameter"] == parameter].sort_values("sensitivity_value")
        x = rows["pv_capacity_mw_median"].to_numpy(dtype=float)
        y = rows["battery_energy_mwh_median"].to_numpy(dtype=float)
        xerr = np.vstack([
            x - rows["pv_capacity_mw_p10"].to_numpy(dtype=float),
            rows["pv_capacity_mw_p90"].to_numpy(dtype=float) - x,
        ])
        yerr = np.vstack([
            y - rows["battery_energy_mwh_p10"].to_numpy(dtype=float),
            rows["battery_energy_mwh_p90"].to_numpy(dtype=float) - y,
        ])
        ax.errorbar(
            x,
            y,
            xerr=np.maximum(xerr, 0.0),
            yerr=np.maximum(yerr, 0.0),
            fmt="o-",
            color=NATURE_PALETTE["blue"],
            ecolor=NATURE_PALETTE["blue_light"],
            elinewidth=0.75,
            capsize=2.0,
            markerfacecolor="white",
            markeredgewidth=0.75,
        )
        for _, row in rows.iterrows():
            ax.annotate(
                f"{float(row['sensitivity_value']):g}",
                (float(row["pv_capacity_mw_median"]), float(row["battery_energy_mwh_median"])),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=5.8,
                color=NATURE_PALETTE["neutral_dark"],
            )
        ax.set_title(f"{chr(ord('a') + index)}  {PARAMETER_LABELS[parameter]}", loc="left")
        ax.set_xlabel("PV capacity (MW)")
        if index == 0:
            ax.set_ylabel("BESS energy capacity (MWh)")
        ax.grid(color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.50)
    base = out / f"{case_key}_{FIGURE_10A_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_figure_10b_robust_cost_composition(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    prepared = validate_robust_sensitivity_mechanism_source(source)
    parameters = ("saidi_min_per_year", "saifi_per_year", "voll_sgd_per_kwh")
    mpl, plt = matplotlib_context(out / f"{case_key}_{FIGURE_10B_NAME}")
    apply_nature_style(mpl)
    fig, axes = plt.subplots(1, len(parameters), figsize=(8.0, 3.2), constrained_layout=False)
    components = [
        ("Grid bill", "grid_cost_million_sgd_per_year_median", NATURE_PALETTE["neutral_light"]),
        ("PV dep. + O&M", "pv_annual_cost_million_sgd_per_year_median", NATURE_PALETTE["teal"]),
        ("Battery dep. + O&M", "battery_annual_cost_million_sgd_per_year_median", NATURE_PALETTE["blue"]),
        ("Lost load", "lost_load_cost_million_sgd_per_year_median", NATURE_PALETTE["red"]),
    ]
    handles = []
    for index, parameter in enumerate(parameters):
        ax = axes[index]
        rows = prepared[prepared["sensitivity_parameter"] == parameter].sort_values("sensitivity_value")
        x = np.arange(len(rows), dtype=float)
        bottom = np.zeros(len(rows), dtype=float)
        for label, column, color in components:
            values = rows[column].to_numpy(dtype=float)
            bars = ax.bar(x, values, bottom=bottom, color=color, edgecolor="white", linewidth=0.35, width=0.72)
            if index == 0:
                handles.append(bars[0])
            bottom += values
        labels = [f"{value:g}" for value in rows["sensitivity_value"].to_numpy(dtype=float)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25 if len(labels) > 3 else 0, ha="right" if len(labels) > 3 else "center")
        ax.set_title(f"{chr(ord('a') + index)}  {PARAMETER_LABELS[parameter]}", loc="left")
        ax.set_xlabel(PARAMETER_AXIS_LABELS[parameter])
        if index == 0:
            ax.set_ylabel("Annualized cost (million SGD yr$^{-1}$)")
        ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.48)
    fig.legend(handles, [item[0] for item in components], loc="upper center", ncol=4, bbox_to_anchor=(0.52, 0.99))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.25, top=0.78, wspace=0.34)
    base = out / f"{case_key}_{FIGURE_10B_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def build_tail_robustness_source(
    uncertainty_baseline: pd.DataFrame,
    uncertainty_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    baseline = build_figure_8_uncertainty_baseline_source(uncertainty_baseline)
    if "unserved_energy_cost_sgd_per_year" in uncertainty_baseline.columns:
        lost_load = uncertainty_baseline[["uncertainty_scenario_id", "unserved_energy_cost_sgd_per_year"]].copy()
        lost_load["uncertainty_scenario_id"] = pd.to_numeric(lost_load["uncertainty_scenario_id"], errors="coerce")
        lost_load["lost_load_cost_million_sgd_per_year"] = pd.to_numeric(
            lost_load["unserved_energy_cost_sgd_per_year"],
            errors="coerce",
        ) / 1e6
        baseline = baseline.merge(
            lost_load[["uncertainty_scenario_id", "lost_load_cost_million_sgd_per_year"]],
            on="uncertainty_scenario_id",
            how="left",
        )
    else:
        baseline["lost_load_cost_million_sgd_per_year"] = np.nan
    robust = _normalize_robust_sensitivity_rows(uncertainty_sensitivity, require_cost_components=False)
    rows: list[dict[str, Any]] = []
    sorted_ue = baseline.sort_values("UE_mwh_per_year", ascending=False).reset_index(drop=True)
    n = len(sorted_ue)
    for index, row in sorted_ue.iterrows():
        rows.append(
            {
                "source_type": "baseline_exceedance",
                "sensitivity_parameter": "baseline",
                "parameter_label": "Baseline",
                "sensitivity_value": np.nan,
                "uncertainty_scenario_id": int(row["uncertainty_scenario_id"]),
                "exceedance_probability": float((index + 1) / max(n, 1)),
                "UE_mwh_per_year": float(row["UE_mwh_per_year"]),
                "lost_load_cost_million_sgd_per_year": float(row["lost_load_cost_million_sgd_per_year"])
                if pd.notna(row["lost_load_cost_million_sgd_per_year"])
                else np.nan,
                "energy_based_LOLP": float(row["scenario_LOLP"]),
                "scenario_count": n,
            }
        )
    _append_tail_summary(rows, "baseline", "Baseline", np.nan, baseline)
    for (parameter, value), group in robust.groupby(["sensitivity_parameter", "sensitivity_value"], sort=False):
        _append_tail_summary(rows, str(parameter), PARAMETER_LABELS[str(parameter)], float(value), group)
    return validate_tail_robustness_source(pd.DataFrame(rows))


def make_figure_11_tail_robustness(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
) -> dict[str, Path]:
    prepared = validate_tail_robustness_source(source)
    base = out / f"{case_key}_{FIGURE_11_NAME}"
    paths: dict[str, Path] = {"source_data": write_source_data(base, prepared)}
    paths.update(plot_figure_11_tail_robustness(out, case_key, prepared))
    return paths


def plot_figure_11_tail_robustness(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    prepared = validate_tail_robustness_source(source)
    mpl, plt = matplotlib_context(out / f"{case_key}_{FIGURE_11_NAME}")
    apply_nature_style(mpl)
    fig = plt.figure(figsize=(7.6, 4.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.25, 1.0])
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_baseline = fig.add_subplot(gs[0, 1])
    ax_ue = fig.add_subplot(gs[1, 0])
    ax_lolp = fig.add_subplot(gs[1, 1])

    exceedance = prepared[prepared["source_type"] == "baseline_exceedance"].sort_values("exceedance_probability")
    ax_curve.plot(
        exceedance["exceedance_probability"] * 100.0,
        exceedance["UE_mwh_per_year"],
        color=NATURE_PALETTE["red"],
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.65,
    )
    ax_curve.set_title("a  Baseline UE exceedance", loc="left")
    ax_curve.set_xlabel("Exceedance probability (%)")
    ax_curve.set_ylabel("UE (MWh yr$^{-1}$)")
    ax_curve.grid(color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.50)

    baseline_summary = prepared[
        (prepared["source_type"] == "tail_summary")
        & (prepared["sensitivity_parameter"] == "baseline")
        & (prepared["tail_metric"] == "UE_mwh_per_year")
    ].iloc[0]
    labels = ["Median", "P90", "P95", "Max"]
    values = [baseline_summary["median"], baseline_summary["p90"], baseline_summary["p95"], baseline_summary["max"]]
    ax_baseline.bar(np.arange(4), values, color=[NATURE_PALETTE["neutral_mid"], NATURE_PALETTE["gold"], NATURE_PALETTE["red_light"], NATURE_PALETTE["red"]])
    ax_baseline.set_title("b  Baseline tail severity", loc="left")
    ax_baseline.set_ylabel("UE (MWh yr$^{-1}$)")
    ax_baseline.set_xticks(np.arange(4))
    ax_baseline.set_xticklabels(labels, rotation=25, ha="right")
    ax_baseline.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.50)

    _plot_tail_interval_panel(ax_ue, prepared, "UE_mwh_per_year", "c  Robust P95 UE", "P95 UE (MWh yr$^{-1}$)")
    _plot_tail_interval_panel(ax_lolp, prepared, "energy_based_LOLP", "d  Robust P95 LOLP", "P95 energy-based LOLP")

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.14, top=0.91, wspace=0.34, hspace=0.46)
    base = out / f"{case_key}_{FIGURE_11_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def validate_robust_sensitivity_band_source(source: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        raise ValueError("Robust sensitivity band source data is empty.")
    required = (
        "sensitivity_parameter",
        "parameter_label",
        "parameter_order",
        "sensitivity_value",
        "metric_key",
        "metric_label",
        "metric_order",
        "unit_type",
        "p10",
        "median",
        "p90",
        "scenario_count",
    )
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError("Robust sensitivity band source is missing required column(s): " + ", ".join(missing))
    prepared = source.copy()
    _require_complete_parameters(prepared)
    for column in ("parameter_order", "sensitivity_value", "metric_order", "p10", "median", "p90", "scenario_count"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if prepared[["sensitivity_value", "p10", "median", "p90", "scenario_count"]].isna().any().any():
        raise ValueError("Robust sensitivity band source contains missing or non-numeric values.")
    if (prepared["scenario_count"] <= 0).any():
        raise ValueError("Robust sensitivity band source requires positive scenario_count.")
    if ((prepared["p10"] > prepared["median"]) | (prepared["median"] > prepared["p90"])).any():
        raise ValueError("Robust sensitivity band quantiles must satisfy p10 <= median <= p90.")
    return prepared.sort_values(["parameter_order", "metric_order", "sensitivity_value"]).reset_index(drop=True)


def validate_robust_sensitivity_mechanism_source(source: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        raise ValueError("Robust sensitivity mechanism source data is empty.")
    base_columns = ("sensitivity_parameter", "parameter_label", "parameter_order", "sensitivity_value", "scenario_count")
    metric_columns: list[str] = []
    for stem in (
        "pv_capacity_mw",
        "battery_energy_mwh",
        "grid_cost_million_sgd_per_year",
        "pv_annual_cost_million_sgd_per_year",
        "battery_annual_cost_million_sgd_per_year",
        "lost_load_cost_million_sgd_per_year",
    ):
        metric_columns.extend([f"{stem}_p10", f"{stem}_median", f"{stem}_p90"])
    missing = [column for column in (*base_columns, *metric_columns) if column not in source.columns]
    if missing:
        raise ValueError("Robust sensitivity mechanism source is missing required column(s): " + ", ".join(missing))
    prepared = source.copy()
    _require_complete_parameters(prepared)
    for column in ("parameter_order", "sensitivity_value", "scenario_count", *metric_columns):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if prepared[["sensitivity_value", "scenario_count", *metric_columns]].isna().any().any():
        raise ValueError("Robust sensitivity mechanism source contains missing or non-numeric values.")
    if (prepared["scenario_count"] <= 0).any():
        raise ValueError("Robust sensitivity mechanism source requires positive scenario_count.")
    return prepared.sort_values(["parameter_order", "sensitivity_value"]).reset_index(drop=True)


def validate_tail_robustness_source(source: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        raise ValueError("Tail robustness source data is empty.")
    required = ("source_type", "sensitivity_parameter", "parameter_label")
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError("Tail robustness source is missing required column(s): " + ", ".join(missing))
    prepared = source.copy()
    if not {"baseline_exceedance", "tail_summary"}.issubset(set(prepared["source_type"].astype(str))):
        raise ValueError("Tail robustness source requires baseline_exceedance and tail_summary rows.")
    for column in (
        "sensitivity_value",
        "uncertainty_scenario_id",
        "exceedance_probability",
        "UE_mwh_per_year",
        "lost_load_cost_million_sgd_per_year",
        "energy_based_LOLP",
        "scenario_count",
        "median",
        "p90",
        "p95",
        "max",
    ):
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    baseline = prepared[prepared["source_type"] == "baseline_exceedance"]
    if baseline[["exceedance_probability", "UE_mwh_per_year", "energy_based_LOLP"]].isna().any().any():
        raise ValueError("Tail robustness baseline exceedance rows contain missing numeric values.")
    if not baseline["exceedance_probability"].between(0.0, 1.0).all():
        raise ValueError("Tail robustness exceedance probabilities must be between 0 and 1.")
    summary = prepared[prepared["source_type"] == "tail_summary"]
    if summary[["scenario_count", "median", "p90", "p95", "max"]].isna().any().any():
        raise ValueError("Tail robustness summary rows contain missing numeric values.")
    return prepared.reset_index(drop=True)


def _normalize_robust_sensitivity_rows(
    uncertainty_sensitivity: pd.DataFrame,
    *,
    require_cost_components: bool,
) -> pd.DataFrame:
    if uncertainty_sensitivity.empty:
        raise ValueError("Robust sensitivity figures require uncertainty sensitivity results.")
    missing = [column for column in ROBUST_REQUIRED_COLUMNS if column not in uncertainty_sensitivity.columns]
    if missing:
        raise ValueError("Robust sensitivity source is missing required column(s): " + ", ".join(missing))
    source = uncertainty_sensitivity.copy()
    source = source[source["sensitivity_parameter"].astype(str).isin(SENSITIVITY_PARAMETERS)].copy()
    if source.empty:
        raise ValueError("Robust sensitivity source contains no supported sensitivity parameters.")
    for parameter in SENSITIVITY_PARAMETERS:
        if parameter not in set(source["sensitivity_parameter"].astype(str)):
            raise ValueError(f"Robust sensitivity source is missing rows for {PARAMETER_LABELS[parameter]}.")
    numeric = [column for column in ROBUST_REQUIRED_COLUMNS if column != "sensitivity_parameter"]
    if "outage_weighted_load_kwh" in source.columns:
        numeric.append("outage_weighted_load_kwh")
    elif "energy_based_LOLP" not in source.columns:
        raise ValueError("Robust sensitivity source requires outage_weighted_load_kwh or energy_based_LOLP.")
    for column in numeric:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source[numeric].isna().any().any():
        raise ValueError("Robust sensitivity source contains missing or non-numeric required values.")
    source["parameter_label"] = source["sensitivity_parameter"].astype(str).map(PARAMETER_LABELS)
    source["pv_capacity_mw"] = source["pv_capacity_kw"].astype(float) / 1000.0
    source["battery_energy_mwh"] = source["battery_energy_kwh"].astype(float) / 1000.0
    source["system_total_cost_million_sgd_per_year"] = source["pv_bess_system_annual_cost_sgd"].astype(float) / 1e6
    source["UE_mwh_per_year"] = source["unserved_energy_kwh_per_year"].astype(float) / 1000.0
    if "energy_based_LOLP" in source.columns:
        source["energy_based_LOLP"] = pd.to_numeric(source["energy_based_LOLP"], errors="coerce")
    else:
        source["energy_based_LOLP"] = np.divide(
            source["unserved_energy_kwh_per_year"].astype(float),
            source["outage_weighted_load_kwh"].astype(float),
            out=np.zeros(len(source), dtype=float),
            where=source["outage_weighted_load_kwh"].astype(float).to_numpy() > 0.0,
        )
    source["energy_based_LOLP"] = source["energy_based_LOLP"].clip(lower=0.0, upper=1.0)
    if "unserved_energy_cost_sgd_per_year" in source.columns:
        source["lost_load_cost_million_sgd_per_year"] = pd.to_numeric(
            source["unserved_energy_cost_sgd_per_year"],
            errors="coerce",
        ) / 1e6
    else:
        source["lost_load_cost_million_sgd_per_year"] = np.nan
    if require_cost_components:
        _add_cost_components(source)
    order = {parameter: index for index, parameter in enumerate(SENSITIVITY_PARAMETERS)}
    source["_parameter_order"] = source["sensitivity_parameter"].astype(str).map(order)
    return source.sort_values(["_parameter_order", "sensitivity_value", "uncertainty_scenario_id"]).drop(columns="_parameter_order").reset_index(drop=True)


def _add_cost_components(source: pd.DataFrame) -> None:
    missing = [column for column in MECHANISM_COST_COLUMNS if column not in source.columns]
    pv_columns = _select_cost_columns(source, PV_COST_OPTIONS, "PV")
    battery_columns = _select_cost_columns(source, BATTERY_COST_OPTIONS, "battery")
    if missing:
        raise ValueError(
            "Figure 10 robust mechanism source requires cost component column(s): "
            + ", ".join(missing)
        )
    source["grid_cost_million_sgd_per_year"] = pd.to_numeric(source["pv_bess_grid_cost_sgd"], errors="coerce") / 1e6
    source["pv_annual_cost_million_sgd_per_year"] = sum(
        pd.to_numeric(source[column], errors="coerce") for column in pv_columns
    ) / 1e6
    source["battery_annual_cost_million_sgd_per_year"] = sum(
        pd.to_numeric(source[column], errors="coerce") for column in battery_columns
    ) / 1e6
    source["lost_load_cost_million_sgd_per_year"] = pd.to_numeric(
        source["unserved_energy_cost_sgd_per_year"],
        errors="coerce",
    ) / 1e6
    component_columns = [
        "grid_cost_million_sgd_per_year",
        "pv_annual_cost_million_sgd_per_year",
        "battery_annual_cost_million_sgd_per_year",
        "lost_load_cost_million_sgd_per_year",
    ]
    if source[component_columns].isna().any().any():
        raise ValueError("Figure 10 robust mechanism source contains non-numeric cost component values.")


def _select_cost_columns(source: pd.DataFrame, options: tuple[tuple[str, ...], ...], label: str) -> tuple[str, ...]:
    for columns in options:
        if all(column in source.columns for column in columns):
            return columns
    expected = " or ".join(" + ".join(columns) for columns in options)
    raise ValueError(f"Figure 10 robust mechanism source requires {label} cost columns: {expected}.")


def _append_tail_summary(
    rows: list[dict[str, Any]],
    parameter: str,
    label: str,
    value: float,
    group: pd.DataFrame,
) -> None:
    metric_map = {
        "UE_mwh_per_year": "UE (MWh yr^-1)",
        "lost_load_cost_million_sgd_per_year": "Lost load cost (M SGD yr^-1)",
        "energy_based_LOLP": "Energy-based LOLP",
    }
    for metric, metric_label in metric_map.items():
        if metric not in group.columns:
            continue
        values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        rows.append(
            {
                "source_type": "tail_summary",
                "sensitivity_parameter": parameter,
                "parameter_label": label,
                "sensitivity_value": value,
                "tail_metric": metric,
                "tail_metric_label": metric_label,
                "scenario_count": int(values.size),
                "median": float(np.quantile(values, 0.50)),
                "p90": float(np.quantile(values, 0.90)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
            }
        )


def _require_complete_parameters(source: pd.DataFrame) -> None:
    present = set(source["sensitivity_parameter"].astype(str).unique())
    missing = [PARAMETER_LABELS[parameter] for parameter in SENSITIVITY_PARAMETERS if parameter not in present]
    if missing:
        raise ValueError("Robust sensitivity source is missing required parameter(s): " + ", ".join(missing))


def _draw_violin_box(ax: Any, values: np.ndarray, color: str, rng: np.random.Generator) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot draw uncertainty distribution with no finite values.")
    if finite.size >= 2 and not np.isclose(float(finite.min()), float(finite.max())):
        violin = ax.violinplot([finite], positions=[1.0], widths=0.70, showmeans=False, showmedians=False, showextrema=False)
        body = violin["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor(NATURE_PALETTE["neutral_black"])
        body.set_alpha(0.32)
        body.set_linewidth(0.65)
    q10, q25, median, q75, q90 = np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90])
    jitter = 1.0 + rng.normal(0.0, 0.035, finite.size)
    ax.scatter(jitter, finite, s=7.0, color=color, alpha=0.45, linewidth=0.0, zorder=3)
    ax.vlines(1.0, q10, q90, color=NATURE_PALETTE["neutral_dark"], linewidth=0.75, zorder=4)
    ax.vlines(1.0, q25, q75, color=NATURE_PALETTE["neutral_black"], linewidth=1.25, zorder=5)
    ax.scatter([1.0], [median], marker="D", s=20, color=NATURE_PALETTE["neutral_black"], zorder=6)
    ax.set_xlim(0.46, 1.54)


def _size_from_values(values: np.ndarray, min_size: float, max_size: float) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or np.isclose(float(finite.min()), float(finite.max())):
        return np.full_like(values, (min_size + max_size) / 2.0, dtype=float)
    low = float(np.quantile(finite, 0.05))
    high = float(np.quantile(finite, 0.95))
    if np.isclose(low, high):
        scaled = np.full_like(values, 0.5, dtype=float)
    else:
        scaled = (np.clip(values, low, high) - low) / (high - low)
    return min_size + (max_size - min_size) * np.sqrt(np.clip(scaled, 0.0, 1.0))


def _add_size_key(fig: Any, sizes: np.ndarray, values: np.ndarray, label: str = "UE (MWh yr$^{-1}$)") -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    quantiles = np.quantile(finite, [0.10, 0.50, 0.90])
    legend_sizes = _size_from_values(quantiles, float(np.nanmin(sizes)), float(np.nanmax(sizes)))
    handles = [
        fig.axes[0].scatter([], [], s=float(size), color=NATURE_PALETTE["neutral_mid"], alpha=0.55, edgecolor="white", linewidth=0.45)
        for size in legend_sizes
    ]
    fig.legend(
        handles,
        [f"{value:.2g}" for value in quantiles],
        title=label.replace(" (", "\n("),
        loc="lower right",
        bbox_to_anchor=(0.985, 0.16),
        labelspacing=1.0,
        handletextpad=0.9,
    )


def _band_y_axis(source: pd.DataFrame, metric: RobustMetricSpec) -> tuple[list[float], tuple[float, float]]:
    rows = source[source["metric_key"] == metric.key]
    values = rows[["p10", "median", "p90"]].to_numpy(dtype=float).ravel()
    ticks = nice_display_ticks(values, unit_type=metric.unit_type, target_count=4)
    lower = min(min(ticks), float(np.nanmin(values)))
    upper = max(max(ticks), float(np.nanmax(values)))
    pad = max((upper - lower) * 0.06, 0.01 if metric.unit_type == "lolp" else 1e-6)
    if metric.unit_type == "lolp":
        return ticks, (max(0.0, lower - pad), min(1.0, upper + pad))
    return ticks, (lower - pad, upper + pad)


def _apply_ticks(ax: Any, values: Any, axis: str, unit_type: str) -> None:
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


def _plot_tail_interval_panel(
    ax: Any,
    source: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    rows = source[
        (source["source_type"] == "tail_summary")
        & (source["sensitivity_parameter"] != "baseline")
        & (source["tail_metric"] == metric)
    ].copy()
    groups = []
    for parameter in SENSITIVITY_PARAMETERS:
        group = rows[rows["sensitivity_parameter"] == parameter]
        values = group["p95"].dropna().to_numpy(dtype=float)
        if values.size:
            groups.append((PARAMETER_LABELS[parameter], float(np.min(values)), float(np.median(values)), float(np.max(values))))
    x = np.arange(len(groups), dtype=float)
    lows = np.array([item[1] for item in groups], dtype=float)
    medians = np.array([item[2] for item in groups], dtype=float)
    highs = np.array([item[3] for item in groups], dtype=float)
    ax.vlines(x, lows, highs, color=NATURE_PALETTE["blue_light"], linewidth=2.4, alpha=0.85)
    ax.scatter(x, medians, color=NATURE_PALETTE["blue"], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([item[0] for item in groups], rotation=25, ha="right")
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.50)
    if metric == "energy_based_LOLP":
        ax.set_ylim(0.0, min(1.0, max(0.05, float(np.nanmax(highs)) * 1.15 if highs.size else 0.05)))
