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
    save_publication_figure,
    write_source_data,
)
from .display_units import (
    DISPLAY_BATTERY_ENERGY,
    DISPLAY_PV_CAPACITY,
    DISPLAY_SYSTEM_COST,
    add_display_unit_columns,
)


FIGURE_COST_NAME = "LOLP_Zero_Uncertainty_Annualized_Cost_Distribution"
FIGURE_PV_NAME = "LOLP_Zero_Uncertainty_PV_Capacity_Distribution"
FIGURE_BATTERY_NAME = "LOLP_Zero_Uncertainty_Battery_Capacity_Distribution"
X_AXIS_PADDING_FRACTION = 0.18

REQUIRED_COLUMNS = (
    "uncertainty_scenario_id",
    "outage_seed",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "system_total_cost_sgd_per_year",
    "unserved_energy_kwh_per_year",
    "scenario_LOLP",
)


@dataclass(frozen=True)
class DistributionSpec:
    figure_name: str
    metric_key: str
    source_column: str
    display_column: str
    title: str
    xlabel: str
    unit_label: str


def make_lolp_zero_uncertainty_figures(
    out: Path,
    case_key: str,
    all_results: pd.DataFrame,
    x_limits_by_metric: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Path]:
    """Create the three LOLP=0 uncertainty distribution figures."""
    source, specs = build_lolp_zero_uncertainty_figure_source(all_results)
    paths: dict[str, Path] = {}
    for spec in specs:
        base = out / f"{case_key}_{spec.figure_name}"
        metric_source = _metric_source(source, spec)
        paths[f"{spec.metric_key}_source_data"] = write_source_data(base, metric_source)
        x_limits = x_limits_by_metric.get(spec.metric_key) if x_limits_by_metric else None
        paths.update(plot_lolp_zero_distribution(out, case_key, source, spec, x_limits=x_limits))
    return paths


def build_lolp_zero_uncertainty_figure_source(
    all_results: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[DistributionSpec, ...]]:
    """Validate and normalize per-scenario rows for LOLP=0 uncertainty figures."""
    missing = [column for column in REQUIRED_COLUMNS if column not in all_results.columns]
    if missing:
        raise ValueError("LOLP=0 uncertainty figure source is missing required column(s): " + ", ".join(missing))
    if all_results.empty:
        raise ValueError("LOLP=0 uncertainty figure source data is empty.")
    source = all_results.copy()
    numeric = list(REQUIRED_COLUMNS)
    for column in numeric:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source[numeric].isna().any().any():
        raise ValueError("LOLP=0 uncertainty figure source contains missing or non-numeric required values.")
    for column in ("pv_capacity_kw", "battery_energy_kwh", "system_total_cost_sgd_per_year", "unserved_energy_kwh_per_year"):
        if (source[column].astype(float) < -1e-12).any():
            raise ValueError(f"LOLP=0 uncertainty figure source contains negative {column}.")
    if not source["scenario_LOLP"].between(0.0, 1.0).all():
        raise ValueError("scenario_LOLP must be between 0 and 1.")

    if "pv_capacity_mw" not in source.columns:
        source["pv_capacity_mw"] = source["pv_capacity_kw"].astype(float) / 1000.0
    if "battery_energy_mwh" not in source.columns:
        source["battery_energy_mwh"] = source["battery_energy_kwh"].astype(float) / 1000.0
    if "system_total_cost_million_sgd_per_year" not in source.columns:
        source["system_total_cost_million_sgd_per_year"] = source["system_total_cost_sgd_per_year"].astype(float) / 1e6
    source, units = add_display_unit_columns(source)
    specs = (
        DistributionSpec(
            figure_name=FIGURE_COST_NAME,
            metric_key="annualized_cost",
            source_column="system_total_cost_sgd_per_year",
            display_column=DISPLAY_SYSTEM_COST,
            title="Manila LOLP=0 annualized cost",
            xlabel=units.annual_cost_axis_label,
            unit_label=units.cost_plain_label,
        ),
        DistributionSpec(
            figure_name=FIGURE_PV_NAME,
            metric_key="pv_capacity",
            source_column="pv_capacity_kw",
            display_column=DISPLAY_PV_CAPACITY,
            title="Manila LOLP=0 PV capacity",
            xlabel=units.pv_axis_label,
            unit_label=units.pv_unit,
        ),
        DistributionSpec(
            figure_name=FIGURE_BATTERY_NAME,
            metric_key="battery_capacity",
            source_column="battery_energy_kwh",
            display_column=DISPLAY_BATTERY_ENERGY,
            title="Manila LOLP=0 BESS capacity",
            xlabel=units.battery_axis_label,
            unit_label=units.battery_unit,
        ),
    )
    return source.sort_values("uncertainty_scenario_id").reset_index(drop=True), specs


def plot_lolp_zero_distribution(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
    spec: DistributionSpec,
    x_limits: tuple[float, float] | None = None,
) -> dict[str, Path]:
    prepared, _ = build_lolp_zero_uncertainty_figure_source(source)
    values = prepared[spec.display_column].astype(float).to_numpy()
    mpl, plt = matplotlib_context(out / f"{case_key}_{spec.figure_name}")
    apply_nature_style(mpl)
    fig, ax = plt.subplots(figsize=(4.75, 3.15), constrained_layout=False)
    density_ax = ax.twinx()
    _draw_histogram_with_density(ax, density_ax, values, x_limits=x_limits)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    cv = 0.0 if abs(mean) <= 1e-12 and std <= 1e-12 else std / mean if abs(mean) > 1e-12 else np.nan
    ax.set_title(spec.title, loc="left")
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel("Frequency")
    density_ax.set_ylabel("Probability density")
    _annotate_distribution(ax, mean, std, cv, spec.unit_label)
    fig.subplots_adjust(left=0.13, right=0.84, bottom=0.18, top=0.88)
    base = out / f"{case_key}_{spec.figure_name}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def shared_lolp_zero_distribution_x_limits(
    *all_results: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """Return common padded x-axis limits for each LOLP=0 distribution metric."""
    frames = [frame for frame in all_results if frame is not None and not frame.empty]
    if not frames:
        raise ValueError("Shared LOLP=0 distribution x-limits require at least one non-empty result table.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    source, specs = build_lolp_zero_uncertainty_figure_source(combined)
    limits: dict[str, tuple[float, float]] = {}
    for spec in specs:
        limits[spec.metric_key] = _padded_x_limits(source[spec.display_column].astype(float).to_numpy())
    return limits


def _draw_histogram_with_density(
    ax: Any,
    density_ax: Any,
    values: np.ndarray,
    x_limits: tuple[float, float] | None = None,
) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("LOLP=0 uncertainty distribution requires at least one finite value.")
    if finite.size >= 2 and not np.isclose(float(finite.min()), float(finite.max())):
        bins = min(6, max(3, int(np.ceil(np.sqrt(finite.size)))))
        counts, bin_edges, _ = ax.hist(
            finite,
            bins=bins,
            color="#E8E6E6",
            edgecolor=NATURE_PALETTE["neutral_mid"],
            linewidth=0.65,
            zorder=2,
        )
        mean = float(np.mean(finite))
        std = float(np.std(finite, ddof=0))
        x_min, x_max = _validated_x_limits(x_limits) if x_limits is not None else _padded_x_limits(finite)
        x = np.linspace(x_min, x_max, 240)
        density = _normal_pdf(x, mean, std)
        density_ax.plot(x, density, color=NATURE_PALETTE["gold"], linewidth=1.05, zorder=4)
        density_ax.set_ylim(0.0, max(float(density.max()) * 1.18, 1e-9))
        ax.set_ylim(0.0, max(float(np.max(counts)) * 1.22, 1.0))
        ax.set_xlim(x_min, x_max)
    else:
        value = float(finite[0])
        width = max(abs(value) * 0.08, 1.0)
        ax.bar(
            [value],
            [finite.size],
            width=width,
            color="#E8E6E6",
            edgecolor=NATURE_PALETTE["neutral_mid"],
            linewidth=0.65,
            zorder=2,
        )
        if x_limits is not None:
            ax.set_xlim(*_validated_x_limits(x_limits))
        else:
            ax.set_xlim(value - 1.8 * width, value + 1.8 * width)
        ax.set_ylim(0.0, max(float(finite.size) * 1.22, 1.0))
        density_ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.50, zorder=0)
    density_ax.grid(False)
    density_ax.spines["right"].set_visible(True)
    density_ax.spines["right"].set_color(NATURE_PALETTE["neutral_black"])
    density_ax.tick_params(labelsize=6.5, length=2.5, width=0.6)


def _padded_x_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot calculate x-axis limits with no finite values.")
    lower = float(finite.min())
    upper = float(finite.max())
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.10, 1.0)
    else:
        pad = max((upper - lower) * X_AXIS_PADDING_FRACTION, 1e-9)
    padded_lower = lower - pad
    if lower >= 0.0 and padded_lower < 0.0:
        padded_lower = 0.0
    return padded_lower, upper + pad


def _validated_x_limits(limits: tuple[float, float] | None) -> tuple[float, float]:
    if limits is None or len(limits) != 2:
        raise ValueError("x_limits must contain exactly two values.")
    lower, upper = float(limits[0]), float(limits[1])
    if not (np.isfinite(lower) and np.isfinite(upper)) or lower >= upper:
        raise ValueError("x_limits must be finite and strictly increasing.")
    return lower, upper


def _normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std <= 0.0 or not np.isfinite(std):
        return np.zeros_like(x, dtype=float)
    z = (x - float(mean)) / float(std)
    return np.exp(-0.5 * z * z) / (float(std) * np.sqrt(2.0 * np.pi))


def _annotate_distribution(ax: Any, mean: float, std: float, cv: float, unit_label: str) -> None:
    cv_text = "undefined" if not np.isfinite(cv) else f"{cv * 100.0:.1f}%"
    ax.text(
        0.72,
        0.92,
        f"mu={mean:.2f} {unit_label}\nsigma={std:.2f} {unit_label}\nCV={cv_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=NATURE_PALETTE["neutral_black"],
    )


def _metric_source(source: pd.DataFrame, spec: DistributionSpec) -> pd.DataFrame:
    values = source[spec.display_column].astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    cv = 0.0 if abs(mean) <= 1e-12 and std <= 1e-12 else std / mean if abs(mean) > 1e-12 else np.nan
    return pd.DataFrame(
        {
            "uncertainty_scenario_id": source["uncertainty_scenario_id"].astype(int),
            "outage_seed": source["outage_seed"].astype(int),
            "metric_key": spec.metric_key,
            "metric_title": spec.title,
            "raw_source_column": spec.source_column,
            "raw_value": source[spec.source_column].astype(float),
            "display_value": values,
            "display_unit": spec.unit_label,
            "mean_display_value": mean,
            "std_display_value": std,
            "coefficient_of_variation": cv,
            "coefficient_of_variation_percent": cv * 100.0 if np.isfinite(cv) else np.nan,
        }
    )
