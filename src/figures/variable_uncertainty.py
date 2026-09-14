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
    plot_smooth_line,
    save_publication_figure,
    write_source_data,
)


FIGURE_PREFIX = "Variable_Uncertainty_Distribution"
DEFAULT_BINS = 12
REQUIRED_COLUMNS = (
    "uncertainty_scenario_id",
    "pv_bess_system_annual_cost_sgd",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "unserved_energy_kwh_per_year",
    "scenario_LOLP",
)


@dataclass(frozen=True)
class HistogramCdfSpec:
    metric_key: str
    source_column: str
    figure_suffix: str
    title: str
    xlabel: str
    unit: str


VARIABLE_UNCERTAINTY_HISTOGRAM_SPECS = (
    HistogramCdfSpec(
        metric_key="system_total_cost",
        source_column="pv_bess_system_annual_cost_sgd",
        figure_suffix="Total_Cost",
        title="Total cost distribution",
        xlabel="Total cost (SGD/year)",
        unit="SGD/year",
    ),
    HistogramCdfSpec(
        metric_key="pv_capacity",
        source_column="pv_capacity_kw",
        figure_suffix="PV_Capacity",
        title="Optimal PV capacity distribution",
        xlabel="Optimal PV capacity (kW)",
        unit="kW",
    ),
    HistogramCdfSpec(
        metric_key="battery_capacity",
        source_column="battery_energy_kwh",
        figure_suffix="Battery_Capacity",
        title="Optimal battery capacity distribution",
        xlabel="Optimal battery capacity (kWh)",
        unit="kWh",
    ),
    HistogramCdfSpec(
        metric_key="unserved_energy",
        source_column="unserved_energy_kwh_per_year",
        figure_suffix="Unserved_Energy",
        title="Unserved energy distribution",
        xlabel="Unserved energy (kWh/year)",
        unit="kWh/year",
    ),
    HistogramCdfSpec(
        metric_key="scenario_lolp",
        source_column="scenario_LOLP",
        figure_suffix="LOLP",
        title="LOLP distribution",
        xlabel="LOLP (fraction)",
        unit="fraction",
    ),
)


def make_variable_uncertainty_distribution_figures(
    out: Path,
    case_key: str,
    scenario_optima: pd.DataFrame,
    selected_designs: pd.DataFrame | None = None,
    bins: int = DEFAULT_BINS,
) -> dict[str, Path]:
    """Create five histogram + CDF figures and one combined panel."""
    source, references = build_variable_uncertainty_histogram_source(
        scenario_optima,
        selected_designs=selected_designs,
        bins=bins,
    )
    paths: dict[str, Path] = {}
    for spec in VARIABLE_UNCERTAINTY_HISTOGRAM_SPECS:
        metric_source = source[source["metric_key"].astype(str) == spec.metric_key].reset_index(drop=True)
        base = out / f"{case_key}_{FIGURE_PREFIX}_{spec.figure_suffix}"
        paths[f"variable_uncertainty_distribution_{spec.metric_key}_source_data"] = write_source_data(
            base,
            metric_source,
        )
        paths.update(plot_variable_uncertainty_histogram_cdf(out, case_key, metric_source, spec))

    combined_base = out / f"{case_key}_{FIGURE_PREFIX}_All_Metrics"
    paths["variable_uncertainty_distribution_all_metrics_source_data"] = write_source_data(combined_base, source)
    paths.update(plot_variable_uncertainty_histogram_cdf_panel(out, case_key, source, references))
    return paths


def build_variable_uncertainty_histogram_source(
    scenario_optima: pd.DataFrame,
    selected_designs: pd.DataFrame | None = None,
    bins: int = DEFAULT_BINS,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return binned frequency/CDF source data for the five requested metrics."""
    if int(bins) <= 0:
        raise ValueError("Variable uncertainty histogram bins must be positive.")
    prepared = _validate_variable_uncertainty_optima(scenario_optima)
    references = _reference_values(selected_designs)
    rows: list[dict[str, Any]] = []
    for spec in VARIABLE_UNCERTAINTY_HISTOGRAM_SPECS:
        values = prepared[spec.source_column].astype(float).to_numpy()
        counts, edges = _histogram(values, bins=int(bins), metric_key=spec.metric_key)
        frequency = counts.astype(float) / float(values.size)
        cdf = np.cumsum(frequency)
        cdf_start = np.concatenate(([0.0], cdf[:-1]))
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = edges[1:] - edges[:-1]
        reference = references.get(spec.metric_key, np.nan)
        for index, (left, right, center, width, count, freq, cdf0, cdf1) in enumerate(
            zip(edges[:-1], edges[1:], centers, widths, counts, frequency, cdf_start, cdf, strict=True),
            start=1,
        ):
            rows.append(
                {
                    "metric_key": spec.metric_key,
                    "metric_title": spec.title,
                    "source_column": spec.source_column,
                    "unit": spec.unit,
                    "bin_index": int(index),
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "bin_center": float(center),
                    "bin_width": float(width),
                    "count": int(count),
                    "frequency": float(freq),
                    "cdf_start": float(cdf0),
                    "cdf": float(cdf1),
                    "cdf_end": float(cdf1),
                    "sample_size": int(values.size),
                    "reference_value": reference,
                }
            )
    source = pd.DataFrame(rows)
    return source, references


def plot_variable_uncertainty_histogram_cdf(
    out: Path,
    case_key: str,
    metric_source: pd.DataFrame,
    spec: HistogramCdfSpec,
) -> dict[str, Path]:
    source = _validate_histogram_source(metric_source, expected_metric=spec.metric_key)
    mpl, plt = matplotlib_context(out / f"{case_key}_{FIGURE_PREFIX}_{spec.figure_suffix}")
    apply_nature_style(mpl)
    fig, ax = plt.subplots(figsize=(5.2, 3.45), constrained_layout=False)
    cdf_ax = ax.twinx()
    _draw_histogram_cdf_axes(ax, cdf_ax, source, spec, show_ylabel=True, show_legend=True)
    fig.subplots_adjust(left=0.14, right=0.84, bottom=0.18, top=0.86)
    base = out / f"{case_key}_{FIGURE_PREFIX}_{spec.figure_suffix}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_variable_uncertainty_histogram_cdf_panel(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
    references: dict[str, float] | None = None,
) -> dict[str, Path]:
    del references
    validated = _validate_histogram_source(source)
    mpl, plt = matplotlib_context(out / f"{case_key}_{FIGURE_PREFIX}_All_Metrics")
    apply_nature_style(mpl)
    fig, axes = plt.subplots(3, 2, figsize=(8.1, 8.2), constrained_layout=False)
    flat_axes = axes.flatten()
    for index, spec in enumerate(VARIABLE_UNCERTAINTY_HISTOGRAM_SPECS):
        ax = flat_axes[index]
        cdf_ax = ax.twinx()
        metric_source = validated[validated["metric_key"].astype(str) == spec.metric_key].reset_index(drop=True)
        _draw_histogram_cdf_axes(ax, cdf_ax, metric_source, spec, show_ylabel=True, show_legend=index == 0)
        ax.set_title(f"{chr(97 + index)}  {spec.title}", loc="left")
    flat_axes[-1].axis("off")
    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.075, top=0.94, wspace=0.46, hspace=0.56)
    base = out / f"{case_key}_{FIGURE_PREFIX}_All_Metrics"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _draw_histogram_cdf_axes(
    ax: Any,
    cdf_ax: Any,
    source: pd.DataFrame,
    spec: HistogramCdfSpec,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    lefts = source["bin_left"].astype(float).to_numpy()
    frequency = source["frequency"].astype(float).to_numpy()
    widths = source["bin_width"].astype(float).to_numpy()
    bars = ax.bar(
        lefts,
        frequency,
        width=widths,
        align="edge",
        color=NATURE_PALETTE["blue_light"],
        edgecolor=NATURE_PALETTE["blue_light"],
        linewidth=0.0,
        alpha=0.88,
        label="Frequency",
        zorder=3,
    )
    cdf_line = _plot_smooth_cdf(
        cdf_ax,
        source,
        color=NATURE_PALETTE["red"],
        linewidth=1.55,
        solid_capstyle="round",
        label="CDF",
        zorder=4,
    )
    _draw_reference_case(ax, cdf_ax, source)
    ax.set_title(spec.title, loc="left")
    ax.set_xlabel(spec.xlabel)
    if show_ylabel:
        ax.set_ylabel("Frequency")
        cdf_ax.set_ylabel("CDF")
    ax.set_ylim(0.0, max(0.05, float(np.nanmax(frequency)) * 1.25))
    cdf_ax.set_ylim(0.0, 1.05)
    ax.set_xlim(*_x_limits_from_source(source, metric_key=spec.metric_key))
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.50, zorder=0)
    cdf_ax.grid(False)
    cdf_ax.spines["right"].set_visible(True)
    cdf_ax.spines["right"].set_color(NATURE_PALETTE["neutral_black"])
    if show_legend:
        ax.figure.legend(
            [bars, cdf_line],
            ["Frequency", "CDF"],
            loc="upper center",
            bbox_to_anchor=(0.50, 0.985),
            ncol=2,
        )


def _draw_reference_case(ax: Any, cdf_ax: Any, source: pd.DataFrame) -> None:
    reference_values = pd.to_numeric(source["reference_value"], errors="coerce").dropna()
    if reference_values.empty:
        return
    reference = float(reference_values.iloc[0])
    if not np.isfinite(reference):
        return
    cdf_x, cdf_y = _cdf_anchor_points(source)
    cdf_at_reference = float(np.interp(reference, cdf_x, cdf_y, left=0.0, right=1.0))
    reference_color = NATURE_PALETTE["gold"]
    ax.axvline(reference, color=reference_color, linestyle=(0, (4, 3)), linewidth=0.95, zorder=5)
    cdf_ax.axhline(cdf_at_reference, color=reference_color, linestyle=(0, (4, 3)), linewidth=0.75, alpha=0.82, zorder=2)
    ax.annotate(
        "Reference case",
        xy=(reference, 0.82),
        xycoords=("data", "axes fraction"),
        xytext=(0.70, 0.66),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": reference_color, "linewidth": 0.75},
        ha="left",
        va="center",
        fontsize=6.2,
        color=NATURE_PALETTE["neutral_black"],
    )


def _validate_variable_uncertainty_optima(scenario_optima: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in scenario_optima.columns]
    if missing:
        raise ValueError(
            "Variable uncertainty histogram source is missing required column(s): " + ", ".join(missing)
        )
    if scenario_optima.empty:
        raise ValueError("Variable uncertainty histogram source data is empty.")
    prepared = scenario_optima.copy()
    if "available" in prepared.columns:
        prepared = prepared[
            prepared["available"].map(lambda value: False if pd.isna(value) else str(value).lower() in {"true", "1"})
        ].copy()
    if prepared.empty:
        raise ValueError("Variable uncertainty histogram source has no available scenarios.")
    for column in REQUIRED_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if prepared[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Variable uncertainty histogram source contains missing or non-numeric required values.")
    for column in (
        "pv_bess_system_annual_cost_sgd",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "unserved_energy_kwh_per_year",
    ):
        if (prepared[column].astype(float) < -1e-12).any():
            raise ValueError(f"Variable uncertainty histogram source contains negative {column}.")
    if not prepared["scenario_LOLP"].between(0.0, 1.0).all():
        raise ValueError("scenario_LOLP must be between 0 and 1.")
    if "outage_seed" not in prepared.columns:
        prepared["outage_seed"] = pd.NA
    return prepared.sort_values("uncertainty_scenario_id").reset_index(drop=True)


def _validate_histogram_source(source: pd.DataFrame, expected_metric: str | None = None) -> pd.DataFrame:
    required = (
        "metric_key",
        "metric_title",
        "source_column",
        "unit",
        "bin_index",
        "bin_left",
        "bin_right",
        "bin_center",
        "bin_width",
        "count",
        "frequency",
        "cdf",
        "cdf_start",
        "cdf_end",
        "sample_size",
        "reference_value",
    )
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError("Variable uncertainty histogram source data is missing column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Variable uncertainty histogram source data is empty.")
    validated = source.copy()
    if expected_metric is not None:
        validated = validated[validated["metric_key"].astype(str) == expected_metric].copy()
        if validated.empty:
            raise ValueError(f"Variable uncertainty histogram source has no rows for {expected_metric}.")
    numeric = (
        "bin_index",
        "bin_left",
        "bin_right",
        "bin_center",
        "bin_width",
        "count",
        "frequency",
        "cdf",
        "cdf_start",
        "cdf_end",
        "sample_size",
    )
    for column in numeric:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(numeric)].isna().any().any():
        raise ValueError("Variable uncertainty histogram source data contains non-numeric bin values.")
    for metric, group in validated.groupby("metric_key", sort=False):
        del metric
        if not np.isclose(float(group["cdf_start"].astype(float).iloc[0]), 0.0):
            raise ValueError("Variable uncertainty histogram CDF must start at 0.")
        if not group["cdf"].astype(float).is_monotonic_increasing:
            raise ValueError("Variable uncertainty histogram CDF must be monotonic increasing.")
        if not np.isclose(float(group["cdf"].astype(float).iloc[-1]), 1.0):
            raise ValueError("Variable uncertainty histogram CDF must end at 1.")
        if not np.isclose(float(group["frequency"].astype(float).sum()), 1.0):
            raise ValueError("Variable uncertainty histogram frequency must sum to 1.")
    return validated.sort_values(["metric_key", "bin_index"]).reset_index(drop=True)


def _cdf_anchor_points(source: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = source.sort_values("bin_index")
    x = np.concatenate(
        (
            [float(ordered["bin_left"].astype(float).iloc[0])],
            ordered["bin_right"].astype(float).to_numpy(),
        )
    )
    y = np.concatenate(([0.0], ordered["cdf"].astype(float).to_numpy()))
    return x, y


def _cdf_smooth_points(source: pd.DataFrame, samples: int = 240) -> tuple[np.ndarray, np.ndarray]:
    x, y = _cdf_anchor_points(source)
    if x.size < 3:
        return x, np.clip(y, 0.0, 1.0)
    from scipy.interpolate import PchipInterpolator

    interpolator = PchipInterpolator(x, y)
    x_smooth = np.linspace(float(x.min()), float(x.max()), max(int(samples), x.size * 4))
    y_smooth = np.clip(interpolator(x_smooth), 0.0, 1.0)
    y_smooth[0] = 0.0
    y_smooth[-1] = 1.0
    return x_smooth, y_smooth


def _plot_smooth_cdf(cdf_ax: Any, source: pd.DataFrame, **plot_kwargs: Any) -> Any:
    cdf_x, cdf_y = _cdf_anchor_points(source)
    cdf_line = plot_smooth_line(cdf_ax, cdf_x, cdf_y, samples=240, **plot_kwargs)
    cdf_line.set_ydata(np.clip(cdf_line.get_ydata(), 0.0, 1.0))
    ydata = cdf_line.get_ydata()
    if len(ydata):
        ydata[0] = 0.0
        ydata[-1] = 1.0
        cdf_line.set_ydata(ydata)
    return cdf_line


def _x_limits_from_source(source: pd.DataFrame, metric_key: str) -> tuple[float, float]:
    lower = float(source["bin_left"].astype(float).min())
    upper = float(source["bin_right"].astype(float).max())
    reference_values = pd.to_numeric(source["reference_value"], errors="coerce").dropna()
    if not reference_values.empty:
        reference = float(reference_values.iloc[0])
        if np.isfinite(reference):
            lower = min(lower, reference)
            upper = max(upper, reference)
    if metric_key == "scenario_lolp":
        return 0.0, 1.0
    span = max(upper - lower, 1e-9)
    pad = max(span * 0.06, 1e-9)
    padded_lower = lower - pad
    if lower >= 0.0 and padded_lower < 0.0:
        padded_lower = 0.0
    return padded_lower, upper + pad


def _histogram(values: np.ndarray, bins: int, metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Variable uncertainty histogram requires at least one finite value.")
    if np.isclose(float(finite.min()), float(finite.max())):
        value = float(finite[0])
        pad = 0.05 if metric_key == "scenario_lolp" else max(abs(value) * 0.08, 1.0)
        edges = np.linspace(value - pad, value + pad, min(max(int(bins), 1), 3) + 1)
        if metric_key == "scenario_lolp":
            edges = np.clip(edges, 0.0, 1.0)
            edges = np.unique(edges)
            if edges.size < 2:
                edges = np.array([0.0, 1.0])
        counts, resolved_edges = np.histogram(finite, bins=edges)
        return counts, resolved_edges
    if metric_key == "scenario_lolp":
        counts, edges = np.histogram(finite, bins=int(bins), range=(0.0, 1.0))
    else:
        counts, edges = np.histogram(finite, bins=int(bins))
    return counts, edges


def _reference_values(selected_designs: pd.DataFrame | None) -> dict[str, float]:
    if selected_designs is None or selected_designs.empty or "design_type" not in selected_designs.columns:
        return {}
    matches = selected_designs[selected_designs["design_type"].astype(str) == "minimum_total_cost_design"]
    if matches.empty:
        return {}
    row = matches.iloc[0]
    mapping = {
        "system_total_cost": "system_total_cost_sgd_per_year",
        "pv_capacity": "pv_capacity_kw",
        "battery_capacity": "battery_energy_kwh",
        "unserved_energy": "mean_UE_kwh_per_year",
        "scenario_lolp": "mean_scenario_LOLP",
    }
    references: dict[str, float] = {}
    for metric, column in mapping.items():
        if column in row.index and pd.notna(row[column]):
            references[metric] = float(row[column])
    return references
