from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    matplotlib_context,
    save_publication_figure,
    write_source_data as _write_source_data,
)


FIGURE_DISTRIBUTION_NAME = "Uncertainty_Baseline_Distribution_Violins"
FIGURE_CAPACITY_BUBBLE_NAME = "Uncertainty_Baseline_Capacity_Cost_Bubble"
REQUIRED_COLUMNS = (
    "uncertainty_scenario_id",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "pv_bess_system_annual_cost_sgd",
    "unserved_energy_kwh_per_year",
    "outage_weighted_load_kwh",
)
SOURCE_COLUMNS = (
    "uncertainty_scenario_id",
    "outage_seed",
    "pv_capacity_kw",
    "pv_capacity_mw",
    "battery_energy_kwh",
    "battery_energy_mwh",
    "pv_bess_system_annual_cost_sgd",
    "system_total_cost_million_sgd_per_year",
    "unserved_energy_kwh_per_year",
    "UE_mwh_per_year",
    "outage_weighted_load_kwh",
    "scenario_LOLP",
    "bubble_cost_normalized",
    "bubble_size_pt2",
)


def build_uncertainty_baseline_figure_source(all_results: pd.DataFrame) -> pd.DataFrame:
    """Return normalized per-scenario rows for uncertainty baseline figures."""
    missing = [column for column in REQUIRED_COLUMNS if column not in all_results.columns]
    if missing:
        raise ValueError("Uncertainty baseline figure source is missing required column(s): " + ", ".join(missing))
    if all_results.empty:
        raise ValueError("Uncertainty baseline figure source data is empty.")

    selected = all_results.copy()

    for column in (
        "uncertainty_scenario_id",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "pv_bess_system_annual_cost_sgd",
        "unserved_energy_kwh_per_year",
        "outage_weighted_load_kwh",
    ):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if selected[
        [
            "uncertainty_scenario_id",
            "pv_capacity_kw",
            "battery_energy_kwh",
            "pv_bess_system_annual_cost_sgd",
            "unserved_energy_kwh_per_year",
            "outage_weighted_load_kwh",
        ]
    ].isna().any().any():
        raise ValueError("Uncertainty baseline figure source contains missing or non-numeric required values.")
    for column in ("pv_capacity_kw", "battery_energy_kwh", "pv_bess_system_annual_cost_sgd", "unserved_energy_kwh_per_year", "outage_weighted_load_kwh"):
        if (selected[column].astype(float) < 0.0).any():
            raise ValueError(f"Uncertainty baseline figure source contains negative {column}.")

    if "outage_seed" not in selected.columns:
        selected["outage_seed"] = pd.NA

    outage_load = selected["outage_weighted_load_kwh"].astype(float).to_numpy()
    ue = selected["unserved_energy_kwh_per_year"].astype(float).to_numpy()
    scenario_lolp = np.divide(ue, outage_load, out=np.zeros_like(ue, dtype=float), where=outage_load > 0.0)
    if np.any(scenario_lolp < -1e-12):
        raise ValueError("Derived scenario_LOLP contains negative values.")
    selected["scenario_LOLP"] = np.clip(scenario_lolp, 0.0, 1.0)
    selected["pv_capacity_mw"] = selected["pv_capacity_kw"].astype(float) / 1000.0
    selected["battery_energy_mwh"] = selected["battery_energy_kwh"].astype(float) / 1000.0
    selected["system_total_cost_million_sgd_per_year"] = (
        selected["pv_bess_system_annual_cost_sgd"].astype(float) / 1e6
    )
    selected["UE_mwh_per_year"] = selected["unserved_energy_kwh_per_year"].astype(float) / 1000.0

    selected = _add_bubble_cost_scaling(selected)
    columns = [column for column in SOURCE_COLUMNS if column in selected.columns]
    return selected[columns].sort_values("uncertainty_scenario_id").reset_index(drop=True)


def make_uncertainty_baseline_figures(
    out: Path,
    case_key: str,
    all_results: pd.DataFrame,
) -> dict[str, Path]:
    """Create all uncertainty baseline figures and matching source-data CSV files."""
    source = build_uncertainty_baseline_figure_source(all_results)
    paths: dict[str, Path] = {}
    distribution_base = out / f"{case_key}_{FIGURE_DISTRIBUTION_NAME}"
    bubble_base = out / f"{case_key}_{FIGURE_CAPACITY_BUBBLE_NAME}"
    paths["uncertainty_baseline_distribution_source_data"] = _write_source_data(distribution_base, source)
    paths["uncertainty_baseline_capacity_bubble_source_data"] = _write_source_data(bubble_base, source)
    paths.update(plot_uncertainty_baseline_distribution_violins(out, case_key, source))
    paths.update(plot_uncertainty_baseline_capacity_cost_bubble(out, case_key, source))
    return paths


def plot_uncertainty_baseline_distribution_violins(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
) -> dict[str, Path]:
    source = validate_uncertainty_baseline_figure_source(source)
    mpl, plt = matplotlib_context(out / f"{case_key}_{FIGURE_DISTRIBUTION_NAME}")
    apply_nature_style(mpl)
    fig, axes = plt.subplots(1, 5, figsize=(7.6, 2.45), constrained_layout=False)
    panels = [
        ("Cost", "system_total_cost_million_sgd_per_year", "Million SGD/year", NATURE_PALETTE["blue"]),
        ("PV", "pv_capacity_mw", "MW", NATURE_PALETTE["teal"]),
        ("BESS", "battery_energy_mwh", "MWh", NATURE_PALETTE["purple"]),
        ("UE", "UE_mwh_per_year", "MWh/year", NATURE_PALETTE["red"]),
        ("LOLP", "scenario_LOLP", "Fraction", NATURE_PALETTE["gold"]),
    ]
    rng = np.random.default_rng(20260714)
    for index, (ax, (title, column, ylabel, color)) in enumerate(zip(axes, panels, strict=True), start=1):
        _draw_single_violin_panel(ax, source[column].to_numpy(dtype=float), title, ylabel, color, rng)
        ax.set_title(f"{chr(96 + index)}  {title}", loc="left")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.24, top=0.84, wspace=0.50)
    base = out / f"{case_key}_{FIGURE_DISTRIBUTION_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_uncertainty_baseline_capacity_cost_bubble(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
) -> dict[str, Path]:
    source = validate_uncertainty_baseline_figure_source(source)
    mpl, plt = matplotlib_context(out / f"{case_key}_{FIGURE_CAPACITY_BUBBLE_NAME}")
    apply_nature_style(mpl)
    fig, ax = plt.subplots(figsize=(4.7, 3.55), constrained_layout=False)
    ax.scatter(
        source["battery_energy_mwh"],
        source["pv_capacity_mw"],
        s=source["bubble_size_pt2"],
        color=NATURE_PALETTE["blue"],
        edgecolor="white",
        linewidth=0.45,
        alpha=0.68,
        zorder=3,
    )
    ax.grid(color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.45, zorder=0)
    ax.set_title("Scenario-level optimal capacity and cost", loc="left")
    ax.set_xlabel("BESS energy capacity (MWh)")
    ax.set_ylabel("PV capacity (MW)")
    _add_bubble_size_legend(ax, source)
    fig.subplots_adjust(left=0.15, right=0.78, bottom=0.16, top=0.90)
    base = out / f"{case_key}_{FIGURE_CAPACITY_BUBBLE_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def validate_uncertainty_baseline_figure_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SOURCE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Uncertainty baseline figure source is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Uncertainty baseline figure source data is empty.")
    validated = source.copy()
    numeric = [
        "uncertainty_scenario_id",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "pv_bess_system_annual_cost_sgd",
        "system_total_cost_million_sgd_per_year",
        "unserved_energy_kwh_per_year",
        "UE_mwh_per_year",
        "outage_weighted_load_kwh",
        "scenario_LOLP",
        "bubble_cost_normalized",
        "bubble_size_pt2",
    ]
    for column in numeric:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[numeric].isna().any().any():
        raise ValueError("Uncertainty baseline figure source contains missing or non-numeric required values.")
    if not validated["scenario_LOLP"].between(0.0, 1.0).all():
        raise ValueError("scenario_LOLP must be between 0 and 1.")
    if (validated["bubble_size_pt2"] <= 0.0).any():
        raise ValueError("bubble_size_pt2 must be positive.")
    return validated.sort_values("uncertainty_scenario_id").reset_index(drop=True)


def _add_bubble_cost_scaling(source: pd.DataFrame) -> pd.DataFrame:
    scaled = source.copy()
    costs = scaled["system_total_cost_million_sgd_per_year"].to_numpy(dtype=float)
    low = float(np.quantile(costs, 0.05))
    high = float(np.quantile(costs, 0.95))
    if np.isclose(low, high):
        normalized = np.full_like(costs, 0.5, dtype=float)
    else:
        clipped = np.clip(costs, low, high)
        normalized = (clipped - low) / (high - low)
    scaled["bubble_cost_normalized"] = normalized
    scaled["bubble_size_pt2"] = 28.0 + 220.0 * np.sqrt(np.clip(normalized, 0.0, 1.0))
    return scaled


def _draw_single_violin_panel(
    ax: Any,
    values: np.ndarray,
    title: str,
    ylabel: str,
    color: str,
    rng: np.random.Generator,
) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"No finite values available for {title}.")
    if finite.size >= 2 and not np.isclose(float(finite.min()), float(finite.max())):
        violins = ax.violinplot(
            [finite],
            positions=[1.0],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body = violins["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor(NATURE_PALETTE["neutral_black"])
        body.set_alpha(0.34)
        body.set_linewidth(0.65)
    else:
        ax.hlines(float(finite[0]), 0.68, 1.32, color=color, linewidth=3.0, alpha=0.45, zorder=2)
    x = 1.0 + rng.normal(0.0, 0.035, finite.size)
    ax.scatter(x, finite, s=7.0, color=color, alpha=0.44, linewidth=0.0, zorder=3)
    q25, median, q75 = np.quantile(finite, [0.25, 0.50, 0.75])
    ax.vlines(1.0, q25, q75, color=NATURE_PALETTE["neutral_black"], linewidth=1.2, zorder=5)
    ax.scatter([1.0], [median], marker="D", s=20, color=NATURE_PALETTE["neutral_black"], zorder=6)
    ax.set_xlim(0.46, 1.54)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.42)
    if title == "LOLP":
        ax.set_ylim(max(0.0, min(0.0, float(finite.min()))), min(1.0, max(0.05, float(finite.max()) * 1.12)))


def _add_bubble_size_legend(ax: Any, source: pd.DataFrame) -> None:
    legend_costs = np.quantile(source["system_total_cost_million_sgd_per_year"].to_numpy(dtype=float), [0.10, 0.50, 0.90])
    low = float(np.quantile(source["system_total_cost_million_sgd_per_year"].to_numpy(dtype=float), 0.05))
    high = float(np.quantile(source["system_total_cost_million_sgd_per_year"].to_numpy(dtype=float), 0.95))
    if np.isclose(low, high):
        legend_sizes = np.full_like(legend_costs, 28.0 + 220.0 * np.sqrt(0.5), dtype=float)
    else:
        normalized = (np.clip(legend_costs, low, high) - low) / (high - low)
        legend_sizes = 28.0 + 220.0 * np.sqrt(np.clip(normalized, 0.0, 1.0))
    handles = [
        ax.scatter(
            [],
            [],
            s=float(size),
            color=NATURE_PALETTE["blue"],
            alpha=0.68,
            edgecolor="white",
            linewidth=0.45,
        )
        for size in legend_sizes
    ]
    labels = [f"P{percent}  {value:.2f}" for percent, value in zip((10, 50, 90), legend_costs, strict=True)]
    legend = ax.legend(
        handles,
        labels,
        title="Total cost\n(M SGD/yr)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        labelspacing=1.0,
        handletextpad=1.0,
    )
    legend.get_title().set_fontsize(6.5)
