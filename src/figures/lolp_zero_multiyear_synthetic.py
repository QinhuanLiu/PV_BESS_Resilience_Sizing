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
from .lolp_zero_uncertainty import _normal_pdf, _padded_x_limits


SYNTHETIC_MULTYEAR_CASE_KEY = "Manila_SingleFamily"
SYNTHETIC_MULTYEAR_STEM = "Manila_SingleFamily_LOLP_Zero_Multiyear_Synthetic_500"
SYNTHETIC_MULTYEAR_SCENARIO_COUNT = 500
SYNTHETIC_MULTYEAR_SIMULATION_YEARS = (1, 3, 5, 10)
SYNTHETIC_MULTYEAR_CV_TREND_YEARS = tuple(range(1, 11))
SYNTHETIC_MULTYEAR_RANDOM_SEED = 20260825


@dataclass(frozen=True)
class SyntheticMetricSpec:
    metric_key: str
    value_column: str
    figure_name: str
    title: str
    xlabel: str
    unit_label: str


@dataclass(frozen=True)
class SyntheticYearParameters:
    years: int
    cost_mean_ksgd_per_year: float
    cost_sd_ksgd_per_year: float
    pv_mean_kw: float
    pv_sd_kw: float
    battery_mean_kwh: float
    battery_sd_kwh: float


SYNTHETIC_YEAR_PARAMETERS = (
    SyntheticYearParameters(1, 1.42, 0.047, 4.07, 0.22, 3.00, 1.30),
    SyntheticYearParameters(3, 1.46, 0.034, 4.04, 0.14, 4.10, 0.92),
    SyntheticYearParameters(5, 1.47, 0.020, 4.03, 0.09, 4.40, 0.58),
    SyntheticYearParameters(10, 1.48, 0.012, 4.02, 0.05, 4.60, 0.35),
)


SYNTHETIC_METRIC_SPECS = (
    SyntheticMetricSpec(
        metric_key="annualized_cost",
        value_column="system_total_cost_ksgd_per_year",
        figure_name="Annualized_Cost_Synthetic_Multiyear_Distribution_Grid",
        title="Annualized cost",
        xlabel="Annualized cost (kSGD yr$^{-1}$)",
        unit_label="kSGD/yr",
    ),
    SyntheticMetricSpec(
        metric_key="pv_capacity",
        value_column="pv_capacity_kw",
        figure_name="PV_Capacity_Synthetic_Multiyear_Distribution_Grid",
        title="PV capacity",
        xlabel="PV capacity (kW)",
        unit_label="kW",
    ),
    SyntheticMetricSpec(
        metric_key="battery_capacity",
        value_column="battery_energy_kwh",
        figure_name="Battery_Capacity_Synthetic_Multiyear_Distribution_Grid",
        title="Battery capacity",
        xlabel="BESS energy (kWh)",
        unit_label="kWh",
    ),
)


def write_lolp_zero_multiyear_synthetic_outputs(
    out: Path,
    case_key: str = SYNTHETIC_MULTYEAR_CASE_KEY,
    scenario_count: int = SYNTHETIC_MULTYEAR_SCENARIO_COUNT,
    seed: int = SYNTHETIC_MULTYEAR_RANDOM_SEED,
) -> dict[str, Path]:
    """Generate deterministic synthetic Manila multi-year LOLP=0 data and figures."""
    out.mkdir(parents=True, exist_ok=True)
    results = build_lolp_zero_multiyear_synthetic_results(scenario_count=int(scenario_count), seed=int(seed))
    cv_summary = summarize_lolp_zero_multiyear_synthetic_results(results)
    cv_trend = build_lolp_zero_multiyear_cv_trend(cv_summary)
    paths: dict[str, Path] = {
        "synthetic_all_results": out / f"{case_key}_LOLP_Zero_Multiyear_Synthetic_500_All_Results.csv",
        "synthetic_cv_summary": out / f"{case_key}_LOLP_Zero_Multiyear_Synthetic_500_CV_Summary.csv",
        "synthetic_cv_trend": out / f"{case_key}_LOLP_Zero_Multiyear_Synthetic_500_CV_Trend_All_Years.csv",
    }
    results.to_csv(paths["synthetic_all_results"], index=False)
    cv_summary.to_csv(paths["synthetic_cv_summary"], index=False)
    cv_trend.to_csv(paths["synthetic_cv_trend"], index=False)
    paths.update(make_lolp_zero_multiyear_synthetic_figures(out, case_key, results))
    paths.update(make_lolp_zero_multiyear_cv_trend_figures(out, case_key, cv_trend))
    return paths


def build_lolp_zero_multiyear_synthetic_results(
    scenario_count: int = SYNTHETIC_MULTYEAR_SCENARIO_COUNT,
    seed: int = SYNTHETIC_MULTYEAR_RANDOM_SEED,
) -> pd.DataFrame:
    """Return a reproducible illustrative data set anchored to the Figure 1/2 results."""
    if int(scenario_count) <= 0:
        raise ValueError("Synthetic scenario_count must be positive.")
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for params in SYNTHETIC_YEAR_PARAMETERS:
        cost = _sample_truncated_normal(
            rng,
            mean=params.cost_mean_ksgd_per_year,
            sd=params.cost_sd_ksgd_per_year,
            size=int(scenario_count),
            lower=0.0,
        )
        pv = _sample_truncated_normal(
            rng,
            mean=params.pv_mean_kw,
            sd=params.pv_sd_kw,
            size=int(scenario_count),
            lower=0.0,
        )
        battery = _sample_truncated_normal(
            rng,
            mean=params.battery_mean_kwh,
            sd=params.battery_sd_kwh,
            size=int(scenario_count),
            lower=0.0,
        )
        pv = _round_to_step(pv, 0.1)
        battery = _round_to_step(battery, 0.5)
        cost = np.round(cost, 6)
        for index in range(int(scenario_count)):
            scenario_id = index + 1
            cost_ksgd = float(cost[index])
            pv_kw = float(pv[index])
            battery_kwh = float(battery[index])
            rows.append(
                {
                    "case_key": SYNTHETIC_MULTYEAR_CASE_KEY,
                    "city_key": "Manila",
                    "building_key": "SingleFamily",
                    "simulation_years": int(params.years),
                    "simulation_period_label": _period_label(params.years),
                    "synthetic_scenario_id": int(scenario_id),
                    "outage_seed": int(seed + params.years * 10_000 + scenario_id),
                    "sample_type": "synthetic_reasonable_constructed_distribution",
                    "system_total_cost_ksgd_per_year": cost_ksgd,
                    "system_total_cost_sgd_per_year": cost_ksgd * 1000.0,
                    "system_total_cost_million_sgd_per_year": cost_ksgd / 1000.0,
                    "pv_capacity_kw": pv_kw,
                    "pv_capacity_mw": pv_kw / 1000.0,
                    "battery_energy_kwh": battery_kwh,
                    "battery_energy_mwh": battery_kwh / 1000.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_lolp_zero_multiyear_synthetic_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for years, group in results.groupby("simulation_years", sort=True):
        label = str(group["simulation_period_label"].iloc[0])
        for spec in SYNTHETIC_METRIC_SPECS:
            values = pd.to_numeric(group[spec.value_column], errors="coerce").dropna()
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            cv = 0.0 if abs(mean) <= 1e-12 and std <= 1e-12 else std / mean if abs(mean) > 1e-12 else np.nan
            rows.append(
                {
                    "simulation_years": int(years),
                    "simulation_period_label": label,
                    "metric": spec.metric_key,
                    "source_column": spec.value_column,
                    "unit": spec.unit_label,
                    "count": int(values.count()),
                    "mean": mean,
                    "std": std,
                    "coefficient_of_variation": cv,
                    "coefficient_of_variation_percent": cv * 100.0 if np.isfinite(cv) else np.nan,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def build_lolp_zero_multiyear_cv_trend(cv_summary: pd.DataFrame) -> pd.DataFrame:
    """Interpolate 1-10 year CV trends while preserving the synthetic distribution anchors."""
    required = {"simulation_years", "metric", "coefficient_of_variation", "coefficient_of_variation_percent"}
    missing = required.difference(cv_summary.columns)
    if missing:
        raise ValueError("Synthetic CV trend summary missing column(s): " + ", ".join(sorted(missing)))
    rows: list[dict[str, Any]] = []
    anchor_years = np.asarray(SYNTHETIC_MULTYEAR_SIMULATION_YEARS, dtype=float)
    for spec in SYNTHETIC_METRIC_SPECS:
        metric_summary = cv_summary[cv_summary["metric"].astype(str).eq(spec.metric_key)].copy()
        metric_summary["simulation_years"] = pd.to_numeric(metric_summary["simulation_years"], errors="coerce")
        metric_summary["coefficient_of_variation_percent"] = pd.to_numeric(
            metric_summary["coefficient_of_variation_percent"],
            errors="coerce",
        )
        metric_summary["coefficient_of_variation"] = pd.to_numeric(
            metric_summary["coefficient_of_variation"],
            errors="coerce",
        )
        metric_summary = metric_summary.dropna(subset=["simulation_years", "coefficient_of_variation_percent"])
        anchor_summary = metric_summary[
            metric_summary["simulation_years"].astype(int).isin(SYNTHETIC_MULTYEAR_SIMULATION_YEARS)
        ].sort_values("simulation_years")
        found_years = set(anchor_summary["simulation_years"].astype(int))
        expected_years = set(SYNTHETIC_MULTYEAR_SIMULATION_YEARS)
        if found_years != expected_years:
            missing_years = ", ".join(str(year) for year in sorted(expected_years.difference(found_years)))
            raise ValueError(f"Synthetic CV trend for {spec.metric_key} missing anchor year(s): {missing_years}")
        anchor_percent = anchor_summary["coefficient_of_variation_percent"].astype(float).to_numpy()
        if np.any(anchor_percent <= 0.0):
            raise ValueError(f"Synthetic CV trend for {spec.metric_key} requires positive anchor CV percentages.")
        anchor_by_year = {
            int(row.simulation_years): (
                float(row.coefficient_of_variation),
                float(row.coefficient_of_variation_percent),
            )
            for row in anchor_summary.itertuples(index=False)
        }
        interpolated_percent = np.exp(
            np.interp(
                np.asarray(SYNTHETIC_MULTYEAR_CV_TREND_YEARS, dtype=float),
                anchor_years,
                np.log(anchor_percent),
            )
        )
        for year, cv_percent in zip(SYNTHETIC_MULTYEAR_CV_TREND_YEARS, interpolated_percent, strict=True):
            if int(year) in anchor_by_year:
                cv_fraction, cv_percent = anchor_by_year[int(year)]
                source_role = "anchor"
            else:
                cv_fraction = float(cv_percent) / 100.0
                source_role = "interpolated"
            rows.append(
                {
                    "simulation_years": int(year),
                    "simulation_period_label": _period_label(int(year)),
                    "metric": spec.metric_key,
                    "metric_title": spec.title,
                    "unit": "%",
                    "coefficient_of_variation": float(cv_fraction),
                    "coefficient_of_variation_percent": float(cv_percent),
                    "source_role": source_role,
                    "interpolation_method": "piecewise_linear_interpolation_on_log_cv_percent",
                }
            )
    return pd.DataFrame(rows)


def make_lolp_zero_multiyear_synthetic_figures(
    out: Path,
    case_key: str,
    results: pd.DataFrame,
) -> dict[str, Path]:
    """Create three 2x2 synthetic multi-year distribution grid figures."""
    paths: dict[str, Path] = {}
    for spec in SYNTHETIC_METRIC_SPECS:
        source = _metric_source(results, spec)
        base = out / f"{case_key}_{spec.figure_name}"
        paths[f"{spec.metric_key}_source_data"] = write_source_data(base, source)
        paths.update(_plot_metric_grid(out, case_key, source, spec))
    return paths


def make_lolp_zero_multiyear_cv_trend_figures(
    out: Path,
    case_key: str,
    cv_trend: pd.DataFrame,
) -> dict[str, Path]:
    """Create one CV convergence trend figure for each synthetic metric."""
    paths: dict[str, Path] = {}
    for spec in SYNTHETIC_METRIC_SPECS:
        source = cv_trend[cv_trend["metric"].astype(str).eq(spec.metric_key)].copy()
        if source.empty:
            raise ValueError(f"Synthetic CV trend source is empty for metric: {spec.metric_key}")
        figure_name = _cv_trend_figure_name(spec)
        base = out / f"{case_key}_{figure_name}"
        paths[f"{spec.metric_key}_cv_trend_source_data"] = write_source_data(base, source)
        paths.update(_plot_cv_trend(out, case_key, source, spec, figure_name))
    return paths


def _plot_metric_grid(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
    spec: SyntheticMetricSpec,
) -> dict[str, Path]:
    limits = _padded_x_limits(source["display_value"].astype(float).to_numpy())
    mpl, plt = matplotlib_context(out / f"{case_key}_{spec.figure_name}")
    apply_nature_style(mpl)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15), constrained_layout=False)
    axes_flat = list(axes.ravel())
    panel_labels = ("a", "b", "c", "d")
    for ax, panel, years in zip(axes_flat, panel_labels, SYNTHETIC_MULTYEAR_SIMULATION_YEARS, strict=True):
        panel_source = source[source["simulation_years"].astype(int).eq(int(years))]
        density_ax = ax.twinx()
        values = panel_source["display_value"].astype(float).to_numpy()
        _draw_panel_distribution(ax, density_ax, values, x_limits=limits)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        cv = 0.0 if abs(mean) <= 1e-12 and std <= 1e-12 else std / mean if abs(mean) > 1e-12 else np.nan
        ax.set_title(f"({panel}) {_period_label(years)}", loc="left", fontsize=7.8)
        ax.set_xlabel(spec.xlabel)
        ax.set_ylabel("Frequency")
        density_ax.set_ylabel("Probability density")
        _annotate_panel(ax, mean, std, cv, len(values), spec.unit_label)
    fig.suptitle(f"Manila synthetic 500-scenario LOLP=0 {spec.title}", x=0.08, y=0.985, ha="left", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.91, wspace=0.46, hspace=0.56)
    paths = save_publication_figure(fig, out / f"{case_key}_{spec.figure_name}")
    plt.close(fig)
    return paths


def _plot_cv_trend(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
    spec: SyntheticMetricSpec,
    figure_name: str,
) -> dict[str, Path]:
    source = source.sort_values("simulation_years")
    years = source["simulation_years"].astype(int).to_numpy()
    cv_percent = source["coefficient_of_variation_percent"].astype(float).to_numpy()
    mpl, plt = matplotlib_context(out / f"{case_key}_{figure_name}")
    apply_nature_style(mpl)
    fig, ax = plt.subplots(figsize=(3.55, 2.65), constrained_layout=False)
    ax.plot(
        years,
        cv_percent,
        color=NATURE_PALETTE["neutral_dark"],
        linestyle=(0, (1.2, 1.8)),
        linewidth=0.9,
        zorder=2,
    )
    ax.scatter(
        years,
        cv_percent,
        s=20,
        color=NATURE_PALETTE["blue"],
        edgecolor=NATURE_PALETTE["neutral_dark"],
        linewidth=0.35,
        zorder=3,
    )
    anchor = source[source["source_role"].astype(str).eq("anchor")]
    ax.scatter(
        anchor["simulation_years"].astype(int),
        anchor["coefficient_of_variation_percent"].astype(float),
        s=26,
        color=NATURE_PALETTE["blue"],
        edgecolor=NATURE_PALETTE["neutral_black"],
        linewidth=0.45,
        zorder=4,
    )
    ax.set_title(f"Manila synthetic CV convergence: {spec.title}", loc="left", fontsize=8.2, fontweight="bold")
    ax.set_xlabel("Simulation period (years)")
    ax.set_ylabel("Coefficient of variation (%)")
    ax.set_xlim(0.8, 10.2)
    ax.set_xticks(list(SYNTHETIC_MULTYEAR_CV_TREND_YEARS))
    ax.set_ylim(_cv_axis_limits(cv_percent))
    ax.grid(True, which="major", color=NATURE_PALETTE["neutral_light"], linewidth=0.45, alpha=0.55, zorder=0)
    ax.tick_params(labelsize=6.2, length=2.5, width=0.55)
    ax.text(
        0.98,
        0.93,
        "Anchors: 1, 3, 5, 10 years\nSynthetic 500-scenario basis",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=NATURE_PALETTE["neutral_dark"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 1.0},
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.88)
    paths = save_publication_figure(fig, out / f"{case_key}_{figure_name}")
    plt.close(fig)
    return paths


def _draw_panel_distribution(ax: Any, density_ax: Any, values: np.ndarray, x_limits: tuple[float, float]) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Synthetic multi-year distribution requires at least one finite value.")
    counts, _, _ = ax.hist(
        finite,
        bins=7,
        color="#E8E6E6",
        edgecolor=NATURE_PALETTE["neutral_mid"],
        linewidth=0.65,
        zorder=2,
    )
    x = np.linspace(float(x_limits[0]), float(x_limits[1]), 300)
    density = _normal_pdf(x, float(np.mean(finite)), float(np.std(finite, ddof=0)))
    density_ax.plot(x, density, color=NATURE_PALETTE["gold"], linewidth=1.0, zorder=4)
    ax.set_xlim(float(x_limits[0]), float(x_limits[1]))
    ax.set_ylim(0.0, max(float(np.max(counts)) * 1.20, 1.0))
    density_ax.set_ylim(0.0, max(float(density.max()) * 1.20, 1e-9))
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.42, alpha=0.48, zorder=0)
    density_ax.grid(False)
    density_ax.spines["right"].set_visible(True)
    density_ax.spines["right"].set_color(NATURE_PALETTE["neutral_black"])
    density_ax.tick_params(labelsize=5.8, length=2.3, width=0.55)
    ax.tick_params(labelsize=5.8, length=2.3, width=0.55)


def _annotate_panel(ax: Any, mean: float, std: float, cv: float, n: int, unit_label: str) -> None:
    cv_text = "undefined" if not np.isfinite(cv) else f"{cv * 100.0:.1f}%"
    ax.text(
        0.62,
        0.91,
        f"mu={mean:.2f} {unit_label}\nsigma={std:.2f} {unit_label}\nCV={cv_text}\nn={int(n)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        color=NATURE_PALETTE["neutral_black"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
    )


def _cv_trend_figure_name(spec: SyntheticMetricSpec) -> str:
    return spec.figure_name.replace("Distribution_Grid", "CV_Trend")


def _cv_axis_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Synthetic CV trend requires at least one finite value.")
    upper = float(np.max(finite))
    if upper <= 0.0:
        return (0.0, 1.0)
    return (0.0, upper * 1.18)


def _metric_source(results: pd.DataFrame, spec: SyntheticMetricSpec) -> pd.DataFrame:
    values = pd.to_numeric(results[spec.value_column], errors="coerce")
    return pd.DataFrame(
        {
            "simulation_years": results["simulation_years"].astype(int),
            "simulation_period_label": results["simulation_period_label"].astype(str),
            "synthetic_scenario_id": results["synthetic_scenario_id"].astype(int),
            "metric_key": spec.metric_key,
            "metric_title": spec.title,
            "display_value": values,
            "display_unit": spec.unit_label,
        }
    )


def _sample_truncated_normal(
    rng: np.random.Generator,
    *,
    mean: float,
    sd: float,
    size: int,
    lower: float,
) -> np.ndarray:
    values = rng.normal(float(mean), float(sd), int(size))
    while np.any(values < float(lower)):
        mask = values < float(lower)
        values[mask] = rng.normal(float(mean), float(sd), int(mask.sum()))
    return values


def _round_to_step(values: np.ndarray, step: float) -> np.ndarray:
    rounded = np.round(np.asarray(values, dtype=float) / float(step)) * float(step)
    return np.maximum(rounded, 0.0)


def _period_label(years: int) -> str:
    return "1 year" if int(years) == 1 else f"{int(years)} years"
