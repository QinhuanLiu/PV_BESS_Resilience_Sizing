"""Plot the evaluated incremental cost--reliability frontier for Brisbane.

The figure reads the completed adaptive sizing result only.  It never reruns
dispatch, creates new scenarios, or evaluates additional capacity designs.

Run with ``python -m src.figures.incremental_cost_reliability_frontier``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .capacity_sweep_contours import file_sha256
from .ccp_3d_surfaces import _completed_records
from .core import NATURE_PALETTE, apply_nature_style, matplotlib_context, save_publication_figure, write_source_data


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_Capacity_Sweep_15Y_100Scenarios_Cost_SSR_SCR"
DEFAULT_SIZING_RESULTS = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_CCP_LOLP_Sizing_0p25_Adaptive_CCP2pct"
DEFAULT_REFERENCE_PV_KW = 3.25
DEFAULT_REFERENCE_BATTERY_KWH = 11.5
DEFAULT_LOLP_THRESHOLD = 0.20
DEFAULT_MIN_ABSOLUTE_CCP = 0.26
DEFAULT_MAX_ABSOLUTE_CCP = 0.95

SOURCE_OUTPUT_STEM = "Incremental_Cost_Reliability_Frontier"
POWER_LAW_FIT_OUTPUT_STEM = "Incremental_Cost_Reliability_Frontier_Power_Law_Fit"
FIGURE_SIZE_INCHES = (7.2, 5.1)
AXES_ASPECT_RATIO = 16.0 / 9.0
AXES_LEFT = 0.145
AXES_RIGHT = 0.98
AXES_BOTTOM = 0.16
AXES_WIDTH_FRACTION = AXES_RIGHT - AXES_LEFT
AXES_HEIGHT_FRACTION = (
    AXES_WIDTH_FRACTION * FIGURE_SIZE_INCHES[0] / AXES_ASPECT_RATIO / FIGURE_SIZE_INCHES[1]
)
AXES_TOP = AXES_BOTTOM + AXES_HEIGHT_FRACTION
X_LIMITS = (0.0, 180.0)
X_TICKS = (0.0, 50.0, 100.0, 150.0)
Y_LIMITS = (0.0, 71.0)
Y_TICKS = tuple(float(value) for value in range(0, 71, 10))
FRONTIER_COLOR = NATURE_PALETTE["blue"]
REFERENCE_COLOR = NATURE_PALETTE["neutral_black"]
FRONTIER_MARKER_SIZE_POINTS_SQUARED = 24.0
REFERENCE_MARKER_SIZE_POINTS_SQUARED = 31.0
POWER_LAW_FIT_SAMPLES = 200
POWER_LAW_FIT_LINE_WIDTH = 2.2

THRESHOLD_FIELDS = {
    0.10: ("LOLP_lt_10pct_passing_scenarios", "CCP_LOLP_lt_10pct"),
    0.20: ("LOLP_lt_20pct_passing_scenarios", "CCP_LOLP_lt_20pct"),
    0.30: ("LOLP_lt_30pct_passing_scenarios", "CCP_LOLP_lt_30pct"),
}
REQUIRED_CANDIDATE_COLUMNS = {
    "pv_capacity_kw",
    "battery_energy_kwh",
    "battery_power_kw",
    "scenario_count",
    "mean_annual_cost_sgd_per_year",
    "candidate_origin",
}


def _threshold_fields(lolp_threshold: float) -> tuple[str, str]:
    """Return the completed-summary columns for a supported strict LOLP threshold."""
    threshold = float(lolp_threshold)
    for supported, fields in THRESHOLD_FIELDS.items():
        if math.isclose(threshold, supported, rel_tol=0.0, abs_tol=1e-12):
            return fields
    choices = ", ".join(f"{value:.2f}" for value in THRESHOLD_FIELDS)
    raise ValueError(f"LOLP threshold must be one of {choices} for this completed sizing result")


def _required_passing_count(probability: float, scenario_count: int, *, rounding: str) -> int:
    value = float(probability) * int(scenario_count)
    if rounding == "ceil":
        return int(math.ceil(value - 1e-12))
    if rounding == "floor":
        return int(math.floor(value + 1e-12))
    raise ValueError(f"Unsupported rounding rule: {rounding}")


def _read_validated_candidates(
    results_path: Path,
    sizing_path: Path,
    *,
    lolp_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str], dict[str, str]]:
    """Load completed adaptive candidates and verify their source and checksums."""
    _records, _source, results_manifest, results_hashes = _completed_records(results_path)
    sizing_manifest = json.loads((sizing_path / "manifest.json").read_text(encoding="utf-8"))
    sizing_validation = json.loads((sizing_path / "validation.json").read_text(encoding="utf-8"))
    sizing_status = json.loads((sizing_path / "run_status.json").read_text(encoding="utf-8"))
    if sizing_status.get("state") != "complete" or sizing_validation.get("passed") is not True:
        raise ValueError("Only a completed, validated adaptive sizing result can be plotted")
    if sizing_validation.get("strict_threshold_verified") is not True:
        raise ValueError("Adaptive sizing result does not verify the strict LOLP comparison")
    if sizing_validation.get("shared_scenarios_verified") is not True:
        raise ValueError("Adaptive sizing result does not verify shared scenarios")
    if sizing_manifest.get("search", {}).get("optimality_status") != "adaptive_high_confidence":
        raise ValueError("Adaptive sizing result is not marked adaptive_high_confidence")
    source_directory = Path(str(sizing_manifest.get("source", {}).get("directory", ""))).resolve()
    if source_directory != results_path:
        raise ValueError("Adaptive sizing result does not reference the requested completed capacity sweep")
    for name, expected_hash in sizing_manifest.get("source", {}).get("file_hashes", {}).items():
        if file_sha256(results_path / name) != expected_hash:
            raise ValueError(f"Completed sweep source changed since adaptive sizing: {name}")

    passing_column, ccp_column = _threshold_fields(lolp_threshold)
    candidate_path = sizing_path / "candidate_summary.csv"
    candidate_hash = file_sha256(candidate_path)
    expected_hash = sizing_validation.get("file_sha256", {}).get(candidate_path.name)
    if candidate_hash != expected_hash:
        raise ValueError("Adaptive candidate summary checksum does not match its validation record")
    candidates = pd.read_csv(candidate_path)
    required = REQUIRED_CANDIDATE_COLUMNS | {passing_column, ccp_column}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate summary is missing required columns: {missing}")
    if candidates.empty:
        raise ValueError("Candidate summary is empty")
    numeric = [
        "pv_capacity_kw", "battery_energy_kwh", "battery_power_kw", "scenario_count",
        "mean_annual_cost_sgd_per_year", passing_column, ccp_column,
    ]
    if not np.isfinite(candidates[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Candidate summary requires finite capacity, cost and CCP values")
    scenario_counts = candidates["scenario_count"].to_numpy(dtype=float)
    if not np.allclose(scenario_counts, np.rint(scenario_counts), rtol=0.0, atol=1e-12):
        raise ValueError("Candidate scenario counts must be integers")
    if len(set(scenario_counts.astype(int).tolist())) != 1:
        raise ValueError("Every candidate must use the same number of scenarios")
    scenario_count = int(scenario_counts[0])
    if scenario_count <= 0:
        raise ValueError("Candidate scenario count must be positive")
    passing = candidates[passing_column].to_numpy(dtype=float)
    if not np.allclose(passing, np.rint(passing), rtol=0.0, atol=1e-12):
        raise ValueError("Strict-LOLP passing counts must be integers")
    if (passing < 0).any() or (passing > scenario_count).any():
        raise ValueError("Strict-LOLP passing counts must lie within the scenario count")
    expected_ccp = passing / scenario_count
    if not np.allclose(candidates[ccp_column].to_numpy(dtype=float), expected_ccp, rtol=0.0, atol=1e-12):
        raise ValueError("Candidate CCP must equal strict-LOLP passing scenarios divided by scenario count")
    return candidates, results_manifest, sizing_manifest, sizing_validation, results_hashes, {
        candidate_path.name: candidate_hash,
    }


def select_incremental_frontier(
    candidates: pd.DataFrame,
    *,
    reference_pv_kw: float = DEFAULT_REFERENCE_PV_KW,
    reference_battery_kwh: float = DEFAULT_REFERENCE_BATTERY_KWH,
    lolp_threshold: float = DEFAULT_LOLP_THRESHOLD,
    min_absolute_ccp: float = DEFAULT_MIN_ABSOLUTE_CCP,
    max_absolute_ccp: float = DEFAULT_MAX_ABSOLUTE_CCP,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the baseline plus non-dominated evaluated capacity designs.

    A retained point has a strictly higher strict-LOLP passing count than every
    lower-cost point.  The cost ordering resolves exact ties by higher CCP,
    then lower PV and lower BESS, so dominated tied-cost designs are excluded.
    """
    passing_column, ccp_column = _threshold_fields(lolp_threshold)
    required = REQUIRED_CANDIDATE_COLUMNS | {passing_column, ccp_column}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate summary is missing required columns: {missing}")
    if not (0.0 <= float(min_absolute_ccp) <= float(max_absolute_ccp) <= 1.0):
        raise ValueError("Absolute CCP limits must satisfy 0 <= min <= max <= 1")
    table = candidates.copy()
    scenario_counts = table["scenario_count"].astype(int)
    if scenario_counts.nunique() != 1 or int(scenario_counts.iloc[0]) <= 0:
        raise ValueError("Candidate summary must contain one positive shared scenario count")
    scenario_count = int(scenario_counts.iloc[0])
    passing = table[passing_column].to_numpy(dtype=float)
    if not np.allclose(passing, np.rint(passing), rtol=0.0, atol=1e-12):
        raise ValueError("Strict-LOLP passing counts must be integers")
    table["_passing_scenarios"] = np.rint(passing).astype(int)
    expected_ccp = table["_passing_scenarios"].to_numpy(dtype=float) / scenario_count
    if not np.allclose(table[ccp_column].to_numpy(dtype=float), expected_ccp, rtol=0.0, atol=1e-12):
        raise ValueError("Candidate CCP must equal passing scenarios divided by scenario count")

    reference_mask = np.isclose(table["pv_capacity_kw"].to_numpy(dtype=float), float(reference_pv_kw), rtol=0.0, atol=1e-12) & np.isclose(
        table["battery_energy_kwh"].to_numpy(dtype=float), float(reference_battery_kwh), rtol=0.0, atol=1e-12
    )
    if int(reference_mask.sum()) != 1:
        raise ValueError("Reference PV/BESS capacity must identify exactly one evaluated candidate")
    reference = table.loc[reference_mask].iloc[0]
    reference_cost = float(reference["mean_annual_cost_sgd_per_year"])
    if not math.isclose(reference_cost, float(table["mean_annual_cost_sgd_per_year"].min()), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Reference capacity is not the global minimum-cost evaluated candidate")
    reference_passing = int(reference["_passing_scenarios"])
    min_passing = _required_passing_count(min_absolute_ccp, scenario_count, rounding="ceil")
    max_passing = _required_passing_count(max_absolute_ccp, scenario_count, rounding="floor")
    if min_passing <= reference_passing:
        raise ValueError("Minimum absolute CCP must be strictly above the reference CCP")

    ordered = table.sort_values(
        ["mean_annual_cost_sgd_per_year", "_passing_scenarios", "pv_capacity_kw", "battery_energy_kwh"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    frontier_indices: list[Any] = []
    greatest_passing_so_far = -1
    for index, row in ordered.iterrows():
        passing_count = int(row["_passing_scenarios"])
        if passing_count > greatest_passing_so_far:
            greatest_passing_so_far = passing_count
            if min_passing <= passing_count <= max_passing:
                frontier_indices.append(index)
    frontier = table.loc[frontier_indices].copy()
    if frontier.empty:
        raise ValueError("No evaluated non-dominated capacity design lies within the requested CCP range")
    frontier = pd.concat([table.loc[[reference.name]].copy(), frontier], ignore_index=True)
    if frontier.duplicated(["pv_capacity_kw", "battery_energy_kwh"]).any():
        raise ValueError("Reference design was unexpectedly duplicated in the plotted frontier")
    frontier["point_type"] = np.where(np.arange(len(frontier)) == 0, "baseline", "frontier")
    frontier["frontier_rank"] = np.arange(len(frontier), dtype=int)
    frontier["actual_ccp_percent"] = np.round(
        frontier["_passing_scenarios"].astype(float) / scenario_count * 100.0, 12
    )
    frontier["incremental_annual_cost_sgd_per_year"] = (
        frontier["mean_annual_cost_sgd_per_year"].astype(float) - reference_cost
    )
    frontier["incremental_ccp_percentage_points"] = np.round(
        frontier["actual_ccp_percent"] - reference_passing / scenario_count * 100.0, 12
    )
    frontier["is_non_dominated"] = True
    if not math.isclose(float(frontier.iloc[0]["incremental_annual_cost_sgd_per_year"]), 0.0, abs_tol=1e-9):
        raise ValueError("Reference design must have zero incremental annual cost")
    if not math.isclose(float(frontier.iloc[0]["incremental_ccp_percentage_points"]), 0.0, abs_tol=1e-9):
        raise ValueError("Reference design must have zero incremental CCP")
    nonbaseline = frontier.iloc[1:]
    if not nonbaseline["mean_annual_cost_sgd_per_year"].is_monotonic_increasing:
        raise ValueError("Cost--reliability frontier costs must increase strictly")
    if not nonbaseline["_passing_scenarios"].is_monotonic_increasing:
        raise ValueError("Cost--reliability frontier CCP must increase strictly")
    source_columns = [
        "frontier_rank", "point_type", "is_non_dominated", "pv_capacity_kw", "battery_energy_kwh",
        "battery_power_kw", "candidate_origin", "mean_annual_cost_sgd_per_year",
        "incremental_annual_cost_sgd_per_year", "scenario_count", "_passing_scenarios",
        "actual_ccp_percent", "incremental_ccp_percentage_points",
    ]
    source = frontier.loc[:, source_columns].rename(columns={"_passing_scenarios": "passing_scenarios"})
    summary = {
        "reference_pv_kw": float(reference["pv_capacity_kw"]),
        "reference_battery_kwh": float(reference["battery_energy_kwh"]),
        "reference_battery_power_kw": float(reference["battery_power_kw"]),
        "reference_annual_cost_sgd_per_year": reference_cost,
        "reference_passing_scenarios": reference_passing,
        "reference_ccp_percent": reference_passing / scenario_count * 100.0,
        "scenario_count": scenario_count,
        "lolp_threshold": float(lolp_threshold),
        "comparison": "LOLP < threshold",
        "min_absolute_ccp_percent": min_passing / scenario_count * 100.0,
        "max_absolute_ccp_percent": max_passing / scenario_count * 100.0,
        "evaluated_candidate_count": int(len(table)),
        "frontier_point_count_including_baseline": int(len(source)),
        "frontier_point_count_excluding_baseline": int(len(source) - 1),
    }
    return source.reset_index(drop=True), summary


def _power_law_through_origin(x: np.ndarray, coefficient: float, exponent: float) -> np.ndarray:
    """Return the positive power-law curve constrained to pass through the origin."""
    return float(coefficient) * np.power(np.asarray(x, dtype=float), float(exponent))


def fit_power_law_through_origin(source: pd.DataFrame) -> dict[str, Any]:
    """Fit ΔCCP = a(ΔC)^b to the plotted evaluated frontier points.

    The fit includes the baseline point at (0, 0), which has exactly zero
    residual under the constrained model. No fitted values are generated beyond
    the observed incremental-cost range.
    """
    required = {"incremental_annual_cost_sgd_per_year", "incremental_ccp_percentage_points"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Power-law fit source is missing required columns: {missing}")
    x_values = source["incremental_annual_cost_sgd_per_year"].to_numpy(dtype=float)
    y_values = source["incremental_ccp_percentage_points"].to_numpy(dtype=float)
    if x_values.size < 3 or not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("Power-law fitting requires at least three finite frontier points")
    if (x_values < 0.0).any() or (y_values < 0.0).any():
        raise ValueError("Power-law fitting requires non-negative incremental coordinates")
    if not np.any(x_values > 0.0) or not np.any(y_values > 0.0):
        raise ValueError("Power-law fitting requires non-zero frontier variation")
    origin = np.isclose(x_values, 0.0, rtol=0.0, atol=1e-12)
    if int(origin.sum()) != 1 or not math.isclose(float(y_values[origin][0]), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Power-law fitting requires exactly one baseline point at the origin")

    positive = x_values > 0.0
    initial_coefficient = float(y_values[positive].max() / np.sqrt(x_values[positive].max()))
    (coefficient, exponent), _covariance = curve_fit(
        _power_law_through_origin,
        x_values,
        y_values,
        p0=(initial_coefficient, 0.5),
        bounds=([np.finfo(float).eps, np.finfo(float).eps], [np.inf, np.inf]),
        maxfev=20_000,
    )
    fitted = _power_law_through_origin(x_values, coefficient, exponent)
    residuals = y_values - fitted
    residual_sum_squares = float(np.sum(residuals**2))
    total_sum_squares = float(np.sum((y_values - y_values.mean())**2))
    r_squared = 1.0 if math.isclose(total_sum_squares, 0.0, abs_tol=1e-15) else float(
        1.0 - residual_sum_squares / total_sum_squares
    )
    x_curve = np.linspace(0.0, float(x_values.max()), POWER_LAW_FIT_SAMPLES)
    y_curve = _power_law_through_origin(x_curve, coefficient, exponent)
    if not np.all(np.diff(y_curve) >= -1e-12):
        raise ValueError("Positive power-law fit must be monotonic over the observed range")
    return {
        "model": "power_law_through_origin",
        "formula": "ΔCCP = a(ΔC)^b",
        "independent_variable": "incremental_annual_cost_sgd_per_year",
        "dependent_variable": "incremental_ccp_percentage_points",
        "coefficient_a": float(coefficient),
        "exponent_b": float(exponent),
        "r_squared": r_squared,
        "rmse_percentage_points": float(np.sqrt(np.mean(residuals**2))),
        "sample_count_including_baseline": int(x_values.size),
        "sample_count_excluding_baseline": int(x_values.size - 1),
        "x_min_sgd_per_year": float(x_curve[0]),
        "x_max_sgd_per_year": float(x_curve[-1]),
        "passes_through_origin": True,
        "monotonic_over_observed_range": True,
        "extrapolation": False,
        "x_curve": x_curve.astype(float).tolist(),
        "y_curve": y_curve.astype(float).tolist(),
    }


def build_frontier_figure(
    source: pd.DataFrame,
    *,
    city_label: str = "Brisbane",
    lolp_threshold: float = DEFAULT_LOLP_THRESHOLD,
    out: Path | str | None = None,
    power_law_fit: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Draw the discrete frontier, optionally with a descriptive power-law fit."""
    required = {
        "point_type", "pv_capacity_kw", "battery_energy_kwh", "incremental_annual_cost_sgd_per_year",
        "actual_ccp_percent", "incremental_ccp_percentage_points",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Frontier source is missing required columns: {missing}")
    baseline = source.loc[source["point_type"].eq("baseline")]
    designs = source.loc[source["point_type"].eq("frontier")]
    if len(baseline) != 1 or designs.empty:
        raise ValueError("Frontier plot requires one baseline point and at least one frontier design")
    if not np.isfinite(source[["incremental_annual_cost_sgd_per_year", "incremental_ccp_percentage_points"]].to_numpy(dtype=float)).all():
        raise ValueError("Frontier plot requires finite incremental coordinates")

    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    mpl.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.6,
            "axes.titlesize": 9.4,
        }
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    fig.subplots_adjust(left=AXES_LEFT, right=AXES_RIGHT, bottom=AXES_BOTTOM, top=AXES_TOP)
    has_power_law_fit = power_law_fit is not None
    title_suffix = " (power-law fit)" if has_power_law_fit else ""
    fig.suptitle(
        f"{city_label}: Incremental cost–reliability frontier{title_suffix}",
        x=0.02,
        ha="left",
        y=0.977,
        fontsize=10.2,
    )
    threshold_percent = float(lolp_threshold) * 100.0
    fig.text(
        0.5,
        0.852,
        (
            "Points: non-dominated evaluated PV–BESS designs; "
            "curve: power-law least-squares fit constrained through the origin"
            if has_power_law_fit
            else "Each point: non-dominated evaluated PV–BESS design; "
            f"CCP = P(LOLP < {threshold_percent:g}%)"
        ),
        ha="center",
        va="center",
        fontsize=7.3,
        color=NATURE_PALETTE["neutral_dark"],
    )
    if has_power_law_fit:
        curve_x = np.asarray(power_law_fit["x_curve"], dtype=float)
        curve_y = np.asarray(power_law_fit["y_curve"], dtype=float)
        if len(curve_x) != POWER_LAW_FIT_SAMPLES or curve_x.shape != curve_y.shape:
            raise ValueError("Power-law fit curve must contain paired fixed-count samples")
        if curve_x[0] != 0.0 or curve_x[-1] > float(designs["incremental_annual_cost_sgd_per_year"].max()) + 1e-12:
            raise ValueError("Power-law fit curve must remain within the observed incremental-cost range")
        ax.plot(
            curve_x,
            curve_y,
            color=FRONTIER_COLOR,
            linewidth=POWER_LAW_FIT_LINE_WIDTH,
            zorder=1,
        )
    ax.scatter(
        designs["incremental_annual_cost_sgd_per_year"],
        designs["incremental_ccp_percentage_points"],
        s=FRONTIER_MARKER_SIZE_POINTS_SQUARED,
        color=FRONTIER_COLOR,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.82 if has_power_law_fit else 1.0,
        zorder=3,
        clip_on=False,
    )
    baseline_row = baseline.iloc[0]
    ax.scatter(
        [float(baseline_row["incremental_annual_cost_sgd_per_year"])],
        [float(baseline_row["incremental_ccp_percentage_points"])],
        s=REFERENCE_MARKER_SIZE_POINTS_SQUARED,
        facecolors="white",
        edgecolors=REFERENCE_COLOR,
        linewidths=1.1,
        zorder=4,
        clip_on=False,
    )
    ax.annotate(
        "Baseline\n3.25 kW PV, 11.5 kWh BESS",
        xy=(0.0, 0.0),
        xytext=(25.0, 7.0),
        textcoords="data",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=REFERENCE_COLOR,
        arrowprops={"arrowstyle": "-", "color": REFERENCE_COLOR, "linewidth": 0.65},
        annotation_clip=False,
    )
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.set_xticks(X_TICKS)
    ax.set_yticks(Y_TICKS)
    ax.set_xlabel("Incremental annual cost, C − C$_0$ (SGD yr$^{-1}$)")
    ax.set_ylabel(
        f"Incremental CCP under LOLP < {threshold_percent:g}% (percentage points)"
    )
    ax.tick_params(direction="out", length=3.0, width=0.65, pad=2.2)
    ax.grid(False)
    position = ax.get_position()
    axes_width_inches = position.width * FIGURE_SIZE_INCHES[0]
    axes_height_inches = position.height * FIGURE_SIZE_INCHES[1]
    metadata = {
        "layout": "single-panel 16:9 plotting area; title and explanatory text outside the axes",
        "figure_size_inches": [float(value) for value in FIGURE_SIZE_INCHES],
        "axes_position": [float(position.x0), float(position.y0), float(position.width), float(position.height)],
        "axes_size_inches": [axes_width_inches, axes_height_inches],
        "axes_aspect_ratio": axes_width_inches / axes_height_inches,
        "title": f"{city_label}: Incremental cost–reliability frontier{title_suffix}",
        "x_axis": ax.get_xlabel(),
        "y_axis": ax.get_ylabel(),
        "x_limits": [float(value) for value in ax.get_xlim()],
        "y_limits": [float(value) for value in ax.get_ylim()],
        "x_ticks": [float(value) for value in ax.get_xticks()],
        "y_ticks": [float(value) for value in ax.get_yticks()],
        "plot_type": "scatter",
        "frontier_points_connected": False,
        "smoothing": False,
        "interpolation": False,
        "trend_fit": has_power_law_fit,
        "fit_type": "power_law_through_origin" if has_power_law_fit else None,
        "frontier_color": FRONTIER_COLOR,
        "baseline_color": REFERENCE_COLOR,
        "frontier_marker_size_points_squared": FRONTIER_MARKER_SIZE_POINTS_SQUARED,
        "baseline_marker_size_points_squared": REFERENCE_MARKER_SIZE_POINTS_SQUARED,
        "frontier_point_count_excluding_baseline": int(len(designs)),
        "baseline_point": {
            "incremental_annual_cost_sgd_per_year": float(baseline_row["incremental_annual_cost_sgd_per_year"]),
            "incremental_ccp_percentage_points": float(baseline_row["incremental_ccp_percentage_points"]),
        },
        "power_law_fit": power_law_fit if has_power_law_fit else None,
        "dpi": 600,
    }
    return fig, metadata


def build_power_law_fit_figure(
    source: pd.DataFrame,
    *,
    city_label: str = "Brisbane",
    lolp_threshold: float = DEFAULT_LOLP_THRESHOLD,
    out: Path | str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Draw the evaluated frontier with its descriptive origin-constrained power-law fit."""
    fit = fit_power_law_through_origin(source)
    return build_frontier_figure(
        source,
        city_label=city_label,
        lolp_threshold=lolp_threshold,
        out=out,
        power_law_fit=fit,
    )


def power_law_fit_summary(fit: dict[str, Any]) -> pd.DataFrame:
    """Return the reproducible scalar summary of the fitted descriptive curve."""
    columns = [
        "model", "formula", "independent_variable", "dependent_variable", "coefficient_a", "exponent_b",
        "r_squared", "rmse_percentage_points", "sample_count_including_baseline",
        "sample_count_excluding_baseline", "x_min_sgd_per_year", "x_max_sgd_per_year",
        "passes_through_origin", "monotonic_over_observed_range", "extrapolation",
    ]
    return pd.DataFrame([{column: fit[column] for column in columns}], columns=columns)


def _readme_text(
    case_key: str,
    summary: dict[str, Any],
    metadata: dict[str, Any],
    power_law_fit: dict[str, Any],
    fit_summary_name: str,
) -> str:
    return f"""# {case_key}: incremental cost--reliability frontier

This figure is derived only from the completed, checksum-validated adaptive
0.25 kW × 0.25 kWh sizing result. It does not rerun dispatch, generate new
scenarios, or evaluate additional capacity combinations.

- Strict reliability condition: `LOLP < {float(summary['lolp_threshold']) * 100:g}%`.
- Baseline: {float(summary['reference_pv_kw']):g} kW PV and
  {float(summary['reference_battery_kwh']):g} kWh BESS, with annual cost
  {float(summary['reference_annual_cost_sgd_per_year']):.6f} SGD yr⁻¹ and
  CCP₀ = {float(summary['reference_ccp_percent']):g}%.
- X coordinate: `C − C₀` in SGD yr⁻¹. Y coordinate: `CCP − CCP₀` in percentage points.
- Only non-dominated evaluated designs with actual CCP from
  {float(summary['min_absolute_ccp_percent']):g}% to {float(summary['max_absolute_ccp_percent']):g}% are plotted.
  The baseline reference point is retained at the origin.
- The original figure contains {int(summary['frontier_point_count_excluding_baseline'])} frontier designs
  plus the baseline. Points are not connected, smoothed, interpolated or fitted.
- The companion power-law figure retains the same points and adds the descriptive least-squares model
  `ΔCCP = a(ΔC)^b`, constrained through the origin. It uses all
  {int(power_law_fit['sample_count_including_baseline'])} plotted points, is drawn only from
  {float(power_law_fit['x_min_sgd_per_year']):g} to {float(power_law_fit['x_max_sgd_per_year']):.6g}
  SGD yr⁻¹, and is not extrapolated. `{fit_summary_name}` records its coefficients, R² and RMSE.
- The plotting area is 16:9, with x range {metadata['x_limits'][0]:g}–{metadata['x_limits'][1]:g}
  SGD yr⁻¹ and y range {metadata['y_limits'][0]:g}–{metadata['y_limits'][1]:g} percentage points.

The front is a cost--reliability frontier among the {int(summary['evaluated_candidate_count'])}
evaluated adaptive-mesh capacity combinations. It does not claim a continuous
capacity optimisation result or extrapolate beyond the evaluated designs.
"""


def run(
    results_dir: Path | str = DEFAULT_RESULTS,
    output_dir: Path | str | None = None,
    *,
    sizing_results_dir: Path | str = DEFAULT_SIZING_RESULTS,
    reference_pv_kw: float = DEFAULT_REFERENCE_PV_KW,
    reference_battery_kwh: float = DEFAULT_REFERENCE_BATTERY_KWH,
    lolp_threshold: float = DEFAULT_LOLP_THRESHOLD,
    min_absolute_ccp: float = DEFAULT_MIN_ABSOLUTE_CCP,
    max_absolute_ccp: float = DEFAULT_MAX_ABSOLUTE_CCP,
) -> dict[str, Any]:
    """Export the discrete evaluated incremental cost--reliability frontier."""
    results_path = Path(results_dir).resolve()
    sizing_path = Path(sizing_results_dir).resolve()
    candidates, results_manifest, sizing_manifest, sizing_validation, results_hashes, sizing_hashes = _read_validated_candidates(
        results_path, sizing_path, lolp_threshold=lolp_threshold
    )
    source, summary = select_incremental_frontier(
        candidates,
        reference_pv_kw=reference_pv_kw,
        reference_battery_kwh=reference_battery_kwh,
        lolp_threshold=lolp_threshold,
        min_absolute_ccp=min_absolute_ccp,
        max_absolute_ccp=max_absolute_ccp,
    )
    output_path = Path(output_dir).resolve() if output_dir else sizing_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    case_key = str(results_manifest["spec"]["case_key"])
    source_path = write_source_data(output_path / f"{case_key}_{SOURCE_OUTPUT_STEM}", source)
    figure_base = output_path / f"{case_key}_{SOURCE_OUTPUT_STEM}"
    figure, metadata = build_frontier_figure(
        source,
        city_label=case_key.split("_", 1)[0],
        lolp_threshold=lolp_threshold,
        out=output_path,
    )
    _, plt = matplotlib_context(output_path)
    try:
        figure_paths = save_publication_figure(figure, figure_base)
    finally:
        plt.close(figure)
    power_law_fit = fit_power_law_through_origin(source)
    fit_summary = power_law_fit_summary(power_law_fit)
    fit_summary_path = output_path / f"{case_key}_{POWER_LAW_FIT_OUTPUT_STEM}_Summary.csv"
    fit_summary.to_csv(fit_summary_path, index=False)
    power_law_figure_base = output_path / f"{case_key}_{POWER_LAW_FIT_OUTPUT_STEM}"
    power_law_figure, power_law_metadata = build_frontier_figure(
        source,
        city_label=case_key.split("_", 1)[0],
        lolp_threshold=lolp_threshold,
        out=output_path,
        power_law_fit=power_law_fit,
    )
    try:
        power_law_figure_paths = save_publication_figure(power_law_figure, power_law_figure_base)
    finally:
        plt.close(power_law_figure)

    for name, expected_hash in results_hashes.items():
        if file_sha256(results_path / name) != expected_hash:
            raise RuntimeError(f"Completed sweep input changed while plotting: {name}")
    for name, expected_hash in sizing_hashes.items():
        if file_sha256(sizing_path / name) != expected_hash:
            raise RuntimeError(f"Adaptive sizing input changed while plotting: {name}")

    readme_path = output_path / f"{SOURCE_OUTPUT_STEM}_README.md"
    readme_path.write_text(
        _readme_text(case_key, summary, metadata, power_law_fit, fit_summary_path.name), encoding="utf-8"
    )
    output_files = [source_path, fit_summary_path, readme_path, *figure_paths.values(), *power_law_figure_paths.values()]
    report = {
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_key": case_key,
        "figure_input": {"kind": "adaptive_fine_candidate_summary", "directory": str(sizing_path)},
        "input_sha256": {**results_hashes, **sizing_hashes},
        "source_fingerprint": sizing_validation.get("source_fingerprint"),
        "fine_precision": sizing_validation.get("fine_precision"),
        "adaptive_optimality_status": sizing_manifest.get("search", {}).get("optimality_status"),
        "comparison": "LOLP < threshold",
        "frontier_rule": (
            "Sort evaluated candidates by annual cost ascending, strict-LOLP passing scenarios descending, "
            "PV ascending and BESS ascending; retain a design only when its passing count exceeds every prior design"
        ),
        "summary": summary,
        "figure": metadata,
        "power_law_fit": fit_summary.iloc[0].to_dict(),
        "power_law_fit_figure": power_law_metadata,
        "output_sha256": {path.name: file_sha256(path) for path in output_files},
    }
    report_path = output_path / "incremental_cost_reliability_frontier_validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "source_data": source_path,
        "figure": figure_paths,
        "power_law_fit": power_law_figure_paths,
        "fit_summary": fit_summary_path,
        "readme": readme_path,
        "validation": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Completed capacity sweep referenced by the adaptive sizing result")
    parser.add_argument("--sizing-results-dir", type=Path, default=DEFAULT_SIZING_RESULTS,
                        help="Completed adaptive sizing directory containing candidate_summary.csv")
    parser.add_argument("--output-dir", type=Path,
                        help="Figure output directory (default: sizing-results-dir/figures)")
    parser.add_argument("--reference-pv-kw", type=float, default=DEFAULT_REFERENCE_PV_KW)
    parser.add_argument("--reference-battery-kwh", type=float, default=DEFAULT_REFERENCE_BATTERY_KWH)
    parser.add_argument("--lolp-threshold", type=float, default=DEFAULT_LOLP_THRESHOLD,
                        help="Strict LOLP threshold; completed summaries support 0.10, 0.20 or 0.30")
    parser.add_argument("--min-absolute-ccp", type=float, default=DEFAULT_MIN_ABSOLUTE_CCP)
    parser.add_argument("--max-absolute-ccp", type=float, default=DEFAULT_MAX_ABSOLUTE_CCP)
    args = parser.parse_args()
    paths = run(
        args.results_dir,
        args.output_dir,
        sizing_results_dir=args.sizing_results_dir,
        reference_pv_kw=args.reference_pv_kw,
        reference_battery_kwh=args.reference_battery_kwh,
        lolp_threshold=args.lolp_threshold,
        min_absolute_ccp=args.min_absolute_ccp,
        max_absolute_ccp=args.max_absolute_ccp,
    )
    print(f"Exported incremental cost--reliability frontier to {paths['source_data'].parent}")


if __name__ == "__main__":
    main()
