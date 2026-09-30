"""Plot 2%-CCP target sizing points from completed fine-resolution data.

The default input is the completed 0.25 kW by 0.25 kWh adaptive CCP--LOLP
sizing result with 2% CCP targets. It only reads validated output and never
reruns dispatch, scenarios or capacity sizing.

Run with ``python -m src.figures.ccp_lolp_target_scatter_plots``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .capacity_sweep_contours import CAPACITY_KEYS, file_sha256
from .ccp_3d_surfaces import _completed_records
from .ccp_lolp_target_line_plots import (
    DEFAULT_LOLP_THRESHOLDS,
    PANEL_SPECS,
    SOURCE_COLUMNS,
    _normalise_probabilities,
    _tick_decimals,
    _threshold_color,
    _threshold_label,
    load_completed_fine_curves,
)
from .core import apply_nature_style, matplotlib_context, save_publication_figure, write_source_data


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_Capacity_Sweep_15Y_100Scenarios_Cost_SSR_SCR"
DEFAULT_SIZING_RESULTS = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_CCP_LOLP_Sizing_0p25_Adaptive_CCP2pct"
DEFAULT_CCP_TARGETS = (*tuple(value / 100.0 for value in range(50, 96, 2)), 0.95)
SOURCE_OUTPUT_STEM = "CCP2pct_Scatter"
MARKER_SIZE_POINTS_SQUARED = 22.0
FIT_CURVE_SAMPLES = 200
AXES_ASPECT_RATIO = 16.0 / 9.0
FIGURE_SIZE_INCHES = (7.2, 5.1)
AXES_LEFT = 0.145
AXES_RIGHT = 0.98
AXES_BOTTOM = 0.16
AXES_WIDTH_FRACTION = AXES_RIGHT - AXES_LEFT
AXES_HEIGHT_FRACTION = (
    AXES_WIDTH_FRACTION * FIGURE_SIZE_INCHES[0] / AXES_ASPECT_RATIO / FIGURE_SIZE_INCHES[1]
)
AXES_TOP = AXES_BOTTOM + AXES_HEIGHT_FRACTION
X_MAJOR_TICKS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
QUADRATIC_FIT_METRICS = frozenset({"bess", "annual_cost"})
DISPLAY_SPECS = {
    "pv": {
        "title": "Cost-optimal PV capacity on a discrete grid",
        "y_limits": (2.9, 4.6),
        "y_ticks": (3.0, 3.5, 4.0, 4.5),
        "y_tick_step": 0.5,
    },
    "bess": {
        "title": "Optimal BESS energy",
        "y_limits": (11.0, 17.5),
        "y_ticks": (11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0),
        "y_tick_step": 1.0,
    },
    "annual_cost": {
        "title": "Mean annual system cost",
        "y_limits": (1.9, 2.3),
        "y_ticks": (1.9, 2.0, 2.1, 2.2, 2.3),
        "y_tick_step": 0.1,
    },
}


def _panel_spec(panel_key: str) -> dict[str, Any]:
    for panel in PANEL_SPECS:
        if panel["key"] == panel_key:
            return {**panel, **DISPLAY_SPECS[panel_key]}
    available = ", ".join(str(panel["key"]) for panel in PANEL_SPECS)
    raise ValueError(f"Unknown CCP-target scatter panel {panel_key!r}; choose one of {available}")


def _scatter_output_stem(panel_key: str) -> str:
    return {
        "pv": "CCP2pct_Scatter_Optimal_PV",
        "bess": "CCP2pct_Scatter_Optimal_BESS",
        "annual_cost": "CCP2pct_Scatter_Mean_Annual_Cost",
    }[panel_key]


def _quadratic_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit y = aX² + bX + c using CCP percentage values, without extrapolation."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != y_values.shape or x_values.size < 3:
        raise ValueError("Quadratic fitting requires at least three paired CCP-target points")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if x_values.size < 3:
        raise ValueError("Quadratic fitting requires at least three finite CCP-target points")
    x_percent = x_values * 100.0
    coefficients = np.polyfit(x_percent, y_values, deg=2)
    fitted = np.polyval(coefficients, x_percent)
    residual_sum_squares = float(np.sum((y_values - fitted) ** 2))
    total_sum_squares = float(np.sum((y_values - y_values.mean()) ** 2))
    r_squared = 1.0 if total_sum_squares == 0.0 else float(1.0 - residual_sum_squares / total_sum_squares)
    x_curve_percent = np.linspace(float(x_percent.min()), float(x_percent.max()), FIT_CURVE_SAMPLES)
    return {
        "formula": "y = aX^2 + bX + c",
        "x_variable": "CCP target (%)",
        "a": float(coefficients[0]),
        "b": float(coefficients[1]),
        "c": float(coefficients[2]),
        "r_squared": r_squared,
        "sample_count": int(x_values.size),
        "x_min_percent": float(x_curve_percent[0]),
        "x_max_percent": float(x_curve_percent[-1]),
        "x_curve": (x_curve_percent / 100.0).tolist(),
        "y_curve": np.polyval(coefficients, x_curve_percent).astype(float).tolist(),
    }


def quadratic_fit_summary(figure_metadata: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Return a clean source-data table for all rendered quadratic trend fits."""
    rows: list[dict[str, Any]] = []
    for metric, metadata in figure_metadata.items():
        for series in metadata["panel"]["series"]:
            fit = series.get("quadratic_fit")
            if fit is None:
                continue
            rows.append(
                {
                    "metric": metric,
                    "lolp_constraint": float(series["lolp_constraint"]),
                    "formula": fit["formula"],
                    "x_variable": fit["x_variable"],
                    "a": float(fit["a"]),
                    "b": float(fit["b"]),
                    "c": float(fit["c"]),
                    "r_squared": float(fit["r_squared"]),
                    "sample_count": int(fit["sample_count"]),
                    "x_min_percent": float(fit["x_min_percent"]),
                    "x_max_percent": float(fit["x_max_percent"]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "metric", "lolp_constraint", "formula", "x_variable", "a", "b", "c", "r_squared",
            "sample_count", "x_min_percent", "x_max_percent",
        ],
    ).sort_values(["metric", "lolp_constraint"], kind="stable").reset_index(drop=True)


def build_metric_figure(
    curves: pd.DataFrame,
    *,
    panel_key: str,
    city_label: str = "Brisbane",
    out: Path | str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Render one 2%-CCP scatter figure, with quadratic trends where specified."""
    required = set(SOURCE_COLUMNS)
    missing = sorted(required - set(curves.columns))
    if missing:
        raise ValueError(f"CCP-target scatter source is missing columns: {missing}")
    if curves.empty:
        raise ValueError("CCP-target scatter source is empty")
    thresholds = tuple(sorted(curves.lolp_constraint.astype(float).unique().tolist()))
    targets = tuple(sorted(curves.ccp_target.astype(float).unique().tolist()))
    if not thresholds or not targets:
        raise ValueError("CCP-target scatter source requires thresholds and targets")
    panel = _panel_spec(panel_key)

    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    mpl.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.6,
            "axes.titlesize": 9.4,
            "legend.fontsize": 7.3,
        }
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    fig.subplots_adjust(left=AXES_LEFT, right=AXES_RIGHT, bottom=AXES_BOTTOM, top=AXES_TOP)
    fig.suptitle(
        f"{city_label}: {panel['title']} under fixed LOLP constraints",
        x=0.02,
        ha="left",
        y=0.977,
        fontsize=10.2,
    )

    scatter_handles: list[Any] = []
    scatter_labels: list[str] = []
    series_metadata: list[dict[str, Any]] = []
    for threshold_index, threshold in enumerate(thresholds):
        series = curves.loc[
            np.isclose(curves.lolp_constraint.astype(float), threshold, rtol=0.0, atol=1e-12)
            & curves.feasible.astype(bool)
        ].sort_values("ccp_target")
        x = series.ccp_target.to_numpy(dtype=float)
        y = series[str(panel["column"])].to_numpy(dtype=float) * float(panel["scale"])
        if x.size == 0:
            continue
        color = _threshold_color(threshold, threshold_index)
        fit: dict[str, Any] | None = None
        if panel_key in QUADRATIC_FIT_METRICS:
            fit = _quadratic_fit(x, y)
            line_width = 2.2 if panel_key == "annual_cost" else 1.55
            ax.plot(
                np.asarray(fit["x_curve"], dtype=float),
                np.asarray(fit["y_curve"], dtype=float),
                color=color,
                linewidth=line_width,
                alpha=0.95,
                solid_capstyle="round",
                zorder=1,
                clip_on=True,
            )
        scatter = ax.scatter(
            x,
            y,
            s=MARKER_SIZE_POINTS_SQUARED,
            marker="o",
            facecolors=color,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.82 if fit is not None else 1.0,
            zorder=3,
            clip_on=True,
            label=_threshold_label(threshold),
        )
        scatter_handles.append(scatter)
        scatter_labels.append(_threshold_label(threshold))
        series_metadata.append(
            {
                "lolp_constraint": float(threshold),
                "color": color,
                "point_count": int(x.size),
                "x_values": [float(value) for value in x],
                "y_values": [float(value) for value in y],
                "marker": "o",
                "marker_size_points_squared": MARKER_SIZE_POINTS_SQUARED,
                "quadratic_fit": fit,
            }
        )

    y_lower, y_upper = (float(value) for value in panel["y_limits"])
    decimals = _tick_decimals(float(panel["y_tick_step"]))
    ax.set_ylabel(str(panel["label"]))
    ax.set_xlabel("CCP target (%)")
    ax.set_xlim(0.495, 0.96)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xticks(X_MAJOR_TICKS)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.get_xticklabels()[-1].set_horizontalalignment("left")
    ax.set_yticks(panel["y_ticks"])
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(f"%.{decimals}f"))
    ax.tick_params(direction="out", length=3.0, width=0.65, pad=2.2)
    legend = fig.legend(
        scatter_handles,
        scatter_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.904),
        ncol=len(scatter_handles),
        borderaxespad=0.0,
        handletextpad=0.55,
        columnspacing=1.5,
    )
    if panel_key in QUADRATIC_FIT_METRICS:
        fig.text(
            0.5,
            0.852,
            "Points: selected optima; curves: quadratic least-squares fit",
            ha="center",
            va="center",
            fontsize=6.4,
            color="#4D4D4D",
        )

    axes_position = ax.get_position()
    axes_width_inches = float(axes_position.width * FIGURE_SIZE_INCHES[0])
    axes_height_inches = float(axes_position.height * FIGURE_SIZE_INCHES[1])

    metadata = {
        "title": fig._suptitle.get_text() if fig._suptitle is not None else "",
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "layout": "single-panel CCP-target plot with a 16:9 plotting area",
        "metric": str(panel["key"]),
        "axes_position": [
            float(axes_position.x0), float(axes_position.y0), float(axes_position.width), float(axes_position.height),
        ],
        "axes_size_inches": [axes_width_inches, axes_height_inches],
        "axes_aspect_ratio": axes_width_inches / axes_height_inches,
        "x_axis": ax.get_xlabel(),
        "x_values": [float(value) for value in targets],
        "x_limits": [float(value) for value in ax.get_xlim()],
        "x_tick_labels": [label.get_text() for label in ax.get_xticklabels()],
        "thresholds": [float(value) for value in thresholds],
        "colors": {
            _threshold_label(threshold): _threshold_color(threshold, index)
            for index, threshold in enumerate(thresholds)
        },
        "legend_labels": [text.get_text() for text in legend.get_texts()],
        "panel": {
            "title": str(panel["title"]),
            "y_axis": ax.get_ylabel(),
            "y_limits": [float(value) for value in ax.get_ylim()],
            "y_tick_step": float(panel["y_tick_step"]),
            "y_tick_labels": [label.get_text() for label in ax.get_yticklabels()],
            "series": series_metadata,
        },
        "plot_type": "scatter",
        "source_points_connected": False,
        "quadratic_trend_lines": panel_key in QUADRATIC_FIT_METRICS,
        "smoothing": False,
        "interpolation": False,
        "extrapolation": False,
        "marker_size_points_squared": MARKER_SIZE_POINTS_SQUARED,
        "major_x_ticks": [float(value) for value in X_MAJOR_TICKS],
        "dpi": 600,
    }
    return fig, metadata


def _readme_text(
    case_key: str,
    thresholds: Sequence[float],
    targets: Sequence[float],
    fine_precision: dict[str, Any],
    figure_metadata: dict[str, dict[str, Any]],
    fit_summary_name: str,
) -> str:
    threshold_text = ", ".join(_threshold_label(float(value)) for value in thresholds)
    target_text = ", ".join(f"{float(value) * 100:g}%" for value in targets)
    axis_text = "\n".join(
        "  - "
        f"{metadata['panel']['title']}: {metadata['panel']['y_limits'][0]:.4g} to "
        f"{metadata['panel']['y_limits'][1]:.4g}; major tick "
        f"{metadata['panel']['y_tick_step']:g}."
        for metadata in figure_metadata.values()
    )
    return f"""# {case_key}: 2%-CCP target scatter plots

This figure bundle is derived from the completed, validated adaptive CCP--LOLP
sizing result. It does not resample outage scenarios, rerun dispatch or rerun
the sizing search.

- The x axis contains the requested CCP targets: {target_text}.
- Each coloured scatter series fixes one strict LOLP constraint: {threshold_text}.
- CCP is calculated as `P(LOLP < threshold)`; a scenario exactly equal to a
  threshold does not pass.
- The input sizing result has final grid precision of {float(fine_precision['pv_kw']):g} kW PV and
  {float(fine_precision['battery_kwh']):g} kWh BESS. Its selected point minimises the arithmetic
  mean `annual_cost_sgd_per_year` across the shared scenarios; exact cost ties select lower PV,
  then lower BESS.
- The bundle contains three independent plots with 16:9 plotting areas: optimal PV capacity,
  optimal BESS energy, and mean annual system cost.
- PV points show discrete cost-optimal capacity choices without a fitted curve. BESS and
  annual-cost plots add same-colour quadratic least-squares trends using all source points;
  fitted curves remain within the observed CCP range and are descriptive only.
- Major CCP labels are 50%, 60%, 70%, 80%, 90% and 95%; all 24 original points remain plotted.
- The fixed display ranges and major y-axis ticks are:
{axis_text}

The Source Data CSV records targets, achieved CCP, passing-scenario counts,
feasibility, selected PV/BESS capacities and annual costs. `{fit_summary_name}` records
the quadratic coefficients and R². The validation JSON records input and output
checksums, the strict comparison and selection rule, and the exported figure metadata.
"""


def run(
    results_dir: Path | str = DEFAULT_RESULTS,
    output_dir: Path | str | None = None,
    *,
    sizing_results_dir: Path | str = DEFAULT_SIZING_RESULTS,
    lolp_thresholds: Sequence[float] = DEFAULT_LOLP_THRESHOLDS,
    ccp_targets: Sequence[float] = DEFAULT_CCP_TARGETS,
) -> dict[str, Any]:
    """Export three 2%-CCP scatter figures from validated fine sizing data."""
    results_path = Path(results_dir).resolve()
    records, _source, manifest, input_hashes = _completed_records(results_path)
    sizing_path = Path(sizing_results_dir).resolve()
    curves, fine_manifest, fine_validation = load_completed_fine_curves(
        sizing_path,
        lolp_thresholds=lolp_thresholds,
        ccp_targets=ccp_targets,
    )
    fine_source = Path(str(fine_manifest.get("source", {}).get("directory", ""))).resolve()
    if fine_source != results_path:
        raise ValueError("Fine sizing result does not reference the requested completed capacity sweep")
    for name, expected_hash in fine_manifest.get("source", {}).get("file_hashes", {}).items():
        if file_sha256(results_path / name) != expected_hash:
            raise ValueError(f"Fine sizing source changed since completion: {name}")
    fine_precision = dict(fine_validation.get("fine_precision", {}))
    if not {"pv_kw", "battery_kwh"}.issubset(fine_precision):
        raise ValueError("Fine sizing validation does not record PV and BESS precision")

    thresholds = _normalise_probabilities(lolp_thresholds, "LOLP threshold", allow_one=False)
    targets = _normalise_probabilities(ccp_targets, "CCP target", allow_one=True)
    output_path = Path(output_dir).resolve() if output_dir else sizing_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    case_key = str(manifest["spec"]["case_key"])
    source_path = write_source_data(output_path / f"{case_key}_{SOURCE_OUTPUT_STEM}", curves)
    paths: dict[str, Any] = {"source_data": source_path}
    figure_metadata: dict[str, dict[str, Any]] = {}
    _, plt = matplotlib_context(output_path)
    for panel in PANEL_SPECS:
        panel_key = str(panel["key"])
        base = output_path / f"{case_key}_{_scatter_output_stem(panel_key)}"
        figure, metadata = build_metric_figure(
            curves,
            panel_key=panel_key,
            city_label=case_key.split("_", 1)[0],
            out=output_path,
        )
        try:
            paths[panel_key] = save_publication_figure(figure, base)
        finally:
            plt.close(figure)
        figure_metadata[panel_key] = metadata

    fit_summary = quadratic_fit_summary(figure_metadata)
    fit_summary_path = output_path / f"{case_key}_CCP2pct_Quadratic_Fit_Summary.csv"
    fit_summary.to_csv(fit_summary_path, index=False)
    paths["fit_summary"] = fit_summary_path

    for name, expected_hash in input_hashes.items():
        if file_sha256(results_path / name) != expected_hash:
            raise RuntimeError(f"Input changed while plotting: {name}")
    readme_path = output_path / f"{SOURCE_OUTPUT_STEM}_README.md"
    readme_path.write_text(
        _readme_text(case_key, thresholds, targets, fine_precision, figure_metadata, fit_summary_path.name),
        encoding="utf-8",
    )
    paths["readme"] = readme_path
    output_files = [source_path, fit_summary_path, readme_path]
    output_files.extend(path for value in paths.values() if isinstance(value, dict) for path in value.values())
    report = {
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_key": case_key,
        "scenario_rows": int(len(records)),
        "capacity_combinations": int(records[CAPACITY_KEYS].drop_duplicates().shape[0]),
        "scenarios_per_capacity": sorted(records.groupby(CAPACITY_KEYS).size().unique().tolist()),
        "analysis_years": int(manifest["spec"]["analysis_years"]),
        "raw_checksums_match_completed_run": True,
        "input_files_unchanged": True,
        "lolp_thresholds": [float(value) for value in thresholds],
        "ccp_targets": [float(value) for value in targets],
        "comparison": "LOLP < threshold",
        "figure_input": {"kind": "adaptive_fine_sizing_2pct", "directory": str(sizing_path)},
        "fine_precision": fine_precision,
        "selection_rule": (
            "Read directly from the completed adaptive fine sizing curve: minimum arithmetic mean "
            "annual_cost_sgd_per_year among feasible candidates; ties resolve by lower PV then lower BESS"
        ),
        "feasible_curve_points": int(curves.feasible.sum()),
        "infeasible_curve_points": int((~curves.feasible).sum()),
        "input_sha256": input_hashes,
        "fine_curve_sha256": file_sha256(sizing_path / "ccp_target_curve_source.csv"),
        "figures": figure_metadata,
        "quadratic_fit_summary": fit_summary.to_dict(orient="records"),
        "output_sha256": {path.name: file_sha256(path) for path in output_files},
    }
    report_path = output_path / "ccp2pct_scatter_validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    paths["validation"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Completed economic capacity-sweep directory referenced by the sizing result")
    parser.add_argument("--sizing-results-dir", type=Path, default=DEFAULT_SIZING_RESULTS,
                        help="Completed adaptive 2%%-CCP sizing directory used as the figure source")
    parser.add_argument("--output-dir", type=Path,
                        help="Figure output directory (default: sizing-results-dir/figures)")
    parser.add_argument("--lolp-thresholds", nargs="+", type=float, default=list(DEFAULT_LOLP_THRESHOLDS),
                        help="Strict LOLP thresholds as fractions (default: 0.10 0.20 0.30)")
    parser.add_argument("--ccp-targets", nargs="+", type=float, default=list(DEFAULT_CCP_TARGETS),
                        help="CCP targets as fractions (default: 0.50--0.94 by 0.02, then 0.95)")
    args = parser.parse_args()
    paths = run(
        args.results_dir,
        args.output_dir,
        sizing_results_dir=args.sizing_results_dir,
        lolp_thresholds=tuple(args.lolp_thresholds),
        ccp_targets=tuple(args.ccp_targets),
    )
    print(f"Exported 2%-CCP scatter figures to {paths['source_data'].parent}")


if __name__ == "__main__":
    main()
