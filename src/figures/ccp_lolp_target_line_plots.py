"""Plot CCP-target sizing curves from completed, fine-resolution sizing data.

The default input is the completed 0.25 kW by 0.25 kWh adaptive CCP--LOLP
sizing result. It never reruns the sizing search, outage scenarios or dispatch.

Run with ``python -m src.figures.ccp_lolp_target_line_plots``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .capacity_sweep_contours import CAPACITY_KEYS, file_sha256
from .ccp_3d_surfaces import _completed_records
from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    matplotlib_context,
    plot_smooth_line,
    save_publication_figure,
    write_source_data,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_Capacity_Sweep_15Y_100Scenarios_Cost_SSR_SCR"
DEFAULT_SIZING_RESULTS = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_CCP_LOLP_Sizing_0p25_Adaptive"
DEFAULT_LOLP_THRESHOLDS = (0.10, 0.20, 0.30)
DEFAULT_CCP_TARGETS = tuple(value / 100.0 for value in range(50, 100, 5))
FIGURE_SIZE_INCHES = (7.2, 4.05)
SMOOTH_SAMPLES = 200
SOURCE_OUTPUT_STEM = "CCP_Target_Line_Plots"
PANEL_SPECS = (
    {
        "key": "pv",
        "column": "optimal_pv_kw",
        "label": "Optimal PV capacity (kW)",
        "title": "Optimal PV capacity",
        "scale": 1.0,
        "tick_step": 0.25,
        "output_stem": "CCP_Target_Optimal_PV",
    },
    {
        "key": "bess",
        "column": "optimal_battery_kwh",
        "label": "Optimal BESS energy (kWh)",
        "title": "Optimal BESS energy",
        "scale": 1.0,
        "tick_step": 0.5,
        "output_stem": "CCP_Target_Optimal_BESS",
    },
    {
        "key": "annual_cost",
        "column": "mean_annual_cost_sgd_per_year",
        "label": "Mean annual system cost (kSGD yr$^{-1}$)",
        "title": "Mean annual system cost",
        "scale": 1.0 / 1000.0,
        "tick_step": 0.05,
        "output_stem": "CCP_Target_Mean_Annual_Cost",
    },
)
THRESHOLD_COLORS = {
    0.10: NATURE_PALETTE["blue"],
    0.20: NATURE_PALETTE["purple"],
    0.30: NATURE_PALETTE["gold"],
}
FALLBACK_COLORS = (
    NATURE_PALETTE["blue"],
    NATURE_PALETTE["purple"],
    NATURE_PALETTE["gold"],
    NATURE_PALETTE["red"],
    NATURE_PALETTE["teal"],
)
SOURCE_COLUMNS = (
    "lolp_constraint",
    "ccp_target",
    "ccp_target_percent",
    "comparison",
    "required_passing_scenarios",
    "actual_ccp",
    "actual_ccp_percent",
    "passing_scenarios",
    "feasible",
    "optimal_pv_kw",
    "optimal_battery_kwh",
    "mean_annual_cost_sgd_per_year",
    "mean_annual_cost_ksgd_per_year",
)
FINE_CURVE_REQUIRED_COLUMNS = {
    "lolp_constraint",
    "ccp_target",
    "comparison",
    "required_passing_scenarios",
    "actual_ccp",
    "passing_scenarios",
    "feasible",
    "optimal_pv_kw",
    "optimal_battery_kwh",
    "mean_annual_cost_sgd_per_year",
}


def _normalise_probabilities(values: Iterable[float], label: str, *, allow_one: bool) -> tuple[float, ...]:
    normalised = tuple(sorted(float(value) for value in values))
    if not normalised:
        raise ValueError(f"At least one {label} is required")
    upper_ok = (lambda value: value <= 1.0) if allow_one else (lambda value: value < 1.0)
    if len(set(normalised)) != len(normalised) or not all(
        np.isfinite(value) and 0.0 < value and upper_ok(value) for value in normalised
    ):
        bound = "(0, 1]" if allow_one else "(0, 1)"
        raise ValueError(f"{label.capitalize()} values must be unique finite probabilities in {bound}")
    return normalised


def _required_passing_scenarios(target: float, scenario_count: int) -> int:
    """Return the exact integer count needed to satisfy a CCP target."""
    return int(math.ceil(float(target) * int(scenario_count) - 1e-12))


def _validate_selection_records(records: pd.DataFrame) -> int:
    required = set(CAPACITY_KEYS + ["scenario_id", "LOLP", "annual_cost_sgd_per_year"])
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"CCP target selection is missing scenario columns: {missing}")
    if records.empty:
        raise ValueError("CCP target selection requires non-empty scenario records")
    numeric = CAPACITY_KEYS + ["scenario_id", "LOLP", "annual_cost_sgd_per_year"]
    if not np.isfinite(records[numeric].to_numpy(dtype=float)).all():
        raise ValueError("CCP target selection requires finite capacity, LOLP and cost values")
    if records.LOLP.lt(0.0).any() or records.LOLP.gt(1.0).any():
        raise ValueError("LOLP values must be probabilities between zero and one")
    if records.duplicated(CAPACITY_KEYS + ["scenario_id"]).any():
        raise ValueError("CCP target selection found duplicate capacity/scenario keys")

    scenario_ids: tuple[int, ...] | None = None
    counts: list[int] = []
    for _, group in records.groupby(CAPACITY_KEYS, sort=True):
        identifiers = tuple(sorted(group.scenario_id.astype(int).tolist()))
        if scenario_ids is None:
            scenario_ids = identifiers
        elif identifiers != scenario_ids:
            raise ValueError("Every capacity point must use the same scenario identifiers")
        counts.append(len(identifiers))
    if scenario_ids is None or not counts or counts[0] == 0:
        raise ValueError("CCP target selection requires scenarios at every capacity point")
    return counts[0]


def select_target_designs(
    records: pd.DataFrame,
    *,
    lolp_thresholds: Sequence[float] = DEFAULT_LOLP_THRESHOLDS,
    ccp_targets: Sequence[float] = DEFAULT_CCP_TARGETS,
) -> pd.DataFrame:
    """Return one lowest-cost design per strict LOLP constraint and CCP target.

    For each fixed constraint the candidates are evaluated as ``P(LOLP <
    threshold)``.  Among candidates that meet a target, the arithmetic mean
    annual system cost is minimised; PV and then BESS capacity resolve exact
    cost ties.  No interpolation or capacity optimisation is performed.
    """
    thresholds = _normalise_probabilities(lolp_thresholds, "LOLP threshold", allow_one=False)
    targets = _normalise_probabilities(ccp_targets, "CCP target", allow_one=True)
    scenario_count = _validate_selection_records(records)
    grouped_cost = records.groupby(CAPACITY_KEYS, as_index=False, sort=True).agg(
        mean_annual_cost_sgd_per_year=("annual_cost_sgd_per_year", "mean"),
    )

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        passing = records.assign(_ccp_pass=records.LOLP.lt(threshold)).groupby(
            CAPACITY_KEYS, as_index=False, sort=True
        )["_ccp_pass"].sum().rename(columns={"_ccp_pass": "passing_scenarios"})
        candidates = grouped_cost.merge(passing, on=CAPACITY_KEYS, validate="one_to_one")
        candidates["passing_scenarios"] = candidates["passing_scenarios"].astype(int)
        candidates["actual_ccp"] = candidates["passing_scenarios"] / scenario_count

        for target in targets:
            required = _required_passing_scenarios(target, scenario_count)
            feasible = candidates.loc[candidates.passing_scenarios.ge(required)].sort_values(
                ["mean_annual_cost_sgd_per_year", "pv_capacity_kw", "battery_energy_kwh"],
                kind="stable",
            )
            if feasible.empty:
                rows.append(
                    {
                        "lolp_constraint": threshold,
                        "ccp_target": target,
                        "ccp_target_percent": target * 100.0,
                        "comparison": "<",
                        "required_passing_scenarios": required,
                        "actual_ccp": np.nan,
                        "actual_ccp_percent": np.nan,
                        "passing_scenarios": 0,
                        "feasible": False,
                        "optimal_pv_kw": np.nan,
                        "optimal_battery_kwh": np.nan,
                        "mean_annual_cost_sgd_per_year": np.nan,
                        "mean_annual_cost_ksgd_per_year": np.nan,
                    }
                )
                continue

            selected = feasible.iloc[0]
            actual_ccp = float(selected.actual_ccp)
            rows.append(
                {
                    "lolp_constraint": threshold,
                    "ccp_target": target,
                    "ccp_target_percent": target * 100.0,
                    "comparison": "<",
                    "required_passing_scenarios": required,
                    "actual_ccp": actual_ccp,
                    "actual_ccp_percent": actual_ccp * 100.0,
                    "passing_scenarios": int(selected.passing_scenarios),
                    "feasible": True,
                    "optimal_pv_kw": float(selected.pv_capacity_kw),
                    "optimal_battery_kwh": float(selected.battery_energy_kwh),
                    "mean_annual_cost_sgd_per_year": float(selected.mean_annual_cost_sgd_per_year),
                    "mean_annual_cost_ksgd_per_year": float(selected.mean_annual_cost_sgd_per_year) / 1000.0,
                }
            )

    curves = pd.DataFrame(rows, columns=SOURCE_COLUMNS).sort_values(
        ["lolp_constraint", "ccp_target"], kind="stable"
    ).reset_index(drop=True)
    if curves.duplicated(["lolp_constraint", "ccp_target"]).any():
        raise RuntimeError("CCP target selection produced duplicate curve points")
    feasible = curves.loc[curves.feasible]
    if not feasible.empty and (feasible.actual_ccp + 1e-12 < feasible.ccp_target).any():
        raise RuntimeError("A selected design does not meet its CCP target")
    return curves


def _normalise_fine_curve_source(source: pd.DataFrame) -> pd.DataFrame:
    """Validate a completed sizing curve and add publication display columns."""
    missing = sorted(FINE_CURVE_REQUIRED_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"Fine sizing curve is missing required columns: {missing}")
    if source.empty:
        raise ValueError("Fine sizing curve is empty")
    curves = source.copy()
    if curves.duplicated(["lolp_constraint", "ccp_target"]).any():
        raise ValueError("Fine sizing curve has duplicate LOLP/CCP target pairs")
    numeric = [
        "lolp_constraint", "ccp_target", "required_passing_scenarios", "actual_ccp",
        "passing_scenarios", "optimal_pv_kw", "optimal_battery_kwh", "mean_annual_cost_sgd_per_year",
    ]
    for column in numeric:
        curves[column] = pd.to_numeric(curves[column], errors="coerce")
    if not curves["comparison"].eq("<").all():
        raise ValueError("Fine sizing curve must use the strict LOLP < threshold comparison")
    feasible = curves["feasible"]
    if not feasible.isin([True, False]).all():
        raise ValueError("Fine sizing curve feasibility flags must be boolean")
    curves["feasible"] = feasible.astype(bool)
    selected = curves.loc[curves.feasible]
    if selected.empty:
        raise ValueError("Fine sizing curve has no feasible target points")
    if not np.isfinite(selected[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Fine sizing curve has non-finite values for a feasible target point")
    if not selected.lolp_constraint.between(0.0, 1.0, inclusive="neither").all():
        raise ValueError("Fine sizing LOLP thresholds must be probabilities strictly between zero and one")
    if not selected.ccp_target.between(0.0, 1.0, inclusive="right").all():
        raise ValueError("Fine sizing CCP targets must be probabilities in (0, 1]")
    if not selected.actual_ccp.between(0.0, 1.0).all() or (selected.actual_ccp + 1e-12 < selected.ccp_target).any():
        raise ValueError("Fine sizing curve contains a selected design below its CCP target")
    if (selected.passing_scenarios < selected.required_passing_scenarios).any():
        raise ValueError("Fine sizing curve passing counts do not meet their targets")

    curves["ccp_target_percent"] = curves["ccp_target"] * 100.0
    curves["actual_ccp_percent"] = curves["actual_ccp"] * 100.0
    curves["mean_annual_cost_ksgd_per_year"] = curves["mean_annual_cost_sgd_per_year"] / 1000.0
    return curves.reindex(columns=[*SOURCE_COLUMNS, *[column for column in curves.columns if column not in SOURCE_COLUMNS]]).sort_values(
        ["lolp_constraint", "ccp_target"], kind="stable"
    ).reset_index(drop=True)


def load_completed_fine_curves(
    sizing_results_dir: Path | str,
    *,
    lolp_thresholds: Sequence[float] = DEFAULT_LOLP_THRESHOLDS,
    ccp_targets: Sequence[float] = DEFAULT_CCP_TARGETS,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load only a complete, checksum-validated fine CCP--LOLP sizing curve."""
    sizing_path = Path(sizing_results_dir).resolve()
    manifest_path = sizing_path / "manifest.json"
    status_path = sizing_path / "run_status.json"
    validation_path = sizing_path / "validation.json"
    curve_path = sizing_path / "ccp_target_curve_source.csv"
    required_paths = (manifest_path, status_path, validation_path, curve_path)
    missing = [path.name for path in required_paths if not path.is_file()]
    if missing:
        raise ValueError(f"Completed fine sizing input is missing files: {missing}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if status.get("state") != "complete" or validation.get("passed") is not True:
        raise ValueError("Only a completed, validated fine sizing result can be plotted")
    expected_hash = validation.get("file_sha256", {}).get(curve_path.name)
    if not expected_hash or file_sha256(curve_path) != expected_hash:
        raise ValueError("Fine sizing curve checksum does not match its validation report")

    curves = _normalise_fine_curve_source(pd.read_csv(curve_path))
    thresholds = _normalise_probabilities(lolp_thresholds, "LOLP threshold", allow_one=False)
    targets = _normalise_probabilities(ccp_targets, "CCP target", allow_one=True)
    selected = curves.copy()
    for column, expected_values, label in (
        ("lolp_constraint", thresholds, "LOLP threshold"),
        ("ccp_target", targets, "CCP target"),
    ):
        matched = np.zeros(len(selected), dtype=bool)
        aligned: list[float] = []
        for index, value in enumerate(selected[column].to_numpy(dtype=float)):
            matches = [expected for expected in expected_values if np.isclose(value, expected, rtol=0.0, atol=1e-12)]
            if len(matches) == 1:
                matched[index] = True
                aligned.append(matches[0])
        selected = selected.loc[matched].copy()
        selected[column] = aligned
    expected = pd.MultiIndex.from_product([thresholds, targets], names=["lolp_constraint", "ccp_target"])
    actual = pd.MultiIndex.from_frame(selected[["lolp_constraint", "ccp_target"]])
    if not actual.equals(expected):
        raise ValueError("Fine sizing curve does not contain exactly the requested LOLP/CCP target matrix")
    return selected.reset_index(drop=True), manifest, validation


def _threshold_label(threshold: float) -> str:
    return f"LOLP < {threshold * 100:g}%"


def _threshold_color(threshold: float, index: int) -> str:
    for key, color in THRESHOLD_COLORS.items():
        if np.isclose(threshold, key, rtol=0.0, atol=1e-12):
            return color
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def _panel_spec(panel_key: str) -> dict[str, Any]:
    for panel in PANEL_SPECS:
        if panel["key"] == panel_key:
            return panel
    available = ", ".join(str(panel["key"]) for panel in PANEL_SPECS)
    raise ValueError(f"Unknown CCP-target panel {panel_key!r}; choose one of {available}")


def _tight_scale(values: np.ndarray, *, tick_step: float) -> tuple[float, float]:
    """Return a data-focused scale with five-percent padding on either side."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("A CCP target panel requires at least one feasible plotted value")
    lower = float(finite.min())
    upper = float(finite.max())
    span = upper - lower
    padding = 0.05 * span if span > 0.0 else max(float(tick_step) * 0.5, abs(lower) * 0.05)
    return lower - padding, upper + padding


def _tick_decimals(tick_step: float) -> int:
    rendered = f"{float(tick_step):.12f}".rstrip("0")
    return len(rendered.partition(".")[2])


def build_metric_figure(
    curves: pd.DataFrame,
    *,
    panel_key: str,
    city_label: str = "Brisbane",
    out: Path | str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Render one marker-free, PCHIP-smoothed CCP-target sizing figure."""
    required = set(SOURCE_COLUMNS)
    missing = sorted(required - set(curves.columns))
    if missing:
        raise ValueError(f"CCP target curve source is missing columns: {missing}")
    if curves.empty:
        raise ValueError("CCP target curve source is empty")
    thresholds = tuple(sorted(curves.lolp_constraint.astype(float).unique().tolist()))
    targets = tuple(sorted(curves.ccp_target.astype(float).unique().tolist()))
    if not thresholds or not targets:
        raise ValueError("CCP target curve source requires thresholds and targets")
    panel = _panel_spec(panel_key)

    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    mpl.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.titlesize": 9.4,
            "legend.fontsize": 7.3,
            "lines.linewidth": 1.5,
        }
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    fig.subplots_adjust(left=0.145, right=0.98, bottom=0.18, top=0.745)
    fig.suptitle(
        f"{city_label}: {panel['title']} under fixed LOLP constraints",
        x=0.02,
        ha="left",
        y=0.972,
        fontsize=10.2,
    )

    line_handles: list[Any] = []
    line_labels: list[str] = []
    all_values: list[float] = []
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
        if x.size < 3:
            raise ValueError("Each CCP-target series needs at least three points for PCHIP smoothing")
        color = _threshold_color(threshold, threshold_index)
        line = plot_smooth_line(
            ax,
            x,
            y,
            samples=SMOOTH_SAMPLES,
            show_points=False,
            color=color,
            linewidth=1.5,
            solid_capstyle="round",
            clip_on=True,
            label=_threshold_label(threshold),
        )
        line_handles.append(line)
        line_labels.append(_threshold_label(threshold))
        all_values.extend(y.tolist())
        series_metadata.append(
            {
                "lolp_constraint": float(threshold),
                "color": color,
                "source_point_count": int(x.size),
                "smooth_sample_count": int(len(line.get_xdata())),
                "x_values": [float(value) for value in x],
                "y_values": [float(value) for value in y],
                "marker": None,
            }
        )

    y_lower, y_upper = _tight_scale(np.asarray(all_values, dtype=float), tick_step=float(panel["tick_step"]))
    decimals = _tick_decimals(float(panel["tick_step"]))
    ax.set_ylabel(str(panel["label"]))
    ax.set_xlabel("CCP target (%)")
    ax.set_xlim(float(targets[0]), float(targets[-1]))
    ax.set_ylim(y_lower, y_upper)
    ax.set_xticks(targets)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(float(panel["tick_step"])))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(f"%.{decimals}f"))
    ax.tick_params(direction="out", length=3.0, width=0.65, pad=2.2)
    legend = fig.legend(
        line_handles,
        line_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.900),
        ncol=len(line_handles),
        borderaxespad=0.0,
        handlelength=2.4,
        columnspacing=1.5,
    )

    metadata = {
        "title": fig._suptitle.get_text() if fig._suptitle is not None else "",
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "layout": "single-panel 16:9 CCP-target line plot",
        "metric": str(panel["key"]),
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
            "y_tick_step": float(panel["tick_step"]),
            "y_tick_labels": [label.get_text() for label in ax.get_yticklabels()],
            "series": series_metadata,
        },
        "smoothing": {
            "method": "shape-preserving PCHIP",
            "samples_per_series": SMOOTH_SAMPLES,
            "extrapolation": False,
        },
        "interpolation": True,
        "show_data_points": False,
        "y_axis_padding_fraction_of_data_span": 0.05,
        "dpi": 600,
    }
    return fig, metadata


def _readme_text(
    case_key: str,
    thresholds: Sequence[float],
    targets: Sequence[float],
    fine_precision: dict[str, Any],
    figure_metadata: dict[str, dict[str, Any]],
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
    return f"""# {case_key}: CCP-target PV--BESS sizing curves

This figure bundle is derived from the completed, validated adaptive CCP--LOLP
sizing result. It does not resample outage scenarios, rerun dispatch or rerun
the sizing search.

- The shared x axis is the requested CCP target: {target_text}.
- Each coloured series fixes one strict LOLP constraint: {threshold_text}.
- CCP is calculated as `P(LOLP < threshold)`; a scenario exactly equal to a
  threshold does not pass.
- The input sizing result has final grid precision of {float(fine_precision['pv_kw']):g} kW PV and
  {float(fine_precision['battery_kwh']):g} kWh BESS. Its selected point minimises the arithmetic
  mean `annual_cost_sgd_per_year` across the shared scenarios; exact cost ties select lower PV,
  then lower BESS.
- The bundle contains three independent 16:9 figures: optimal PV capacity,
  optimal BESS energy, and mean annual system cost.
- Each line is a 200-sample shape-preserving PCHIP connection through the selected
  source points. Markers are hidden and no extrapolation beyond the requested CCP
  target range is applied.
- Each y axis spans the data range plus 5% of its total span at either end:
{axis_text}

The Source Data CSV records targets, achieved CCP, passing-scenario counts,
feasibility, selected PV/BESS capacities and annual costs. The validation JSON
records input and output checksums, the strict comparison and selection rule, and
the exported figure metadata.
"""


def run(
    results_dir: Path | str = DEFAULT_RESULTS,
    output_dir: Path | str | None = None,
    *,
    sizing_results_dir: Path | str | None = DEFAULT_SIZING_RESULTS,
    lolp_thresholds: Sequence[float] = DEFAULT_LOLP_THRESHOLDS,
    ccp_targets: Sequence[float] = DEFAULT_CCP_TARGETS,
) -> dict[str, Any]:
    """Export three CCP-target figures from validated fine sizing data.

    Passing ``sizing_results_dir=None`` retains the original coarse-sweep
    derivation for backwards-compatible exploratory use.
    """
    results_path = Path(results_dir).resolve()
    records, _source, manifest, input_hashes = _completed_records(results_path)
    fine_manifest: dict[str, Any] | None = None
    fine_validation: dict[str, Any] | None = None
    if sizing_results_dir is None:
        curves = select_target_designs(records, lolp_thresholds=lolp_thresholds, ccp_targets=ccp_targets)
        output_path = Path(output_dir).resolve() if output_dir else results_path / "figures"
        figure_input = {"kind": "coarse_capacity_sweep", "directory": str(results_path)}
        fine_precision = {"pv_kw": 1.0, "battery_kwh": 2.5}
    else:
        sizing_path = Path(sizing_results_dir).resolve()
        curves, fine_manifest, fine_validation = load_completed_fine_curves(
            sizing_path, lolp_thresholds=lolp_thresholds, ccp_targets=ccp_targets,
        )
        fine_source = Path(str(fine_manifest.get("source", {}).get("directory", ""))).resolve()
        if fine_source != results_path:
            raise ValueError("Fine sizing result does not reference the requested completed capacity sweep")
        for name, expected_hash in fine_manifest.get("source", {}).get("file_hashes", {}).items():
            if file_sha256(results_path / name) != expected_hash:
                raise ValueError(f"Fine sizing source changed since completion: {name}")
        output_path = Path(output_dir).resolve() if output_dir else sizing_path / "figures"
        figure_input = {"kind": "adaptive_fine_sizing", "directory": str(sizing_path)}
        fine_precision = dict(fine_validation.get("fine_precision", {}))
        if not {"pv_kw", "battery_kwh"}.issubset(fine_precision):
            raise ValueError("Fine sizing validation does not record PV and BESS precision")
    thresholds = tuple(sorted(curves.lolp_constraint.astype(float).unique().tolist()))
    targets = tuple(sorted(curves.ccp_target.astype(float).unique().tolist()))
    output_path.mkdir(parents=True, exist_ok=True)
    case_key = str(manifest["spec"]["case_key"])
    source_base = output_path / f"{case_key}_{SOURCE_OUTPUT_STEM}"

    source_path = write_source_data(source_base, curves)
    paths: dict[str, Any] = {"source_data": source_path}
    figure_metadata: dict[str, dict[str, Any]] = {}
    _, plt = matplotlib_context(output_path)
    for panel in PANEL_SPECS:
        panel_key = str(panel["key"])
        base = output_path / f"{case_key}_{panel['output_stem']}"
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

    for name, expected_hash in input_hashes.items():
        if file_sha256(results_path / name) != expected_hash:
            raise RuntimeError(f"Input changed while plotting: {name}")
    readme_path = output_path / f"{SOURCE_OUTPUT_STEM}_README.md"
    readme_path.write_text(
        _readme_text(case_key, thresholds, targets, fine_precision, figure_metadata),
        encoding="utf-8",
    )
    paths["readme"] = readme_path
    output_files = [source_path, readme_path]
    output_files.extend(path for panel_paths in paths.values() if isinstance(panel_paths, dict) for path in panel_paths.values())
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
        "figure_input": figure_input,
        "fine_precision": fine_precision,
        "selection_rule": (
            "Read directly from the completed adaptive fine sizing curve: minimum arithmetic mean "
            "annual_cost_sgd_per_year among feasible candidates; ties resolve by lower PV then lower BESS"
        ),
        "feasible_curve_points": int(curves.feasible.sum()),
        "infeasible_curve_points": int((~curves.feasible).sum()),
        "input_sha256": input_hashes,
        "fine_curve_sha256": file_sha256(Path(sizing_results_dir).resolve() / "ccp_target_curve_source.csv")
        if sizing_results_dir is not None else None,
        "figures": figure_metadata,
        "output_sha256": {path.name: file_sha256(path) for path in output_files},
    }
    report_path = output_path / "ccp_target_line_plots_validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    paths["validation"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Completed economic capacity-sweep directory referenced by the sizing result")
    parser.add_argument("--sizing-results-dir", type=Path, default=DEFAULT_SIZING_RESULTS,
                        help="Completed 0.25-grid adaptive sizing directory used as the figure source")
    parser.add_argument("--use-coarse-sweep", action="store_true",
                        help="Derive exploratory curves from the coarse capacity sweep instead of fine sizing output")
    parser.add_argument("--output-dir", type=Path,
                        help="Figure output directory (default: RESULTS/figures)")
    parser.add_argument("--lolp-thresholds", nargs="+", type=float, default=list(DEFAULT_LOLP_THRESHOLDS),
                        help="Strict LOLP thresholds as fractions (default: 0.10 0.20 0.30)")
    parser.add_argument("--ccp-targets", nargs="+", type=float, default=list(DEFAULT_CCP_TARGETS),
                        help="CCP targets as fractions (default: 0.50 through 0.95 by 0.05)")
    args = parser.parse_args()
    paths = run(
        args.results_dir,
        args.output_dir,
        sizing_results_dir=None if args.use_coarse_sweep else args.sizing_results_dir,
        lolp_thresholds=tuple(args.lolp_thresholds),
        ccp_targets=tuple(args.ccp_targets),
    )
    print(f"Exported CCP-target PV/BESS/cost figures to {paths['source_data'].parent}")


if __name__ == "__main__":
    main()
