"""Plot constrained Brisbane PV and BESS CAPEX sensitivity curves.

The module only accepts the completed, checksum-validated output produced by
``src.ccp_cost_sensitivity``.  It does not rerun optimisation or dispatch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..ccp_cost_sensitivity import DEFAULT_OUTPUT as DEFAULT_RESULTS_DIR
from . import _capacity_contour_style as contour_style
from .core import NATURE_PALETTE, apply_nature_style, matplotlib_context, save_publication_figure, write_source_data
from .sensitivity_grid_contours import METRICS, display_values, file_sha256
from .sensitivity_grid_line_plots import (
    SMOOTH_SAMPLES,
    _draw_smoothed_line,
    _plot_scr_domain,
    _zero_based_y_scale,
    metric_line_color,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_CCP_LOLP_Cost_Sensitivity_Figures_15Y"

X_LABELS = {
    "pv_capex": "PV CAPEX (SGD kW$^{-1}$)",
    "battery_capex": "Battery energy CAPEX (SGD kWh$^{-1}$)",
}

LINE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "pv_capex",
        "directory": "PV_CAPEX",
        "title": "PV CAPEX sensitivity",
        "x_column": "pv_capex_sgd_per_kw",
        "x_label": X_LABELS["pv_capex"],
        "fixed_column": "battery_capex_sgd_per_kwh",
        "fixed_label": "Battery energy CAPEX",
        "fixed_unit": "SGD/kWh",
        "fixed_value": 400.0,
    },
    {
        "key": "battery_capex",
        "directory": "Battery_CAPEX",
        "title": "Battery energy CAPEX sensitivity",
        "x_column": "battery_capex_sgd_per_kwh",
        "x_label": X_LABELS["battery_capex"],
        "fixed_column": "pv_capex_sgd_per_kw",
        "fixed_label": "PV CAPEX",
        "fixed_unit": "SGD/kW",
        "fixed_value": 1200.0,
    },
)

_RESULT_COLUMNS = {
    "sweep_key", "sweep_parameter", "sweep_value", "sweep_unit",
    "pv_capex_sgd_per_kw", "battery_capex_sgd_per_kwh", "battery_power_capex_sgd_per_kw",
    "case_key", "analysis_years", "lolp_threshold", "ccp_target", "comparison",
    "scenario_count", "required_passing_scenarios", "feasible", "passing_scenarios",
    "actual_ccp", "domain_limited", "annual_cost_sgd_per_year", "pv_capacity_kw",
    "battery_energy_kwh", "battery_power_kw", "EUE_kwh_per_year", "LOLP", "SSR",
    "SCR", "SCR_defined", "SCR_status",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing CCP CAPEX sensitivity provenance file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid CCP CAPEX sensitivity provenance file: {path.name}")
    return value


def _as_bool(series: pd.Series, column: str) -> pd.Series:
    if series.dtype == bool:
        return series
    values = series.astype(str).str.strip().str.lower()
    if not values.isin({"true", "false"}).all():
        raise ValueError(f"CCP CAPEX sensitivity {column} is not boolean")
    return values.eq("true")


def load_completed_results(results_dir: Path = DEFAULT_RESULTS_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one finished constrained run after verifying its input checksum."""
    results_dir = Path(results_dir).resolve()
    manifest = _read_json(results_dir / "manifest.json")
    validation = _read_json(results_dir / "validation.json")
    status = _read_json(results_dir / "run_status.json")
    results_path = results_dir / "ccp_cost_sensitivity_results.csv"
    if status.get("state") != "complete":
        raise ValueError("Only a completed CCP CAPEX sensitivity run can be plotted")
    if validation.get("passed") is not True:
        raise ValueError("Only a validated CCP CAPEX sensitivity run can be plotted")
    fingerprint = manifest.get("fingerprint")
    if not fingerprint or validation.get("fingerprint") != fingerprint:
        raise ValueError("CCP CAPEX sensitivity manifest and validation fingerprints differ")
    if validation.get("result_sha256") != file_sha256(results_path):
        raise ValueError("Completed CCP CAPEX sensitivity checksum does not match")
    frame = pd.read_csv(results_path)
    missing = sorted(_RESULT_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"CCP CAPEX sensitivity result table is missing: {', '.join(missing)}")
    points = manifest.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("CCP CAPEX sensitivity manifest lacks requested price points")
    if len(frame) != len(points) or validation.get("result_row_count") != len(frame):
        raise ValueError("CCP CAPEX sensitivity result row count does not match its manifest")
    if not frame.case_key.astype(str).eq("Brisbane_SingleFamily").all():
        raise ValueError("CCP CAPEX sensitivity figures only support Brisbane_SingleFamily")
    numeric = [
        "sweep_value", "pv_capex_sgd_per_kw", "battery_capex_sgd_per_kwh",
        "battery_power_capex_sgd_per_kw", "analysis_years", "lolp_threshold", "ccp_target",
        "scenario_count", "required_passing_scenarios", "passing_scenarios", "actual_ccp",
        "annual_cost_sgd_per_year", "pv_capacity_kw", "battery_energy_kwh", "battery_power_kw",
        "EUE_kwh_per_year", "LOLP", "SSR",
    ]
    values = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("CCP CAPEX sensitivity results have non-finite required values")
    frame = frame.copy()
    for column in ("feasible", "domain_limited", "SCR_defined"):
        frame[column] = _as_bool(frame[column], column)
    if not frame.feasible.all() or frame.domain_limited.any():
        raise ValueError("CCP CAPEX sensitivity contains infeasible or domain-limited selections")
    if not frame.comparison.astype(str).eq("LOLP < threshold").all():
        raise ValueError("CCP CAPEX sensitivity does not use the strict LOLP comparison")
    if not np.isclose(frame.lolp_threshold.to_numpy(dtype=float), 0.20, rtol=0.0, atol=1e-12).all():
        raise ValueError("CCP CAPEX sensitivity LOLP threshold must be 20%")
    if not np.isclose(frame.ccp_target.to_numpy(dtype=float), 0.90, rtol=0.0, atol=1e-12).all():
        raise ValueError("CCP CAPEX sensitivity CCP target must be 90%")
    actual_ccp = frame.passing_scenarios.to_numpy(dtype=float) / frame.scenario_count.to_numpy(dtype=float)
    if (
        (frame.passing_scenarios.to_numpy(dtype=float) < frame.required_passing_scenarios.to_numpy(dtype=float)).any()
        or not np.allclose(frame.actual_ccp.to_numpy(dtype=float), actual_ccp, rtol=0.0, atol=1e-12)
        or (frame.actual_ccp.to_numpy(dtype=float) + 1e-12 < 0.90).any()
    ):
        raise ValueError("CCP CAPEX sensitivity selections do not satisfy the required passing-scenario count")
    if not np.isclose(frame.battery_power_capex_sgd_per_kw.to_numpy(dtype=float), 150.0, rtol=0.0, atol=1e-9).all():
        raise ValueError("CCP CAPEX sensitivity battery power CAPEX must remain fixed at 150 SGD/kW")
    has_pv = frame.SCR_defined.to_numpy(dtype=bool)
    if frame.loc[has_pv, "SCR"].isna().any() or frame.loc[~has_pv, "SCR"].notna().any():
        raise ValueError("CCP CAPEX sensitivity SCR values do not match their definition status")
    return frame, {
        "manifest": manifest,
        "validation": validation,
        "status": status,
        "results_path": results_path,
        "input_sha256": {
            "ccp_cost_sensitivity_results.csv": file_sha256(results_path),
            "manifest.json": file_sha256(results_dir / "manifest.json"),
            "validation.json": file_sha256(results_dir / "validation.json"),
            "run_status.json": file_sha256(results_dir / "run_status.json"),
        },
    }


def extract_line(frame: pd.DataFrame, spec: Mapping[str, Any], manifest: Mapping[str, Any]) -> pd.DataFrame:
    """Extract one exact, validated CAPEX cross-section from the combined table."""
    key = str(spec["key"])
    source = frame.loc[frame.sweep_key.astype(str).eq(key)].copy()
    if source.empty:
        raise ValueError(f"Missing CCP CAPEX sensitivity curve: {key}")
    expected = [
        float(point["sweep_value"])
        for point in manifest["points"]
        if isinstance(point, dict) and point.get("sweep_key") == key
    ]
    if len(source) != len(expected):
        raise ValueError(f"CCP CAPEX sensitivity {key} curve does not match its manifest")
    x_column = str(spec["x_column"])
    x = pd.to_numeric(source[x_column], errors="coerce")
    if x.isna().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError(f"CCP CAPEX sensitivity {key} has non-finite x values")
    source = source.assign(_x_value=x.astype(float)).sort_values("_x_value").reset_index(drop=True)
    actual = source["_x_value"].to_numpy(dtype=float)
    if len(actual) < 2 or np.any(np.diff(actual) <= 0.0):
        raise ValueError(f"CCP CAPEX sensitivity {key} requires unique, strictly increasing x values")
    if not np.allclose(actual, np.sort(np.asarray(expected, dtype=float)), rtol=0.0, atol=1e-9):
        raise ValueError(f"CCP CAPEX sensitivity {key} x values differ from the requested input range")
    fixed_column, fixed_value = str(spec["fixed_column"]), float(spec["fixed_value"])
    if not np.isclose(source[fixed_column].to_numpy(dtype=float), fixed_value, rtol=0.0, atol=1e-9).all():
        raise ValueError(f"CCP CAPEX sensitivity {key} does not retain {fixed_column}={fixed_value:g}")
    if not source.sweep_parameter.astype(str).eq(
        "pv_capex_sgd_per_kw" if key == "pv_capex" else "battery_capex_sgd_per_kwh"
    ).all():
        raise ValueError(f"CCP CAPEX sensitivity {key} has an inconsistent scan parameter")
    return source


def _fixed_text(spec: Mapping[str, Any]) -> str:
    return (
        f"Constraint: P(LOLP < 20%) ≥ 90%; {spec['fixed_label']} = {float(spec['fixed_value']):g} "
        f"{spec['fixed_unit']}; BESS power CAPEX = 150 SGD/kW"
    )


def build_figure(source: pd.DataFrame, spec: Mapping[str, Any], *, out: Path | None = None) -> tuple[Any, dict[str, Any]]:
    """Render a seven-panel one-column constrained sensitivity curve."""
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    contour_style._apply_single_outage_readable_font_style(mpl)
    mpl.rcParams.update({
        "font.size": 9.4, "axes.labelsize": 8.5, "xtick.labelsize": 8.1,
        "ytick.labelsize": 8.1, "axes.titlesize": 10.2, "lines.markersize": 4.2,
    })
    fig = plt.figure(figsize=(7.25, 14.8))
    grid = fig.add_gridspec(7, 1, hspace=0.45)
    axes = [fig.add_subplot(grid[0, 0])]
    axes.extend(fig.add_subplot(grid[index, 0], sharex=axes[0]) for index in range(1, len(METRICS)))
    x = source["_x_value"].to_numpy(dtype=float)
    title = f"Brisbane: {spec['title']}"
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.052, top=0.91)
    fig.suptitle(title, x=0.02, ha="left", y=0.982, fontsize=14.0, fontweight="bold")
    fig.text(0.02, 0.952, _fixed_text(spec), ha="left", va="top", fontsize=10.2, color=NATURE_PALETTE["neutral_dark"])

    panels: dict[str, Any] = {}
    for index, (ax, metric) in enumerate(zip(axes, METRICS)):
        values = display_values(source, metric).to_numpy(dtype=float)
        plot_x, plot_y, note = x, values, None
        if metric["key"] == "scr":
            plot_x, plot_y, note = _plot_scr_domain(x, values)
        color = metric_line_color(mpl, metric)
        line, method = _draw_smoothed_line(ax, plot_x, plot_y, color=color)
        ax.set_title(f"{chr(97 + index)}  {metric['title']}", loc="left", pad=4)
        ax.set_ylabel(str(metric["label"]))
        ax.set_xlim(float(x.min()), float(x.max()))
        limits, ticks = _zero_based_y_scale(plot_y, str(metric["unit_type"]))
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        if metric["unit_type"] == "probability":
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=None))
        ax.tick_params(direction="out", length=2.8, width=0.65, pad=2.0)
        if index < len(axes) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(str(spec["x_label"]))
        if note:
            ax.text(0.015, 0.04, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5,
                    color=NATURE_PALETTE["neutral_dark"])
        fig.canvas.draw()
        panels[str(metric["key"])] = {
            "title": ax.get_title(), "y_axis": ax.get_ylabel(), "color": color,
            "x_values": [float(value) for value in x],
            "y_values": [None if not np.isfinite(value) else float(value) for value in values],
            "y_limits": [float(value) for value in ax.get_ylim()],
            "y_tick_labels": [label.get_text() for label in ax.get_yticklabels()],
            "undefined_value_count": int((~np.isfinite(values)).sum()), "note": note,
            "curve_method": method, "rendered_sample_count": int(len(line.get_xdata())),
            "marker": str(line.get_marker()),
        }
    return fig, {
        "title": title, "constraint": "P(LOLP < 20%) >= 90%",
        "x_axis": str(spec["x_label"]), "fixed_conditions": {
            str(spec["fixed_column"]): float(spec["fixed_value"]),
            "battery_power_capex_sgd_per_kw": 150.0,
        },
        "panel_count": len(panels), "metrics": panels,
        "layout": "seven-row single-column shared x-axis",
        "shared_x_limits": [float(value) for value in axes[0].get_xlim()],
        "smoothing": "PCHIP shape-preserving interpolation (200 samples; no extrapolation)",
        "interpolation": True, "show_data_points": False,
        "y_scale_policy": "zero baseline with a per-panel rounded upper bound", "dpi": 600,
    }


def _source_data(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sweep_key", "sweep_parameter", "sweep_value", "sweep_unit", "pv_capex_sgd_per_kw",
        "battery_capex_sgd_per_kwh", "battery_power_capex_sgd_per_kw", "case_key", "analysis_years",
        "lolp_threshold", "ccp_target", "comparison", "scenario_count", "required_passing_scenarios",
        "passing_scenarios", "actual_ccp", "annual_cost_sgd_per_year", "annual_system_cost_ksgd_per_year",
        "pv_capacity_kw", "battery_energy_kwh", "battery_power_kw", "EUE_kwh_per_year", "LOLP", "SSR",
        "SCR", "SCR_defined", "SCR_status",
    ]
    source = frame.copy()
    source["annual_system_cost_ksgd_per_year"] = source["annual_cost_sgd_per_year"].astype(float) / 1000.0
    return source.loc[:, columns]


def run(results_dir: Path = DEFAULT_RESULTS_DIR, output_dir: Path | None = None) -> dict[str, Path]:
    """Export both seven-panel constrained CAPEX sensitivity figures."""
    results_dir = Path(results_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    frame, provenance = load_completed_results(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = {str(spec["key"]): extract_line(frame, spec, provenance["manifest"]) for spec in LINE_SPECS}
    paths: dict[str, Path] = {}
    figures: dict[str, Any] = {}
    for spec in LINE_SPECS:
        key, directory_name = str(spec["key"]), str(spec["directory"])
        directory = output_dir / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        base_name = f"Brisbane_SingleFamily_{directory_name}_CCP_LOLP_Cost_Sensitivity"
        source_path = write_source_data(directory / base_name, _source_data(lines[key]))
        paths[f"{key}_source_data"] = source_path
        figure, metadata = build_figure(lines[key], spec, out=output_dir)
        _, plt = matplotlib_context(output_dir)
        try:
            exports = save_publication_figure(figure, directory / base_name)
        finally:
            plt.close(figure)
        for suffix in ("png", "svg", "pdf"):
            paths[f"{key}_{suffix}"] = exports[f"{base_name}_{suffix}"]
        figures[key] = {
            "directory": directory_name, "source_data": source_path.name, "sample_count": int(len(lines[key])),
            "fixed_conditions": metadata["fixed_conditions"], "figure": metadata,
        }
    for filename, digest in provenance["input_sha256"].items():
        if file_sha256(results_dir / filename) != digest:
            raise RuntimeError(f"CCP CAPEX sensitivity input changed while plotting: {filename}")
    readme = output_dir / "README.md"
    readme.write_text(_readme_text(), encoding="utf-8")
    paths["readme"] = readme
    report = {
        "passed": True, "created_at": datetime.now(timezone.utc).isoformat(),
        "constraint": "P(LOLP < 20%) >= 90%", "input_sha256": provenance["input_sha256"],
        "input_run_validation": provenance["validation"], "figures": figures,
        "SCR_policy": "Undefined SCR remains blank; it is never zero-filled or extrapolated.",
        "exports": {key: str(path.relative_to(output_dir)) for key, path in paths.items()},
        "output_sha256": {path.relative_to(output_dir).as_posix(): file_sha256(path) for path in paths.values()},
    }
    validation = output_dir / "ccp_cost_sensitivity_line_validation.json"
    validation.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    paths["validation"] = validation
    return paths


def _readme_text() -> str:
    return """# Brisbane CCP--LOLP CAPEX sensitivity figures

These two seven-panel figures read a completed constrained cost-sensitivity
result directly; optimisation and dispatch are not rerun.  Every selected
design satisfies the strict scenario criterion `P(LOLP < 20%) >= 90%`.

- `PV_CAPEX`: PV CAPEX varies from 600 to 1800 SGD/kW; battery energy CAPEX is fixed at 400 SGD/kWh.
- `Battery_CAPEX`: battery energy CAPEX varies from 200 to 600 SGD/kWh; PV CAPEX is fixed at 1200 SGD/kW.

Battery power CAPEX remains fixed at 150 SGD/kW.  Each figure shows annual
system cost, optimal PV capacity, optimal BESS energy capacity, EUE, LOLP, SSR
and SCR.  Curves use 200 PCHIP samples inside the original CAPEX range only;
all y axes begin at zero.  Undefined SCR values remain blank and are never
converted to zero.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = run(args.results_dir, args.output_dir)
    print(f"Exported 2 CCP CAPEX sensitivity figures to {paths['validation'].parent}")


if __name__ == "__main__":
    main()
