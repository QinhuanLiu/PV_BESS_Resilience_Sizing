"""Plot validated Brisbane PV--BESS sensitivity-grid contour maps.

Run with ``python -m src.figures.sensitivity_grid_contours``.  This module only
reads completed grid summaries; it never reruns optimisation or dispatch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import _capacity_contour_style as style
from .core import apply_nature_style, matplotlib_context, save_publication_figure, write_source_data


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRID_DIR = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_Sensitivity_Grid_15Y"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_Sensitivity_Figures_15Y"

GRID_GROUPS: dict[str, dict[str, str]] = {
    "cost_capex": {
        "summary_filename": "Cost_CAPEX_Grid_Summary.csv",
        "directory": "Cost_CAPEX",
        "display_name": "Cost CAPEX",
        "row_parameter": "pv_capex_sgd_per_kw",
        "column_parameter": "battery_capex_sgd_per_kwh",
        "x_label": "Battery energy CAPEX (SGD kWh$^{-1}$)",
        "y_label": "PV CAPEX (SGD kW$^{-1}$)",
    },
    "saidi_voll": {
        "summary_filename": "SAIDI_VoLL_Grid_Summary.csv",
        "directory": "SAIDI_VoLL",
        "display_name": "SAIDI--VoLL",
        "row_parameter": "saidi_min_per_year",
        "column_parameter": "voll_sgd_per_kwh",
        "x_label": "VoLL (SGD kWh$^{-1}$)",
        "y_label": "SAIDI (min yr$^{-1}$)",
    },
}

METRICS: tuple[dict[str, str], ...] = (
    {
        "key": "annual_system_cost",
        "column": "annual_cost_sgd_per_year",
        "title": "Annual system cost",
        "label": "Annual system cost (kSGD yr$^{-1}$)",
        "unit_type": "cost",
        "cmap": "Blues",
        "filename": "Annual_System_Cost",
    },
    {
        "key": "optimal_pv_capacity",
        "column": "pv_capacity_kw",
        "title": "Optimal PV capacity",
        "label": "Optimal PV capacity (kW)",
        "unit_type": "capacity",
        "cmap": "YlGn",
        "filename": "Optimal_PV_Capacity",
    },
    {
        "key": "optimal_battery_energy",
        "column": "battery_energy_kwh",
        "title": "Optimal BESS energy capacity",
        "label": "Optimal BESS energy capacity (kWh)",
        "unit_type": "energy",
        "cmap": "Purples",
        "filename": "Optimal_BESS_Energy_Capacity",
    },
    {
        "key": "eue",
        "column": "EUE_kwh_per_year",
        "title": "EUE",
        "label": "EUE (kWh yr$^{-1}$)",
        "unit_type": "ue",
        "cmap": "Reds",
        "filename": "EUE",
    },
    {
        "key": "lolp",
        "column": "LOLP",
        "title": "LOLP",
        "label": "LOLP (%)",
        "unit_type": "probability",
        "cmap": "YlGnBu",
        "filename": "LOLP",
    },
    {
        "key": "ssr",
        "column": "SSR",
        "title": "SSR",
        "label": "SSR (%)",
        "unit_type": "probability",
        "cmap": "Greens",
        "filename": "SSR",
    },
    {
        "key": "scr",
        "column": "SCR",
        "title": "SCR",
        "label": "SCR (%)",
        "unit_type": "probability",
        "cmap": "Oranges",
        "filename": "SCR",
    },
)

REQUIRED_COLUMNS = {
    "grid_type",
    "row_parameter",
    "row_value",
    "row_unit",
    "column_parameter",
    "column_value",
    "column_unit",
    "case_key",
    "city_key",
    "building_key",
    "analysis_years",
    "annual_cost_sgd_per_year",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "EUE_kwh_per_year",
    "LOLP",
    "SSR",
    "SCR",
    "SCR_defined",
    "SCR_status",
}
FINITE_METRIC_COLUMNS = (
    "annual_cost_sgd_per_year",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "EUE_kwh_per_year",
    "LOLP",
    "SSR",
)
NONNEGATIVE_COLUMNS = (
    "annual_cost_sgd_per_year",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "EUE_kwh_per_year",
)
PROBABILITY_COLUMNS = ("LOLP", "SSR")
SOURCE_COLUMNS = (
    "grid_type",
    "row_parameter",
    "row_value",
    "row_unit",
    "column_parameter",
    "column_value",
    "column_unit",
    "case_key",
    "city_key",
    "building_key",
    "analysis_years",
    "annual_cost_sgd_per_year",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "EUE_kwh_per_year",
    "LOLP",
    "SSR",
    "SCR",
    "SCR_defined",
    "SCR_status",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_completed_grids(grid_dir: Path = DEFAULT_GRID_DIR) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load both summaries only when their completed-run provenance is intact."""
    grid_dir = Path(grid_dir).resolve()
    manifest = _read_json(grid_dir / "manifest.json")
    status = _read_json(grid_dir / "run_status.json")
    validation = _read_json(grid_dir / "validation.json")
    if status.get("state") != "complete":
        raise ValueError("Only a completed sensitivity-grid run can be plotted")
    if validation.get("passed") is not True:
        raise ValueError("Only a validated sensitivity-grid run can be plotted")

    frames: dict[str, pd.DataFrame] = {}
    checksums: dict[str, str] = {}
    grid_validation = validation.get("grids")
    if not isinstance(grid_validation, dict):
        raise ValueError("Sensitivity-grid validation lacks per-grid checksums")
    for grid_key, group in GRID_GROUPS.items():
        path = grid_dir / group["summary_filename"]
        if not path.is_file():
            raise ValueError(f"Missing sensitivity-grid summary: {path.name}")
        expected = grid_validation.get(grid_key, {})
        expected_checksum = expected.get("file_sha256") if isinstance(expected, dict) else None
        actual_checksum = file_sha256(path)
        if not expected_checksum or actual_checksum != expected_checksum:
            raise ValueError(f"Completed sensitivity-grid checksum does not match: {path.name}")
        frame = _validate_grid_frame(pd.read_csv(path), grid_key, group)
        expected_rows = expected.get("row_count") if isinstance(expected, dict) else None
        if expected_rows is not None and int(expected_rows) != len(frame):
            raise ValueError(f"Sensitivity-grid validation row count does not match: {path.name}")
        frames[grid_key] = frame
        checksums[path.name] = actual_checksum

    _validate_pair_consistency(frames, manifest)
    provenance = {
        "manifest": manifest,
        "validation": validation,
        "input_sha256": checksums,
        "grid_dir": str(grid_dir),
    }
    return frames, provenance


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing sensitivity-grid provenance file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Sensitivity-grid provenance must be a JSON object: {path.name}")
    return value


def _validate_grid_frame(frame: pd.DataFrame, grid_key: str, group: Mapping[str, str]) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{grid_key} summary is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{grid_key} summary is empty")
    frame = frame.copy()
    if not frame["grid_type"].astype(str).eq(grid_key).all():
        raise ValueError(f"{grid_key} summary has an inconsistent grid_type")
    for field in ("row_parameter", "column_parameter"):
        expected = group[field]
        if not frame[field].astype(str).eq(expected).all():
            raise ValueError(f"{grid_key} summary has an inconsistent {field}")
    for field in ("row_unit", "column_unit", "case_key", "city_key", "building_key"):
        if frame[field].isna().any() or frame[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"{grid_key} summary has a missing {field}")

    numeric_columns = ("row_value", "column_value", "analysis_years", *FINITE_METRIC_COLUMNS)
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{grid_key} summary has non-finite {column}")
        frame[column] = values.astype(float)
    if (frame["analysis_years"] <= 0).any() or frame["analysis_years"].nunique() != 1:
        raise ValueError(f"{grid_key} summary has an invalid analysis_years value")
    for column in NONNEGATIVE_COLUMNS:
        if frame[column].lt(-1e-9).any():
            raise ValueError(f"{grid_key} summary has negative {column}")
    for column in PROBABILITY_COLUMNS:
        if frame[column].lt(-1e-9).any() or frame[column].gt(1 + 1e-9).any():
            raise ValueError(f"{grid_key} summary has {column} outside [0, 1]")

    frame["SCR_defined"] = _parse_boolean(frame["SCR_defined"], f"{grid_key} SCR_defined")
    scr = pd.to_numeric(frame["SCR"], errors="coerce")
    defined = frame["SCR_defined"]
    if scr.loc[defined].isna().any() or not np.isfinite(scr.loc[defined].to_numpy(dtype=float)).all():
        raise ValueError(f"{grid_key} summary has undefined or non-finite SCR where SCR is defined")
    if scr.loc[defined].lt(-1e-9).any() or scr.loc[defined].gt(1 + 1e-9).any():
        raise ValueError(f"{grid_key} summary has SCR outside [0, 1]")
    if scr.loc[~defined].notna().any():
        raise ValueError(f"{grid_key} summary has SCR values where SCR is undefined")
    if not frame.loc[defined, "SCR_status"].astype(str).eq("defined").all():
        raise ValueError(f"{grid_key} summary has an invalid SCR_status where SCR is defined")
    if not frame.loc[~defined, "SCR_status"].astype(str).eq("not_applicable_no_pv").all():
        raise ValueError(f"{grid_key} summary has an invalid SCR_status where SCR is undefined")
    frame["SCR"] = scr.astype(float)

    row_values = np.sort(frame["row_value"].unique())
    column_values = np.sort(frame["column_value"].unique())
    if len(row_values) < 2 or len(column_values) < 2:
        raise ValueError(f"{grid_key} summary requires at least two values on each contour axis")
    actual = pd.MultiIndex.from_frame(frame[["row_value", "column_value"]])
    expected = pd.MultiIndex.from_product([row_values, column_values], names=actual.names)
    if actual.has_duplicates or len(actual) != len(expected) or not actual.sort_values().equals(expected.sort_values()):
        raise ValueError(f"{grid_key} summary does not cover a complete rectangular parameter grid")
    return frame.sort_values(["row_value", "column_value"]).reset_index(drop=True)


def _parse_boolean(values: pd.Series, context: str) -> pd.Series:
    parsed: list[bool] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        if pd.isna(value):
            raise ValueError(f"{context} must be a non-missing boolean")
        normalized = str(value).strip().lower()
        if normalized in {"true", "1"}:
            parsed.append(True)
        elif normalized in {"false", "0"}:
            parsed.append(False)
        else:
            raise ValueError(f"{context} must be a boolean")
    return pd.Series(parsed, index=values.index, dtype=bool)


def _validate_pair_consistency(frames: Mapping[str, pd.DataFrame], manifest: Mapping[str, Any]) -> None:
    case_keys = {str(frame["case_key"].iloc[0]) for frame in frames.values()}
    city_keys = {str(frame["city_key"].iloc[0]) for frame in frames.values()}
    building_keys = {str(frame["building_key"].iloc[0]) for frame in frames.values()}
    horizons = {float(frame["analysis_years"].iloc[0]) for frame in frames.values()}
    if not (len(case_keys) == len(city_keys) == len(building_keys) == 1):
        raise ValueError("Sensitivity-grid summaries do not describe the same case")
    if len(horizons) != 1:
        raise ValueError("Sensitivity-grid summaries use different analysis horizons")
    spec = manifest.get("spec") if isinstance(manifest.get("spec"), dict) else {}
    if spec.get("case_key") and str(spec["case_key"]) != next(iter(case_keys)):
        raise ValueError("Sensitivity-grid manifest case_key differs from summary data")
    if spec.get("analysis_years") is not None and not np.isclose(float(spec["analysis_years"]), next(iter(horizons))):
        raise ValueError("Sensitivity-grid manifest analysis_years differs from summary data")


def display_values(frame: pd.DataFrame, metric: Mapping[str, str]) -> pd.Series:
    """Return the plotted values without changing the source-data units."""
    values = frame[str(metric["column"])].astype(float)
    return values / 1000.0 if metric["key"] == "annual_system_cost" else values


def group_metric_levels(frames: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, np.ndarray]]:
    """Compute independent contour scales for every metric in each sensitivity group."""
    levels: dict[str, dict[str, np.ndarray]] = {}
    for grid_key, frame in frames.items():
        levels[grid_key] = {}
        for metric in METRICS:
            values = display_values(frame, metric).dropna().to_numpy(dtype=float)
            if values.size == 0:
                raise ValueError(f"Cannot build a contour scale for {grid_key} {metric['title']}")
            levels[grid_key][metric["key"]] = _levels_cover_data(
                style._metric_levels(values, metric["unit_type"]), values
            )
    return levels


def _levels_cover_data(levels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Expand contour endpoints by one representable float to avoid unfilled extrema."""
    result = np.asarray(levels, dtype=float).copy()
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or result.size < 2:
        raise ValueError("Contour levels require at least two finite boundaries")
    result[0] = min(result[0], np.nextafter(float(finite.min()), -np.inf))
    result[-1] = max(result[-1], np.nextafter(float(finite.max()), np.inf))
    if np.any(np.diff(result) <= 0.0):
        raise ValueError("Contour levels must remain strictly increasing after covering data bounds")
    return result


def build_figure(
    source: pd.DataFrame,
    group: Mapping[str, str],
    metric: Mapping[str, str],
    levels: np.ndarray,
    *,
    city_label: str = "Brisbane",
    out: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Render one unsmoothed regular-grid contour map and its own colorbar."""
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    style._apply_single_outage_readable_font_style(mpl)
    display = source.assign(_display_value=display_values(source, metric))
    x_values, y_values, z_values, scr_note = _contour_domain(display, metric)
    xx, yy = np.meshgrid(x_values, y_values)

    fig = plt.figure(figsize=(6.7, 5.6), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.052], wspace=0.07)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    cmap = style._contour_colormap(mpl, dict(metric))
    artist = ax.contourf(xx, yy, z_values, levels=levels, cmap=cmap, alpha=0.88)
    line_levels = style._sparse_contour_lines(levels)
    valid_line_levels = line_levels[(line_levels > float(np.nanmin(z_values))) & (line_levels < float(np.nanmax(z_values)))]
    if valid_line_levels.size:
        ax.contour(xx, yy, z_values, levels=valid_line_levels, colors="white", linewidths=0.32, alpha=0.65)

    title = f"{city_label}: {metric['title']}"
    ax.set_title(title, loc="left", pad=6)
    ax.set_xlabel(group["x_label"])
    ax.set_ylabel(group["y_label"])
    ax.set_xlim(float(display["column_value"].min()), float(display["column_value"].max()))
    ax.set_ylim(float(display["row_value"].min()), float(display["row_value"].max()))
    ax.set_box_aspect(1.0)
    if scr_note:
        ax.text(
            0.02,
            0.02,
            scr_note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=mpl.rcParams["xtick.labelsize"],
            color=mpl.rcParams["text.color"],
        )
    colorbar = fig.colorbar(artist, cax=cax)
    style._format_metric_colorbar(colorbar, levels, dict(metric))
    fig.canvas.draw()
    metadata = {
        "title": title,
        "x_axis": ax.get_xlabel(),
        "y_axis": ax.get_ylabel(),
        "levels": [float(value) for value in levels],
        "colorbar_ticks": [float(value) for value in colorbar.get_ticks()],
        "colorbar_tick_labels": [label.get_text() for label in cax.get_yticklabels()],
        "x_limits": [float(value) for value in ax.get_xlim()],
        "y_limits": [float(value) for value in ax.get_ylim()],
        "colormap": artist.cmap.name,
        "fill_alpha": float(artist.get_alpha()),
        "grid_shape": [int(len(y_values)), int(len(x_values))],
        "undefined_value_count": int(display["_display_value"].isna().sum()),
        "note": scr_note,
        "smoothing": False,
        "dpi": 600,
    }
    return fig, metadata


def _contour_domain(
    display: pd.DataFrame,
    metric: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    pivot = display.pivot(index="row_value", columns="column_value", values="_display_value").sort_index().sort_index(axis=1)
    if not pivot.isna().any().any():
        return (
            pivot.columns.to_numpy(dtype=float),
            pivot.index.to_numpy(dtype=float),
            pivot.to_numpy(dtype=float),
            None,
        )
    if metric["key"] != "scr":
        raise ValueError(f"{metric['title']} contains undefined values")
    valid = ~pivot.isna()
    row_indices = np.flatnonzero(valid.any(axis=1).to_numpy())
    column_indices = np.flatnonzero(valid.any(axis=0).to_numpy())
    if row_indices.size < 2 or column_indices.size < 2:
        raise ValueError("SCR contour requires at least two defined values on each axis")
    row_slice = slice(int(row_indices.min()), int(row_indices.max()) + 1)
    column_slice = slice(int(column_indices.min()), int(column_indices.max()) + 1)
    defined_rectangle = valid.iloc[row_slice, column_slice]
    outside = valid.copy()
    outside.iloc[row_slice, column_slice] = False
    if not defined_rectangle.all().all() or outside.any().any():
        raise ValueError("Undefined SCR must be confined to contiguous outer grid boundaries")
    domain = pivot.iloc[row_slice, column_slice]
    return (
        domain.columns.to_numpy(dtype=float),
        domain.index.to_numpy(dtype=float),
        domain.to_numpy(dtype=float),
        "SCR undefined outside the defined PV domain.\nNo zero-fill or extrapolation.",
    )


def _source_data(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame.loc[:, list(SOURCE_COLUMNS)].copy()
    source.insert(
        source.columns.get_loc("annual_cost_sgd_per_year") + 1,
        "annual_system_cost_ksgd_per_year",
        source["annual_cost_sgd_per_year"] / 1000.0,
    )
    return source


def run(grid_dir: Path = DEFAULT_GRID_DIR, output_dir: Path | None = None) -> dict[str, Path]:
    """Export both validated sensitivity figure groups and their provenance bundle."""
    grid_dir = Path(grid_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    frames, provenance = load_completed_grids(grid_dir)
    levels_by_group = group_metric_levels(frames)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_key = str(next(iter(frames.values()))["case_key"].iloc[0])
    city_label = str(next(iter(frames.values()))["city_key"].iloc[0])
    paths: dict[str, Path] = {}
    figure_metadata: dict[str, Any] = {}
    for grid_key, group in GRID_GROUPS.items():
        group_dir = output_dir / group["directory"]
        group_dir.mkdir(parents=True, exist_ok=True)
        source_path = write_source_data(group_dir / f"{case_key}_{group['directory']}_Contour", _source_data(frames[grid_key]))
        paths[f"{grid_key}_source_data"] = source_path
        figure_metadata[grid_key] = {
            "display_name": group["display_name"],
            "source_data": source_path.name,
            "figures": {},
        }
        _, plt = matplotlib_context(output_dir)
        for metric in METRICS:
            figure, metadata = build_figure(
                frames[grid_key],
                group,
                metric,
                levels_by_group[grid_key][metric["key"]],
                city_label=city_label,
                out=output_dir,
            )
            base = group_dir / f"{case_key}_{group['directory']}_{metric['filename']}_Contour"
            try:
                exports = save_publication_figure(figure, base)
            finally:
                plt.close(figure)
            paths.update(exports)
            figure_metadata[grid_key]["figures"][metric["key"]] = metadata

    for filename, digest in provenance["input_sha256"].items():
        if file_sha256(grid_dir / filename) != digest:
            raise RuntimeError(f"Sensitivity-grid input changed while plotting: {filename}")
    readme_path = output_dir / "README.md"
    readme_path.write_text(_readme_text(case_key, city_label), encoding="utf-8")
    paths["readme"] = readme_path
    report = {
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_key": case_key,
        "city_key": city_label,
        "analysis_years": float(next(iter(frames.values()))["analysis_years"].iloc[0]),
        "input_sha256": provenance["input_sha256"],
        "input_run_validation": provenance["validation"],
        "color_scale_policy": "Independent, precision-safe contour levels for each metric within each sensitivity group",
        "group_levels": {
            grid_key: {metric_key: [float(value) for value in values] for metric_key, values in metric_levels.items()}
            for grid_key, metric_levels in levels_by_group.items()
        },
        "metrics": [dict(metric) for metric in METRICS],
        "groups": figure_metadata,
        "SCR_policy": "Undefined values are permitted only outside one contiguous edge-aligned defined rectangle; they are left blank with no zero-fill or extrapolation.",
        "exports": {key: str(path.relative_to(output_dir)) for key, path in paths.items() if path != readme_path},
        "output_sha256": {path.relative_to(output_dir).as_posix(): file_sha256(path) for path in paths.values() if path != readme_path},
    }
    report_path = output_dir / "sensitivity_contour_validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    paths["validation"] = report_path
    return paths


def _readme_text(case_key: str, city_label: str) -> str:
    return f"""# {city_label}: PV--BESS sensitivity contour maps

This directory contains two standalone contour-map groups derived directly from a completed,
validated sensitivity-grid run. No optimisation or dispatch was rerun.

## Groups and axes

- `Cost_CAPEX`: x = battery energy CAPEX (SGD/kWh); y = PV CAPEX (SGD/kW).
- `SAIDI_VoLL`: x = VoLL (SGD/kWh); y = SAIDI (min/year).

Each group contains seven figures for annual system cost, optimal PV capacity, optimal BESS
energy capacity, EUE, LOLP, SSR, and SCR. The annual cost is the existing annual system cost:
annualised PV/BESS investment, O&M, net grid cost, and outage loss. Battery capacity means energy
capacity. Cost is displayed in kSGD/year; LOLP, SSR, and SCR are displayed as percentages.

Each sensitivity group uses its own metric-specific contour levels and colourbar scale, so that
its local response remains legible. Every colourbar extends by one representable floating-point
value beyond the local data extrema, preventing maximum or minimum grid values from being blank.
Each group also includes a source-data CSV with the untouched input values plus the displayed
annual-cost conversion. PNG exports are 600 dpi; SVG text remains editable and PDF is included.

SCR is not zero-filled: if it is undefined because optimal PV capacity is zero, only an
edge-aligned defined domain can be plotted and the omitted region is explicitly left blank.

Reproduce with:

```bash
.venv/bin/python -m src.figures.sensitivity_grid_contours --grid-dir outputs/Brisbane/Brisbane_SingleFamily_Sensitivity_Grid_15Y
```

The validation JSON records source checksums, group-specific contour levels, figure metadata, and output
checksums for all generated files.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR, help="Completed sensitivity-grid output directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for sensitivity contour figures")
    args = parser.parse_args()
    paths = run(args.grid_dir, args.output_dir)
    figure_count = sum(path.suffix == ".png" for path in paths.values())
    print(f"Exported {figure_count} sensitivity contour maps to {paths['validation'].parent}")


if __name__ == "__main__":
    main()
