"""Plot two strict-threshold CCP surfaces for the completed Brisbane capacity sweep.

Run with ``python -m src.figures.ccp_3d_surfaces --threshold-pair 20-30`` to
select the 20% and 30% surfaces. The module reads completed scenario records
only; it never resamples outages or runs dispatch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import _capacity_contour_style as style
from .capacity_sweep_contours import (
    CAPACITY_KEYS,
    DEFAULT_RESULTS,
    aggregate_scenarios,
    file_sha256,
    reconcile_summary,
)
from .core import apply_nature_style, matplotlib_context, save_publication_figure, write_source_data


FIGURE_SIZE_INCHES = (11.2, 11.4)
GRID_SPEC = {
    "nrows": 5,
    "ncols": 5,
    "left": 0.075,
    "right": 0.945,
    "bottom": 0.075,
    "top": 0.935,
    "wspace": 0.24,
    "hspace": 0.22,
    "width_ratios": [1.0] * 4 + [0.055],
}
SURFACE_STYLES = {
    "10-20": {
        "lt20": {"column": "CCP_LOLP_lt20", "count_column": "LOLP_lt20_scenario_count",
                 "threshold": 0.20, "cmap": "Purples", "color_position": 0.72, "alpha": 0.62,
                 "label": "CCP (LOLP < 20%)"},
        "lt10": {"column": "CCP_LOLP_lt10", "count_column": "LOLP_lt10_scenario_count",
                 "threshold": 0.10, "cmap": "YlGnBu", "color_position": 0.72, "alpha": 0.88,
                 "label": "CCP (LOLP < 10%)"},
    },
    "20-30": {
        "lt30": {"column": "CCP_LOLP_lt30", "count_column": "LOLP_lt30_scenario_count",
                 "threshold": 0.30, "cmap": "Oranges", "color_position": 0.72, "alpha": 0.62,
                 "label": "CCP (LOLP < 30%)"},
        "lt20": {"column": "CCP_LOLP_lt20", "count_column": "LOLP_lt20_scenario_count",
                 "threshold": 0.20, "cmap": "Purples", "color_position": 0.72, "alpha": 0.88,
                 "label": "CCP (LOLP < 20%)"},
    },
}
SURFACE_STYLE = SURFACE_STYLES["10-20"]
PAIR_OUTPUTS = {
    "10-20": {"filename": "CCP_LOLP_LT10_LT20_3D_Surface", "validation": "ccp_3d_surface_validation.json"},
    "20-30": {"filename": "CCP_LOLP_LT20_LT30_3D_Surface", "validation": "ccp_3d_surface_lt20_lt30_validation.json"},
}
WHITE_GRID_WIDTH = 0.32
WHITE_GRID_ALPHA = 0.65
VIEW_ELEVATION = 28
VIEW_AZIMUTH = -135


def _completed_records(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, str]]:
    """Load only a completed, checksum-validated capacity sweep."""
    manifest = json.loads((results_dir / "manifest.json").read_text())
    status = json.loads((results_dir / "run_status.json").read_text())
    validation = json.loads((results_dir / "validation.json").read_text())
    if status.get("state") != "complete" or validation.get("passed") is not True:
        raise ValueError("Only a completed, validated sweep can be plotted")
    checksums: dict[str, str] = {}
    for name in ("scenario_results.csv", "capacity_summary.csv"):
        path = results_dir / name
        checksums[name] = file_sha256(path)
        if checksums[name] != validation.get("file_sha256", {}).get(name):
            raise ValueError(f"Completed sweep checksum does not match: {name}")
    records = pd.read_csv(results_dir / "scenario_results.csv")
    source = aggregate_scenarios(records, manifest["spec"])
    reconcile_summary(source, pd.read_csv(results_dir / "capacity_summary.csv"))
    return records, source, manifest, checksums


def dual_threshold_source(
    records: pd.DataFrame, base_source: pd.DataFrame, *, surface_style: dict[str, dict[str, Any]] = SURFACE_STYLE,
) -> pd.DataFrame:
    """Recompute strict CCPs for a selected pair of per-scenario LOLP thresholds."""
    source = base_source.copy()
    capacity_index = pd.MultiIndex.from_frame(source[CAPACITY_KEYS])
    for key, spec in surface_style.items():
        threshold = float(spec["threshold"])
        passing = records.assign(_passes=records.LOLP.lt(threshold)).groupby(
            CAPACITY_KEYS, sort=True
        )["_passes"].sum()
        source[f"LOLP_threshold_{key}"] = threshold
        source[spec["count_column"]] = passing.reindex(capacity_index).to_numpy(dtype=int)
        source[spec["column"]] = source[spec["count_column"]] / source["scenario_count"]
    if "lt20" in surface_style and not np.array_equal(
        source["LOLP_lt20_scenario_count"], base_source["LOLP_lt20_scenario_count"]
    ):
        raise ValueError("Recomputed strict LOLP < 20% counts differ from the established CCP source")
    ordered = sorted(surface_style.items(), key=lambda item: float(item[1]["threshold"]))
    for _, spec in ordered:
        if not np.all(source[str(spec["column"])].between(0.0, 1.0)):
            raise ValueError("CCP values must be fractions between 0 and 1")
    for (_, stricter), (_, looser) in zip(ordered, ordered[1:]):
        if not np.all(source[str(stricter["column"])] <= source[str(looser["column"])]):
            raise ValueError("A stricter LOLP threshold cannot have a higher CCP")
    return source


def surface_grid(source: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the original capacity grid, with no interpolation or densification."""
    pivot = source.pivot(index="pv_capacity_kw", columns="battery_energy_kwh", values=column)
    pivot = pivot.sort_index().sort_index(axis=1)
    if pivot.isna().any().any() or pivot.shape[0] < 2 or pivot.shape[1] < 2:
        raise ValueError("CCP surface requires a complete two-dimensional capacity grid")
    x, y = np.meshgrid(
        pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float)
    )
    return x, y, pivot.to_numpy(dtype=float) * 100.0


def build_figure(
    source: pd.DataFrame, *, city_label: str = "Brisbane", out: Path | None = None,
    surface_style: dict[str, dict[str, Any]] = SURFACE_STYLE,
) -> tuple[Any, dict[str, Any]]:
    """Draw both CCP surfaces with the established contour typography and export geometry."""
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    style._apply_single_outage_readable_font_style(mpl)
    fig = plt.figure(figsize=FIGURE_SIZE_INCHES, constrained_layout=False)
    grid = fig.add_gridspec(**GRID_SPEC)
    ax = fig.add_subplot(grid[1:4, 0:3], projection="3d")
    ax.set_box_aspect((1.35, 1.0, 0.80))

    artists = {}
    legend_handles = []
    ordered = sorted(surface_style.items(), key=lambda item: float(item[1]["threshold"]))
    for key, spec in ordered:
        x, y, z = surface_grid(source, str(spec["column"]))
        color = mpl.colormaps[str(spec["cmap"])](float(spec["color_position"]))
        artists[key] = ax.plot_surface(
            x, y, z,
            rstride=1,
            cstride=1,
            color=color,
            edgecolor=(1.0, 1.0, 1.0, WHITE_GRID_ALPHA),
            linewidth=WHITE_GRID_WIDTH,
            alpha=float(spec["alpha"]),
            shade=False,
            antialiased=True,
        )
        legend_handles.append(
            mpl.patches.Patch(color=color, alpha=float(spec["alpha"]), label=str(spec["label"]))
        )

    ax.set_title(city_label, loc="left", pad=14)
    ax.set_xlabel("BESS energy (kWh)", labelpad=11)
    ax.set_ylabel("PV capacity (kW)", labelpad=11)
    ax.set_zlabel("CCP (%)", labelpad=12)
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_zlim(0.0, 100.0)
    ax.set_xticks([0.0, 5.0, 10.0, 15.0, 20.0])
    ax.set_yticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    zticks = np.arange(0.0, 101.0, 20.0)
    ax.set_zticks(zticks)
    ax.set_zticklabels([f"{int(value)}%" for value in zticks])
    ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.05, 1.0), borderaxespad=0.0)

    metadata = {
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "grid_spec": GRID_SPEC,
        "axes": {"x": "BESS energy (kWh)", "y": "PV capacity (kW)", "z": "CCP (%)"},
        "x_limits": [0.0, 20.0],
        "y_limits": [0.0, 10.0],
        "z_limits": [0.0, 100.0],
        "z_tick_labels": [f"{int(value)}%" for value in zticks],
        "view": {"elevation": VIEW_ELEVATION, "azimuth": VIEW_AZIMUTH},
        "surface_grid_shape": list(surface_grid(source, str(ordered[0][1]["column"]))[2].shape),
        "surfaces": {
            key: {
                "label": spec["label"], "threshold": spec["threshold"], "comparison": "<",
                "colormap": spec["cmap"], "alpha": spec["alpha"],
                "white_grid_width": WHITE_GRID_WIDTH, "white_grid_alpha": WHITE_GRID_ALPHA,
            }
            for key, spec in surface_style.items()
        },
        "smoothing": False,
        "interpolation": False,
        "dpi": 600,
    }
    return fig, metadata


def _readme_text(case_key: str, surface_style: dict[str, dict[str, Any]]) -> str:
    ordered = sorted(surface_style.items(), key=lambda item: float(item[1]["threshold"]))
    lower = ordered[0][1]
    upper = ordered[-1][1]
    return f"""# {case_key}: dual-threshold CCP 3D surface

This directory contains a single three-dimensional figure derived from the completed,
validated capacity sweep. No outage scenario was regenerated and no dispatch was rerun.

- The x axis is BESS energy capacity (kWh), the y axis is PV capacity (kW), and the
  z axis is CCP shown as a percentage.
- Each surface uses the original 11 by 9 capacity grid. No interpolation, smoothing,
  or grid densification is applied.
- `{lower['label']}` uses the existing `{lower['cmap']}` palette. `{upper['label']}`
  uses the existing `{upper['cmap']}` palette. White surface grid lines use the same
  width and transparency as the contour lines.
- Both thresholds use strict comparisons. A scenario with LOLP exactly equal to either
  threshold is not counted as passing that criterion.

The Source Data CSV contains all 99 capacity points, scenario counts, passing counts,
thresholds and CCPs. The validation report records raw-input checksums, the fixed view,
surface style, threshold definitions and output checksums.
"""


def run(
    results_dir: Path | str = DEFAULT_RESULTS, output_dir: Path | str | None = None,
    threshold_pair: str = "10-20",
) -> dict[str, Path]:
    """Export the dual-threshold CCP surface without altering two-dimensional outputs."""
    if threshold_pair not in SURFACE_STYLES:
        raise ValueError(f"Unsupported threshold pair: {threshold_pair}")
    results_path = Path(results_dir).resolve()
    output_path = Path(output_dir).resolve() if output_dir else results_path / "figures"
    records, base_source, manifest, input_hashes = _completed_records(results_path)
    surface_style = SURFACE_STYLES[threshold_pair]
    output_spec = PAIR_OUTPUTS[threshold_pair]
    source = dual_threshold_source(records, base_source, surface_style=surface_style)
    output_path.mkdir(parents=True, exist_ok=True)
    case_key = str(manifest["spec"]["case_key"])
    base = output_path / f"{case_key}_{output_spec['filename']}"
    source_path = write_source_data(base, source)
    paths: dict[str, Path] = {"source_data": source_path}
    _, plt = matplotlib_context(output_path)
    figure, metadata = build_figure(
        source, city_label=case_key.split("_", 1)[0], out=output_path, surface_style=surface_style
    )
    try:
        paths.update(save_publication_figure(figure, base))
    finally:
        plt.close(figure)
    for name, digest in input_hashes.items():
        if file_sha256(results_path / name) != digest:
            raise RuntimeError(f"Input changed while plotting: {name}")

    readme_path = output_path / f"{output_spec['filename']}_README.md"
    readme_path.write_text(_readme_text(case_key, surface_style), encoding="utf-8")
    paths["readme"] = readme_path
    report = {
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_key": case_key,
        "scenario_rows": len(records),
        "capacity_combinations": len(source),
        "scenarios_per_capacity": sorted(source.scenario_count.unique().tolist()),
        "analysis_years": manifest["spec"]["analysis_years"],
        "raw_checksums_match_completed_run": True,
        "input_files_unchanged": True,
        "complete_grid_verified": True,
        "shared_scenarios_verified": True,
        "thresholds": {
            key: {
                "threshold": spec["threshold"], "comparison": "<",
                "count_column": spec["count_column"], "ccp_column": spec["column"],
                "range": [float(source[spec["column"]].min()), float(source[spec["column"]].max())],
            }
            for key, spec in surface_style.items()
        },
        "threshold_pair": threshold_pair,
        "strict_threshold_ccp_order_verified": True,
        "reference_tag": style.REFERENCE_TAG,
        "reference_revision": style.REFERENCE_REVISION,
        "input_sha256": input_hashes,
        "figure": metadata,
        "output_sha256": {path.name: file_sha256(path) for path in paths.values()},
    }
    report_path = output_path / str(output_spec["validation"])
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    paths["validation"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Completed Brisbane capacity-sweep directory")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory (default: RESULTS/figures)")
    parser.add_argument("--threshold-pair", choices=tuple(SURFACE_STYLES), default="10-20",
                        help="Strict LOLP thresholds to overlay (default: 10-20)")
    args = parser.parse_args()
    paths = run(args.results_dir, args.output_dir, args.threshold_pair)
    print(f"Exported the dual-threshold CCP 3D surface, source data and validation to {paths['source_data'].parent}")


if __name__ == "__main__":
    main()
