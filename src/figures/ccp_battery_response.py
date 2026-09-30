"""Plot strict CCP<20% battery-capacity response curves for selected PV capacities.

Run with ``python -m src.figures.ccp_battery_response``.  The module only reads
the completed Brisbane capacity-sweep records; it never reruns dispatch.
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
from .capacity_sweep_contours import DEFAULT_RESULTS, file_sha256
from .ccp_3d_surfaces import _completed_records, dual_threshold_source
from .core import (
    NATURE_PALETTE, apply_nature_style, matplotlib_context, plot_smooth_line,
    save_publication_figure, write_source_data,
)


PV_CAPACITIES = (2.0, 5.0, 8.0)
BATTERY_CAPACITIES = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0)
SMOOTH_SAMPLES = 200
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
THREE_LINE_PURPLE_POSITIONS = (0.45, 0.65, 0.85)
PV_COLOR_POSITIONS = dict(zip(PV_CAPACITIES, THREE_LINE_PURPLE_POSITIONS, strict=True))
CCP_THRESHOLD = 0.20
OUTPUT_STEM = "CCP_LOLP_LT20_Battery_Response_PV2_PV5_PV8"
DEFAULT_PROFILE = "pv2-pv5-pv8"
RESPONSE_PROFILES = {
    DEFAULT_PROFILE: {
        "pv_capacities": PV_CAPACITIES,
        "output_stem": OUTPUT_STEM,
        "readme_description": "Colors are light-to-dark `Purples` for PV 2, 5 and 8 kW, consistent with the CCP contour.",
        "cmap": "Purples",
        "color_positions": PV_COLOR_POSITIONS,
        "design_marker": None,
    },
    "pv3-pv6-pv9": {
        "pv_capacities": (3.0, 6.0, 9.0),
        "output_stem": "CCP_LOLP_LT20_Battery_Response_PV3_PV6_PV9",
        "readme_description": "High-contrast core colours distinguish PV 3, 6 and 9 kW: blue, red and purple.",
        "line_colors": {
            3.0: NATURE_PALETTE["blue"],
            6.0: NATURE_PALETTE["red"],
            9.0: NATURE_PALETTE["purple"],
        },
        "design_marker": {"pv_capacity_kw": 3.0, "battery_energy_kwh": 15.0},
    },
}


def _profile(profile_name: str) -> dict[str, Any]:
    try:
        return RESPONSE_PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"Unknown CCP battery-response profile: {profile_name}") from exc


def extract_response_source(
    source: pd.DataFrame, *, pv_capacities: tuple[float, ...] = PV_CAPACITIES
) -> pd.DataFrame:
    """Select the exact original capacity points used by one three-curve profile."""
    required = {
        "pv_capacity_kw", "battery_energy_kwh", "scenario_count", "analysis_years",
        "LOLP_threshold_lt20", "LOLP_lt20_scenario_count", "CCP_LOLP_lt20",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"CCP source is missing columns: {missing}")
    response = source.loc[
        source.pv_capacity_kw.isin(pv_capacities)
        & source.battery_energy_kwh.isin(BATTERY_CAPACITIES),
        [
            "pv_capacity_kw", "battery_energy_kwh", "scenario_count", "analysis_years",
            "LOLP_threshold_lt20", "LOLP_lt20_scenario_count", "CCP_LOLP_lt20",
        ],
    ].sort_values(["pv_capacity_kw", "battery_energy_kwh"]).reset_index(drop=True)
    expected = pd.MultiIndex.from_product(
        [pv_capacities, BATTERY_CAPACITIES], names=["pv_capacity_kw", "battery_energy_kwh"]
    )
    actual = pd.MultiIndex.from_frame(response[["pv_capacity_kw", "battery_energy_kwh"]])
    if not actual.equals(expected):
        raise ValueError("Response curves require the complete PV-capacity by BESS 0--20 kWh grid")
    if not response.scenario_count.eq(response.scenario_count.iloc[0]).all():
        raise ValueError("Each response point must contain the same scenario count")
    if not response.LOLP_threshold_lt20.eq(CCP_THRESHOLD).all():
        raise ValueError("Response source must use the strict LOLP < 20% threshold")
    if not response.CCP_LOLP_lt20.between(0.0, 1.0).all():
        raise ValueError("CCP values must be fractions between zero and one")
    response.insert(response.columns.get_loc("CCP_LOLP_lt20") + 1, "CCP_LOLP_lt20_percent",
                    response.CCP_LOLP_lt20 * 100.0)
    return response


def _ccp_y_scale(values: pd.Series | np.ndarray) -> tuple[float, np.ndarray]:
    """Use the contour display-step rule to choose a readable zero-based CCP scale."""
    maximum_percent = float(np.asarray(values, dtype=float).max()) * 100.0
    step = style.nice_step(maximum_percent / 5.0) if maximum_percent > 0.0 else 1.0
    upper_percent = min(100.0, max(step, float(np.ceil(maximum_percent / step) * step)))
    return upper_percent / 100.0, np.arange(0.0, upper_percent + step * 0.5, step) / 100.0


def build_ccp_response_figure(
    response: pd.DataFrame,
    *,
    x_column: str,
    x_label: str,
    x_limits: tuple[float, float],
    x_ticks: tuple[float, ...],
    series_column: str,
    series_values: tuple[float, ...],
    series_key_prefix: str,
    series_label: str,
    series_unit: str,
    color_positions: dict[float, float] | None = None,
    line_colors: dict[float, str] | None = None,
    design_marker: dict[str, float] | None = None,
    city_label: str = "Brisbane",
    out: Path | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Draw a CCP response chart using the shared battery-response visual contract."""
    if (color_positions is None) == (line_colors is None):
        raise ValueError("Specify exactly one of color_positions or line_colors")
    required = {x_column, series_column, "CCP_LOLP_lt20"}
    missing = sorted(required - set(response.columns))
    if missing:
        raise ValueError(f"Response figure source is missing columns: {missing}")
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    style._apply_single_outage_readable_font_style(mpl)
    fig = plt.figure(figsize=FIGURE_SIZE_INCHES, constrained_layout=False)
    grid = fig.add_gridspec(**GRID_SPEC)
    ax = fig.add_subplot(grid[1:4, 0:3])
    ax.set_box_aspect(1.0)

    lines: dict[str, Any] = {}
    colors: dict[str, str] = {}
    for series_value in series_values:
        curve = response.loc[response[series_column].eq(series_value)].sort_values(x_column)
        if len(curve) < 3:
            raise ValueError("Each response curve requires at least three source points")
        color = (
            line_colors[series_value]
            if line_colors is not None
            else mpl.colormaps["Purples"](color_positions[series_value])
        )
        series_key = f"{series_key_prefix}_{series_value:g}"
        lines[series_key] = plot_smooth_line(
            ax,
            curve[x_column].to_numpy(dtype=float),
            curve.CCP_LOLP_lt20.to_numpy(dtype=float),
            samples=SMOOTH_SAMPLES,
            show_points=False,
            color=color,
            linewidth=1.45,
            clip_on=True,
            label=f"{series_label} {series_value:g} {series_unit}",
        )
        colors[series_key] = mpl.colors.to_hex(color)

    marker_metadata = None
    if design_marker is not None:
        marker_source = response
        for column, value in design_marker.items():
            if column not in response:
                raise ValueError(f"The configured design marker uses an unknown column: {column}")
            marker_source = marker_source.loc[marker_source[column].eq(value)]
        if len(marker_source) != 1:
            raise ValueError("The configured design marker must match exactly one original response point")
        marker_x = float(marker_source[x_column].iloc[0])
        marker_ccp = float(marker_source.CCP_LOLP_lt20.iloc[0])
        star = ax.scatter(
            [marker_x], [marker_ccp], marker="*", s=56, color="#272727",
            edgecolors="white", linewidths=0.45, zorder=10,
        )
        marker_metadata = {
            **design_marker,
            "CCP_LOLP_lt20": marker_ccp,
            "marker": "*", "size_points_squared": 56,
            "facecolor": "#272727", "edgecolor": "white", "edgewidth": 0.45,
            "zorder": int(star.get_zorder()),
        }

    ax.set_title(city_label, loc="left", pad=5)
    ax.set_xlabel(x_label)
    ax.set_ylabel("CCP (LOLP < 20%)")
    ax.set_xlim(*x_limits)
    ax.set_xticks(x_ticks)
    y_upper, yticks = _ccp_y_scale(response.CCP_LOLP_lt20)
    ax.set_ylim(0.0, y_upper)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{value * 100:g}%" for value in yticks])
    ax.grid(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), borderaxespad=0.0)

    metadata = {
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "grid_spec": GRID_SPEC,
        "x_axis": ax.get_xlabel(),
        "y_axis": ax.get_ylabel(),
        "x_limits": list(x_limits),
        "y_limits": [0.0, y_upper],
        "y_tick_labels": [label.get_text() for label in ax.get_yticklabels()],
        "colors": colors,
        "curve_method": "PCHIP shape-preserving interpolation",
        "rendered_samples_per_curve": SMOOTH_SAMPLES,
        "show_data_points": False,
        "markers": {key: str(line.get_marker()) for key, line in lines.items()},
        "design_marker": marker_metadata,
        "line_width": 1.45,
        "legend": {"loc": "upper left", "bbox_to_anchor": [1.05, 1.0], "borderaxespad": 0.0},
        "style_provider": "ccp_battery_response",
        "smoothing": True,
        "extrapolation": False,
        "dpi": 600,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return fig, metadata


def build_figure(
    response: pd.DataFrame, *, city_label: str = "Brisbane", profile_name: str = DEFAULT_PROFILE,
    out: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Draw marker-free PCHIP battery-capacity response curves with the shared style."""
    profile = _profile(profile_name)
    pv_capacities = tuple(profile["pv_capacities"])
    return build_ccp_response_figure(
        response,
        x_column="battery_energy_kwh",
        x_label="BESS energy (kWh)",
        x_limits=(0.0, 20.0),
        x_ticks=BATTERY_CAPACITIES,
        series_column="pv_capacity_kw",
        series_values=pv_capacities,
        series_key_prefix="PV",
        series_label="PV",
        series_unit="kW",
        color_positions=profile.get("color_positions"),
        line_colors=profile.get("line_colors"),
        design_marker=profile["design_marker"],
        city_label=city_label,
        out=out,
        metadata_extra={
            "profile": profile_name,
            "pv_capacities_kw": list(pv_capacities),
            "battery_capacities_kwh": list(BATTERY_CAPACITIES),
        },
    )


def _readme_text(case_key: str, profile: dict[str, Any]) -> str:
    pv_text = ", ".join(f"{pv:g}" for pv in profile["pv_capacities"])
    marker_note = ""
    if profile["design_marker"] is not None:
        marker = profile["design_marker"]
        marker_note = (
            f"- A small dark star with white outline marks the original PV {marker['pv_capacity_kw']:g} kW, "
            f"BESS {marker['battery_energy_kwh']:g} kWh response point.\n"
        )
    return f"""# {case_key}: CCP<20% battery response curves

This directory contains a three-line battery-capacity response figure derived directly
from the completed, validated Brisbane capacity sweep. No scenario was regenerated and
no dispatch was rerun.

- Curves use PV capacities {pv_text} kW, with BESS energy values from 0 to 20 kWh in 2.5 kWh steps.
- CCP is the strict fraction of scenarios satisfying `LOLP < 0.20`; a scenario exactly at
  20% is not counted as passing.
- The 15 source values remain unsmoothed in the Source Data CSV. The figure uses 200-sample
  PCHIP shape-preserving interpolation per curve, has no data markers, and does not extend
  beyond the original 0--20 kWh domain.
{marker_note}- {profile['readme_description']}
"""


def run(
    results_dir: Path | str = DEFAULT_RESULTS, output_dir: Path | str | None = None,
    *, profile_name: str = DEFAULT_PROFILE,
) -> dict[str, Path]:
    """Export the validated CCP<20% battery response figure and source-data bundle."""
    profile = _profile(profile_name)
    results_path = Path(results_dir).resolve()
    output_path = Path(output_dir).resolve() if output_dir else results_path / "figures"
    records, base_source, manifest, input_hashes = _completed_records(results_path)
    source = dual_threshold_source(records, base_source)
    response = extract_response_source(source, pv_capacities=tuple(profile["pv_capacities"]))
    output_path.mkdir(parents=True, exist_ok=True)
    case_key = str(manifest["spec"]["case_key"])
    output_stem = str(profile["output_stem"])
    base = output_path / f"{case_key}_{output_stem}"
    source_path = write_source_data(base, response)
    paths: dict[str, Path] = {"source_data": source_path}
    _, plt = matplotlib_context(output_path)
    figure, metadata = build_figure(
        response, city_label=case_key.split("_", 1)[0], profile_name=profile_name, out=output_path,
    )
    try:
        paths.update(save_publication_figure(figure, base))
    finally:
        plt.close(figure)
    for name, digest in input_hashes.items():
        if file_sha256(results_path / name) != digest:
            raise RuntimeError(f"Input changed while plotting: {name}")

    readme_path = output_path / f"{output_stem}_README.md"
    readme_path.write_text(_readme_text(case_key, profile), encoding="utf-8")
    paths["readme"] = readme_path
    report = {
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_key": case_key,
        "scenario_rows": len(records),
        "capacity_combinations": len(base_source),
        "scenarios_per_capacity": sorted(base_source.scenario_count.unique().tolist()),
        "analysis_years": manifest["spec"]["analysis_years"],
        "raw_checksums_match_completed_run": True,
        "input_files_unchanged": True,
        "strict_ccp_threshold": CCP_THRESHOLD,
        "comparison": "<",
        "response_points": len(response),
        "profile": profile_name,
        "response_pv_capacities_kw": list(profile["pv_capacities"]),
        "response_battery_capacities_kwh": list(BATTERY_CAPACITIES),
        "ccp_range": [float(response.CCP_LOLP_lt20.min()), float(response.CCP_LOLP_lt20.max())],
        "reference_tag": style.REFERENCE_TAG,
        "reference_revision": style.REFERENCE_REVISION,
        "input_sha256": input_hashes,
        "figure": metadata,
        "output_sha256": {path.name: file_sha256(path) for path in paths.values()},
    }
    report_name = "ccp_battery_response_validation.json" if profile_name == DEFAULT_PROFILE else f"{output_stem}_Validation.json"
    report_path = output_path / report_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    paths["validation"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Completed Brisbane capacity-sweep directory")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory (default: RESULTS/figures)")
    parser.add_argument("--profile", choices=tuple(RESPONSE_PROFILES), default=DEFAULT_PROFILE,
                        help="Named PV-capacity and styling profile")
    args = parser.parse_args()
    paths = run(args.results_dir, args.output_dir, profile_name=args.profile)
    print(f"Exported CCP<20% battery response curves, source data and validation to {paths['source_data'].parent}")


if __name__ == "__main__":
    main()
