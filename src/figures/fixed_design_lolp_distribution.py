"""Plot the LOLP distribution for Brisbane's fixed PV--BESS design.

Run with::

    python -m src.figures.fixed_design_lolp_distribution --input-csv RESULTS.csv

The input is a completed set of 1,000 fifteen-year outage scenarios.  This
module only reads that result table; it never runs dispatch or resamples
outages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import _capacity_contour_style as contour_style
from .core import (
    NATURE_PALETTE, apply_nature_style, matplotlib_context, save_publication_figure,
    write_source_data,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_"
    "PV3p25_Battery11p5_15Y_1000Scenarios/figures"
)

EXPECTED_SCENARIO_COUNT = 1000
EXPECTED_ANALYSIS_YEARS = 15
HISTOGRAM_BIN_WIDTH_PERCENT = 5.0
HISTOGRAM_MINIMUM_UPPER_PERCENT = 80.0
CDF_UPPER_PERCENT = 80.0
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
REQUIRED_COLUMNS = (
    "outage_seed",
    "analysis_years",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "LOLP",
)


def file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for an input or generated artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate(input_csv: Path | str) -> pd.DataFrame:
    """Load and validate the exact completed fixed-design scenario table."""
    path = Path(input_csv)
    if not path.is_file():
        raise ValueError(f"Input CSV does not exist: {path}")
    try:
        records = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Input CSV is empty") from exc

    missing = sorted(set(REQUIRED_COLUMNS) - set(records.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    if len(records) != EXPECTED_SCENARIO_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SCENARIO_COUNT} scenario records, found {len(records)}"
        )
    required = records.loc[:, list(REQUIRED_COLUMNS)]
    if required.isna().any().any():
        raise ValueError("Required scenario fields contain missing values")
    numeric = required.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Required scenario fields contain non-finite values")

    seeds = records["outage_seed"].to_numpy(dtype=float)
    if not np.allclose(seeds, np.rint(seeds), rtol=0.0, atol=1e-12):
        raise ValueError("outage_seed values must be integers")
    seed_values = np.rint(seeds).astype(int)
    expected_seeds = np.arange(1, EXPECTED_SCENARIO_COUNT + 1, dtype=int)
    if not np.array_equal(np.sort(seed_values), expected_seeds):
        raise ValueError("outage_seed must contain each integer from 1 through 1000 exactly once")

    years = records["analysis_years"].to_numpy(dtype=float)
    if not np.allclose(years, EXPECTED_ANALYSIS_YEARS, rtol=0.0, atol=1e-12):
        raise ValueError("All scenario records must cover the expected 15 analysis years")
    pv_values = records["pv_capacity_kw"].to_numpy(dtype=float)
    battery_values = records["battery_energy_kwh"].to_numpy(dtype=float)
    if np.any(pv_values < 0.0) or np.any(battery_values < 0.0):
        raise ValueError("PV and battery capacities must be non-negative")
    if not np.allclose(pv_values, pv_values[0], rtol=0.0, atol=1e-12):
        raise ValueError("Input must contain exactly one PV capacity for a single-design distribution")
    if not np.allclose(battery_values, battery_values[0], rtol=0.0, atol=1e-12):
        raise ValueError("Input must contain exactly one battery capacity for a single-design distribution")
    lolp = records["LOLP"].to_numpy(dtype=float)
    if np.any((lolp < 0.0) | (lolp > 1.0)):
        raise ValueError("LOLP values must be fractions between 0 and 1")
    return records.copy()


def design_metadata(records: pd.DataFrame) -> dict[str, float | int | str]:
    """Return the single fixed design carried by a validated scenario table."""
    pv_values = records["pv_capacity_kw"].to_numpy(dtype=float)
    battery_values = records["battery_energy_kwh"].to_numpy(dtype=float)
    if not np.allclose(pv_values, pv_values[0], rtol=0.0, atol=1e-12):
        raise ValueError("A distribution figure requires a single PV capacity")
    if not np.allclose(battery_values, battery_values[0], rtol=0.0, atol=1e-12):
        raise ValueError("A distribution figure requires a single battery capacity")
    pv = float(pv_values[0])
    battery = float(battery_values[0])
    return {
        "pv_capacity_kw": pv,
        "battery_energy_kwh": battery,
        "scenario_count": int(len(records)),
        "legend_label": f"PV {pv:g} kW, BESS {battery:g} kWh",
    }


def ranked_source_data(records: pd.DataFrame) -> pd.DataFrame:
    """Return all scenario LOLP values in deterministic descending order."""
    columns = [
        "outage_seed",
        "analysis_years",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "LOLP",
    ]
    source = records.loc[:, columns].sort_values(
        ["LOLP", "outage_seed"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    source.insert(0, "rank_descending", np.arange(len(source), dtype=int))
    source.insert(source.columns.get_loc("LOLP") + 1, "LOLP_percent", source["LOLP"] * 100.0)
    return source


def histogram_bin_data(lolp: pd.Series | np.ndarray) -> pd.DataFrame:
    """Calculate 5-percentage-point bins that fully cover the LOLP range."""
    percentages = np.asarray(lolp, dtype=float) * 100.0
    maximum = float(percentages.max())
    upper = max(
        HISTOGRAM_MINIMUM_UPPER_PERCENT,
        float(np.ceil(maximum / HISTOGRAM_BIN_WIDTH_PERCENT) * HISTOGRAM_BIN_WIDTH_PERCENT),
    )
    edges = np.arange(0.0, upper + HISTOGRAM_BIN_WIDTH_PERCENT * 0.5, HISTOGRAM_BIN_WIDTH_PERCENT)
    frequencies, edges = np.histogram(percentages, bins=edges)
    return pd.DataFrame(
        {
            "bin_index": np.arange(len(frequencies), dtype=int),
            "LOLP_lower_fraction": edges[:-1] / 100.0,
            "LOLP_upper_fraction": edges[1:] / 100.0,
            "LOLP_lower_percent": edges[:-1],
            "LOLP_upper_percent": edges[1:],
            "frequency": frequencies.astype(int),
        }
    )


def cdf_upper_percent(lolp: pd.Series | np.ndarray) -> float:
    """Return a 5%-aligned CDF limit that visibly contains every raw value."""
    values = np.asarray(lolp, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("CDF upper bound requires one non-empty finite LOLP series")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("CDF upper bound LOLP values must be fractions between zero and one")
    maximum_percent = float(values.max() * 100.0)
    upper = max(
        CDF_UPPER_PERCENT,
        float(np.ceil(maximum_percent / HISTOGRAM_BIN_WIDTH_PERCENT) * HISTOGRAM_BIN_WIDTH_PERCENT),
    )
    if upper <= maximum_percent + 1e-12:
        upper += HISTOGRAM_BIN_WIDTH_PERCENT
    return upper


def empirical_cdf_source_data(
    lolp: pd.Series | np.ndarray, *, upper_percent: float = CDF_UPPER_PERCENT
) -> pd.DataFrame:
    """Return the standard right-continuous empirical CDF and visible upper tail."""
    values = np.asarray(lolp, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("CDF requires one non-empty finite LOLP series")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("CDF LOLP values must be fractions between zero and one")
    upper_fraction = float(upper_percent) / 100.0
    if upper_fraction <= float(values.max()):
        raise ValueError("CDF upper display bound must exceed the largest LOLP value")
    observed_values, counts = np.unique(values, return_counts=True)
    cumulative_counts = np.cumsum(counts, dtype=int)
    output_values = np.append(observed_values, upper_fraction)
    output_counts = np.append(cumulative_counts, len(values))
    return pd.DataFrame(
        {
            "LOLP_value_fraction": output_values,
            "LOLP_value_percent": output_values * 100.0,
            "cumulative_scenario_count": output_counts.astype(int),
            "CDF": output_counts / len(values),
            "value_source": ["observed_LOLP"] * len(observed_values) + ["display_upper_bound"],
        }
    )


def _contour_style_figure(out: Path | None = None) -> tuple[Any, Any, tuple[Any, Any]]:
    """Build a large single panel using the contour module's canvas and typography."""
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    contour_style._apply_single_outage_readable_font_style(mpl)
    fig = plt.figure(figsize=FIGURE_SIZE_INCHES, constrained_layout=False)
    grid = fig.add_gridspec(**GRID_SPEC)
    ax = fig.add_subplot(grid[1:4, 0:3])
    ax.set_box_aspect(1.0)
    return mpl, plt, (fig, ax)


def _design_title() -> str:
    return "Brisbane"


def _set_percent_axis(ax: Any, axis: str, upper_percent: float) -> None:
    ticks = np.arange(0.0, upper_percent + 0.1, 10.0)
    if not np.isclose(ticks[-1], upper_percent, rtol=0.0, atol=1e-12):
        ticks = np.append(ticks, upper_percent)
    labels = [f"{value:g}%" for value in ticks]
    if axis == "x":
        ax.set_xlim(0.0, upper_percent)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    elif axis == "y":
        ax.set_ylim(0.0, upper_percent)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    else:
        raise ValueError("axis must be either 'x' or 'y'")


def build_histogram_figure(
    histogram: pd.DataFrame, *, design: dict[str, float | int | str], out: Path | None = None
) -> tuple[Any, dict[str, Any]]:
    """Draw the fixed-design LOLP frequency histogram in the contour visual style."""
    mpl, _, (fig, ax) = _contour_style_figure(out)
    bar_color = (*mpl.colors.to_rgb(NATURE_PALETTE["teal"]), 0.88)
    ax.bar(
        histogram["LOLP_lower_percent"],
        histogram["frequency"],
        width=HISTOGRAM_BIN_WIDTH_PERCENT,
        align="edge",
        color=bar_color,
        edgecolor=(1.0, 1.0, 1.0, 0.65),
        linewidth=0.32,
        label=str(design["legend_label"]),
    )
    upper = float(histogram["LOLP_upper_percent"].iloc[-1])
    _set_percent_axis(ax, "x", upper)
    ax.set_xlabel("LOLP (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(_design_title(), loc="left", pad=5)
    ax.grid(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), borderaxespad=0.0)
    return fig, {
        "kind": "histogram",
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "grid_spec": GRID_SPEC,
        "box_aspect": ax.get_box_aspect(),
        "color": NATURE_PALETTE["teal"],
        "color_mapping": "core.py teal histogram series",
        "fill_alpha": 0.88,
        "white_edge_width": 0.32,
        "white_edge_alpha": 0.65,
        "bin_width_percent": HISTOGRAM_BIN_WIDTH_PERCENT,
        "bin_upper_percent": upper,
        "frequency_total": int(histogram["frequency"].sum()),
        "x_axis": ax.get_xlabel(),
        "y_axis": ax.get_ylabel(),
        "title": ax.get_title(loc="left"),
        "legend_label": str(design["legend_label"]),
        "smoothing": False,
        "dpi": 600,
    }


def build_empirical_cdf_figure(
    cdf: pd.DataFrame, *, design: dict[str, float | int | str], out: Path | None = None
) -> tuple[Any, dict[str, Any]]:
    """Draw the standard right-continuous empirical CDF for one fixed design."""
    mpl, _, (fig, ax) = _contour_style_figure(out)
    line_color = NATURE_PALETTE["blue"]
    ax.step(
        cdf["LOLP_value_percent"],
        cdf["CDF"] * 100.0,
        where="post", color=line_color,
        linewidth=mpl.rcParams["lines.linewidth"],
        solid_capstyle="round",
        label=str(design["legend_label"]),
    )
    upper = float(cdf["LOLP_value_percent"].iloc[-1])
    _set_percent_axis(ax, "x", upper)
    _set_percent_axis(ax, "y", 100.0)
    ax.set_xlabel("LOLP (%)")
    ax.set_ylabel("CDF (%)")
    ax.set_title(_design_title(), loc="left", pad=5)
    ax.grid(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), borderaxespad=0.0)
    return fig, {
        "kind": "empirical_cdf",
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "grid_spec": GRID_SPEC,
        "box_aspect": ax.get_box_aspect(),
        "color": NATURE_PALETTE["blue"],
        "line_width": float(mpl.rcParams["lines.linewidth"]),
        "step_where": "post",
        "x_limits": [0.0, upper],
        "y_upper_percent": 100.0,
        "cdf_definition": "Count of LOLP values less than or equal to each observed value divided by all scenarios",
        "comparison": "<=",
        "x_axis": ax.get_xlabel(),
        "y_axis": ax.get_ylabel(),
        "title": ax.get_title(loc="left"),
        "legend_label": str(design["legend_label"]),
        "smoothing": False,
        "dpi": 600,
    }


def _readme_text(input_name: str, design: dict[str, float | int | str], cdf_upper: float) -> str:
    return f"""# Brisbane fixed-design LOLP distribution figures

This directory contains two standalone figures from `{input_name}`. No outage
profiles were resampled and no dispatch simulation was rerun.

- Design: {design['legend_label']}.
- Sample: 1,000 unique outage seeds, each spanning 15 analysis years.
- LOLP is retained as a 0--1 fraction in the source table and shown as a
  percentage in both figures.
- The histogram uses 5-percentage-point bins and the `core.py` teal series colour.
  The second figure is a standard right-continuous empirical CDF: at each observed
  LOLP value it reports the fraction of scenarios with LOLP less than or equal to
  that value. It uses no smoothing or interpolation. Its horizontal range is at
  least 0--80% and expands to the next 5% tick when the raw sample needs it
  (this figure: 0--{cdf_upper:g}%).
- Titles identify Brisbane; the shared design is shown in each figure's series legend
  so future designs can use the same comparison convention.

`Brisbane_FixedDesign_LOLP_Source_Data.csv` contains all ranked scenario values;
`Brisbane_FixedDesign_LOLP_Histogram_Bins_Source_Data.csv` contains bin edges and
frequencies; `Brisbane_FixedDesign_LOLP_CDF_Source_Data.csv` contains the
empirical CDF values. `lolp_distribution_validation.json` records the input
checksum, validation, summary statistics, shared contour-style settings, and output checksums.
"""


def run(input_csv: Path | str, output_dir: Path | str | None = None) -> dict[str, Path]:
    """Validate one completed run and export both fixed-design distribution figures."""
    input_path = Path(input_csv).resolve()
    output_path = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    input_digest = file_sha256(input_path)
    records = load_and_validate(input_path)
    design = design_metadata(records)
    ranked = ranked_source_data(records)
    histogram = histogram_bin_data(ranked["LOLP"])
    cdf_upper = cdf_upper_percent(ranked["LOLP"])
    cdf = empirical_cdf_source_data(ranked["LOLP"], upper_percent=cdf_upper)
    if int(histogram["frequency"].sum()) != EXPECTED_SCENARIO_COUNT:
        raise RuntimeError("Histogram frequencies do not account for every scenario")
    if not np.isclose(cdf["CDF"].iloc[-1], 1.0):
        raise RuntimeError("CDF does not reach 100% above the observed LOLP range")

    output_path.mkdir(parents=True, exist_ok=True)
    source_path = write_source_data(output_path / "Brisbane_FixedDesign_LOLP", ranked)
    bins_path = write_source_data(output_path / "Brisbane_FixedDesign_LOLP_Histogram_Bins", histogram)
    cdf_path = write_source_data(output_path / "Brisbane_FixedDesign_LOLP_CDF", cdf)
    paths: dict[str, Path] = {
        "source_data": source_path, "histogram_bins": bins_path, "cdf_source": cdf_path,
    }
    figures: dict[str, Any] = {}
    _, plt = matplotlib_context(output_path)
    for key, builder, base in (
        ("histogram", lambda: build_histogram_figure(histogram, design=design, out=output_path),
         "Brisbane_FixedDesign_LOLP_Histogram"),
        ("cdf", lambda: build_empirical_cdf_figure(cdf, design=design, out=output_path),
         "Brisbane_FixedDesign_LOLP_CDF"),
    ):
        figure, metadata = builder()
        try:
            paths.update(save_publication_figure(figure, output_path / base))
        finally:
            plt.close(figure)
        figures[key] = metadata

    for filename in (
        *(f"Brisbane_FixedDesign_LOLP_Descending_Rank.{suffix}" for suffix in ("png", "svg", "pdf")),
        *(f"Brisbane_FixedDesign_LOLP_CCP_Curve.{suffix}" for suffix in ("png", "svg", "pdf")),
        "Brisbane_FixedDesign_LOLP_CCP_Curve_Source_Data.csv",
    ):
        obsolete = output_path / filename
        if obsolete.exists():
            obsolete.unlink()

    if file_sha256(input_path) != input_digest:
        raise RuntimeError("Input CSV changed while figures were being generated")
    readme_path = output_path / "README.md"
    readme_path.write_text(_readme_text(input_path.name, design, cdf_upper), encoding="utf-8")
    paths["readme"] = readme_path
    report = {
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_file": input_path.name,
        "input_sha256": input_digest,
        "input_unchanged": True,
        "scenario_count": len(ranked),
        "outage_seed_range": [int(ranked.outage_seed.min()), int(ranked.outage_seed.max())],
        "analysis_years": EXPECTED_ANALYSIS_YEARS,
        "pv_capacity_kw": design["pv_capacity_kw"],
        "battery_energy_kwh": design["battery_energy_kwh"],
        "design_legend_label": design["legend_label"],
        "LOLP_definition": "Per-scenario energy-based unmet energy divided by outage demand over the same horizon",
        "LOLP_fraction_range": [float(ranked.LOLP.min()), float(ranked.LOLP.max())],
        "LOLP_fraction_mean": float(ranked.LOLP.mean()),
        "LOLP_fraction_median": float(ranked.LOLP.median()),
        "histogram": {
            "bin_width_percent": HISTOGRAM_BIN_WIDTH_PERCENT,
            "bin_upper_percent": float(histogram.LOLP_upper_percent.iloc[-1]),
            "frequency_total": int(histogram.frequency.sum()),
        },
        "sorting": "LOLP descending; outage_seed ascending only to break exact LOLP ties",
        "empirical_cdf": {
            "definition": "Count of scenario LOLP values less than or equal to each observed value divided by all 1,000 scenarios",
            "comparison": "<=",
            "upper_display_percent": cdf_upper,
            "source_rows": len(cdf),
        },
        "style_reference": {
            "contour_reference_tag": contour_style.REFERENCE_TAG,
            "contour_reference_revision": contour_style.REFERENCE_REVISION,
            "figure_size_inches": list(FIGURE_SIZE_INCHES),
            "grid_spec": GRID_SPEC,
            "font_sizes": {
                "base": contour_style.SINGLE_OUTAGE_BASE_FONT_SIZE,
                "tick": contour_style.SINGLE_OUTAGE_TICK_FONT_SIZE,
                "title": contour_style.SINGLE_OUTAGE_TITLE_FONT_SIZE,
            },
            "histogram_color": NATURE_PALETTE["teal"],
            "histogram_fill_alpha": 0.88,
            "histogram_white_edge_width": 0.32,
            "histogram_white_edge_alpha": 0.65,
            "cdf_color": NATURE_PALETTE["blue"],
        },
        "figures": figures,
        "output_sha256": {path.name: file_sha256(path) for path in paths.values()},
    }
    report_path = output_path / "lolp_distribution_validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["validation"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True,
                        help="Completed 1,000-scenario CSV for one fixed PV/BESS design")
    parser.add_argument("--output-dir", type=Path,
                        help="Figure output directory (default: Brisbane fixed-design figures directory)")
    args = parser.parse_args()
    paths = run(args.input_csv, args.output_dir)
    print(f"Exported two LOLP figures (PNG/SVG/PDF), source data and validation to {paths['source_data'].parent}")


if __name__ == "__main__":
    main()
