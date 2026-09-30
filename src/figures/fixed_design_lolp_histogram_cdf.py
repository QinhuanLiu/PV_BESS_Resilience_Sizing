"""Export combined frequency-histogram and empirical-CDF figures for fixed designs.

With no arguments, the module renders the four completed Brisbane fixed designs.
It only reads their retained 1,000-scenario LOLP records and never reruns dispatch
or resamples outage profiles.
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

from . import fixed_design_lolp_distribution as distribution
from .core import NATURE_PALETTE, matplotlib_context, save_publication_figure, write_source_data


ROOT = Path(__file__).resolve().parents[2]
BASE_NAME = "Brisbane_FixedDesign_LOLP_Histogram_CDF"
BIN_WIDTH_PERCENT = 5.0
X_UPPER_PERCENT = 100.0
FREQUENCY_UPPER_PERCENT = 40.0
CDF_UPPER_PERCENT = 100.0
DEFAULT_DESIGNS = (
    {
        "input_csv": ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV3p25_Battery8p5_15Y_1000Scenarios/scenario_results.csv",
        "output_dir": ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV3p25_Battery8p5_15Y_1000Scenarios/figures",
        "pv_capacity_kw": 3.25,
        "battery_energy_kwh": 8.5,
    },
    {
        "input_csv": ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV3p25_Battery11p5_15Y_1000Scenarios/figures/Brisbane_FixedDesign_LOLP_Source_Data.csv",
        "output_dir": ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV3p25_Battery11p5_15Y_1000Scenarios/figures",
        "pv_capacity_kw": 3.25,
        "battery_energy_kwh": 11.5,
    },
    {
        "input_csv": ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV3p25_Battery14p5_15Y_1000Scenarios/scenario_results.csv",
        "output_dir": ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV3p25_Battery14p5_15Y_1000Scenarios/figures",
        "pv_capacity_kw": 3.25,
        "battery_energy_kwh": 14.5,
    },
    {
        "input_csv": ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV5_Battery11p5_15Y_1000Scenarios/scenario_results.csv",
        "output_dir": ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV5_Battery11p5_15Y_1000Scenarios/figures",
        "pv_capacity_kw": 5.0,
        "battery_energy_kwh": 11.5,
    },
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a source or delivered artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_histogram_bin_data(lolp: pd.Series | np.ndarray) -> pd.DataFrame:
    """Return twenty 5%-wide bins over the shared 0--100% LOLP range."""
    values = np.asarray(lolp, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Histogram requires one non-empty finite LOLP series")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Histogram LOLP values must be fractions between zero and one")
    edges = np.arange(0.0, X_UPPER_PERCENT + BIN_WIDTH_PERCENT * 0.5, BIN_WIDTH_PERCENT)
    frequency, edges = np.histogram(values * 100.0, bins=edges)
    return pd.DataFrame(
        {
            "bin_index": np.arange(len(frequency), dtype=int),
            "LOLP_lower_fraction": edges[:-1] / 100.0,
            "LOLP_upper_fraction": edges[1:] / 100.0,
            "LOLP_lower_percent": edges[:-1],
            "LOLP_upper_percent": edges[1:],
            "frequency": frequency.astype(int),
            "frequency_percent": frequency / len(values) * 100.0,
        }
    )


def fixed_empirical_cdf_source_data(lolp: pd.Series | np.ndarray) -> pd.DataFrame:
    """Return a right-continuous empirical CDF with a fixed 100% endpoint."""
    values = np.asarray(lolp, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("CDF requires one non-empty finite LOLP series")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("CDF LOLP values must be fractions between zero and one")
    observed, counts = np.unique(values, return_counts=True)
    cumulative = np.cumsum(counts, dtype=int)
    if observed[-1] < 1.0:
        output_values = np.append(observed, 1.0)
        output_counts = np.append(cumulative, len(values))
        source = ["observed_LOLP"] * len(observed) + ["display_upper_bound"]
    else:
        output_values, output_counts = observed, cumulative
        source = ["observed_LOLP"] * len(observed)
    return pd.DataFrame(
        {
            "LOLP_value_fraction": output_values,
            "LOLP_value_percent": output_values * 100.0,
            "cumulative_scenario_count": output_counts.astype(int),
            "CDF": output_counts / len(values),
            "value_source": source,
        }
    )


def _set_frequency_axis(ax: Any, upper_percent: float = FREQUENCY_UPPER_PERCENT) -> None:
    ticks = np.arange(0.0, upper_percent + 0.1, 10.0)
    ax.set_ylim(0.0, upper_percent)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{value:g}%" for value in ticks])


def build_figure(
    histogram: pd.DataFrame,
    cdf: pd.DataFrame,
    *,
    design: dict[str, float | int | str],
    out: Path | None = None,
    frequency_upper_percent: float = FREQUENCY_UPPER_PERCENT,
) -> tuple[Any, dict[str, Any]]:
    """Draw one unified 5%-bin histogram and right-continuous CDF panel."""
    mpl, _, (figure, frequency_axis) = distribution._contour_style_figure(out)
    cdf_axis = frequency_axis.twinx()
    teal = NATURE_PALETTE["teal"]
    blue = NATURE_PALETTE["blue"]
    bars = frequency_axis.bar(
        histogram["LOLP_lower_percent"], histogram["frequency_percent"],
        width=BIN_WIDTH_PERCENT, align="edge", color=(*mpl.colors.to_rgb(teal), 0.88),
        edgecolor=(1.0, 1.0, 1.0, 0.65), linewidth=0.32,
        label="Histogram (frequency)", zorder=1,
    )
    line = cdf_axis.step(
        cdf["LOLP_value_percent"], cdf["CDF"] * 100.0,
        where="post", color=blue, linewidth=mpl.rcParams["lines.linewidth"],
        solid_capstyle="round", label="CDF", zorder=2,
    )[0]
    distribution._set_percent_axis(frequency_axis, "x", X_UPPER_PERCENT)
    _set_frequency_axis(frequency_axis, frequency_upper_percent)
    distribution._set_percent_axis(cdf_axis, "y", CDF_UPPER_PERCENT)
    frequency_axis.set_xlabel("LOLP (%)")
    frequency_axis.set_ylabel("Frequency (%)", color=teal)
    cdf_axis.set_ylabel("CDF (%)", color=blue)
    frequency_axis.tick_params(axis="y", colors=teal)
    cdf_axis.tick_params(axis="y", colors=blue)
    frequency_axis.spines["left"].set_color(teal)
    cdf_axis.spines["right"].set_visible(True)
    cdf_axis.spines["right"].set_color(blue)
    frequency_axis.set_title("Brisbane", loc="left", pad=5)
    frequency_axis.grid(False)
    cdf_axis.grid(False)
    legend = frequency_axis.legend(
        handles=[bars, line], labels=["Histogram (frequency)", "CDF"],
        title=str(design["legend_label"]), loc="upper left", bbox_to_anchor=(1.31, 1.0),
        borderaxespad=0.0,
    )
    legend.get_title().set_fontsize(mpl.rcParams["legend.fontsize"])
    # ``twinx`` initially spans the GridSpec cell while the square primary
    # axis is narrowed by ``set_box_aspect``. Align the two data rectangles
    # after Matplotlib has resolved the primary axis geometry.
    figure.canvas.draw()
    cdf_axis.set_position(frequency_axis.get_position())
    figure.canvas.draw()
    return figure, {
        "kind": "histogram_empirical_cdf_dual_y_axis",
        "figure_size_inches": list(distribution.FIGURE_SIZE_INCHES),
        "grid_spec": distribution.GRID_SPEC,
        "box_aspect": frequency_axis.get_box_aspect(),
        "x_limits_percent": [0.0, X_UPPER_PERCENT],
        "frequency_y_limits_percent": [0.0, frequency_upper_percent],
        "cdf_y_limits_percent": [0.0, CDF_UPPER_PERCENT],
        "bin_count": int(len(histogram)),
        "bin_width_percent": BIN_WIDTH_PERCENT,
        "frequency_total": int(histogram["frequency"].sum()),
        "frequency_percent_total": float(histogram["frequency_percent"].sum()),
        "cdf_definition": "Count of LOLP values less than or equal to x divided by all scenarios",
        "cdf_comparison": "<=",
        "cdf_step_where": "post",
        "histogram_color": teal,
        "histogram_fill_alpha": 0.88,
        "histogram_white_edge_width": 0.32,
        "histogram_white_edge_alpha": 0.65,
        "cdf_color": blue,
        "cdf_line_width": float(mpl.rcParams["lines.linewidth"]),
        "frequency_axis_label": frequency_axis.get_ylabel(),
        "cdf_axis_label": cdf_axis.get_ylabel(),
        "legend_title": legend.get_title().get_text(),
        "legend_labels": [text.get_text() for text in legend.get_texts()],
        "title": frequency_axis.get_title(loc="left"),
        "smoothing": False,
        "dpi": 600,
    }


def _readme(design: dict[str, float | int | str], frequency_upper_percent: float) -> str:
    return f"""# Brisbane fixed-design LOLP histogram and CDF

This figure combines the 5-percentage-point LOLP histogram and the standard
right-continuous empirical CDF for {design['legend_label']}. It reads one
completed set of 1,000 fifteen-year outage scenarios and does not rerun
dispatch or resample outages.

- The common horizontal axis is LOLP from 0--100%.
- The teal left axis is bin frequency as a percentage of all 1,000 scenarios.
  It is fixed at 0--{frequency_upper_percent:g}% for direct comparison with the selected fixed designs.
- The blue right axis is `CDF (%) = P(LOLP ≤ x)`, fixed at 0--100%.
- Bars use 20 fixed 5%-wide bins; the CDF uses raw observed values without
  smoothing, interpolation, resampling or a LOLP threshold line.

The two source tables contain the fixed-bin counts and percentages plus the CDF
steps. The validation report records input identity, style settings, axis limits,
data checks and delivered-file checksums.
"""


def run(
    input_csv: Path | str,
    output_dir: Path | str,
    *,
    frequency_upper_percent: float = FREQUENCY_UPPER_PERCENT,
) -> dict[str, Path]:
    """Export one combined figure from a validated 1,000-scenario fixed design."""
    input_path = Path(input_csv).resolve()
    output = Path(output_dir).resolve()
    input_digest = file_sha256(input_path)
    records = distribution.load_and_validate(input_path)
    design = distribution.design_metadata(records)
    histogram = fixed_histogram_bin_data(records["LOLP"])
    cdf = fixed_empirical_cdf_source_data(records["LOLP"])
    if (not np.isfinite(frequency_upper_percent) or frequency_upper_percent < FREQUENCY_UPPER_PERCENT
            or frequency_upper_percent > 100.0
            or not np.isclose(frequency_upper_percent / 10.0, round(frequency_upper_percent / 10.0))):
        raise ValueError("Frequency axis upper bound must be a 10% step between 40% and 100%")
    if float(histogram["frequency_percent"].max()) > frequency_upper_percent + 1e-9:
        raise ValueError("Frequency axis upper bound would clip a histogram bar")
    if len(histogram) != int(X_UPPER_PERCENT / BIN_WIDTH_PERCENT):
        raise RuntimeError("Histogram does not use the required twenty bins")
    if int(histogram["frequency"].sum()) != len(records) or not np.isclose(histogram["frequency_percent"].sum(), 100.0):
        raise RuntimeError("Histogram does not account for every scenario as a percentage")
    if not cdf["CDF"].is_monotonic_increasing or not np.isclose(cdf["CDF"].iloc[-1], 1.0):
        raise RuntimeError("CDF must be monotonic and end at 100%")

    output.mkdir(parents=True, exist_ok=True)
    bins_path = write_source_data(output / f"{BASE_NAME}_Bins", histogram)
    cdf_path = write_source_data(output / f"{BASE_NAME}_CDF", cdf)
    figure, figure_metadata = build_figure(
        histogram, cdf, design=design, out=output,
        frequency_upper_percent=frequency_upper_percent,
    )
    _, plt = matplotlib_context(output)
    try:
        paths = save_publication_figure(figure, output / BASE_NAME)
    finally:
        plt.close(figure)
    paths["bins_source"] = bins_path
    paths["cdf_source"] = cdf_path
    if file_sha256(input_path) != input_digest:
        raise RuntimeError("Input CSV changed while the combined figure was being generated")
    readme_path = output / "LOLP_Histogram_CDF_README.md"
    readme_path.write_text(_readme(design, frequency_upper_percent), encoding="utf-8")
    paths["readme"] = readme_path
    report = {
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": input_digest,
        "input_unchanged": True,
        "design": design,
        "scenario_count": int(len(records)),
        "outage_seed_range": [int(records.outage_seed.min()), int(records.outage_seed.max())],
        "analysis_years": distribution.EXPECTED_ANALYSIS_YEARS,
        "histogram": {
            "bin_count": int(len(histogram)), "bin_width_percent": BIN_WIDTH_PERCENT,
            "range_percent": [0.0, X_UPPER_PERCENT],
            "frequency_total": int(histogram.frequency.sum()),
            "frequency_percent_total": float(histogram.frequency_percent.sum()),
        },
        "empirical_cdf": {
            "definition": "F(x)=P(LOLP <= x)", "comparison": "<=",
            "right_continuous": True, "ends_at_one": True,
        },
        "style_reference": {
            "contour_reference_tag": distribution.contour_style.REFERENCE_TAG,
            "contour_reference_revision": distribution.contour_style.REFERENCE_REVISION,
            "figure_size_inches": list(distribution.FIGURE_SIZE_INCHES),
            "grid_spec": distribution.GRID_SPEC,
            "histogram_color": NATURE_PALETTE["teal"],
            "cdf_color": NATURE_PALETTE["blue"],
            "histogram_fill_alpha": 0.88,
            "histogram_white_edge_width": 0.32,
            "histogram_white_edge_alpha": 0.65,
            "x_limits_percent": [0.0, X_UPPER_PERCENT],
            "frequency_y_limits_percent": [0.0, frequency_upper_percent],
            "cdf_y_limits_percent": [0.0, CDF_UPPER_PERCENT],
        },
        "figure": figure_metadata,
        "output_sha256": {path.name: file_sha256(path) for path in paths.values()},
    }
    report_path = output / "lolp_histogram_cdf_validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["validation"] = report_path
    return paths


def run_all() -> dict[str, dict[str, Path]]:
    """Render the planned combined figure for all four completed fixed designs."""
    outputs: dict[str, dict[str, Path]] = {}
    for item in DEFAULT_DESIGNS:
        paths = run(item["input_csv"], item["output_dir"])
        actual = distribution.load_and_validate(item["input_csv"])
        design = distribution.design_metadata(actual)
        if not (np.isclose(float(design["pv_capacity_kw"]), item["pv_capacity_kw"])
                and np.isclose(float(design["battery_energy_kwh"]), item["battery_energy_kwh"])):
            raise ValueError("Default fixed-design input has an unexpected capacity label")
        outputs[str(design["legend_label"])] = paths
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, help="One completed 1,000-scenario fixed-design CSV")
    parser.add_argument("--output-dir", type=Path, help="Output directory for one combined figure")
    args = parser.parse_args()
    if (args.input_csv is None) != (args.output_dir is None):
        parser.error("--input-csv and --output-dir must be supplied together")
    if args.input_csv is None:
        outputs = run_all()
        print(f"Exported {len(outputs)} combined histogram-CDF figures for the default fixed designs")
    else:
        paths = run(args.input_csv, args.output_dir)
        print(f"Exported combined histogram-CDF figure to {paths['bins_source'].parent}")


if __name__ == "__main__":
    main()
