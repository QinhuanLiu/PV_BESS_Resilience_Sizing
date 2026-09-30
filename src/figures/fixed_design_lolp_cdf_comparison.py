"""Export a four-design empirical LOLP CDF comparison for Brisbane.

The comparison only reads completed fixed-design result tables.  It uses the
right-continuous empirical definition ``F(x)=P(LOLP <= x)`` and never
resamples outage profiles or reruns dispatch.
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
TWO_BY_TWO_ROOT = (
    ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_2x2_"
    "PV3p25_PV6p5_Battery11p5_Battery14p5_15Y_1000Scenarios"
)
TWO_BY_TWO_SUMMARY = TWO_BY_TWO_ROOT / "four_design_summary.csv"
TWO_BY_TWO_OUTPUT_DIR = TWO_BY_TWO_ROOT / "figures/cdf_comparison"
DEFAULT_INPUTS = (
    ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV3p25_Battery8p5_15Y_1000Scenarios/scenario_results.csv",
    ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV3p25_Battery11p5_15Y_1000Scenarios/figures/Brisbane_FixedDesign_LOLP_Source_Data.csv",
    ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV3p25_Battery14p5_15Y_1000Scenarios/scenario_results.csv",
    ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_PV5_Battery11p5_15Y_1000Scenarios/scenario_results.csv",
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_FixedDesign_LOLP_CDF_Comparison_1000Scenarios/figures"
DESIGNS = (
    (3.25, 8.5, NATURE_PALETTE["teal"]),
    (3.25, 11.5, NATURE_PALETTE["blue"]),
    (3.25, 14.5, NATURE_PALETTE["red"]),
    (5.0, 11.5, NATURE_PALETTE["purple"]),
)
TWO_BY_TWO_DESIGNS = (
    (3.25, 11.5, NATURE_PALETTE["teal"]),
    (3.25, 14.5, NATURE_PALETTE["blue"]),
    (6.5, 11.5, NATURE_PALETTE["red"]),
    (6.5, 14.5, NATURE_PALETTE["purple"]),
)
TWO_BY_TWO_LEGEND_LABELS = (
    "Low PV / Low Battery",
    "Low PV / High Battery",
    "High PV / Low Battery",
    "High PV / High Battery",
)
BASE_NAME = "Brisbane_FixedDesign_LOLP_CDF_Comparison"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(value: float, expected: float) -> bool:
    return bool(np.isclose(value, expected, rtol=0.0, atol=1e-12))


def load_designs(
    input_csvs: tuple[Path | str, ...] | list[Path | str],
    design_specs: tuple[tuple[float, float, str], ...] = DESIGNS,
) -> list[dict[str, Any]]:
    """Load four validated 1,000-scenario inputs in the declared design order."""
    if len(input_csvs) != len(design_specs):
        raise ValueError("CDF comparison requires exactly four fixed-design CSV inputs")
    designs: list[dict[str, Any]] = []
    expected_seeds = np.arange(1, distribution.EXPECTED_SCENARIO_COUNT + 1, dtype=int)
    for raw_path, (expected_pv, expected_battery, color) in zip(input_csvs, design_specs, strict=True):
        path = Path(raw_path).resolve()
        records = distribution.load_and_validate(path)
        metadata = distribution.design_metadata(records)
        if not (_same(float(metadata["pv_capacity_kw"]), expected_pv)
                and _same(float(metadata["battery_energy_kwh"]), expected_battery)):
            raise ValueError(
                f"Expected PV={expected_pv:g} kW, BESS={expected_battery:g} kWh in {path.name}"
            )
        seeds = np.sort(records["outage_seed"].astype(int).to_numpy())
        if not np.array_equal(seeds, expected_seeds):
            raise ValueError(f"Input does not contain each outage seed from 1 through 1000: {path.name}")
        cdf = distribution.empirical_cdf_source_data(records["LOLP"], upper_percent=100.0)
        label = str(metadata["legend_label"])
        cdf.insert(0, "design_label", label)
        cdf.insert(1, "pv_capacity_kw", expected_pv)
        cdf.insert(2, "battery_energy_kwh", expected_battery)
        cdf.insert(3, "series_color", color)
        designs.append({"path": path, "records": records, "metadata": metadata, "cdf": cdf, "color": color})
    return designs


def two_by_two_inputs(summary_csv: Path | str = TWO_BY_TWO_SUMMARY) -> tuple[tuple[Path, ...], Path]:
    """Resolve the verified four-way run without duplicating its scenario tables."""
    summary_path = Path(summary_csv).resolve()
    root = summary_path.parent
    validation_path = root / "validation.json"
    status_path = root / "run_status.json"
    manifest_path = root / "manifest.json"
    for path in (summary_path, validation_path, status_path, manifest_path):
        if not path.is_file():
            raise ValueError(f"Completed 2x2 source is missing {path.name}: {path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_pairs = [(pv, battery) for pv, battery, _ in TWO_BY_TWO_DESIGNS]
    if (validation.get("passed") is not True or status.get("state") != "complete"
            or validation.get("shared_outage_profile_hash_by_seed_verified") is not True
            or validation.get("scenario_count_per_design") != distribution.EXPECTED_SCENARIO_COUNT
            or validation.get("analysis_years") != distribution.EXPECTED_ANALYSIS_YEARS
            or validation.get("four_designs") != [list(pair) for pair in expected_pairs]):
        raise ValueError("The 2x2 source has not passed complete shared-scenario validation")
    if validation.get("output_sha256", {}).get(summary_path.name) != file_sha256(summary_path):
        raise ValueError("The 2x2 summary changed after validation")
    summary = pd.read_csv(summary_path)
    if len(summary) != len(expected_pairs):
        raise ValueError("The 2x2 summary must contain exactly four designs")
    paths: list[Path] = []
    for row, (pv, battery) in zip(summary.itertuples(index=False), expected_pairs, strict=True):
        if not (_same(float(row.pv_capacity_kw), pv) and _same(float(row.battery_energy_kwh), battery)):
            raise ValueError("The 2x2 summary design order does not match the requested four curves")
        path = Path(str(row.source_csv)).resolve()
        label = f"PV{str(pv).replace('.', 'p')}_Battery{str(battery).replace('.', 'p')}"
        recorded_hash = (
            manifest.get("existing_input_sha256", {}).get(label)
            if pv == 3.25 else validation.get("output_sha256", {}).get(path.name)
        )
        if not path.is_file() or recorded_hash != file_sha256(path):
            raise ValueError(f"The 2x2 source file changed or is missing: {path}")
        paths.append(path)
    return tuple(paths), validation_path


def build_figure(
    designs: list[dict[str, Any]], *, out: Path | None = None,
    display_labels: tuple[str, ...] | None = None, legend_inside: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Build the common-scale four series plot using the fixed-design figure canvas."""
    if display_labels is None:
        display_labels = tuple(str(item["metadata"]["legend_label"]) for item in designs)
    if len(display_labels) != len(designs):
        raise ValueError("One display label is required for each CDF curve")
    mpl, _, (fig, ax) = distribution._contour_style_figure(out)
    series: list[dict[str, Any]] = []
    for design, label in zip(designs, display_labels, strict=True):
        cdf = design["cdf"]
        ax.step(
            cdf["LOLP_value_percent"], cdf["CDF"] * 100.0, where="post",
            color=design["color"], linewidth=mpl.rcParams["lines.linewidth"],
            solid_capstyle="round", label=label,
        )
        series.append({
            "label": label,
            "pv_capacity_kw": design["metadata"]["pv_capacity_kw"],
            "battery_energy_kwh": design["metadata"]["battery_energy_kwh"],
            "color": design["color"], "cdf_rows": int(len(cdf)),
        })
    distribution._set_percent_axis(ax, "x", 100.0)
    distribution._set_percent_axis(ax, "y", 100.0)
    ax.set_xlabel("LOLP (%)")
    ax.set_ylabel("CDF (%)")
    ax.set_title("Brisbane", loc="left", pad=5)
    ax.grid(False)
    if legend_inside:
        ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.96), borderaxespad=0.0,
                  frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), borderaxespad=0.0)
    return fig, {
        "kind": "four_design_empirical_cdf_comparison",
        "figure_size_inches": list(distribution.FIGURE_SIZE_INCHES),
        "grid_spec": distribution.GRID_SPEC,
        "box_aspect": ax.get_box_aspect(),
        "x_limits_percent": [0.0, 100.0],
        "y_limits_percent": [0.0, 100.0],
        "cdf_definition": "Count of LOLP values less than or equal to x divided by all scenarios",
        "comparison": "<=",
        "step_where": "post",
        "smoothing": False,
        "series": series,
        "legend_inside_axes": legend_inside,
        "dpi": 600,
    }


def _readme(designs: list[dict[str, Any]]) -> str:
    labels = "\n".join(f"- {item['metadata']['legend_label']}: `{item['path']}`" for item in designs)
    return f"""# Brisbane fixed-design LOLP CDF comparison

This figure overlays four completed 1,000-scenario fixed-design evaluations.
Each curve is a right-continuous empirical CDF, `F(x)=P(LOLP ≤ x)`, without
smoothing, interpolation, resampling, or a reliability-threshold line.

{labels}

The horizontal and vertical axes both use fixed 0--100% percentage ranges.
The four series use the shared core palette: teal, deep blue, red and purple.
"""


def run(input_csvs: tuple[Path | str, ...] | list[Path | str] = DEFAULT_INPUTS,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR, *,
        design_specs: tuple[tuple[float, float, str], ...] = DESIGNS,
        upstream_validation: Path | str | None = None,
        display_labels: tuple[str, ...] | None = None,
        legend_inside: bool = False,
        refresh_figure_only: bool = False) -> dict[str, Path]:
    """Validate inputs, export the comparison figure and auditable source data."""
    designs = load_designs(input_csvs, design_specs)
    upstream_path: Path | None = None
    if upstream_validation is not None:
        upstream_path = Path(upstream_validation).resolve()
        upstream_report = json.loads(upstream_path.read_text(encoding="utf-8"))
        if (upstream_report.get("passed") is not True
                or upstream_report.get("shared_outage_profile_hash_by_seed_verified") is not True):
            raise ValueError("Upstream four-design shared-scenario validation has not passed")
    cdf_checks: list[dict[str, Any]] = []
    for item in designs:
        cdf = item["cdf"]
        if not cdf["CDF"].is_monotonic_increasing or not _same(float(cdf["CDF"].iloc[-1]), 1.0):
            raise ValueError(f"CDF is not monotonic or complete: {item['path']}")
        cdf_checks.append({
            "design_label": item["metadata"]["legend_label"],
            "monotonic": True,
            "final_cdf_percent": float(cdf["CDF"].iloc[-1] * 100.0),
        })
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cdf_source = pd.concat([item["cdf"] for item in designs], ignore_index=True)
    source_path = output / f"{BASE_NAME}_Source_Data.csv"
    if refresh_figure_only:
        if not source_path.is_file() or source_path.read_bytes() != cdf_source.to_csv(index=False).encode("utf-8"):
            raise ValueError("Existing CDF source data is missing or differs from the validated inputs")
    else:
        source_path = write_source_data(output / BASE_NAME, cdf_source)
    readme_path = output / "README.md"
    if refresh_figure_only:
        if not readme_path.is_file():
            raise ValueError(f"Existing CDF README is missing: {readme_path}")
    else:
        readme_path.write_text(_readme(designs), encoding="utf-8")
    figure, metadata = build_figure(designs, out=output,
                                    display_labels=display_labels, legend_inside=legend_inside)
    _, plt = matplotlib_context(output)
    try:
        paths = save_publication_figure(figure, output / BASE_NAME,
                                        allow_inside_legend=legend_inside)
    finally:
        plt.close(figure)
    paths["source_data"] = source_path
    paths["readme"] = readme_path
    report = {
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario_count_per_design": distribution.EXPECTED_SCENARIO_COUNT,
        "analysis_years": distribution.EXPECTED_ANALYSIS_YEARS,
        "cdf_definition": "F(x)=P(LOLP <= x)",
        "comparison": "<=",
        "shared_scenarios_required": True,
        "cdf_checks": cdf_checks,
        "inputs": [
            {"path": str(item["path"]), "sha256": file_sha256(item["path"]),
             "design": item["metadata"]}
            for item in designs
        ],
        "style_reference": {
            "contour_reference_tag": distribution.contour_style.REFERENCE_TAG,
            "contour_reference_revision": distribution.contour_style.REFERENCE_REVISION,
            "core_palette": [item["color"] for item in designs],
            "figure_size_inches": list(distribution.FIGURE_SIZE_INCHES),
            "grid_spec": distribution.GRID_SPEC,
            "fixed_axes_percent": [0.0, 100.0],
        },
        "figure": metadata,
        "output_sha256": {path.name: file_sha256(path) for path in paths.values()},
    }
    if upstream_path is not None:
        report["shared_scenarios_verified"] = True
        report["upstream_validation"] = {
            "path": str(upstream_path), "sha256": file_sha256(upstream_path),
        }
    report_path = output / "lolp_cdf_comparison_validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["validation"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-set", choices=("legacy", "2x2"), default="legacy",
                        help="Use the existing four-design comparison or the completed PV/BESS 2x2 result")
    parser.add_argument("--input-csv", type=Path, action="append",
                        help="One fixed-design 1,000-scenario CSV; supply all four in stated order")
    parser.add_argument("--output-dir", type=Path,
                        help="Figure directory (defaults to the selected design set's directory)")
    args = parser.parse_args()
    if args.design_set == "2x2":
        if args.input_csv:
            parser.error("--design-set 2x2 reads its inputs from the completed four-design summary")
        input_csvs, upstream = two_by_two_inputs()
        output_dir = args.output_dir or TWO_BY_TWO_OUTPUT_DIR
        paths = run(input_csvs, output_dir,
                    design_specs=TWO_BY_TWO_DESIGNS, upstream_validation=upstream,
                    display_labels=TWO_BY_TWO_LEGEND_LABELS, legend_inside=True,
                    refresh_figure_only=output_dir.resolve() == TWO_BY_TWO_OUTPUT_DIR.resolve())
    else:
        paths = run(tuple(args.input_csv) if args.input_csv else DEFAULT_INPUTS,
                    args.output_dir or DEFAULT_OUTPUT_DIR)
    print(f"Exported comparison CDF and validation to {paths['source_data'].parent}")


if __name__ == "__main__":
    main()
