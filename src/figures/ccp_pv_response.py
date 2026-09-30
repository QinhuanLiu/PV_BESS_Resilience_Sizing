"""Plot strict CCP<20% PV-capacity response curves for selected BESS capacities.

Run with ``python -m src.figures.ccp_pv_response``.  The module reads only the
completed Brisbane capacity-sweep records; it does not rerun dispatch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import _capacity_contour_style as style
from .capacity_sweep_contours import DEFAULT_RESULTS, file_sha256
from .ccp_battery_response import (
    CCP_THRESHOLD,
    FIGURE_SIZE_INCHES,
    GRID_SPEC,
    SMOOTH_SAMPLES,
    THREE_LINE_PURPLE_POSITIONS,
    build_ccp_response_figure,
)
from .ccp_3d_surfaces import _completed_records, dual_threshold_source
from .core import matplotlib_context, save_publication_figure, write_source_data


BESS_CAPACITIES = (10.0, 12.5, 15.0)
PV_CAPACITIES = tuple(float(value) for value in range(0, 9))
COLOR_POSITIONS = dict(zip(BESS_CAPACITIES, THREE_LINE_PURPLE_POSITIONS, strict=True))
OUTPUT_STEM = "CCP_LOLP_LT20_PV_Response_BESS10_BESS12p5_BESS15"


def extract_response_source(source: pd.DataFrame) -> pd.DataFrame:
    """Select the exact original capacity points used by the three response curves."""
    required = {
        "pv_capacity_kw",
        "battery_energy_kwh",
        "scenario_count",
        "analysis_years",
        "LOLP_threshold_lt20",
        "LOLP_lt20_scenario_count",
        "CCP_LOLP_lt20",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"CCP source is missing columns: {missing}")
    response = source.loc[
        source.battery_energy_kwh.isin(BESS_CAPACITIES)
        & source.pv_capacity_kw.isin(PV_CAPACITIES),
        [
            "pv_capacity_kw",
            "battery_energy_kwh",
            "scenario_count",
            "analysis_years",
            "LOLP_threshold_lt20",
            "LOLP_lt20_scenario_count",
            "CCP_LOLP_lt20",
        ],
    ].sort_values(["battery_energy_kwh", "pv_capacity_kw"]).reset_index(drop=True)
    expected = pd.MultiIndex.from_product(
        [BESS_CAPACITIES, PV_CAPACITIES], names=["battery_energy_kwh", "pv_capacity_kw"]
    )
    actual = pd.MultiIndex.from_frame(response[["battery_energy_kwh", "pv_capacity_kw"]])
    if not actual.equals(expected):
        raise ValueError("PV response curves require the complete BESS by PV 0--8 kW grid")
    if response.scenario_count.empty or not response.scenario_count.eq(response.scenario_count.iloc[0]).all():
        raise ValueError("Each response point must contain the same scenario count")
    if not response.LOLP_threshold_lt20.eq(CCP_THRESHOLD).all():
        raise ValueError("Response source must use the strict LOLP < 20% threshold")
    if not response.CCP_LOLP_lt20.between(0.0, 1.0).all():
        raise ValueError("CCP values must be fractions between zero and one")
    response.insert(
        response.columns.get_loc("CCP_LOLP_lt20") + 1,
        "CCP_LOLP_lt20_percent",
        response.CCP_LOLP_lt20 * 100.0,
    )
    return response


def build_figure(
    response: pd.DataFrame, *, city_label: str = "Brisbane", out: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Draw marker-free PCHIP PV response curves using the battery-response style."""
    return build_ccp_response_figure(
        response,
        x_column="pv_capacity_kw",
        x_label="PV capacity (kW)",
        x_limits=(0.0, 8.0),
        x_ticks=PV_CAPACITIES,
        series_column="battery_energy_kwh",
        series_values=BESS_CAPACITIES,
        series_key_prefix="BESS",
        series_label="BESS",
        series_unit="kWh",
        color_positions=COLOR_POSITIONS,
        city_label=city_label,
        out=out,
        metadata_extra={
            "bess_capacities_kwh": list(BESS_CAPACITIES),
            "pv_capacities_kw": list(PV_CAPACITIES),
        },
    )


def _readme_text(case_key: str) -> str:
    bess_text = ", ".join(f"{bess:g}" for bess in BESS_CAPACITIES)
    return f"""# {case_key}: CCP<20% PV response curves

This directory contains a three-line PV-capacity response figure derived directly
from the completed, validated Brisbane capacity sweep. No outage scenario was
regenerated and no dispatch was rerun.

- Curves use BESS energy capacities {bess_text} kWh, with PV capacities from 0 to 8 kW in 1 kW steps.
- CCP is the strict fraction of scenarios satisfying `LOLP < 0.20`; a scenario exactly at
  20% is not counted as passing.
- The 27 source values remain unsmoothed in the Source Data CSV. The figure uses 200-sample
  PCHIP shape-preserving interpolation per curve, has no data markers, and does not extend
  beyond the original 0--8 kW domain.
- Colors are light-to-dark `Purples` for BESS 10, 12.5 and 15 kWh, matching the existing
  CCP response-curve style.
"""


def run(
    results_dir: Path | str = DEFAULT_RESULTS, output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Export the validated CCP<20% PV response figure and source-data bundle."""
    results_path = Path(results_dir).resolve()
    output_path = Path(output_dir).resolve() if output_dir else results_path / "figures"
    records, base_source, manifest, input_hashes = _completed_records(results_path)
    source = dual_threshold_source(records, base_source)
    response = extract_response_source(source)
    output_path.mkdir(parents=True, exist_ok=True)
    case_key = str(manifest["spec"]["case_key"])
    base = output_path / f"{case_key}_{OUTPUT_STEM}"
    source_path = write_source_data(base, response)
    paths: dict[str, Path] = {"source_data": source_path}
    _, plt = matplotlib_context(output_path)
    figure, metadata = build_figure(response, city_label=case_key.split("_", 1)[0], out=output_path)
    try:
        paths.update(save_publication_figure(figure, base))
    finally:
        plt.close(figure)
    for name, digest in input_hashes.items():
        if file_sha256(results_path / name) != digest:
            raise RuntimeError(f"Input changed while plotting: {name}")

    readme_path = output_path / f"{OUTPUT_STEM}_README.md"
    readme_path.write_text(_readme_text(case_key), encoding="utf-8")
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
        "complete_grid_verified": True,
        "shared_scenarios_verified": True,
        "strict_ccp_threshold": CCP_THRESHOLD,
        "comparison": "<",
        "response_points": len(response),
        "response_bess_capacities_kwh": list(BESS_CAPACITIES),
        "response_pv_capacities_kw": list(PV_CAPACITIES),
        "ccp_range": [float(response.CCP_LOLP_lt20.min()), float(response.CCP_LOLP_lt20.max())],
        "reference_tag": style.REFERENCE_TAG,
        "reference_revision": style.REFERENCE_REVISION,
        "input_sha256": input_hashes,
        "figure": metadata,
        "output_sha256": {path.name: file_sha256(path) for path in paths.values()},
    }
    report_path = output_path / f"{OUTPUT_STEM}_Validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    paths["validation"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Completed Brisbane capacity-sweep directory")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory (default: RESULTS/figures)")
    args = parser.parse_args()
    paths = run(args.results_dir, args.output_dir)
    print(f"Exported CCP<20% PV response curves, source data and validation to {paths['source_data'].parent}")


if __name__ == "__main__":
    main()
