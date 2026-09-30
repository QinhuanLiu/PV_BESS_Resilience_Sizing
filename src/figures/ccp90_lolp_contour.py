"""Draw Brisbane's empirical Q90(LOLP) capacity contour.

For every PV/BESS grid point, the displayed value is the inverse empirical CDF
at 90%: the 90th ascending LOLP order statistic from the 100 paired outage
scenarios.  The accompanying source table also records direct strict
``P(LOLP < 20%) >= 90%`` feasibility, so no strict chance-constraint decision
is inferred from a rounded percentile value.
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

from . import capacity_sweep_contours as contours
from .core import matplotlib_context, save_publication_figure, write_source_data


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_Capacity_Sweep_15Y_100Scenarios"
CCP_TARGET = 0.90
LOLP_CONSTRAINT = 0.20
METRIC = {
    "key": "q90_lolp",
    "column": "Q90_LOLP",
    "title": "Q90 LOLP",
    "label": "Q90 LOLP (%)",
    "unit_type": "probability",
    "cmap": "YlGnBu",
    "filename": "Q90_LOLP_Contour",
}
LEGACY_METRIC = {
    "key": "lolp_ccp90_legacy",
    "column": "LOLP_at_CCP90",
    "title": "LOLP at 90% CCP",
    "label": "LOLP at 90% CCP (%)",
    "unit_type": "probability",
    "cmap": "YlGnBu",
    "filename": "LOLP_at_90pct_CCP_Contour",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ccp_order_statistic_index(scenario_count: int, ccp_target: float = CCP_TARGET) -> int:
    """Return the zero-based inverse empirical-CDF index for a CCP target."""
    if scenario_count <= 0:
        raise ValueError("Scenario count must be positive")
    if not 0.0 < ccp_target <= 1.0:
        raise ValueError("CCP target must be in (0, 1]")
    return int(np.ceil(scenario_count * ccp_target)) - 1


def aggregate_q90_lolp(records: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Validate raw records then add empirical Q90 LOLP and strict CCP feasibility."""
    source = contours.aggregate_scenarios(records, spec)
    scenario_count = int(spec["scenario_count"])
    index = ccp_order_statistic_index(scenario_count)
    quantiles = records.groupby(contours.CAPACITY_KEYS, sort=True)["LOLP"].apply(
        lambda values: float(np.sort(values.to_numpy(dtype=float))[index])
    )
    source["CCP_target"] = CCP_TARGET
    source["CCP_order_statistic_rank"] = index + 1
    source["Q90_LOLP"] = quantiles.reindex(
        pd.MultiIndex.from_frame(source[contours.CAPACITY_KEYS])
    ).to_numpy(dtype=float)
    source["LOLP_constraint"] = LOLP_CONSTRAINT
    source["strict_CCP_LOLP_lt20_feasible"] = source["CCP_LOLP_lt20"].ge(CCP_TARGET)
    if not source["Q90_LOLP"].between(0.0, 1.0).all():
        raise RuntimeError("Computed Q90 LOLP values are outside [0, 1]")
    return source


# Retain the earlier callable name for notebooks or scripts written before the
# Q90-labelled figure was introduced.
aggregate_ccp90_lolp = aggregate_q90_lolp


def _readme(case_key: str, source: pd.DataFrame, spec: dict[str, Any], *, title: str) -> str:
    feasible = int(source["strict_CCP_LOLP_lt20_feasible"].sum())
    return f"""# {case_key}: {title} capacity contour

This standalone contour is recalculated from the completed `{spec['scenario_count']}`
paired raw scenarios at each PV/BESS capacity point. No dispatch was rerun.

- The displayed Q90(LOLP) is the empirical inverse-CDF value at 90%: ascending
  rank {int(source['CCP_order_statistic_rank'].iloc[0])} of {spec['scenario_count']}
  (the smallest value whose empirical `P(LOLP ≤ x)` reaches at least 90%).
- `strict_CCP_LOLP_lt20_feasible` is independently calculated as
  `P(LOLP < 20%) ≥ 90%`; it uses strict `<` and therefore remains authoritative
  if a scenario lies exactly at 20% LOLP.
- {feasible}/{len(source)} capacity points satisfy this strict 90% CCP constraint
  in this completed grid.
- The source CSV stores probabilities as 0–1 fractions. The colour bar uses
  percentages. The axes are BESS energy (kWh) and PV capacity (kW).

The rendering reuses the existing capacity-contour helper, regular 11×9 grid,
YlGnBu risk palette, 0.88 fill transparency, white 0.32-point contour lines,
typography, layout checks and 600 dpi PNG/SVG/PDF export.
No smoothing, interpolation or extra capacity points are introduced.
"""


def run(results_dir: Path | str = DEFAULT_RESULTS, output_dir: Path | str | None = None) -> dict[str, Path]:
    """Export the Q90(LOLP) contour from one complete, checksum-validated sweep."""
    results = Path(results_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else results / "figures"
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((results / "run_status.json").read_text(encoding="utf-8"))
    validation = json.loads((results / "validation.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete" or validation.get("passed") is not True:
        raise ValueError("Only a completed, validated sweep can be plotted")
    input_hashes = {}
    for name in ("scenario_results.csv", "capacity_summary.csv"):
        path = results / name
        input_hashes[name] = file_sha256(path)
        if input_hashes[name] != validation.get("file_sha256", {}).get(name):
            raise ValueError(f"Completed sweep checksum does not match: {name}")
    records = pd.read_csv(results / "scenario_results.csv", float_precision="round_trip")
    source = aggregate_q90_lolp(records, manifest["spec"])
    summary_differences = contours.reconcile_summary(
        source, pd.read_csv(results / "capacity_summary.csv", float_precision="round_trip")
    )
    output.mkdir(parents=True, exist_ok=True)
    case_key = str(manifest["spec"]["case_key"])
    source_path = write_source_data(output / f"{case_key}_{METRIC['filename']}", source)
    figure, figure_metadata = contours.build_figure(
        source, METRIC, city_label=case_key.split("_", 1)[0], out=output
    )
    _, plt = matplotlib_context(output)
    try:
        paths = save_publication_figure(figure, output / f"{case_key}_{METRIC['filename']}")
    finally:
        plt.close(figure)
    paths["source_data"] = source_path
    legacy_source = source.rename(columns={"Q90_LOLP": "LOLP_at_CCP90"})
    legacy_source_path = write_source_data(
        output / f"{case_key}_{LEGACY_METRIC['filename']}", legacy_source
    )
    legacy_figure, legacy_figure_metadata = contours.build_figure(
        legacy_source, LEGACY_METRIC, city_label=case_key.split("_", 1)[0], out=output
    )
    try:
        legacy_paths = save_publication_figure(
            legacy_figure, output / f"{case_key}_{LEGACY_METRIC['filename']}"
        )
    finally:
        plt.close(legacy_figure)
    legacy_paths["source_data"] = legacy_source_path
    for name, digest in input_hashes.items():
        if file_sha256(results / name) != digest:
            raise RuntimeError(f"Input changed while plotting: {name}")
    readme_path = output / f"{METRIC['filename']}_README.md"
    readme_path.write_text(
        _readme(case_key, source, manifest["spec"], title="Q90(LOLP)"), encoding="utf-8"
    )
    paths["readme"] = readme_path
    legacy_readme_path = output / f"{LEGACY_METRIC['filename']}_README.md"
    legacy_readme_path.write_text(
        _readme(case_key, legacy_source, manifest["spec"], title="LOLP at 90% CCP (legacy compatibility)"),
        encoding="utf-8",
    )
    legacy_paths["readme"] = legacy_readme_path
    order_index = ccp_order_statistic_index(int(manifest["spec"]["scenario_count"]))
    common_report = {
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_key": case_key,
        "scenario_rows": int(len(records)),
        "capacity_combinations": int(len(source)),
        "scenarios_per_capacity": int(manifest["spec"]["scenario_count"]),
        "analysis_years": int(manifest["spec"]["analysis_years"]),
        "raw_checksums_match_completed_run": True,
        "input_files_unchanged": True,
        "complete_grid_verified": True,
        "shared_scenarios_verified": True,
        "summary_max_absolute_differences": summary_differences,
        "strict_chance_constraint": {
            "definition": "P(LOLP < 0.20) >= 0.90",
            "comparison": "<",
            "lolp_threshold": LOLP_CONSTRAINT,
            "ccp_target": CCP_TARGET,
            "feasible_capacity_count": int(source.strict_CCP_LOLP_lt20_feasible.sum()),
            "infeasible_capacity_count": int((~source.strict_CCP_LOLP_lt20_feasible).sum()),
            "threshold_equality_scenarios": int(records.LOLP.eq(LOLP_CONSTRAINT).sum()),
        },
        "style_reference": {
            "reference_tag": contours.style.REFERENCE_TAG,
            "reference_revision": contours.style.REFERENCE_REVISION,
            "plotting_helper_sha256": file_sha256(Path(contours.style.__file__)),
            "colormap": METRIC["cmap"],
            "fill_alpha": 0.88,
            "white_line_width": 0.32,
            "white_line_alpha": 0.65,
            "smoothing": False,
            "dpi": 600,
        },
        "input_sha256": input_hashes,
    }
    report = {
        **common_report,
        "display_metric": {
            "name": "Q90_LOLP",
            "definition": "90th ascending per-scenario LOLP value (inverse empirical CDF at 0.90)",
            "cdf_comparison": "<=",
            "ccp_target": CCP_TARGET,
            "order_statistic_rank": order_index + 1,
            "order_statistic_zero_based_index": order_index,
            "range": [float(source.Q90_LOLP.min()), float(source.Q90_LOLP.max())],
        },
        "figure": figure_metadata,
        "output_sha256": {path.name: file_sha256(path) for path in paths.values()},
    }
    report_path = output / "q90_lolp_contour_validation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["validation"] = report_path
    legacy_report = {
        **common_report,
        "display_metric": {
            "name": "LOLP_at_CCP90",
            "definition": "Compatibility label for the same 90th ascending per-scenario LOLP value",
            "cdf_comparison": "<=",
            "ccp_target": CCP_TARGET,
            "order_statistic_rank": order_index + 1,
            "order_statistic_zero_based_index": order_index,
            "range": [float(legacy_source.LOLP_at_CCP90.min()), float(legacy_source.LOLP_at_CCP90.max())],
        },
        "figure": legacy_figure_metadata,
        "output_sha256": {path.name: file_sha256(path) for path in legacy_paths.values()},
    }
    legacy_report_path = output / "ccp90_lolp_contour_validation.json"
    legacy_report_path.write_text(
        json.dumps(legacy_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["legacy_validation"] = legacy_report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = run(args.results_dir, args.output_dir)
    print(f"Exported Q90(LOLP) contour (PNG/SVG/PDF), source data and validation to {paths['source_data'].parent}")


if __name__ == "__main__":
    main()
