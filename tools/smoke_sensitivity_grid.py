"""Smoke-test a small two-parameter factorial run in the unified sensitivity module."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.io.config import load_config  # noqa: E402
from src.sensitivity import run_sensitivity  # noqa: E402


PARAMETER_RANGES = {
    "pv_capex_sgd_per_kw": ("1200", "1400", "200"),
    "battery_capex_sgd_per_kwh": ("400", "450", "50"),
}


def main() -> None:
    input_root = _prepare_input()
    output_root = Path("/tmp/pv_bess_sensitivity_grid_smoke_output")
    config_path = Path("/tmp/pv_bess_sensitivity_grid_smoke_config.yaml")
    if output_root.exists():
        shutil.rmtree(output_root)
    config_path.write_text(
        "\n".join(
            [
                "input:",
                f"  directory: \"{input_root}\"",
                "outputs:",
                "  directory: \"/tmp/pv_bess_sensitivity_grid_smoke_output\"",
                "  save_candidate_dispatch_workbooks: false",
                "dispatch:",
                "  backend: greedy",
                "capacity_search:",
                "  coarse_intervals: 1",
                "  local_refine_divisions: 1",
                "  max_upper_expansions: 0",
                "  local_search_validation_radius: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    paths = run_sensitivity(
        config=config,
        input_dir=input_root,
        output_dir=output_root,
        requested_parameters=list(PARAMETER_RANGES),
        mode="factorial",
        analysis_years=2,
        workers=2,
        warm_start_enabled=True,
        case_filters=["Brisbane_SingleFamily"],
    )
    summary = paths["Brisbane_SingleFamily_sensitivity_results"]
    frame = pd.read_csv(summary)
    if len(frame) != 4:
        raise RuntimeError(f"Factorial smoke expected four unique points; got {len(frame)}")
    if int(frame["is_baseline"].astype(bool).sum()) != 1:
        raise RuntimeError("Unified sensitivity output does not contain exactly one baseline.")
    _check_metrics(frame)
    later = frame[(~frame["is_baseline"].astype(bool)) & (frame["chain_position"].astype(int) > 0)]
    if later.empty or not later["warm_start_used"].astype(bool).all():
        raise RuntimeError("Factorial sensitivity did not warm-start later points in its chains.")
    if list(output_root.rglob("*Dispatch*.csv")):
        raise RuntimeError("Sensitivity smoke wrote dispatch CSV files.")
    print("Unified factorial sensitivity smoke passed.")
    print(f"Output directory: {summary.parent}")


def _check_metrics(frame: pd.DataFrame) -> None:
    required = [
        "annual_cost_sgd_per_year",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "battery_power_kw",
        "EUE_kwh_per_year",
        "LOLP",
        "SSR",
        "SCR",
        "SCR_defined",
        "SCR_status",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Sensitivity summary is missing required columns: {missing}")
    finite = [
        "annual_cost_sgd_per_year",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "battery_power_kw",
        "EUE_kwh_per_year",
        "LOLP",
        "SSR",
    ]
    values = frame[finite].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise RuntimeError("Sensitivity summary has missing required metrics.")
    if (values[["annual_cost_sgd_per_year", "pv_capacity_kw", "battery_energy_kwh", "battery_power_kw", "EUE_kwh_per_year"]] < 0).any().any():
        raise RuntimeError("Sensitivity summary has negative required metrics.")
    if not frame["LOLP"].between(0, 1).all() or not frame["SSR"].between(0, 1).all():
        raise RuntimeError("Sensitivity probability metrics are outside [0, 1].")


def _prepare_input() -> Path:
    input_root = Path("/tmp/pv_bess_sensitivity_grid_smoke_input")
    if input_root.exists():
        shutil.rmtree(input_root)
    case_dir = input_root / "Brisbane_SingleFamily"
    case_dir.mkdir(parents=True)
    source_dir = REPO_ROOT / "Input" / "Brisbane_SingleFamily"
    for source in source_dir.glob("Brisbane_SingleFamily_*.csv"):
        shutil.copy2(source, case_dir / source.name)
    _shrink_parameters(case_dir / "Brisbane_SingleFamily_Parameter.csv")
    return input_root


def _shrink_parameters(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise RuntimeError(f"Parameter file has no header: {path}")
    for row in rows:
        parameter = row["parameter"]
        if parameter in PARAMETER_RANGES:
            start, end, step = PARAMETER_RANGES[parameter]
            row["sensitivity_start"] = start
            row["sensitivity_end"] = end
            row["sensitivity_step"] = step
        elif parameter in {"local_refine_window_kw", "local_refine_window_kwh"}:
            row["base_value"] = "0"
        elif parameter in {"local_refine_step_kw", "local_refine_step_kwh"}:
            row["base_value"] = "1"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
