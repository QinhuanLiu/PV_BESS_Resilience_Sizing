from __future__ import annotations

import csv
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.app.workflow import run_selected  # noqa: E402
from src.io.config import load_config  # noqa: E402
from src.io.data_loader import discover_input_cases, load_input_case  # noqa: E402


def main() -> None:
    input_root = _prepare_smoke_input()
    output_root = Path("/tmp/pv_bess_core_smoke")

    cases = discover_input_cases(input_root)
    if not any(case.case_key == "SG_UTown" for case in cases):
        raise RuntimeError("Smoke check expected the SG_UTown input case.")

    config = load_config(REPO_ROOT / "config.yaml")
    config.setdefault("dispatch", {})["backend"] = "greedy"
    config.setdefault("outputs", {})["save_candidate_dispatch_workbooks"] = False
    config.setdefault("capacity_search", {})["coarse_intervals"] = 1
    config.setdefault("capacity_search", {})["local_refine_divisions"] = 1
    config.setdefault("capacity_search", {})["max_upper_expansions"] = 0

    sg_case = next(case for case in cases if case.case_key == "SG_UTown")
    frame, _, _, _ = load_input_case(sg_case, config)
    if len(frame) != 8760:
        raise RuntimeError(f"Expected 8760 hourly rows for SG_UTown; got {len(frame)}.")

    if output_root.exists():
        shutil.rmtree(output_root)
    paths = run_selected(
        config,
        input_root,
        output_root,
        case_filters=["SG_UTown"],
        tou_analysis=True,
    )
    required_keys = {
        "SG_UTown_baseline_optimal_design",
        "SG_UTown_baseline_candidate_summary",
        "SG_UTown_baseline_dispatch",
        "SG_UTown_baseline_outage_summary",
        "SG_UTown_baseline_yearly_summary",
        "SG_UTown_tou_analysis_summary",
    }
    missing = sorted(required_keys - set(paths))
    if missing:
        raise RuntimeError(f"Smoke check did not produce expected outputs: {missing}")
    for key in required_keys:
        if not paths[key].exists():
            raise RuntimeError(f"Smoke output path does not exist for {key}: {paths[key]}")
    tou_analysis = output_root / "SG" / "SG_UTown_TOU_Analysis" / "SG_UTown_TOU_Analysis_Summary.csv"
    if not tou_analysis.exists():
        raise RuntimeError(f"Smoke check expected TOU analysis summary: {tou_analysis}")
    with tou_analysis.open(newline="", encoding="utf-8") as handle:
        tou_rows = list(csv.DictReader(handle))
    expected_k = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    actual_k = sorted(round(float(row["tou_spread_k"]), 6) for row in tou_rows)
    if actual_k != expected_k:
        raise RuntimeError(f"Smoke check expected TOU k values {expected_k}, got {actual_k}")
    required_tou_columns = {
        "tou_spread_k",
        "tou_adjusted_mean_sgd_per_kwh",
        "tou_adjusted_spread_sgd_per_kwh",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "system_total_cost_sgd_per_year",
        "energy_based_LOLP",
    }
    missing_tou_columns = sorted(required_tou_columns - set(tou_rows[0]))
    if missing_tou_columns:
        raise RuntimeError(f"Smoke check TOU summary missing columns: {missing_tou_columns}")

    print("Core smoke check passed.")
    print(f"Output directory: {output_root}")


def _prepare_smoke_input() -> Path:
    input_root = Path("/tmp/pv_bess_core_smoke_input")
    if input_root.exists():
        shutil.rmtree(input_root)
    case_dir = input_root / "SG"
    case_dir.mkdir(parents=True, exist_ok=True)
    source_dir = REPO_ROOT / "Input" / "SG"
    for source in source_dir.glob("SG_UTown_*.csv"):
        shutil.copy2(source, case_dir / source.name)
    _shrink_smoke_parameter_ranges(case_dir / "SG_UTown_Parameter.csv")
    return input_root


def _shrink_smoke_parameter_ranges(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise RuntimeError(f"Parameter file has no header: {path}")

    for row in rows:
        parameter = row["parameter"]
        if parameter in {"local_refine_window_kw", "local_refine_window_kwh"}:
            row["base_value"] = "0"
        elif parameter in {"local_refine_step_kw", "local_refine_step_kwh"}:
            row["base_value"] = "1"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
