from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.config import load_config  # noqa: E402
from src.io.data_loader import discover_input_cases, load_input_case  # noqa: E402


SOURCE_DIR = REPO_ROOT / "Input" / "SG_SingleFamily"
STAGING_ROOT = Path("/private/tmp/pv_bess_sg_singlefamily_input")
STAGING_DIR = STAGING_ROOT / "SG"
CASE_KEY = "SG_SingleFamily"

LOAD_COLUMNS = {
    "Cooling load (kWh_th)": "Cooling",
    "Heating load (kWh_th)": "Heating",
    "Lighting load (kWh_e)": "Lighting",
    "Equipment load (kWh_e)": "Equipment",
}
STANDARD_LOAD_COLUMNS = ["Cooling", "Heating", "Lighting", "Equipment"]


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Missing source input directory: {SOURCE_DIR}")
    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    _write_standard_load()
    _copy_case_file("PV")
    _copy_case_file("TOU")
    _write_parameter_table()
    _validate_staging_case()


def _write_standard_load() -> None:
    source = SOURCE_DIR / f"{CASE_KEY}_Load.csv"
    load = pd.read_csv(source)
    missing = [column for column in LOAD_COLUMNS if column not in load.columns]
    if missing:
        raise ValueError(f"{source} is missing expected load column(s): {', '.join(missing)}")
    standardized = load.rename(columns=LOAD_COLUMNS)[STANDARD_LOAD_COLUMNS].copy()
    if len(standardized) != 8760:
        raise ValueError(f"{source} must contain 8760 rows, got {len(standardized)}")
    standardized.to_csv(STAGING_DIR / f"{CASE_KEY}_Load.csv", index=False)


def _copy_case_file(suffix: str) -> None:
    source = SOURCE_DIR / f"{CASE_KEY}_{suffix}.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing source input file: {source}")
    shutil.copy2(source, STAGING_DIR / source.name)


def _write_parameter_table() -> None:
    source = SOURCE_DIR / f"{CASE_KEY}_Parameter.csv"
    parameters = pd.read_csv(source)
    if "parameter" not in parameters.columns or "base_value" not in parameters.columns:
        raise ValueError(f"{source} must include parameter and base_value columns")
    mask = parameters["parameter"].astype(str).eq("building_name")
    if not mask.any():
        raise ValueError(f"{source} does not include building_name")
    parameters.loc[mask, "base_value"] = "SingleFamily"
    parameters.to_csv(STAGING_DIR / f"{CASE_KEY}_Parameter.csv", index=False)


def _validate_staging_case() -> None:
    cases = discover_input_cases(STAGING_ROOT)
    matching = [case for case in cases if case.case_key == CASE_KEY]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one staged {CASE_KEY} case, got {len(matching)}")
    config = load_config(REPO_ROOT / "config.yaml")
    frame, pv_meta, case_config, parameters = load_input_case(matching[0], config)
    pv_candidates = parameters.base.get("pv_capacity_candidates_kw") or case_config["pv"]["capacity_kw_candidates"]
    battery_candidates = parameters.base.get("battery_energy_candidates_kwh") or case_config["battery"]["energy_kwh_candidates"]
    print(f"staging_root={STAGING_ROOT}")
    print(f"case_key={matching[0].case_key}")
    print(f"display_name={case_config['scenarios'][CASE_KEY]['name']}")
    print(f"load_rows={len(frame)}")
    print(f"pv_capacity_active_kw={pv_meta['pv_capacity_kw']}")
    print(f"pv_candidates={min(pv_candidates):g}..{max(pv_candidates):g} kW ({len(pv_candidates)} points)")
    print(f"battery_candidates={min(battery_candidates):g}..{max(battery_candidates):g} kWh ({len(battery_candidates)} points)")
    print(f"uncertainty_baseline_count={case_config['uncertainty']['baseline_scenario_count']}")
    print(f"uncertainty_seed_start={case_config['uncertainty']['seed_start']}")
    print(f"parallel_workers={case_config['uncertainty']['parallel_workers']}")


if __name__ == "__main__":
    main()
