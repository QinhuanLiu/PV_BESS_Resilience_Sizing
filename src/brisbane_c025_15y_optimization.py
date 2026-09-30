"""Run the Brisbane single-family 15-year optimization with an in-memory 0.25C override."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .app.outputs import write_case_outputs
from .design_evaluation import evaluate_design_case
from .io.config import load_config
from .io.data_loader import discover_input_cases, load_input_case


ROOT = Path(__file__).resolve().parents[1]
CASE_KEY = "Brisbane_SingleFamily"
RESULTS_DIR = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_15Y_C0p25_Optimization"
OVERRIDES = {"analysis_years": 15, "battery_max_c_rate": 0.25}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _changed_config_paths(before: dict, after: dict, prefix: str = "") -> list[str]:
    changes: list[str] = []
    for key in sorted(before.keys() | after.keys()):
        path = f"{prefix}.{key}" if prefix else key
        left, right = before.get(key), after.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            changes.extend(_changed_config_paths(left, right, path))
        elif left != right:
            changes.append(path)
    return changes


def run(output_dir: Path = RESULTS_DIR) -> dict[str, Path]:
    """Optimize without changing any source input and save auditable results."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing optimization run: {output_dir}")
    case = next(
        (item for item in discover_input_cases(ROOT / "Input") if item.case_key == CASE_KEY),
        None,
    )
    if case is None:
        raise FileNotFoundError(f"Input case does not exist: {CASE_KEY}")
    input_paths = (case.load_csv, case.pv_csv, case.tou_csv, case.parameter_csv)
    input_hashes = {path.name: _sha256(path) for path in input_paths}
    global_config = load_config(ROOT / "config.yaml")
    _, _, baseline_config, _ = load_input_case(case, global_config)
    _, _, effective_config, _ = load_input_case(case, global_config, parameter_overrides=OVERRIDES)
    changed_paths = _changed_config_paths(baseline_config, effective_config)
    if changed_paths != ["analysis.years", "battery.max_c_rate"]:
        raise ValueError(f"Unexpected effective configuration changes: {changed_paths}")
    if effective_config["outage"]["seed"] != 42:
        raise ValueError("The run must retain outage seed 42")

    print("Optimizing Brisbane_SingleFamily for 15 years at 0.25C...", flush=True)
    (optimal, profile, events, dispatch, candidates, candidate_dispatches,
     outage_summary, yearly) = evaluate_design_case(case, effective_config, CASE_KEY)
    if len(dispatch) != 15 * 8760 or len(profile) != len(dispatch):
        raise ValueError("Optimization did not cover the complete 15-year horizon")
    if not dispatch["dispatch_backend"].eq("greedy").all():
        raise ValueError("The 15-year optimization did not use greedy dispatch")
    if any(_sha256(path) != input_hashes[path.name] for path in input_paths):
        raise ValueError("An original input changed during the optimization")

    output_dir.mkdir(parents=True)
    paths = write_case_outputs(
        output_dir, CASE_KEY, effective_config, optimal, profile, events,
        pd.DataFrame([outage_summary]), yearly, dispatch, candidates, candidate_dispatches,
    )
    config_path = output_dir / "effective_config.json"
    config_path.write_text(json.dumps(effective_config, indent=2, ensure_ascii=False) + "\n")
    paths["effective_config"] = config_path
    manifest = {
        "case_key": CASE_KEY,
        "parameter_overrides": OVERRIDES,
        "changed_effective_config_paths": changed_paths,
        "source_input_sha256": input_hashes,
        "analysis_years": 15,
        "outage_seed": 42,
        "dispatch_backend": "greedy",
        "outage_event_count": int(len(events)),
        "optimized_pv_capacity_kw": float(optimal.iloc[0]["pv_capacity_kw"]),
        "optimized_battery_energy_kwh": float(optimal.iloc[0]["battery_energy_kwh"]),
        "output_sha256": {name: _sha256(path) for name, path in paths.items()},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    paths["manifest"] = manifest_path
    print(f"Optimization complete: {output_dir}", flush=True)
    return paths


if __name__ == "__main__":
    for label, path in run().items():
        print(f"{label}: {path}")
