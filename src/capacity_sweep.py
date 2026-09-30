"""Reproducible, fixed-capacity outage sweeps with process-level checkpoints.

Run the requested Brisbane experiment with ``python -m src.capacity_sweep``.
The existing dispatch model is used unchanged; only its outputs are aggregated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

from .dispatch import solve_dispatch_greedy
from .io.config import load_config
from .io.data_loader import InputCase, build_case_frame, discover_input_cases, load_input_case
from .optimizer import battery_sizing, scenario_config
from .outage import simulate_outages, summarize_outages
from .resilience_metrics import DEFAULT_TOLERANCE, electric_equivalent_building_load
from .sweep_economics import (
    PRIMARY_COLUMNS,
    ECONOMIC_METRICS_README,
    ENRICHMENT_README,
    economic_metrics,
    validate_economic_results,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
_WORKER: dict[str, Any] = {}
KEYS = ["pv_capacity_kw", "battery_energy_kwh", "scenario_id"]
IDENTIFIER_COLUMNS = [
    "case_key", "pv_capacity_kw", "battery_energy_kwh", "battery_power_kw", "battery_c_rate",
    "scenario_id", "outage_seed", "outage_profile_sha256",
]
SCENARIO_COLUMNS = IDENTIFIER_COLUMNS + [
    "analysis_years", "EUE_kwh_total", "EUE_kwh_per_year", "outage_load_kwh_total", "LOLP",
    "no_outage_load", "outage_hours_total", "dispatch_backend",
]
YEARLY_COLUMNS = IDENTIFIER_COLUMNS + [
    "analysis_year", "EUE_kwh", "outage_load_kwh", "LOLP", "no_outage_load", "outage_hours",
]


@dataclass(frozen=True)
class SweepSpec:
    case_key: str = "Brisbane_SingleFamily"
    analysis_years: int = 15
    scenario_count: int = 100
    seed_start: int = 42
    pv_capacities: tuple[float, ...] = tuple(float(i) for i in range(11))
    battery_capacities: tuple[float, ...] = tuple(i * 2.5 for i in range(9))

    def validate(self) -> None:
        for name in ("analysis_years", "scenario_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if int(self.seed_start) != self.seed_start or self.seed_start < 0:
            raise ValueError("seed_start must be a nonnegative integer")
        for name in ("pv_capacities", "battery_capacities"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
            if not all(np.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"{name} must contain finite, nonnegative capacities")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _availability_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provenance(case: InputCase, config: dict[str, Any], spec: SweepSpec) -> dict[str, Any]:
    input_paths = [case.load_csv, case.pv_csv, case.tou_csv, case.parameter_csv]
    source_paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "requirements.txt"]
    return {
        "schema_version": SCHEMA_VERSION,
        "spec": asdict(spec),
        "effective_config": config,
        "input_hashes": {p.name: _file_hash(p) for p in input_paths},
        "source_hashes": {str(p.relative_to(ROOT)): _file_hash(p) for p in source_paths},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "unmet_energy_tolerance_kwh_per_hour": DEFAULT_TOLERANCE,
    }


def _snapshot_inputs(case: InputCase, output: Path) -> InputCase:
    directory = output / "inputs"
    directory.mkdir(exist_ok=True)
    updated = {}
    for role in ("load_csv", "pv_csv", "tou_csv", "parameter_csv"):
        original = getattr(case, role)
        destination = directory / original.name
        shutil.copy2(original, destination)
        updated[role] = destination
    return replace(case, city_dir=directory, **updated)


def _load_enrichment_source(source: Path) -> dict[str, Any]:
    source = source.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    manifest_provenance = {k: v for k, v in manifest.items() if k not in {"fingerprint", "created_at", "git_revision"}}
    if _digest(manifest_provenance) != manifest["fingerprint"]:
        raise ValueError("Source manifest fingerprint is invalid")
    state = json.loads((source / "run_status.json").read_text())
    validation = json.loads((source / "validation.json").read_text())
    if state["state"] != "complete" or not validation["passed"]:
        raise ValueError("Enrichment requires a completed, validated source run")
    if state["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Source completion status does not match its manifest")
    for name, digest in validation["file_sha256"].items():
        if _file_hash(source / name) != digest:
            raise ValueError(f"Source result checksum mismatch: {name}")
    for name, digest in manifest["input_hashes"].items():
        if _file_hash(source / "inputs" / name) != digest:
            raise ValueError(f"Source input checksum mismatch: {name}")
    for name, digest in manifest["source_hashes"].items():
        if _file_hash(source / "source_snapshot" / name) != digest:
            raise ValueError(f"Source code snapshot checksum mismatch: {name}")
    config = json.loads((source / "effective_config.json").read_text())
    if _digest(config) != _digest(manifest["effective_config"]):
        raise ValueError("Source effective configuration differs from its manifest")
    values = dict(manifest["spec"])
    for field in ("pv_capacities", "battery_capacities"):
        values[field] = tuple(values[field])
    spec = SweepSpec(**values)
    spec.validate()
    if config["analysis"]["years"] != spec.analysis_years or config["dispatch"]["backend"] != "greedy":
        raise ValueError("Source horizon/backend does not match this fixed greedy-dispatch workflow")
    def input_path(role: str) -> Path:
        names = [name for name in manifest["input_hashes"]
                 if name.endswith(f"_{role}.csv") or (role == "Load" and name.endswith("_hourly_loads_kWh.csv"))]
        if len(names) != 1:
            raise ValueError(f"Ambiguous source input role: {role}")
        return source / "inputs" / names[0]
    city, building = spec.case_key.rsplit("_", 1)
    case = InputCase(city, building, spec.case_key, source / "inputs",
                     input_path("Load"), input_path("PV"), input_path("TOU"), input_path("Parameter"))
    read = lambda name: pd.read_csv(source / name, float_precision="round_trip")
    results, yearly = read("scenario_results.csv"), read("yearly_results.csv")
    scenarios, events = read("outage_scenarios.csv"), read("outage_events.csv")
    validate_results(results, yearly, scenarios, spec)
    payloads = json.loads((source / "outage_profiles_sparse.json").read_text())
    expected_ids = set(range(1, spec.scenario_count + 1))
    if len(payloads) != spec.scenario_count or {p["scenario_id"] for p in payloads} != expected_ids:
        raise ValueError("Source outage payloads do not cover the required scenarios")
    if len(scenarios) != spec.scenario_count or scenarios.scenario_id.duplicated().any():
        raise ValueError("Source outage scenario table is incomplete or duplicated")
    scenario_index = scenarios.set_index("scenario_id")
    for payload in payloads:
        reference = scenario_index.loc[payload["scenario_id"]]
        if (payload["case_key"] != spec.case_key or payload["outage_seed"] != reference.outage_seed
                or payload["outage_profile_sha256"] != reference.outage_profile_sha256):
            raise ValueError("Source sparse profile identity mismatch")
        indices = np.asarray(payload["hour_indices"], dtype=int)
        values = np.asarray(payload["availability"], dtype=float)
        if (len(indices) != len(values) or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= 8760 * spec.analysis_years)
                or not np.isfinite(values).all() or np.any(values < 0) or np.any(values >= 1)):
            raise ValueError("Invalid sparse outage profile")
        available = np.ones(8760 * spec.analysis_years)
        available[indices] = values
        if _availability_hash(available) != payload["outage_profile_sha256"]:
            raise ValueError("Source outage profile checksum mismatch")
    names = ("manifest.json", "effective_config.json", "validation.json", "scenario_results.csv",
             "yearly_results.csv", "outage_scenarios.csv", "outage_events.csv", "outage_profiles_sparse.json")
    return {
        "spec": spec, "case": case, "config": config, "payloads": payloads,
        "scenarios": scenarios, "events": events, "results": results, "yearly": yearly,
        "identity": {"directory": str(source), "source_fingerprint": manifest["fingerprint"],
                     "file_hashes": {name: _file_hash(source / name) for name in names}},
    }


def _check_source_dependencies(identity: dict[str, Any]) -> None:
    for name, digest in identity["file_hashes"].items():
        if _file_hash(Path(identity["directory"]) / name) != digest:
            raise ValueError(f"Source run changed during enrichment: {name}")


def _retain_reference_columns(calculated: pd.DataFrame, reference: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    reference = reference.sort_values(keys).reset_index(drop=True)
    calculated = calculated.sort_values(keys).reset_index(drop=True)
    columns = YEARLY_COLUMNS if "analysis_year" in keys else SCENARIO_COLUMNS
    for column in columns:
        left, right = reference[column], calculated[column]
        if pd.api.types.is_numeric_dtype(left):
            equal = np.allclose(left, right, rtol=1e-12, atol=1e-9)
        else:
            equal = left.equals(right)
        if not equal:
            raise ValueError(f"Replayed result differs from source: {column}")
    # Preserve the source's actual values, not merely numerically close replacements.
    retained = reference[columns].copy()
    for column in calculated.columns:
        if column not in columns:
            retained[column] = calculated[column].to_numpy()
    return retained


def _make_scenarios(config: dict[str, Any], spec: SweepSpec) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    scenario = config["active_scenario"]
    outage = config["outage"]
    payloads, summaries, event_frames = [], [], []
    for index in range(spec.scenario_count):
        seed = spec.seed_start + index
        profile, events = simulate_outages(
            saifi_per_year=float(scenario["saifi_per_year"]),
            saidi_min_per_year=float(scenario["saidi_min_per_year"]),
            hours=8760 * spec.analysis_years,
            seed=seed,
            forced_event_mode=outage["forced_event_mode"],
            night_window_start_hour=float(outage["night_window_start_hour"]),
            night_window_end_hour=float(outage["night_window_end_hour"]),
            analysis_years=spec.analysis_years,
            aggregate_horizon=spec.analysis_years > 1,
        )
        available = profile["grid_availability"].to_numpy(dtype=float)
        indices = np.flatnonzero(available < 1.0)
        identifiers = {
            "case_key": spec.case_key,
            "scenario_id": index + 1,
            "outage_seed": seed,
            "outage_profile_sha256": _availability_hash(available),
        }
        summaries.append({
            **identifiers,
            **summarize_outages(
                profile, events, spec.analysis_years,
                float(scenario["saidi_min_per_year"]), float(scenario["saifi_per_year"]),
            ),
        })
        event_frames.append(events.assign(**identifiers))
        payloads.append({
            **identifiers,
            "hour_indices": indices.tolist(),
            "availability": available[indices].tolist(),
        })
    return payloads, pd.DataFrame(summaries), pd.concat(event_frames, ignore_index=True)


def _init_worker(case: InputCase, config: dict[str, Any], reference_pv: float) -> None:
    frame, _ = build_case_frame(case, config, reference_pv)
    technical = config["technical"]
    _WORKER.clear()
    _WORKER.update(
        frame=frame[[
            "hour_index", "analysis_year", "electric_load_kwh", "cooling_load_kwh_th",
            "heating_load_kwh_th", "pv_kwh", "tou_sgd_per_kwh", "pv_export_sgd_per_kwh",
        ]].copy(),
        pv_per_kw=frame["pv_kwh"].to_numpy() / reference_pv if reference_pv > 0 else np.zeros(len(frame)),
        total_load=electric_equivalent_building_load(
            frame, technical["cooling_cop"], technical["heating_cop"],
        ).to_numpy(),
        config=config,
    )


def summarize_dispatch(
    frame: pd.DataFrame,
    dispatch: pd.DataFrame,
    total_load: np.ndarray,
    analysis_years: int,
    identifiers: dict[str, Any],
) -> tuple[dict, list[dict]]:
    """Aggregate one continuous dispatch; numerator and denominator share the horizon."""
    if not np.array_equal(frame["hour_index"].to_numpy(), dispatch["hour_index"].to_numpy()):
        raise ValueError("Dispatch and input hour indices are not aligned")
    unmet = dispatch["unmet_load_kwh_equiv"].to_numpy(dtype=float)
    fractions = frame["outage_fraction"].to_numpy(dtype=float)
    load = np.asarray(total_load, dtype=float) * fractions
    if not all(np.isfinite(values).all() for values in (unmet, load, fractions)):
        raise ValueError("Nonfinite dispatch or outage values")
    if np.any(unmet < -1e-9) or np.any(unmet > load + 1e-6):
        raise ValueError("Unserved energy outside outage demand bounds")
    # Match the existing reliability module, using the same threshold for yearly totals.
    unmet = np.where(unmet > DEFAULT_TOLERANCE, unmet, 0.0)
    year_ids = frame["analysis_year"].to_numpy(dtype=int)
    if np.any(year_ids < 1) or np.any(year_ids > analysis_years):
        raise ValueError("Analysis-year labels outside the configured horizon")
    annual_unmet = np.bincount(year_ids, weights=unmet, minlength=analysis_years + 1)[1:]
    annual_load = np.bincount(year_ids, weights=load, minlength=analysis_years + 1)[1:]
    annual_hours = np.bincount(year_ids, weights=fractions, minlength=analysis_years + 1)[1:]
    total_unmet, total_outage_load = float(annual_unmet.sum()), float(annual_load.sum())
    row = {
        **identifiers,
        "analysis_years": analysis_years,
        "EUE_kwh_total": total_unmet,
        "EUE_kwh_per_year": total_unmet / analysis_years,
        "outage_load_kwh_total": total_outage_load,
        "LOLP": total_unmet / total_outage_load if total_outage_load > 0 else 0.0,
        "no_outage_load": bool(total_outage_load <= 0),
        "outage_hours_total": float(annual_hours.sum()),
        "dispatch_backend": str(dispatch["dispatch_backend"].iloc[0]),
    }
    annual_rows = [{
        **identifiers,
        "analysis_year": year + 1,
        "EUE_kwh": float(annual_unmet[year]),
        "outage_load_kwh": float(annual_load[year]),
        "LOLP": float(annual_unmet[year] / annual_load[year]) if annual_load[year] > 0 else 0.0,
        "no_outage_load": bool(annual_load[year] <= 0),
        "outage_hours": float(annual_hours[year]),
    } for year in range(analysis_years)]
    return row, annual_rows


def _evaluate_block(scenario: dict, pv: float, batteries: tuple[float, ...]) -> dict:
    started = time.perf_counter()
    frame = _WORKER["frame"].copy()
    available = np.ones(len(frame), dtype=float)
    available[np.asarray(scenario["hour_indices"], dtype=int)] = scenario["availability"]
    if _availability_hash(available) != scenario["outage_profile_sha256"]:
        raise ValueError("Shared outage profile did not reconstruct exactly")
    frame["grid_availability"] = available
    frame["outage_fraction"] = 1.0 - available
    frame["pv_kwh"] = _WORKER["pv_per_kw"] * pv
    config = _WORKER["config"]
    enrich = bool(config.get("capacity_sweep", {}).get("include_economic_metrics", False))
    scenario_rows, annual_rows = [], []
    for battery in batteries:
        sizing = battery_sizing(battery, config)
        identifiers = {key: scenario[key] for key in ("case_key", "scenario_id", "outage_seed", "outage_profile_sha256")}
        identifiers.update(pv_capacity_kw=pv, **sizing)
        dispatch = solve_dispatch_greedy(frame, sizing, config, metrics_only=True, collect_totals=enrich)
        row, yearly = summarize_dispatch(
            frame, dispatch, _WORKER["total_load"], int(config["analysis"]["years"]), identifiers,
        )
        if enrich:
            row.update(economic_metrics(dispatch.attrs["energy_totals"], sizing, pv, config))
            row["EUE_threshold_difference_kwh_total"] = row["unserved_energy_kwh_total"] - row["EUE_kwh_total"]
        scenario_rows.append(row)
        annual_rows.extend(yearly)
        del dispatch
    return {
        "scenario_rows": scenario_rows,
        "yearly_rows": annual_rows,
        "worker_pid": os.getpid(),
        "compute_seconds": time.perf_counter() - started,
    }


def _checkpoint_path(output: Path, scenario_id: int, pv_index: int) -> Path:
    return output / "checkpoints" / f"scenario_{scenario_id:03d}_pv_{pv_index:02d}.json"


def _read_checkpoint(path: Path, fingerprint: str, scenario_id: int, pv: float, spec: SweepSpec) -> dict:
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    if envelope["fingerprint"] != fingerprint or envelope["payload_sha256"] != _digest(payload):
        raise ValueError(f"Checkpoint fingerprint/checksum mismatch: {path}")
    rows = pd.DataFrame(payload["scenario_rows"])
    yearly = pd.DataFrame(payload["yearly_rows"])
    if len(rows) != len(spec.battery_capacities) or len(yearly) != len(rows) * spec.analysis_years:
        raise ValueError(f"Incomplete checkpoint: {path}")
    for table in (rows, yearly):
        if not table["scenario_id"].eq(scenario_id).all() or not table["pv_capacity_kw"].eq(pv).all():
            raise ValueError(f"Checkpoint task identity mismatch: {path}")
    if set(rows["battery_energy_kwh"]) != set(spec.battery_capacities):
        raise ValueError(f"Checkpoint capacity mismatch: {path}")
    return payload


def validate_results(results: pd.DataFrame, yearly: pd.DataFrame, scenarios: pd.DataFrame, spec: SweepSpec) -> dict:
    expected = pd.MultiIndex.from_product(
        [spec.pv_capacities, spec.battery_capacities, range(1, spec.scenario_count + 1)], names=KEYS,
    )
    actual = pd.MultiIndex.from_frame(results[KEYS])
    if actual.has_duplicates or len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError("Scenario results do not cover exactly the requested capacity/scenario grid")
    yearly_keys = KEYS + ["analysis_year"]
    expected_yearly = pd.MultiIndex.from_product(
        [spec.pv_capacities, spec.battery_capacities, range(1, spec.scenario_count + 1), range(1, spec.analysis_years + 1)],
        names=yearly_keys,
    )
    actual_yearly = pd.MultiIndex.from_frame(yearly[yearly_keys])
    if actual_yearly.has_duplicates or len(actual_yearly) != len(expected_yearly) or len(expected_yearly.difference(actual_yearly)):
        raise ValueError("Yearly results are incomplete or duplicated")
    for table, columns in (
        (results, ["EUE_kwh_total", "EUE_kwh_per_year", "outage_load_kwh_total", "outage_hours_total", "LOLP"]),
        (yearly, ["EUE_kwh", "outage_load_kwh", "outage_hours", "LOLP"]),
    ):
        values = table[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < -1e-9) or table["LOLP"].gt(1 + 1e-9).any():
            raise ValueError("Metric values are nonfinite or outside valid bounds")
        if not table["case_key"].eq(spec.case_key).all():
            raise ValueError("Case identity mismatch")
        for row in scenarios.itertuples(index=False):
            selected = table[table["scenario_id"] == row.scenario_id]
            if not selected["outage_seed"].eq(row.outage_seed).all() or not selected["outage_profile_sha256"].eq(row.outage_profile_sha256).all():
                raise ValueError("Capacities did not share identical outage scenarios")
    if not results["analysis_years"].eq(spec.analysis_years).all() or not results["dispatch_backend"].eq("greedy").all():
        raise ValueError("Unexpected horizon or dispatch backend")
    indexed = results.set_index(KEYS).sort_index()
    totals = yearly.groupby(KEYS)[["EUE_kwh", "outage_load_kwh", "outage_hours"]].sum().sort_index()
    comparisons = [
        (indexed["EUE_kwh_total"], totals["EUE_kwh"]),
        (indexed["outage_load_kwh_total"], totals["outage_load_kwh"]),
        (indexed["outage_hours_total"], totals["outage_hours"]),
        (results["EUE_kwh_per_year"] * spec.analysis_years, results["EUE_kwh_total"]),
    ]
    for table, ue_column, demand_column in (
        (results, "EUE_kwh_total", "outage_load_kwh_total"),
        (yearly, "EUE_kwh", "outage_load_kwh"),
    ):
        demand = table[demand_column].to_numpy()
        ratio = np.divide(table[ue_column].to_numpy(), demand, out=np.zeros(len(table)), where=demand > 0)
        comparisons.append((table["LOLP"], ratio))
        if not np.array_equal(table["no_outage_load"].to_numpy(), demand <= 0):
            raise ValueError("Zero-outage-demand flag mismatch")
    if not all(np.allclose(a, b, rtol=1e-10, atol=1e-9) for a, b in comparisons):
        raise ValueError("Annual/cumulative EUE, demand, hours or LOLP failed reconciliation")
    zero = results[(results.pv_capacity_kw == 0) & (results.battery_energy_kwh == 0)]
    # The established 1e-6 kWh hourly tolerance can discard negligible unmet energy.
    if not np.allclose(zero.EUE_kwh_total, zero.outage_load_kwh_total, rtol=0, atol=8760 * spec.analysis_years * DEFAULT_TOLERANCE):
        raise ValueError("Zero-PV, zero-battery reference failed")
    if results.groupby("scenario_id")["outage_load_kwh_total"].nunique().gt(1).any():
        raise ValueError("Outage demand differs between capacities")
    return {
        "passed": True,
        "scenario_result_rows": len(results),
        "yearly_result_rows": len(yearly),
        "capacity_combinations": len(spec.pv_capacities) * len(spec.battery_capacities),
        "scenarios_per_capacity": spec.scenario_count,
        "unique_outage_profiles": int(scenarios.outage_profile_sha256.nunique()),
        "shared_scenarios_verified": True,
        "yearly_totals_and_ratios_verified": True,
        "dispatch_backend": "greedy",
        "completed_at": _utc_now(),
    }


def capacity_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pv, battery), group in results.groupby(["pv_capacity_kw", "battery_energy_kwh"], sort=True):
        row = {"pv_capacity_kw": pv, "battery_energy_kwh": battery, "scenario_count": len(group)}
        metrics = ["EUE_kwh_total", "EUE_kwh_per_year", "LOLP"]
        metrics += [name for name in ("annual_cost_sgd_per_year", "SSR", "SCR") if name in group]
        for metric in metrics:
            values = group[metric].dropna()
            if values.empty:
                row.update({f"{metric}_{name}": None for name in ("mean", "std", "median", "p95", "min", "max")})
                continue
            row.update({f"{metric}_{name}": float(value) for name, value in {
                "mean": values.mean(), "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                "median": values.median(), "p95": values.quantile(0.95), "min": values.min(), "max": values.max(),
            }.items()})
        if "SCR" in group:
            row["SCR_defined_scenario_count"] = int(group.SCR.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_readme(output: Path, spec: SweepSpec, *, include_economic_metrics: bool) -> None:
    metric_file = "- `数据说明.md`: 五项逐情景指标、单位及边界情况的中文说明。\n" if include_economic_metrics else ""
    economic_note = ""
    if include_economic_metrics:
        economic_note = """

`annual_cost_sgd_per_year` includes annualized PV/BESS investment, O&M, net grid cost and
outage loss. `SSR` is PV direct service plus PV-origin battery discharge divided by total
electric-equivalent demand. `SCR` is PV supplied directly or used to charge the battery divided
by available PV generation. These are retained for every capacity/scenario row. For zero PV,
`SSR=0` and `SCR` is empty with `SCR_defined=false` and `SCR_status=not_applicable_no_pv`.
All economic and PV-use totals are accumulated during the same streaming dispatch that produces
EUE and LOLP, so no second dispatch run is required.
"""
    (output / "README.md").write_text(f"""# {spec.case_key} fixed-capacity outage sweep

Each of {spec.scenario_count} equally weighted scenarios spans {spec.analysis_years} consecutive
8760-hour model years. The one-year load, PV and TOU profiles repeat; SOC does not reset at
year boundaries. Battery capacity is fixed and power is energy capacity times the input C-rate.
The existing greedy dispatch and its fractional-hour outage treatment are used unchanged.

## Files and units

- `scenario_results.csv`: one row per PV/battery/scenario; all individual results are retained.
- `yearly_results.csv`: one row per PV/battery/scenario/model year.
- `capacity_summary.csv`: equally weighted scenario statistics for each capacity pair.
- `outage_scenarios.csv`, `outage_events.csv`: shared seeds, actual interruption counts,
  durations and timing. `outage_profiles_sparse.json` stores exact non-unit availability values.
- `manifest.json`, `effective_config.json`, `inputs/`, `source_snapshot/`: provenance and snapshots.
- `run_status.json`, `validation.json`, `checkpoints/`: completion, QA and resumable task results.
- `failures.jsonl`, when present: append-only history of failed task attempts.
{metric_file}

`EUE_kwh_total` is scenario-specific unserved electricity-equivalent energy over the full
horizon; `EUE_kwh_per_year = EUE_kwh_total / {spec.analysis_years}`. Only the mean across scenarios
is the Monte Carlo estimate of expected unserved energy. These per-scenario field names follow
the requested EUE terminology. The hourly unmet-energy tolerance is {DEFAULT_TOLERANCE} kWh,
matching the existing reliability module. Yearly `EUE_kwh` uses the same tolerance.

`LOLP = EUE_kwh_total / outage_load_kwh_total` is the project's **energy-based outage ratio**,
not a time-based loss-of-load probability. Yearly LOLP uses that year's numerator/denominator.
When outage demand is zero, LOLP is 0 and `no_outage_load` is true. No percentage conversion is
applied (0.1 means 10%). Mean LOLP is the arithmetic mean of scenario ratios, not the ratio of
mean energies. Standard deviation is the sample standard deviation (ddof=1; 0 for a single
scenario); P95 uses pandas' linear interpolation.

The full-building electric-equivalent load includes lighting and equipment plus cooling/COP
and heating/COP. All capacity pairs use exactly the same outage profiles, verified by SHA-256.
The retained multi-year outage model draws a Poisson count with mean SAIFI times the horizon,
allows a zero-event realization, and rescales nonzero realizations to SAIDI times the horizon.
It does not represent climate variability or physical capacity fade across years. Degradation
is not used to reduce available battery energy.{economic_note}

`analysis_year` is the 8760-hour model-year index, not a Gregorian calendar year.
No full hourly dispatch tables are saved. Resume using the same invocation plus `--resume`;
input, effective config, source and runtime fingerprints must match. Worker count may change.
""", encoding="utf-8")


def run_sweep(
    spec: SweepSpec,
    output: Path,
    *,
    workers: int = 8,
    resume: bool = False,
    input_dir: Path = ROOT / "Input",
    config_path: Path = ROOT / "config.yaml",
    enrich_from: Path | None = None,
    replay_from: Path | None = None,
    include_economic_metrics: bool = False,
) -> dict[str, Any]:
    spec.validate()
    if workers <= 0 or int(workers) != workers:
        raise ValueError("workers must be a positive integer")
    output = Path(output).resolve()
    if enrich_from is not None and replay_from is not None:
        raise ValueError("Use either enrich_from or replay_from, not both")
    source = None
    source_mode: str | None = None
    if enrich_from is not None:
        if include_economic_metrics:
            raise ValueError("--enrich-from already computes economic metrics; do not also request them directly")
        if output == Path(enrich_from).resolve():
            raise ValueError("Enriched output must not overwrite the source run")
        source = _load_enrichment_source(Path(enrich_from))
        source_mode = "enrichment"
        spec, case, config = source["spec"], source["case"], source["config"]
        config["dispatch"]["hourly_output"] = "reliability_with_streaming_totals"
        config["capacity_sweep"] = {
            "include_economic_metrics": True,
            "annual_cost_definition": "system_total_including_raw_unserved_energy_cost",
            "SSR_definition": "pv_direct_plus_pv_origin_battery_discharge_over_total_demand",
            "SCR_definition": "pv_direct_plus_pv_battery_charge_over_available_generation",
            "zero_pv_SCR": None,
        }
    elif replay_from is not None:
        if include_economic_metrics:
            raise ValueError("Frozen-profile replay is reliability-only; do not request economic metrics")
        if output == Path(replay_from).resolve():
            raise ValueError("Replay output must not overwrite the source run")
        source = _load_enrichment_source(Path(replay_from))
        source_mode = "frozen_profile_replay"
        source_spec = source["spec"]
        spec = SweepSpec(
            case_key=source_spec.case_key,
            analysis_years=source_spec.analysis_years,
            scenario_count=source_spec.scenario_count,
            seed_start=source_spec.seed_start,
            pv_capacities=spec.pv_capacities,
            battery_capacities=spec.battery_capacities,
        )
        spec.validate()
        case, config = source["case"], source["config"]
        config.pop("capacity_sweep", None)
        config["dispatch"]["backend"] = "greedy"
        config["dispatch"]["hourly_output"] = "reliability_only"
    else:
        cases = {case.case_key: case for case in discover_input_cases(input_dir)}
        if spec.case_key not in cases:
            raise ValueError(f"Unknown case: {spec.case_key}")
        case = cases[spec.case_key]
        _, _, config, _ = load_input_case(
            case, load_config(config_path), {"analysis_years": spec.analysis_years}, max(spec.pv_capacities),
        )
        config = scenario_config(config, spec.case_key)
        config["dispatch"]["backend"] = "greedy"
        config["dispatch"]["hourly_output"] = (
            "reliability_with_streaming_totals" if include_economic_metrics else "reliability_only"
        )
        if include_economic_metrics:
            config["capacity_sweep"] = {
                "include_economic_metrics": True,
                "annual_cost_definition": "system_total_including_raw_unserved_energy_cost",
                "SSR_definition": "pv_direct_plus_pv_origin_battery_discharge_over_total_demand",
                "SCR_definition": "pv_direct_plus_pv_battery_charge_over_available_generation",
                "zero_pv_SCR": None,
            }

    has_economic_metrics = bool(config.get("capacity_sweep", {}).get("include_economic_metrics", False))

    def current_provenance() -> dict[str, Any]:
        value = _provenance(case, config, spec)
        if source is not None:
            _check_source_dependencies(source["identity"])
            value[f"{source_mode}_source"] = source["identity"]
        return value

    provenance = current_provenance()
    fingerprint = _digest(provenance)
    manifest_path = output / "manifest.json"
    previous_elapsed = 0.0
    if manifest_path.exists():
        if not resume:
            raise ValueError(f"Output already exists; use --resume: {output}")
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Resume refused: input, configuration, program or runtime fingerprint changed")
        status_path = output / "run_status.json"
        if status_path.exists():
            previous_elapsed = float(json.loads(status_path.read_text()).get("elapsed_seconds", 0.0))
        snapshot_case = replace(case, city_dir=output / "inputs", **{
            role: output / "inputs" / getattr(case, role).name
            for role in ("load_csv", "pv_csv", "tou_csv", "parameter_csv")
        })
        for name, digest in provenance["input_hashes"].items():
            if _file_hash(output / "inputs" / name) != digest:
                raise ValueError("Saved input snapshot changed")
        for name, digest in provenance["source_hashes"].items():
            if _file_hash(output / "source_snapshot" / name) != digest:
                raise ValueError("Saved program snapshot changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Refusing to overwrite a nonempty output directory without a manifest: {output}")
        output.mkdir(parents=True, exist_ok=True)
        snapshot_case = _snapshot_inputs(case, output)
        for name in provenance["source_hashes"]:
            destination = output / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        manifest = {"fingerprint": fingerprint, "created_at": _utc_now(), "git_revision": revision, **provenance}
        _atomic_json(manifest_path, manifest)
    _atomic_json(output / "effective_config.json", config)
    if source is None:
        _write_readme(output, spec, include_economic_metrics=has_economic_metrics)
        if has_economic_metrics:
            (output / "数据说明.md").write_text(ECONOMIC_METRICS_README, encoding="utf-8")
        payloads, scenarios, events = _make_scenarios(config, spec)
    elif source_mode == "enrichment":
        _atomic_json(output / "source_reference.json", source["identity"])
        (output / "数据说明.md").write_text(ENRICHMENT_README, encoding="utf-8")
        (output / "README.md").write_text(ENRICHMENT_README, encoding="utf-8")
        payloads, scenarios, events = source["payloads"], source["scenarios"], source["events"]
    else:
        _atomic_json(output / "source_reference.json", {
            "mode": "frozen_profile_replay", **source["identity"],
        })
        _write_readme(output, spec, include_economic_metrics=False)
        payloads, scenarios, events = source["payloads"], source["scenarios"], source["events"]
    _atomic_csv(output / "outage_scenarios.csv", scenarios)
    _atomic_csv(output / "outage_events.csv", events)
    _atomic_json(output / "outage_profiles_sparse.json", payloads)

    tasks, completed_payloads = [], []
    for scenario in payloads:
        for pv_index, pv in enumerate(spec.pv_capacities):
            path = _checkpoint_path(output, scenario["scenario_id"], pv_index)
            if path.exists():
                completed_payloads.append(_read_checkpoint(path, fingerprint, scenario["scenario_id"], pv, spec))
            else:
                tasks.append((scenario, pv, path))
    total_tasks = spec.scenario_count * len(spec.pv_capacities)
    cached_tasks = len(completed_payloads)
    started = time.perf_counter()
    failed = 0
    worker_pids = {item["worker_pid"] for item in completed_payloads}

    def status(state: str) -> dict:
        data = {
            "state": state, "updated_at": _utc_now(), "fingerprint": fingerprint,
            "workers_requested": workers, "worker_pids": sorted(worker_pids),
            "metrics": "cost_EUE_LOLP_SSR_SCR" if has_economic_metrics else "EUE_LOLP",
            "tasks_total": total_tasks, "tasks_completed": len(completed_payloads),
            "tasks_resumed": cached_tasks, "tasks_failed_this_attempt": failed,
            "scenario_results_completed": len(completed_payloads) * len(spec.battery_capacities),
            "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
        }
        _atomic_json(output / "run_status.json", data)
        return data

    def record_result(payload: dict, path: Path) -> None:
        _atomic_json(path, {"fingerprint": fingerprint, "payload_sha256": _digest(payload), "payload": payload})
        completed_payloads.append(payload)
        worker_pids.add(payload["worker_pid"])
        data = status("running")
        finished = len(completed_payloads) - cached_tasks
        elapsed = time.perf_counter() - started
        eta = elapsed / finished * (total_tasks - len(completed_payloads)) if finished else 0.0
        print(f"Progress: {len(completed_payloads)}/{total_tasks} tasks; "
              f"{data['scenario_results_completed']} scenario results; "
              f"elapsed {elapsed:.1f}s; ETA {eta:.1f}s", flush=True)

    def record_failure(exc: Exception, task: tuple) -> None:
        nonlocal failed
        failed += 1
        error = {"time": _utc_now(), "scenario_id": task[0]["scenario_id"],
                 "pv_capacity_kw": task[1], "error": repr(exc), "traceback": traceback.format_exc()}
        with (output / "failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_json_bytes(error).decode("utf-8") + "\n")
        status("running")
        print(f"FAILED scenario={task[0]['scenario_id']} PV={task[1]}: {exc}", flush=True)

    status("running")
    print(f"Starting {spec.case_key}: {total_tasks} tasks, {cached_tasks} resumed, {workers} workers", flush=True)
    try:
        if tasks and workers == 1:
            _init_worker(snapshot_case, config, max(spec.pv_capacities))
            for task in tasks:
                try:
                    record_result(_evaluate_block(task[0], task[1], spec.battery_capacities), task[2])
                except Exception as exc:
                    record_failure(exc, task)
        elif tasks:
            # Explicit spawn is portable on macOS and avoids inherited RNG/thread state.
            with ProcessPoolExecutor(
                max_workers=min(workers, len(tasks)), mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker, initargs=(snapshot_case, config, max(spec.pv_capacities)),
            ) as executor:
                futures = {executor.submit(_evaluate_block, scenario, pv, spec.battery_capacities): (scenario, pv, path)
                           for scenario, pv, path in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        record_result(future.result(), task[2])
                    except Exception as exc:
                        record_failure(exc, task)
        if failed:
            raise RuntimeError(f"{failed} tasks failed; inspect failures.jsonl and resume to retry")
        if _digest(current_provenance()) != fingerprint:
            raise RuntimeError("Input or program changed during execution; results cannot be certified")
        results = pd.DataFrame([row for item in completed_payloads for row in item["scenario_rows"]]).sort_values(KEYS).reset_index(drop=True)
        columns = SCENARIO_COLUMNS.copy()
        if has_economic_metrics:
            columns += PRIMARY_COLUMNS
            columns += sorted(set(results.columns) - set(columns))
        results = results[columns]
        yearly = pd.DataFrame([row for item in completed_payloads for row in item["yearly_rows"]], columns=YEARLY_COLUMNS).sort_values(KEYS + ["analysis_year"]).reset_index(drop=True)
        validation = validate_results(results, yearly, scenarios, spec)
        if source_mode == "enrichment":
            results = _retain_reference_columns(results, source["results"], KEYS)
            yearly = _retain_reference_columns(yearly, source["yearly"], KEYS + ["analysis_year"])
            validation["reference_reliability_rows_verified"] = len(results)
            validation["reference_yearly_rows_verified"] = len(yearly)
            validation["original_reliability_columns_preserved"] = True
        if has_economic_metrics:
            validation.update(validate_economic_results(results, config))
        summary = capacity_summary(results)
        for name, table in (("scenario_results", results), ("yearly_results", yearly), ("capacity_summary", summary)):
            _atomic_csv(output / f"{name}.csv", table)
        # Verify the delivered CSVs as well as the in-memory tables.
        delivered = pd.read_csv(output / "scenario_results.csv", float_precision="round_trip")
        delivered_yearly = pd.read_csv(output / "yearly_results.csv", float_precision="round_trip")
        validate_results(delivered, delivered_yearly, scenarios, spec)
        if has_economic_metrics:
            validate_economic_results(delivered, config)
        if source_mode == "enrichment":
            _retain_reference_columns(delivered, source["results"], KEYS)
            _retain_reference_columns(delivered_yearly, source["yearly"], KEYS + ["analysis_year"])
        validation["file_sha256"] = {name: _file_hash(output / name) for name in ("scenario_results.csv", "yearly_results.csv", "capacity_summary.csv")}
        _atomic_json(output / "validation.json", validation)
        final_status = status("complete")
        print(f"Complete: {len(results)} scenario rows, {len(yearly)} yearly rows, {len(summary)} capacities. Output: {output}", flush=True)
        return final_status
    except BaseException:
        status("failed")
        raise
    finally:
        _WORKER.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="Brisbane_SingleFamily")
    parser.add_argument("--analysis-years", type=int, default=15)
    parser.add_argument("--scenario-count", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--pv-capacities", nargs="+", type=float, default=list(SweepSpec.pv_capacities))
    parser.add_argument("--battery-capacities", nargs="+", type=float, default=list(SweepSpec.battery_capacities))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "Input")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--enrich-from", type=Path, help="Replay a completed source run with streaming cost, SSR and SCR statistics.")
    parser.add_argument(
        "--replay-from", type=Path,
        help="Reuse the exact frozen outage profiles from a completed source run while evaluating this PV/BESS grid.",
    )
    parser.add_argument(
        "--include-economic-metrics",
        action="store_true",
        help="Compute annual cost, SSR and SCR in the initial streaming dispatch, alongside EUE and LOLP.",
    )
    args = parser.parse_args()
    if args.enrich_from is not None and args.replay_from is not None:
        parser.error("--enrich-from and --replay-from cannot be used together.")
    if args.enrich_from is not None:
        source_controlled = {"--case", "--analysis-years", "--scenario-count", "--seed-start", "--pv-capacities",
                             "--battery-capacities", "--input-dir", "--config"}
        if any(token.split("=", 1)[0] in source_controlled for token in sys.argv[1:]):
            parser.error("--enrich-from uses the saved case, grid, horizon, scenarios and inputs; overrides are not allowed.")
    if args.replay_from is not None:
        source_controlled = {"--case", "--analysis-years", "--scenario-count", "--seed-start", "--input-dir", "--config"}
        if any(token.split("=", 1)[0] in source_controlled for token in sys.argv[1:]):
            parser.error("--replay-from uses the saved case, horizon, scenarios and inputs; only PV/BESS capacities may change.")
    spec = SweepSpec(args.case, args.analysis_years, args.scenario_count, args.seed_start,
                     tuple(args.pv_capacities), tuple(args.battery_capacities))
    output = args.output_dir or ROOT / "outputs" / args.case.split("_", 1)[0] / f"{args.case}_Capacity_Sweep_{args.analysis_years}Y_{args.scenario_count}Scenarios"
    if args.enrich_from is not None and args.output_dir is None:
        output = args.enrich_from.resolve().with_name(args.enrich_from.resolve().name + "_Cost_SSR_SCR")
    run_sweep(spec, output, workers=args.workers, resume=args.resume, input_dir=args.input_dir,
              config_path=args.config, enrich_from=args.enrich_from,
              replay_from=args.replay_from,
              include_economic_metrics=args.include_economic_metrics)


if __name__ == "__main__":
    main()
