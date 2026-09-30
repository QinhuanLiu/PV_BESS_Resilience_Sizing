from __future__ import annotations

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
import time
from typing import Any

import numpy as np
import pandas as pd

from ..app.workflow import filter_cases
from ..design_evaluation import evaluate_design_case
from ..io.data_loader import (
    CaseParameters,
    InputCase,
    discover_input_cases,
    load_case_parameters,
    load_input_case,
    validate_sensitivity_config,
)
from ..io.reporting import write_csv
from .planning import SensitivityPoint, SensitivitySpec, build_sensitivity_plan, catalog_rows, normalize_requested_parameters


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PreparedSensitivityCase:
    case: InputCase
    parameters: CaseParameters
    spec: SensitivitySpec
    points: tuple[SensitivityPoint, ...]


def run_sensitivity(
    *,
    config: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    requested_parameters: list[str] | tuple[str, ...],
    mode: str = "oat",
    analysis_years: int | None = None,
    workers: int = 1,
    allow_large: bool = False,
    warm_start_enabled: bool = True,
    city_filters: list[str] | None = None,
    case_filters: list[str] | None = None,
    resume: bool = True,
) -> dict[str, Path]:
    """Run reusable one- or multi-parameter sensitivity experiments for selected cases."""
    base_spec = SensitivitySpec(
        parameters=tuple(requested_parameters),
        mode=str(mode),
        analysis_years=analysis_years,
        workers=int(workers),
        allow_large=bool(allow_large),
        warm_start_enabled=bool(warm_start_enabled),
    )
    base_spec.validate()
    cases = filter_cases(discover_input_cases(input_dir), city_filters, case_filters)
    # Validate every selected case and every prospective point before creating
    # any run directory. This prevents a later invalid case from leaving a
    # partial multi-case sensitivity batch on disk.
    prepared = [_prepare_case(case, config, base_spec) for case in cases]
    paths: dict[str, Path] = {}
    for item in prepared:
        paths.update(_run_case(item, config, output_dir, resume=resume))
    return paths


def _prepare_case(
    case: InputCase,
    config: dict[str, Any],
    requested_spec: SensitivitySpec,
) -> PreparedSensitivityCase:
    parameters = load_case_parameters(case.parameter_csv)
    selected = normalize_requested_parameters(requested_spec.parameters, parameters)
    _, _, initial_config, _ = load_input_case(case, config)
    resolved_years = int(requested_spec.analysis_years or initial_config["analysis"]["years"])
    spec = replace(requested_spec, parameters=selected, analysis_years=resolved_years)
    spec.validate()
    points = build_sensitivity_plan(parameters, spec)
    _preflight(case, config, spec, points)
    return PreparedSensitivityCase(case=case, parameters=parameters, spec=spec, points=tuple(points))


def _run_case(
    prepared: PreparedSensitivityCase,
    config: dict[str, Any],
    output_root: Path,
    *,
    resume: bool,
) -> dict[str, Path]:
    case = prepared.case
    parameters = prepared.parameters
    spec = prepared.spec
    points = list(prepared.points)

    output = _case_output_dir(output_root, case.case_key, spec)
    output.mkdir(parents=True, exist_ok=True)
    effective_config = _effective_config(case, config, int(spec.analysis_years or 1))
    fingerprint = _fingerprint(case, effective_config, spec, points)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError(f"Sensitivity output already exists; choose another output directory: {output}")
        manifest = _read_json(manifest_path)
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Sensitivity resume refused: input, configuration, source, runtime, or spec changed")
    else:
        _snapshot_inputs(case, output)
        _atomic_json(
            manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": _utc_now(),
                "fingerprint": fingerprint,
                "spec": asdict(spec),
                "points": [asdict(point) for point in points],
                "effective_config": effective_config,
                "input_hashes": _input_hashes(case),
                "source_hashes": _source_hashes(),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            },
        )
    _atomic_json(output / "effective_config.json", effective_config)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    _atomic_json(output / "run_status.json", _status_payload("running", spec, points, checkpoint_dir))

    try:
        _execute_points(case, config, spec, points, checkpoint_dir, fingerprint)
        rows = _completed_rows(points, checkpoint_dir, fingerprint)
        _validate_results(rows, points, spec)
        result = _order_results(pd.DataFrame(rows), spec.parameters)
        result_path = write_csv(output / "sensitivity_results.csv", result)
        catalog_path = write_csv(output / "sensitivity_parameter_catalog.csv", pd.DataFrame(catalog_rows(parameters, spec.parameters)))
        validation = {
            "passed": True,
            "completed_at": _utc_now(),
            "expected_run_count": len(points),
            "result_row_count": len(result),
            "result_sha256": _file_hash(result_path),
            "catalog_sha256": _file_hash(catalog_path),
        }
        _atomic_json(output / "validation.json", validation)
        _atomic_json(output / "run_status.json", _status_payload("complete", spec, points, checkpoint_dir))
    except BaseException:
        _atomic_json(output / "run_status.json", _status_payload("failed", spec, points, checkpoint_dir))
        raise
    return {
        f"{case.case_key}_sensitivity_results": output / "sensitivity_results.csv",
        f"{case.case_key}_sensitivity_catalog": output / "sensitivity_parameter_catalog.csv",
        f"{case.case_key}_sensitivity_manifest": manifest_path,
    }


def _preflight(case: InputCase, config: dict[str, Any], spec: SensitivitySpec, points: list[SensitivityPoint]) -> None:
    for point in points:
        overrides = dict(point.overrides)
        overrides["analysis_years"] = int(spec.analysis_years or 1)
        _, _, candidate_config, _ = load_input_case(case, config, parameter_overrides=overrides)
        validate_sensitivity_config(candidate_config)


def _effective_config(case: InputCase, config: dict[str, Any], analysis_years: int) -> dict[str, Any]:
    _, _, effective_config, _ = load_input_case(case, config, parameter_overrides={"analysis_years": analysis_years})
    effective_config.setdefault("outputs", {})["save_candidate_dispatch_workbooks"] = False
    return effective_config


def _execute_points(
    case: InputCase,
    config: dict[str, Any],
    spec: SensitivitySpec,
    points: list[SensitivityPoint],
    checkpoint_dir: Path,
    fingerprint: str,
) -> None:
    baseline = [point for point in points if point.is_baseline]
    nonbaseline = [point for point in points if not point.is_baseline]
    _run_chain_worker(_worker_payload(case, config, spec, baseline, checkpoint_dir, fingerprint))
    chains = _chains(nonbaseline)
    if not chains:
        return
    if spec.workers == 1 or len(chains) == 1:
        for chain in chains:
            _run_chain_worker(_worker_payload(case, config, spec, chain, checkpoint_dir, fingerprint))
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(spec.workers, len(chains)), mp_context=context) as executor:
        futures = [
            executor.submit(_run_chain_worker, _worker_payload(case, config, spec, chain, checkpoint_dir, fingerprint))
            for chain in chains
        ]
        for future in as_completed(futures):
            future.result()


def _chains(points: list[SensitivityPoint]) -> list[list[SensitivityPoint]]:
    groups: dict[str, list[SensitivityPoint]] = {}
    for point in points:
        groups.setdefault(point.chain_id, []).append(point)
    return [sorted(chain, key=lambda point: point.chain_position) for _, chain in sorted(groups.items())]


def _run_chain_worker(payload: dict[str, Any]) -> None:
    case = _case_from_payload(payload["case"])
    config = payload["config"]
    spec = SensitivitySpec(**payload["spec"])
    points = [SensitivityPoint(**point) for point in payload["points"]]
    checkpoint_dir = Path(payload["checkpoint_dir"])
    fingerprint = str(payload["fingerprint"])
    previous_best: dict[str, float] | None = None
    for point in points:
        checkpoint = _checkpoint_path(checkpoint_dir, point)
        if checkpoint.exists():
            row = _read_checkpoint(checkpoint, fingerprint)
            previous_best = _anchor_from_row(row)
            continue
        overrides = {**point.overrides, "analysis_years": int(spec.analysis_years or 1)}
        _, _, run_config, _ = load_input_case(case, config, parameter_overrides=overrides)
        run_config.setdefault("outputs", {})["save_candidate_dispatch_workbooks"] = False
        run_config.setdefault("capacity_search", {})["warm_start_enabled"] = bool(spec.warm_start_enabled)
        warm_anchor = (
            previous_best
            if spec.mode == "factorial" and spec.warm_start_enabled and point.chain_position > 0
            else None
        )
        started = time.perf_counter()
        design, _, _, dispatch, _, candidate_dispatches, _, _ = evaluate_design_case(
            case,
            run_config,
            case.case_key,
            warm_start_best=warm_anchor,
        )
        if candidate_dispatches:
            raise AssertionError("Sensitivity runs must not retain candidate dispatch workbooks")
        row = _result_row(point, spec, design, dispatch, time.perf_counter() - started)
        row["worker_pid"] = os.getpid()
        _atomic_json(checkpoint, {"fingerprint": fingerprint, "payload_sha256": _digest(row), "payload": row})
        previous_best = _anchor_from_row(row)


def _result_row(
    point: SensitivityPoint,
    spec: SensitivitySpec,
    design: pd.DataFrame,
    dispatch: pd.DataFrame,
    elapsed_s: float,
) -> dict[str, Any]:
    if design.empty:
        raise ValueError(f"Sensitivity point {point.run_id} produced no design result")
    row = dict(design.iloc[0])
    outage_load = float(row.get("outage_weighted_load_kwh", 0.0))
    eue = float(row.get("unserved_energy_kwh_per_year", row.get("UE_kwh_per_year", 0.0)))
    result = {
        "run_id": point.run_id,
        "execution_index": int(point.execution_index),
        "sensitivity_mode": spec.mode,
        "is_baseline": bool(point.is_baseline),
        "chain_id": point.chain_id,
        "chain_position": int(point.chain_position),
        "point_elapsed_s": float(elapsed_s),
        "annual_cost_sgd_per_year": float(row["pv_bess_system_annual_cost_sgd"]),
        "EUE_kwh_per_year": eue,
        "LOLP": 0.0 if outage_load <= 0.0 else eue / outage_load,
        **_ssr_scr_metrics(dispatch),
        **{f"parameter_{name}": float(point.overrides[name]) for name in spec.parameters},
        **row,
    }
    result["annual_cost_sgd_per_year"] = float(row["pv_bess_system_annual_cost_sgd"])
    result["EUE_kwh_per_year"] = eue
    result["LOLP"] = 0.0 if outage_load <= 0.0 else eue / outage_load
    return result


def _ssr_scr_metrics(dispatch: pd.DataFrame) -> dict[str, Any]:
    required = [
        "total_load_e_equiv_kwh",
        "pv_available_kwh",
        "pv_to_load_kwh",
        "pv_to_battery_kwh",
        "battery_discharge_from_pv_kwh",
    ]
    missing = [column for column in required if column not in dispatch.columns]
    if missing:
        raise ValueError(f"Dispatch is missing SSR/SCR fields: {missing}")
    load = float(dispatch["total_load_e_equiv_kwh"].sum())
    pv_available = float(dispatch["pv_available_kwh"].sum())
    pv_to_load = float(dispatch["pv_to_load_kwh"].sum())
    pv_to_battery = float(dispatch["pv_to_battery_kwh"].sum())
    discharge_from_pv = float(dispatch["battery_discharge_from_pv_kwh"].sum())
    if load <= 0.0:
        raise ValueError("SSR requires positive total load")
    scr_defined = pv_available > 0.0
    return {
        "SSR": float((pv_to_load + discharge_from_pv) / load),
        "SCR": float((pv_to_load + pv_to_battery) / pv_available) if scr_defined else None,
        "SCR_defined": bool(scr_defined),
        "SCR_status": "defined" if scr_defined else "not_applicable_no_pv",
    }


def _completed_rows(points: list[SensitivityPoint], checkpoint_dir: Path, fingerprint: str) -> list[dict[str, Any]]:
    return [_read_checkpoint(_checkpoint_path(checkpoint_dir, point), fingerprint) for point in points]


def _validate_results(rows: list[dict[str, Any]], points: list[SensitivityPoint], spec: SensitivitySpec) -> None:
    frame = pd.DataFrame(rows)
    expected = {point.run_id for point in points}
    if len(frame) != len(points) or set(frame["run_id"]) != expected or frame["run_id"].duplicated().any():
        raise ValueError("Sensitivity checkpoints do not cover the planned runs exactly once")
    if int(frame["is_baseline"].astype(bool).sum()) != 1:
        raise ValueError("Sensitivity results must contain exactly one baseline point")
    required = ["annual_cost_sgd_per_year", "pv_capacity_kw", "battery_energy_kwh", "EUE_kwh_per_year", "LOLP", "SSR"]
    for column in required:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Sensitivity results contain non-finite {column}")
    if frame["LOLP"].lt(0).any() or frame["LOLP"].gt(1).any() or frame["SSR"].lt(0).any() or frame["SSR"].gt(1).any():
        raise ValueError("Sensitivity probability metrics are outside [0, 1]")
    if spec.mode == "oat" and frame.loc[~frame["is_baseline"], "warm_start_used"].astype(bool).any():
        raise ValueError("OAT sensitivity runs must not use warm-start")


def _order_results(frame: pd.DataFrame, parameters: tuple[str, ...]) -> pd.DataFrame:
    front = [
        "run_id", "execution_index", "sensitivity_mode", "is_baseline", "chain_id", "chain_position",
        *[f"parameter_{name}" for name in parameters],
        "case_key", "city_key", "building_key", "analysis_years", "annual_cost_sgd_per_year",
        "pv_capacity_kw", "battery_energy_kwh", "battery_power_kw", "EUE_kwh_per_year", "LOLP", "SSR", "SCR",
        "SCR_defined", "SCR_status", "point_elapsed_s", "warm_start_used", "warm_start_source_pv_capacity_kw",
        "warm_start_source_battery_energy_kwh", "warm_start_fallback_triggered", "warm_start_fallback_reason",
        "warm_start_candidate_count", "capacity_search_method", "worker_pid",
    ]
    columns = [column for column in front if column in frame.columns]
    columns.extend(column for column in frame.columns if column not in columns)
    return frame.sort_values("execution_index").reset_index(drop=True)[columns]


def _case_output_dir(output_root: Path, case_key: str, spec: SensitivitySpec) -> Path:
    city = case_key.split("_", 1)[0]
    names = "-".join(spec.parameters)
    if len(names) > 120:
        identity = _digest({"parameters": list(spec.parameters)})[:10]
        names = f"{len(spec.parameters)}P_{spec.parameters[0]}_to_{spec.parameters[-1]}_{identity}"
    return Path(output_root) / city / f"{case_key}_Sensitivity_{spec.mode}_{names}_{spec.analysis_years}Y"


def _worker_payload(
    case: InputCase,
    config: dict[str, Any],
    spec: SensitivitySpec,
    points: list[SensitivityPoint],
    checkpoint_dir: Path,
    fingerprint: str,
) -> dict[str, Any]:
    return {
        "case": _case_payload(case),
        "config": config,
        "spec": asdict(spec),
        "points": [asdict(point) for point in points],
        "checkpoint_dir": str(checkpoint_dir),
        "fingerprint": fingerprint,
    }


def _case_payload(case: InputCase) -> dict[str, str]:
    return {
        "city_key": case.city_key,
        "building_key": case.building_key,
        "case_key": case.case_key,
        "city_dir": str(case.city_dir),
        "load_csv": str(case.load_csv),
        "pv_csv": str(case.pv_csv),
        "tou_csv": str(case.tou_csv),
        "parameter_csv": str(case.parameter_csv),
    }


def _case_from_payload(payload: dict[str, str]) -> InputCase:
    return InputCase(
        city_key=payload["city_key"],
        building_key=payload["building_key"],
        case_key=payload["case_key"],
        city_dir=Path(payload["city_dir"]),
        load_csv=Path(payload["load_csv"]),
        pv_csv=Path(payload["pv_csv"]),
        tou_csv=Path(payload["tou_csv"]),
        parameter_csv=Path(payload["parameter_csv"]),
    )


def _checkpoint_path(directory: Path, point: SensitivityPoint) -> Path:
    return directory / f"{point.execution_index:04d}_{point.run_id}.json"


def _read_checkpoint(path: Path, fingerprint: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Missing sensitivity checkpoint: {path.name}")
    envelope = _read_json(path)
    payload = envelope.get("payload")
    if envelope.get("fingerprint") != fingerprint or not isinstance(payload, dict) or envelope.get("payload_sha256") != _digest(payload):
        raise ValueError(f"Sensitivity checkpoint fingerprint or checksum mismatch: {path}")
    return payload


def _anchor_from_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        "pv_capacity_kw": float(row["pv_capacity_kw"]),
        "battery_energy_kwh": float(row["battery_energy_kwh"]),
    }


def _status_payload(state: str, spec: SensitivitySpec, points: list[SensitivityPoint], checkpoint_dir: Path) -> dict[str, Any]:
    completed = len(list(checkpoint_dir.glob("*.json"))) if checkpoint_dir.exists() else 0
    return {
        "state": state,
        "updated_at": _utc_now(),
        "mode": spec.mode,
        "parameters": list(spec.parameters),
        "analysis_years": int(spec.analysis_years or 1),
        "warm_start_enabled": bool(spec.warm_start_enabled and spec.mode == "factorial"),
        "workers_requested": int(spec.workers),
        "runs_completed": completed,
        "runs_expected": len(points),
    }


def _fingerprint(case: InputCase, effective_config: dict[str, Any], spec: SensitivitySpec, points: list[SensitivityPoint]) -> str:
    return _digest(
        {
            "schema_version": SCHEMA_VERSION,
            "case_key": case.case_key,
            "spec": asdict(spec),
            "points": [asdict(point) for point in points],
            "effective_config": effective_config,
            "input_hashes": _input_hashes(case),
            "source_hashes": _source_hashes(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        }
    )


def _input_hashes(case: InputCase) -> dict[str, str]:
    return {path.name: _file_hash(path) for path in (case.load_csv, case.pv_csv, case.tou_csv, case.parameter_csv)}


def _source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): _file_hash(path) for path in sorted((ROOT / "src").rglob("*.py"))}


def _snapshot_inputs(case: InputCase, output: Path) -> None:
    target = output / "inputs"
    target.mkdir(exist_ok=True)
    for source in (case.load_csv, case.pv_csv, case.tou_csv, case.parameter_csv):
        shutil.copy2(source, target / source.name)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _json_safe(value: Any) -> Any:
    """Normalize pandas/NumPy scalars so checkpoints preserve their real types."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} to sensitivity JSON")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
