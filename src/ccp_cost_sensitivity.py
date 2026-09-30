"""Brisbane one-factor CAPEX sensitivity under a strict CCP--LOLP constraint.

The module keeps the outage scenarios, hourly inputs and dispatch settings frozen
from a completed enriched capacity sweep.  PV and battery CAPEX do not alter a
candidate's dispatch or LOLP, so physical candidate records are shared across
price points and only their economic fields are repriced.

Run with ``python -m src.ccp_cost_sensitivity`` after the source Brisbane sweep
has completed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .ccp_lolp_sizing import (
    DEFAULT_SOURCE,
    CapacityGrid,
    CandidateCache,
    SearchSpec,
    SizingRun,
    _digest,
    _file_hash,
    _target_count,
    _validate_source,
)
from .io.reporting import write_csv
from .optimizer import battery_sizing
from .sweep_economics import economic_metrics, validate_economic_results


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMETER_CSV = ROOT / "Input/Brisbane_SingleFamily/Brisbane_SingleFamily_Parameter.csv"
DEFAULT_OUTPUT = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_CCP_LOLP_Cost_Sensitivity_15Y"
SCHEMA_VERSION = 1
LOLP_THRESHOLD = 0.20
CCP_TARGET = 0.90
PV_PARAMETER = "pv_capex_sgd_per_kw"
BATTERY_PARAMETER = "battery_capex_sgd_per_kwh"


@dataclass(frozen=True)
class CostSensitivitySpec:
    """Fixed specification for the two Brisbane one-factor CAPEX curves."""

    case_key: str = "Brisbane_SingleFamily"
    lolp_threshold: float = LOLP_THRESHOLD
    ccp_target: float = CCP_TARGET
    pv_step_kw: float = 0.25
    battery_step_kwh: float = 0.25
    intermediate_pv_step_kw: float = 0.5
    intermediate_battery_step_kwh: float = 0.5
    bess_batch_size: int = 12

    def validate(self) -> None:
        if self.case_key != "Brisbane_SingleFamily":
            raise ValueError("CCP CAPEX sensitivity currently supports Brisbane_SingleFamily only")
        if not np.isclose(self.lolp_threshold, LOLP_THRESHOLD):
            raise ValueError("CCP CAPEX sensitivity fixes the strict LOLP threshold at 0.20")
        if not np.isclose(self.ccp_target, CCP_TARGET):
            raise ValueError("CCP CAPEX sensitivity fixes the CCP target at 0.90")
        if self.pv_step_kw <= 0 or self.battery_step_kwh <= 0:
            raise ValueError("Final PV and BESS grid steps must be positive")
        if self.intermediate_pv_step_kw <= 0 or self.intermediate_battery_step_kwh <= 0:
            raise ValueError("Intermediate PV and BESS grid steps must be positive")
        if self.bess_batch_size <= 0:
            raise ValueError("BESS batch size must be positive")


@dataclass(frozen=True)
class PricePoint:
    sweep_key: str
    sweep_parameter: str
    sweep_value: float
    sweep_unit: str
    pv_capex_sgd_per_kw: float
    battery_capex_sgd_per_kwh: float

    @property
    def price_key(self) -> tuple[float, float]:
        return (round(float(self.pv_capex_sgd_per_kw), 12), round(float(self.battery_capex_sgd_per_kwh), 12))

    @property
    def directory_token(self) -> str:
        return f"pv_{self.pv_capex_sgd_per_kw:g}_bess_{self.battery_capex_sgd_per_kwh:g}".replace(".", "p")


_TOTAL_COLUMNS = (
    "grid_import_kwh_total",
    "grid_cost_sgd_total",
    "pv_export_kwh_total",
    "pv_export_revenue_sgd_total",
    "unserved_energy_kwh_total",
    "dispatch_timesteps",
    "battery_soc_kwh_sum",
    "battery_charge_kwh_total",
    "battery_discharge_kwh_total",
    "load_kwh_total",
    "pv_to_load_kwh_total",
    "battery_discharge_from_pv_kwh_total",
    "pv_to_battery_kwh_total",
    "pv_available_kwh_total",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def _parameter_rows(parameter_csv: Path) -> pd.DataFrame:
    if not parameter_csv.is_file():
        raise ValueError(f"Missing Brisbane Parameter.csv: {parameter_csv}")
    rows = pd.read_csv(parameter_csv)
    required = {"parameter", "base_value", "unit", "sensitivity_start", "sensitivity_end", "sensitivity_step"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Brisbane Parameter.csv is missing columns: {', '.join(missing)}")
    if rows.parameter.duplicated().any():
        raise ValueError("Brisbane Parameter.csv has duplicate parameter names")
    return rows.set_index("parameter", drop=False)


def _numeric(row: pd.Series, field: str, parameter: str) -> float:
    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parameter} requires a numeric {field} in Brisbane Parameter.csv") from exc
    if not np.isfinite(value):
        raise ValueError(f"{parameter} {field} must be finite")
    return value


def _range_from_parameter(rows: pd.DataFrame, parameter: str) -> tuple[float, tuple[float, ...], str]:
    if parameter not in rows.index:
        raise ValueError(f"Brisbane Parameter.csv does not define {parameter}")
    row = rows.loc[parameter]
    base = _numeric(row, "base_value", parameter)
    start = _numeric(row, "sensitivity_start", parameter)
    end = _numeric(row, "sensitivity_end", parameter)
    step = _numeric(row, "sensitivity_step", parameter)
    if step <= 0 or end < start:
        raise ValueError(f"{parameter} requires sensitivity_end >= sensitivity_start and a positive step")
    count = int(round((end - start) / step))
    values = tuple(round(start + index * step, 12) for index in range(count + 1))
    if not np.isclose(values[-1], end, rtol=0.0, atol=1e-9):
        raise ValueError(f"{parameter} sensitivity range does not land on sensitivity_end")
    unit = str(row["unit"])
    return base, values, unit


def build_price_points(parameter_csv: Path = DEFAULT_PARAMETER_CSV) -> tuple[PricePoint, ...]:
    """Build the requested two self-contained one-factor curves from current inputs."""
    rows = _parameter_rows(Path(parameter_csv))
    pv_base, pv_values, pv_unit = _range_from_parameter(rows, PV_PARAMETER)
    battery_base, battery_values, battery_unit = _range_from_parameter(rows, BATTERY_PARAMETER)
    points = [
        PricePoint("pv_capex", PV_PARAMETER, value, pv_unit, value, battery_base)
        for value in pv_values
    ]
    points.extend(
        PricePoint("battery_capex", BATTERY_PARAMETER, value, battery_unit, pv_base, value)
        for value in battery_values
    )
    if not any(point.price_key == (round(pv_base, 12), round(battery_base, 12)) for point in points):
        raise ValueError("Both Brisbane CAPEX sensitivity ranges must contain their base values")
    return tuple(points)


def _price_config(base: dict[str, Any], point: PricePoint) -> dict[str, Any]:
    config = deepcopy(base)
    config.setdefault("pv", {})["capex_sgd_per_kw"] = float(point.pv_capex_sgd_per_kw)
    economics = config.setdefault("economics", {})
    economics["battery_capex_sgd_per_kwh"] = float(point.battery_capex_sgd_per_kwh)
    active = config.setdefault("active_scenario", {})
    active_economics = active.setdefault("battery_economics", {})
    active_economics["battery_capex_sgd_per_kwh"] = float(point.battery_capex_sgd_per_kwh)
    scenario_key = str(config.get("active_scenario_key") or "Brisbane_SingleFamily")
    scenarios = config.setdefault("scenarios", {})
    scenario = scenarios.setdefault(scenario_key, {})
    scenario_economics = scenario.setdefault("battery_economics", {})
    scenario_economics["battery_capex_sgd_per_kwh"] = float(point.battery_capex_sgd_per_kwh)
    return config


def _reprice_rows(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Recompute CAPEX-dependent metrics while retaining dispatch/reliability fields."""
    missing = sorted(set(_TOTAL_COLUMNS) - set(rows.columns))
    if missing:
        raise ValueError(f"Physical candidate records cannot be repriced; missing: {', '.join(missing)}")
    required = {"pv_capacity_kw", "battery_energy_kwh"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Physical candidate records lack capacity fields: {', '.join(missing)}")
    result = rows.copy()
    # Repricing a 100-scenario source sweep changes several economic columns
    # per row.  Accumulating updates and assigning whole columns avoids slow
    # mixed-dtype scalar writes while preserving the normal cost function.
    update_rows: list[dict[str, Any]] = []
    for record in result.to_dict(orient="records"):
        totals = {column: float(record[column]) for column in _TOTAL_COLUMNS}
        sizing = battery_sizing(float(record["battery_energy_kwh"]), config)
        metrics = economic_metrics(totals, sizing, float(record["pv_capacity_kw"]), config)
        update = dict(metrics)
        update["battery_power_kw"] = float(sizing["battery_power_kw"])
        if "EUE_kwh_total" in result.columns:
            update["EUE_threshold_difference_kwh_total"] = (
                float(metrics["unserved_energy_kwh_total"]) - float(record["EUE_kwh_total"])
            )
        update_rows.append(update)
    updates = pd.DataFrame(update_rows, index=result.index)
    for name in updates.columns:
        result[name] = updates[name]
    return result


class SharedPhysicalCache:
    """Price-independent scenario records, persisted separately from price results."""

    def __init__(
        self,
        source: dict[str, Any],
        *,
        pv_grid: Any,
        battery_grid: Any,
        output: Path,
        fingerprint: str,
    ) -> None:
        self.pv_grid = pv_grid
        self.battery_grid = battery_grid
        self.output = output
        self.fingerprint = fingerprint
        self.rows_by_key: dict[tuple[int, int], pd.DataFrame] = {}
        self.external_paths: dict[tuple[int, int], Path] = {}
        scenarios = source["scenarios"].sort_values("scenario_id")
        self.scenario_ids = tuple(scenarios.scenario_id.astype(int).tolist())
        self._seed(source["results"])
        self._load()

    def key(self, pv_kw: float, battery_kwh: float) -> tuple[int, int]:
        return self.pv_grid.index(float(pv_kw)), self.battery_grid.index(float(battery_kwh))

    def _seed(self, rows: pd.DataFrame) -> None:
        for _, group in rows.groupby(["pv_capacity_kw", "battery_energy_kwh"], sort=True):
            key = self.key(float(group.pv_capacity_kw.iloc[0]), float(group.battery_energy_kwh.iloc[0]))
            self.rows_by_key[key] = group.sort_values("scenario_id").reset_index(drop=True).copy()

    def _path(self, key: tuple[int, int]) -> Path:
        return self.output / "physical_checkpoints" / f"pv_{key[0]:05d}_bess_{key[1]:05d}.json"

    def _load(self) -> None:
        directory = self.output / "physical_checkpoints"
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("pv_*_bess_*.json")):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope.get("payload")
            if not isinstance(payload, dict):
                raise ValueError(f"Shared physical checkpoint payload is invalid: {path}")
            if envelope.get("fingerprint") != self.fingerprint or envelope.get("payload_sha256") != _digest(payload):
                raise ValueError(f"Shared physical checkpoint checksum/fingerprint mismatch: {path}")
            frame = pd.DataFrame(payload.get("scenario_rows"))
            if frame.empty:
                raise ValueError(f"Shared physical checkpoint is empty: {path}")
            key = self.key(float(frame.pv_capacity_kw.iloc[0]), float(frame.battery_energy_kwh.iloc[0]))
            if tuple(payload.get("key", [])) != key:
                raise ValueError(f"Shared physical checkpoint coordinate mismatch: {path}")
            self.add(frame, persist=False)

    def has(self, key: tuple[int, int]) -> bool:
        return key in self.rows_by_key or key in self.external_paths

    def rows(self, key: tuple[int, int]) -> pd.DataFrame:
        if key not in self.rows_by_key:
            path = self.external_paths.get(key)
            if path is None:
                raise KeyError(key)
            rows = self._read_external_checkpoint(path, key)
            self.add(rows, persist=False)
        return self.rows_by_key[key].copy()

    @property
    def candidate_count(self) -> int:
        return len(set(self.rows_by_key) | set(self.external_paths))

    def add(self, rows: pd.DataFrame, *, persist: bool = True) -> None:
        if rows.empty or rows.pv_capacity_kw.nunique() != 1 or rows.battery_energy_kwh.nunique() != 1:
            raise ValueError("Shared physical cache rows must describe exactly one non-empty capacity point")
        ids = tuple(sorted(rows.scenario_id.astype(int).tolist()))
        if ids != self.scenario_ids or rows.scenario_id.duplicated().any():
            raise ValueError("Shared physical cache rows do not cover the frozen scenario identifiers exactly once")
        key = self.key(float(rows.pv_capacity_kw.iloc[0]), float(rows.battery_energy_kwh.iloc[0]))
        ordered = rows.sort_values("scenario_id").reset_index(drop=True).copy()
        existing = self.rows_by_key.get(key)
        if existing is not None:
            for column in ("LOLP", "EUE_kwh_total", "outage_load_kwh_total"):
                if column in existing and column in ordered and not np.allclose(
                    existing[column].to_numpy(dtype=float), ordered[column].to_numpy(dtype=float), rtol=1e-10, atol=1e-10
                ):
                    raise ValueError("Conflicting physical reliability records for one capacity point")
            return
        self.rows_by_key[key] = ordered
        if persist:
            payload = {"key": list(key), "scenario_rows": ordered.to_dict(orient="records")}
            _atomic_json(self._path(key), {"fingerprint": self.fingerprint, "payload_sha256": _digest(payload), "payload": payload})

    def import_sizing_checkpoints(self, directory: Path, source_identity: dict[str, Any]) -> int:
        """Reuse compatible, price-independent records from a prior sizing run.

        This is deliberately stricter than a directory copy: the donor must
        identify the exact frozen sweep and every per-capacity payload must
        retain its original checksum.  Costs are still recalculated for every
        price point before any candidate is compared.
        """
        directory = Path(directory).resolve()
        reference_path = directory / "source_reference.json"
        manifest_path = directory / "manifest.json"
        if not reference_path.is_file() or not manifest_path.is_file():
            raise ValueError("Physical checkpoint import requires manifest.json and source_reference.json")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference != source_identity:
            raise ValueError("Physical checkpoint import refused: frozen source identity differs")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        search = manifest.get("search") if isinstance(manifest, dict) else None
        if not isinstance(search, dict) or not np.isclose(float(search.get("pv_step_kw", np.nan)), self.pv_grid.step) or not np.isclose(
            float(search.get("battery_step_kwh", np.nan)), self.battery_grid.step
        ):
            raise ValueError("Physical checkpoint import final capacity grid differs")
        checkpoint_dir = directory / "checkpoints"
        if not checkpoint_dir.is_dir():
            raise ValueError("Physical checkpoint import directory has no checkpoints")
        imported = 0
        for path in sorted(checkpoint_dir.glob("pv_*_bess_*.json")):
            key = self._read_external_checkpoint(path)
            if key in self.rows_by_key:
                # Source-grid points can occur in the donor.  Load these only
                # long enough to verify they agree with their frozen source.
                self.add(self._read_external_checkpoint(path, key), persist=False)
                continue
            if key in self.external_paths:
                raise ValueError(f"Physical checkpoint import contains duplicate capacity key: {path}")
            self.external_paths[key] = path
            imported += 1
        if not imported:
            raise ValueError("Physical checkpoint import supplied no new capacity records")
        return imported

    def _read_external_checkpoint(self, path: Path, expected_key: tuple[int, int] | None = None) -> tuple[int, int] | pd.DataFrame:
        """Validate one donor payload; return its key or its rows when requested."""
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope.get("payload")
        if not isinstance(payload, dict) or envelope.get("payload_sha256") != _digest(payload):
            raise ValueError(f"Physical checkpoint import checksum mismatch: {path}")
        rows = pd.DataFrame(payload.get("scenario_rows"))
        if rows.empty:
            raise ValueError(f"Physical checkpoint import is empty: {path}")
        key = self.key(float(rows.pv_capacity_kw.iloc[0]), float(rows.battery_energy_kwh.iloc[0]))
        if tuple(payload.get("key", [])) != key:
            raise ValueError(f"Physical checkpoint import coordinate mismatch: {path}")
        # Calling add performs exact scenario-identifier and one-coordinate checks.
        ids = tuple(sorted(rows.scenario_id.astype(int).tolist()))
        if ids != self.scenario_ids or rows.scenario_id.duplicated().any():
            raise ValueError(f"Physical checkpoint import scenario coverage differs: {path}")
        if expected_key is not None:
            if key != expected_key:
                raise ValueError(f"Physical checkpoint import key changed unexpectedly: {path}")
            return rows
        return key


class PriceSizingRun(SizingRun):
    """A price-specific adaptive search backed by a shared physical cache."""

    def __init__(self, source: dict[str, Any], output: Path, spec: SearchSpec, workers: int, fingerprint: str,
                 trace: list[dict[str, Any]], shared: SharedPhysicalCache) -> None:
        self.shared = shared
        super().__init__(source, output, spec, workers, fingerprint, trace)
        # Source sweep points already exist in the normal cache. Add only fine
        # candidates discovered by prior price points.
        for key, rows in sorted(self.shared.rows_by_key.items()):
            if self.in_domain(key) and not self.cache.has(key):
                self.cache.add(_reprice_rows(rows, self.source["config"]), "shared_physical")

    def evaluate(self, keys: Iterable[tuple[int, int]], *, reason: str) -> None:
        requested = self._new_keys(keys)
        for key in [key for key in requested if self.shared.has(key)]:
            self.cache.add(_reprice_rows(self.shared.rows(key), self.source["config"]), "shared_physical")
        to_dispatch = self._new_keys(keys)
        if not to_dispatch:
            return
        super().evaluate(to_dispatch, reason=reason)
        for key in to_dispatch:
            self.shared.add(self.cache.rows(key))


def _price_source(source: dict[str, Any], point: PricePoint) -> dict[str, Any]:
    adjusted = {key: value for key, value in source.items()}
    adjusted["config"] = _price_config(source["config"], point)
    adjusted["results"] = _reprice_rows(source["results"], adjusted["config"])
    validate_economic_results(adjusted["results"], adjusted["config"])
    return adjusted


def _search_spec(spec: CostSensitivitySpec) -> SearchSpec:
    return SearchSpec(
        lolp_thresholds=(spec.lolp_threshold,),
        ccp_targets=(spec.ccp_target,),
        pv_step_kw=spec.pv_step_kw,
        battery_step_kwh=spec.battery_step_kwh,
        intermediate_pv_step_kw=spec.intermediate_pv_step_kw,
        intermediate_battery_step_kwh=spec.intermediate_battery_step_kwh,
        bess_batch_size=spec.bess_batch_size,
    )


def _point_result(point: PricePoint, run: PriceSizingRun, selection: tuple[tuple[int, int] | None, dict[str, Any] | None],
                  spec: CostSensitivitySpec) -> dict[str, Any]:
    key, summary = selection
    scenario_count = len(run.scenario_ids)
    required = _target_count(spec.ccp_target, scenario_count)
    result: dict[str, Any] = {
        "sweep_key": point.sweep_key,
        "sweep_parameter": point.sweep_parameter,
        "sweep_value": float(point.sweep_value),
        "sweep_unit": point.sweep_unit,
        "pv_capex_sgd_per_kw": float(point.pv_capex_sgd_per_kw),
        "battery_capex_sgd_per_kwh": float(point.battery_capex_sgd_per_kwh),
        "battery_power_capex_sgd_per_kw": float(run.source["config"]["economics"]["battery_power_capex_sgd_per_kw"]),
        "case_key": spec.case_key,
        "analysis_years": int(run.source["config"]["analysis"]["years"]),
        "lolp_threshold": float(spec.lolp_threshold),
        "ccp_target": float(spec.ccp_target),
        "comparison": "LOLP < threshold",
        "scenario_count": scenario_count,
        "required_passing_scenarios": required,
        "feasible": key is not None and summary is not None,
    }
    if key is None or summary is None:
        return {**result, "passing_scenarios": 0, "actual_ccp": np.nan, "domain_limited": False}
    rows = run.cache.rows(key)
    passing = run.cache.count(key, spec.lolp_threshold)
    scr = rows["SCR"].astype(float)
    scr_defined = rows["SCR_defined"].astype(bool)
    upper_pv = key[0] == run.domain[1]
    upper_battery = key[1] == run.domain[3]
    return {
        **result,
        "passing_scenarios": int(passing),
        "actual_ccp": float(passing / scenario_count),
        "domain_limited": bool(upper_pv or upper_battery),
        "boundary_status": "upper_domain" if upper_pv or upper_battery else "interior_or_screened_boundary",
        "candidate_origin": str(summary["candidate_origin"]),
        "annual_cost_sgd_per_year": float(summary["mean_annual_cost_sgd_per_year"]),
        "annual_system_cost_ksgd_per_year": float(summary["mean_annual_cost_sgd_per_year"]) / 1000.0,
        "pv_capacity_kw": float(summary["pv_capacity_kw"]),
        "battery_energy_kwh": float(summary["battery_energy_kwh"]),
        "battery_power_kw": float(summary["battery_power_kw"]),
        "EUE_kwh_per_year": float(summary["mean_EUE_kwh_per_year"]),
        "LOLP": float(summary["mean_LOLP"]),
        "SSR": float(rows["SSR"].mean()),
        "SCR": float(scr.mean()) if scr_defined.all() else np.nan,
        "SCR_defined": bool(scr_defined.all()),
        "SCR_status": "defined" if scr_defined.all() else "not_applicable_no_pv",
    }


def _validate_results(frame: pd.DataFrame, points: tuple[PricePoint, ...], spec: CostSensitivitySpec) -> dict[str, Any]:
    if len(frame) != len(points):
        raise ValueError("CCP CAPEX sensitivity result count does not match requested price points")
    expected_by_sweep: dict[str, tuple[float, ...]] = {}
    for point in points:
        expected_by_sweep.setdefault(point.sweep_key, tuple())
        expected_by_sweep[point.sweep_key] += (float(point.sweep_value),)
    for key, expected_values in expected_by_sweep.items():
        group = frame.loc[frame.sweep_key.eq(key)]
        expected = len(expected_values)
        if len(group) != expected:
            raise ValueError(f"CCP CAPEX sensitivity expects {expected} {key} points")
        values = group.sweep_value.to_numpy(dtype=float)
        if (
            len(np.unique(values)) != expected
            or not np.all(np.diff(np.sort(values)) > 0)
            or not np.allclose(np.sort(values), np.sort(np.asarray(expected_values, dtype=float)), rtol=0.0, atol=1e-9)
        ):
            raise ValueError(f"CCP CAPEX sensitivity {key} axis is incomplete or duplicated")
    if not frame.feasible.astype(bool).all():
        raise ValueError("At least one requested CAPEX point is infeasible under the CCP--LOLP constraint")
    if frame.domain_limited.astype(bool).any():
        raise ValueError("At least one CAPEX optimum touches the search-domain upper boundary")
    if (frame.passing_scenarios.astype(int) < frame.required_passing_scenarios.astype(int)).any():
        raise ValueError("A selected CAPEX design does not meet the required passing-scenario count")
    if (frame.actual_ccp.astype(float) + 1e-12 < spec.ccp_target).any():
        raise ValueError("A selected CAPEX design does not meet the CCP target")
    expected_ccp = frame.passing_scenarios.astype(float) / frame.scenario_count.astype(float)
    if not np.allclose(frame.actual_ccp.astype(float), expected_ccp, rtol=0.0, atol=1e-12):
        raise ValueError("CCP values do not equal passing-scenario counts divided by scenario count")
    numeric = [
        "annual_cost_sgd_per_year", "pv_capacity_kw", "battery_energy_kwh", "battery_power_kw",
        "EUE_kwh_per_year", "LOLP", "SSR",
    ]
    values = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("CCP CAPEX sensitivity has non-finite selected metrics")
    if (frame.LOLP < -1e-12).any() or (frame.LOLP > 1 + 1e-12).any() or (frame.SSR < -1e-12).any() or (frame.SSR > 1 + 1e-12).any():
        raise ValueError("CCP CAPEX sensitivity probability metrics are outside [0, 1]")
    for column, step in (("pv_capacity_kw", spec.pv_step_kw), ("battery_energy_kwh", spec.battery_step_kwh)):
        values = frame[column].to_numpy(dtype=float)
        if (values < -1e-12).any() or not np.allclose(values / step, np.round(values / step), rtol=0.0, atol=1e-8):
            raise ValueError(f"CCP CAPEX sensitivity {column} values do not align to the final capacity grid")
    if (frame.battery_power_kw.to_numpy(dtype=float) < -1e-12).any():
        raise ValueError("CCP CAPEX sensitivity has a negative battery power capacity")
    defined = frame.SCR_defined.astype(bool)
    if frame.loc[defined, "SCR"].isna().any() or frame.loc[~defined, "SCR"].notna().any():
        raise ValueError("CCP CAPEX sensitivity has inconsistent SCR values")
    if defined.any() and ((frame.loc[defined, "SCR"] < -1e-12).any() or (frame.loc[defined, "SCR"] > 1 + 1e-12).any()):
        raise ValueError("CCP CAPEX sensitivity SCR is outside [0, 1]")
    return {
        "passed": True,
        "result_row_count": int(len(frame)),
        "pv_capex_point_count": int(frame.sweep_key.eq("pv_capex").sum()),
        "battery_capex_point_count": int(frame.sweep_key.eq("battery_capex").sum()),
        "constraint": "P(LOLP < 0.20) >= 0.90",
    }


def _fingerprint(source: dict[str, Any], parameter_csv: Path, spec: CostSensitivitySpec, points: tuple[PricePoint, ...]) -> str:
    return _json_digest({
        "schema_version": SCHEMA_VERSION,
        "source": source["identity"],
        "parameter_csv_sha256": _file_hash(parameter_csv),
        "spec": asdict(spec),
        "points": [asdict(point) for point in points],
    })


def _completed_result_is_valid(
    results_path: Path, validation_path: Path, fingerprint: str,
) -> bool:
    """Return true only for a complete, checksum-matching prior result."""
    if not results_path.is_file() or not validation_path.is_file():
        return False
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    return bool(
        validation.get("passed") is True
        and validation.get("result_sha256") == _file_hash(results_path)
        and validation.get("fingerprint") == fingerprint
    )


def _readme(spec: CostSensitivitySpec) -> str:
    return f"""# Brisbane CCP--LOLP CAPEX sensitivity

This completed run varies PV CAPEX and battery energy CAPEX independently while
minimising mean annual system cost subject to the strict scenario criterion
`P(LOLP < {spec.lolp_threshold:.0%}) >= {spec.ccp_target:.0%}`.

The run reuses one frozen 15-year / 100-scenario Brisbane source sweep.  Physical
dispatch and reliability records are shared by capacity across price points;
CAPEX-dependent economic fields are recalculated using the model's normal cost
function. Battery power CAPEX, VoLL, load, PV, TOU and outage scenarios remain fixed.

`ccp_cost_sensitivity_results.csv` contains 7 PV-CAPEX and 9 battery-energy-CAPEX
rows. The common base price pair appears once in each curve so both are self-contained.
"""


def run_cost_sensitivity(
    source_results_dir: Path | str = DEFAULT_SOURCE,
    output_dir: Path | str | None = None,
    *,
    parameter_csv: Path | str = DEFAULT_PARAMETER_CSV,
    workers: int = min(8, os.cpu_count() or 1),
    resume: bool = False,
    spec: CostSensitivitySpec = CostSensitivitySpec(),
    physical_checkpoint_dirs: Iterable[Path | str] = (),
) -> dict[str, Path]:
    """Run both constrained Brisbane CAPEX curves and write one validated result table."""
    spec.validate()
    if workers <= 0 or int(workers) != workers:
        raise ValueError("workers must be a positive integer")
    parameter_path = Path(parameter_csv).resolve()
    source = _validate_source(Path(source_results_dir).resolve())
    if str(source["case"].case_key) != spec.case_key:
        raise ValueError(f"Frozen source case must be {spec.case_key}")
    points = build_price_points(parameter_path)
    output = (Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT.resolve())
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(source, parameter_path, spec, points)
    manifest_path = output / "manifest.json"
    results_path = output / "ccp_cost_sensitivity_results.csv"
    validation_path = output / "validation.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("CCP CAPEX sensitivity resume refused: source, parameter ranges or specification changed")
        if _completed_result_is_valid(results_path, validation_path, fingerprint):
            return {"results": results_path, "manifest": manifest_path, "validation": validation_path}
        if not resume:
            raise ValueError("CCP CAPEX sensitivity output already exists; use --resume or choose another directory")
    else:
        _atomic_json(manifest_path, {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "fingerprint": fingerprint,
            "spec": asdict(spec),
            "points": [asdict(point) for point in points],
            "parameter_csv": str(parameter_path),
            "parameter_csv_sha256": _file_hash(parameter_path),
            "source_identity": source["identity"],
            "source_results_dir": str(Path(source_results_dir).resolve()),
        })
    _atomic_json(output / "run_status.json", {
        "state": "running", "updated_at": _utc_now(), "fingerprint": fingerprint,
        "workers_requested": int(workers), "point_count": len(points),
    })
    try:
        search = _search_spec(spec)
        shared = SharedPhysicalCache(
            source,
            pv_grid=CapacityGrid(search.pv_step_kw, "PV"),
            battery_grid=CapacityGrid(search.battery_step_kwh, "BESS"),
            output=output,
            fingerprint=fingerprint,
        )
        imported_physical: dict[str, int] = {}
        for directory in physical_checkpoint_dirs:
            path = Path(directory).resolve()
            imported_physical[str(path)] = shared.import_sizing_checkpoints(path, source["identity"])
        unique: dict[tuple[float, float], PricePoint] = {}
        for point in points:
            unique.setdefault(point.price_key, point)
        summaries: dict[tuple[float, float], dict[str, Any]] = {}
        for price_key, point in unique.items():
            point_output = output / "point_runs" / point.directory_token
            selection_path = point_output / "selection.json"
            if resume and selection_path.is_file():
                saved = json.loads(selection_path.read_text(encoding="utf-8"))
                if saved.get("price_point") != asdict(point) or not isinstance(saved.get("selection"), dict):
                    raise ValueError(f"CCP CAPEX sensitivity resume refused: invalid selection checkpoint {selection_path}")
                summaries[price_key] = dict(saved["selection"])
                continue
            point_source = _price_source(source, point)
            point_fingerprint = _json_digest({"root": fingerprint, "price_key": price_key})
            trace: list[dict[str, Any]] = []
            run = PriceSizingRun(point_source, point_output, search, int(workers), point_fingerprint, trace, shared)
            try:
                if resume:
                    run.load_checkpoints()
                best, _ = run.adaptive_mesh_search()
                selected = best[(spec.lolp_threshold, spec.ccp_target)]
                summaries[price_key] = _point_result(point, run, selected, spec)
                _atomic_json(selection_path, {
                    "root_fingerprint": fingerprint, "price_point": asdict(point), "selection": summaries[price_key], "trace": trace,
                    "worker_pids": sorted(run.worker_pids), "dispatch_seconds": run.dispatch_seconds,
                })
            finally:
                run.close_workers()
        rows = []
        for point in points:
            row = dict(summaries[point.price_key])
            row.update({
                "sweep_key": point.sweep_key,
                "sweep_parameter": point.sweep_parameter,
                "sweep_value": point.sweep_value,
                "sweep_unit": point.sweep_unit,
            })
            rows.append(row)
        frame = pd.DataFrame(rows).sort_values(["sweep_key", "sweep_value"], kind="stable").reset_index(drop=True)
        validation = _validate_results(frame, points, spec)
        result_path = write_csv(results_path, frame)
        validation.update({
            "completed_at": _utc_now(),
            "fingerprint": fingerprint,
            "result_sha256": _file_hash(result_path),
            "physical_candidate_count": shared.candidate_count,
            "imported_physical_candidate_counts": imported_physical,
            "unique_price_point_count": len(unique),
            "output_files": {"results": str(result_path), "manifest": str(manifest_path)},
        })
        _atomic_json(validation_path, validation)
        _atomic_json(output / "source_reference.json", source["identity"])
        (output / "README.md").write_text(_readme(spec), encoding="utf-8")
        _atomic_json(output / "run_status.json", {
            "state": "complete", "updated_at": _utc_now(), "fingerprint": fingerprint,
            "workers_requested": int(workers), "point_count": len(points),
            "physical_candidate_count": shared.candidate_count,
            "imported_physical_candidate_counts": imported_physical,
        })
        return {"results": result_path, "manifest": manifest_path, "validation": validation_path}
    except BaseException:
        _atomic_json(output / "run_status.json", {
            "state": "failed", "updated_at": _utc_now(), "fingerprint": fingerprint,
            "workers_requested": int(workers), "point_count": len(points),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parameter-csv", type=Path, default=DEFAULT_PARAMETER_CSV)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--physical-checkpoint-dir", type=Path, action="append", default=[],
        help="Compatible completed CCP sizing output whose checked dispatch records may be reused",
    )
    args = parser.parse_args()
    paths = run_cost_sensitivity(
        args.source_results_dir, args.output_dir, parameter_csv=args.parameter_csv,
        workers=args.workers, resume=args.resume, physical_checkpoint_dirs=args.physical_checkpoint_dir,
    )
    print(f"Completed CCP CAPEX sensitivity: {paths['results']}")


if __name__ == "__main__":
    main()
