"""Adaptive, auditable PV--BESS sizing under strict CCP--LOLP constraints.

The command reads a completed, economically enriched greedy capacity sweep and
reuses its frozen 15-year inputs and outage profiles.  It never resamples
outages.  New candidate evaluations use the same greedy dispatcher in
streaming-only mode, so no hourly or yearly dispatch results are retained.

Example
-------
``python -m src.ccp_lolp_sizing --workers 8``
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .capacity_sweep import (
    ROOT,
    _availability_hash,
    _check_source_dependencies,
    _file_hash,
    _load_enrichment_source,
)
from .costing import pv_annual_cost
from .degradation import annualized_battery_cost
from .dispatch import solve_dispatch_greedy
from .io.data_loader import InputCase, build_case_frame
from .optimizer import battery_sizing
from .sweep_economics import economic_metrics, validate_economic_results


DEFAULT_SOURCE = ROOT / "outputs/Brisbane/Brisbane_SingleFamily_Capacity_Sweep_15Y_100Scenarios_Cost_SSR_SCR"
DEFAULT_LOLP_THRESHOLDS = (0.10, 0.20, 0.30)
# The 2%-spaced sequence cannot land exactly on 95%, so retain the requested
# high-reliability endpoint explicitly after 94%.
DEFAULT_CCP_TARGETS = tuple(round(value / 100.0, 10) for value in range(50, 95, 2)) + (0.95,)
_WORKER: dict[str, Any] = {}
_EPS = 1e-12


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


@dataclass(frozen=True)
class CapacityGrid:
    """An integer representation that prevents floating-point cache duplicates."""

    step: float
    label: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.step) or self.step <= 0:
            raise ValueError(f"{self.label} step must be positive and finite")

    def index(self, value: float) -> int:
        index = int(round(float(value) / self.step))
        if not np.isclose(index * self.step, float(value), rtol=0.0, atol=1e-8):
            raise ValueError(f"{self.label} value {value} does not align with step {self.step}")
        return index

    def value(self, index: int) -> float:
        return round(int(index) * self.step, 10)

    def inclusive(self, low: int, high: int) -> range:
        if high < low:
            raise ValueError(f"Invalid {self.label} interval")
        return range(low, high + 1)


@dataclass(frozen=True)
class SearchSpec:
    lolp_thresholds: tuple[float, ...] = DEFAULT_LOLP_THRESHOLDS
    ccp_targets: tuple[float, ...] = DEFAULT_CCP_TARGETS
    pv_step_kw: float = 0.1
    battery_step_kwh: float = 0.1
    pv_min_kw: float | None = None
    pv_max_kw: float | None = None
    battery_min_kwh: float | None = None
    battery_max_kwh: float | None = None
    bess_batch_size: int = 12
    intermediate_pv_step_kw: float | None = None
    intermediate_battery_step_kwh: float | None = None
    cost_screen_margin: float = 0.05
    import_checkpoints_from: Path | str | None = None

    def validate(self) -> None:
        if not self.lolp_thresholds or not self.ccp_targets:
            raise ValueError("At least one LOLP threshold and CCP target are required")
        for label, values in (("LOLP threshold", self.lolp_thresholds), ("CCP target", self.ccp_targets)):
            if len(set(values)) != len(values):
                raise ValueError(f"{label}s must be unique")
            if not all(np.isfinite(value) and 0.0 < value < 1.0 for value in values):
                raise ValueError(f"{label}s must be strictly between zero and one")
        CapacityGrid(self.pv_step_kw, "PV")
        CapacityGrid(self.battery_step_kwh, "BESS")
        if self.bess_batch_size <= 0:
            raise ValueError("bess_batch_size must be positive")
        if (self.intermediate_pv_step_kw is None) != (self.intermediate_battery_step_kwh is None):
            raise ValueError("Intermediate PV and BESS steps must be supplied together")
        if self.intermediate_pv_step_kw is not None:
            intermediate_pv = CapacityGrid(self.intermediate_pv_step_kw, "Intermediate PV")
            intermediate_battery = CapacityGrid(self.intermediate_battery_step_kwh, "Intermediate BESS")
            final_pv = CapacityGrid(self.pv_step_kw, "PV")
            final_battery = CapacityGrid(self.battery_step_kwh, "BESS")
            if self.intermediate_pv_step_kw <= self.pv_step_kw or self.intermediate_battery_step_kwh <= self.battery_step_kwh:
                raise ValueError("Intermediate steps must be coarser than final steps")
            # A nested final grid makes every intermediate dispatch reusable.
            final_pv.index(float(self.intermediate_pv_step_kw))
            final_battery.index(float(self.intermediate_battery_step_kwh))
            del intermediate_pv, intermediate_battery
        if not np.isfinite(self.cost_screen_margin) or not 0.0 <= self.cost_screen_margin <= 1.0:
            raise ValueError("cost_screen_margin must be between zero and one")


@dataclass(frozen=True)
class CoarseCell:
    pv_low: int
    pv_high: int
    battery_low: int
    battery_high: int

    def key(self) -> tuple[int, int, int, int]:
        return self.pv_low, self.pv_high, self.battery_low, self.battery_high


@dataclass(frozen=True)
class MeshCell:
    """A rectangular leaf in the final-grid coordinate system."""

    pv_low: int
    pv_high: int
    battery_low: int
    battery_high: int
    level: str
    parent: str | None = None

    def key(self) -> tuple[int, int, int, int, str]:
        return self.pv_low, self.pv_high, self.battery_low, self.battery_high, self.level

    def identifier(self) -> str:
        return f"{self.level}:p{self.pv_low}-{self.pv_high}:b{self.battery_low}-{self.battery_high}"


@dataclass
class CandidateCache:
    """Stores source and newly evaluated scenario rows by exact grid coordinate."""

    pv_grid: CapacityGrid
    battery_grid: CapacityGrid
    scenario_ids: tuple[int, ...]
    thresholds: tuple[float, ...]
    _rows: dict[tuple[int, int], pd.DataFrame] = field(default_factory=dict)
    _origins: dict[tuple[int, int], str] = field(default_factory=dict)
    _summaries: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)

    def key(self, pv: float, battery: float) -> tuple[int, int]:
        return self.pv_grid.index(pv), self.battery_grid.index(battery)

    def has(self, key: tuple[int, int]) -> bool:
        return key in self._rows

    def add(self, rows: pd.DataFrame, origin: str) -> tuple[int, int]:
        if rows.empty:
            raise ValueError("Cannot cache empty candidate rows")
        if rows.pv_capacity_kw.nunique() != 1 or rows.battery_energy_kwh.nunique() != 1:
            raise ValueError("Candidate cache rows must have one PV/BESS coordinate")
        key = self.key(float(rows.pv_capacity_kw.iloc[0]), float(rows.battery_energy_kwh.iloc[0]))
        ids = tuple(sorted(rows.scenario_id.astype(int).tolist()))
        if ids != self.scenario_ids or rows.scenario_id.duplicated().any():
            raise ValueError("Candidate rows do not cover the shared scenario identifiers exactly once")
        current = self._rows.get(key)
        ordered = rows.sort_values("scenario_id").reset_index(drop=True).copy()
        if current is not None:
            left = current.sort_values("scenario_id").reset_index(drop=True)
            numeric = [name for name in ("LOLP", "EUE_kwh_total", "annual_cost_sgd_per_year") if name in left]
            if len(left) != len(ordered) or not all(
                np.allclose(left[name], ordered[name], rtol=1e-10, atol=1e-10) for name in numeric
            ):
                raise ValueError("Conflicting cached values for the same candidate")
            return key
        self._rows[key] = ordered
        self._origins[key] = origin
        self._summaries.pop(key, None)
        return key

    def rows(self, key: tuple[int, int]) -> pd.DataFrame:
        return self._rows[key]

    def origin(self, key: tuple[int, int]) -> str:
        return self._origins[key]

    def summary(self, key: tuple[int, int]) -> dict[str, Any]:
        if key in self._summaries:
            return self._summaries[key]
        rows = self.rows(key)
        summary: dict[str, Any] = {
            "pv_capacity_kw": self.pv_grid.value(key[0]),
            "battery_energy_kwh": self.battery_grid.value(key[1]),
            "battery_power_kw": float(rows.battery_power_kw.mean()),
            "scenario_count": len(rows),
            "mean_LOLP": float(rows.LOLP.mean()),
            "mean_EUE_kwh_total": float(rows.EUE_kwh_total.mean()),
            "mean_EUE_kwh_per_year": float(rows.EUE_kwh_per_year.mean()),
            "mean_annual_cost_sgd_per_year": float(rows.annual_cost_sgd_per_year.mean()),
            "candidate_origin": self.origin(key),
        }
        for threshold in self.thresholds:
            token = _threshold_token(threshold)
            passing = int(rows.LOLP.lt(threshold).sum())
            summary[f"LOLP_lt_{token}_passing_scenarios"] = passing
            summary[f"CCP_LOLP_lt_{token}"] = passing / len(rows)
        self._summaries[key] = summary
        return summary

    def count(self, key: tuple[int, int], threshold: float) -> int:
        return int(self.summary(key)[f"LOLP_lt_{_threshold_token(threshold)}_passing_scenarios"])

    def all_summaries(self) -> pd.DataFrame:
        rows = [self.summary(key) for key in sorted(self._rows)]
        return pd.DataFrame(rows).sort_values(["pv_capacity_kw", "battery_energy_kwh"]).reset_index(drop=True)

    def new_rows(self) -> pd.DataFrame:
        frames = [self.rows(key).assign(candidate_origin=self.origin(key)) for key in sorted(self._rows)
                  if self.origin(key) != "source"]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(
            ["pv_capacity_kw", "battery_energy_kwh", "scenario_id"]
        ).reset_index(drop=True)


def _threshold_token(threshold: float) -> str:
    return f"{int(round(threshold * 100)):02d}pct"


def strict_ccp_count(lolp: Iterable[float], threshold: float) -> int:
    """Number of scenarios satisfying the intentionally strict LOLP criterion."""
    return int(np.count_nonzero(np.asarray(list(lolp), dtype=float) < float(threshold)))


def _target_count(target: float, scenario_count: int) -> int:
    # The tiny tolerance prevents 0.55 * 100 being rounded to 56 on a binary float.
    return int(np.ceil(float(target) * scenario_count - 1e-12))


def _validate_source(source_dir: Path) -> dict[str, Any]:
    """Use the sweep's own complete-run/provenance validation before refining."""
    loaded = _load_enrichment_source(source_dir)
    results = loaded["results"]
    required = {"annual_cost_sgd_per_year", "battery_power_kw", "dispatch_backend"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Source sweep lacks required enriched greedy fields: {missing}")
    if not results.dispatch_backend.eq("greedy").all():
        raise ValueError("Only greedy-dispatch source sweeps are accepted")
    validate_economic_results(results, loaded["config"])
    # The capacity-sweep loader already validates these frozen files before
    # returning.  Retain their hashes as part of this run's identity too, so a
    # source input cannot silently change while worker processes are reading it.
    loaded["identity"]["input_snapshot_hashes"] = {
        getattr(loaded["case"], role).name: _file_hash(getattr(loaded["case"], role))
        for role in ("load_csv", "pv_csv", "tou_csv", "parameter_csv")
    }
    return loaded


def _check_source_still_frozen(source: dict[str, Any]) -> None:
    _check_source_dependencies(source["identity"])
    expected = source["identity"]["input_snapshot_hashes"]
    actual = {
        getattr(source["case"], role).name: _file_hash(getattr(source["case"], role))
        for role in ("load_csv", "pv_csv", "tou_csv", "parameter_csv")
    }
    if actual != expected:
        raise ValueError("Frozen source input snapshot changed during sizing")


def _source_grid(values: pd.Series, grid: CapacityGrid, label: str) -> tuple[int, ...]:
    indexes = tuple(sorted({grid.index(float(value)) for value in values.unique()}))
    if len(indexes) < 2:
        raise ValueError(f"Source requires at least two {label} capacity values")
    return indexes


def _resolve_domain(
    source_pv: tuple[int, ...], source_battery: tuple[int, ...], spec: SearchSpec,
    pv_grid: CapacityGrid, battery_grid: CapacityGrid,
) -> tuple[int, int, int, int]:
    pv_low = pv_grid.index(spec.pv_min_kw) if spec.pv_min_kw is not None else source_pv[0]
    pv_high = pv_grid.index(spec.pv_max_kw) if spec.pv_max_kw is not None else source_pv[-1]
    battery_low = battery_grid.index(spec.battery_min_kwh) if spec.battery_min_kwh is not None else source_battery[0]
    battery_high = battery_grid.index(spec.battery_max_kwh) if spec.battery_max_kwh is not None else source_battery[-1]
    if pv_low >= pv_high or battery_low >= battery_high:
        raise ValueError("Each search-domain upper bound must exceed its lower bound")
    for value, source, label in (
        (pv_low, source_pv, "PV lower"), (pv_high, source_pv, "PV upper"),
        (battery_low, source_battery, "BESS lower"), (battery_high, source_battery, "BESS upper"),
    ):
        if value not in source:
            raise ValueError(f"{label} domain bound must be an existing source-grid point")
    return pv_low, pv_high, battery_low, battery_high


def _cells(
    source_pv: tuple[int, ...], source_battery: tuple[int, ...], domain: tuple[int, int, int, int],
) -> list[CoarseCell]:
    pv_low, pv_high, battery_low, battery_high = domain
    selected_pv = [value for value in source_pv if pv_low <= value <= pv_high]
    selected_battery = [value for value in source_battery if battery_low <= value <= battery_high]
    return [
        CoarseCell(p0, p1, b0, b1)
        for p0, p1 in zip(selected_pv, selected_pv[1:])
        for b0, b1 in zip(selected_battery, selected_battery[1:])
    ]


def _init_worker(case: InputCase, config: dict[str, Any], reference_pv_kw: float) -> None:
    frame, _ = build_case_frame(case, config, reference_pv_kw)
    keep = [
        "hour_index", "analysis_year", "electric_load_kwh", "cooling_load_kwh_th", "heating_load_kwh_th",
        "pv_kwh", "tou_sgd_per_kwh", "pv_export_sgd_per_kwh",
    ]
    _WORKER.clear()
    _WORKER.update(
        frame=frame[keep].copy(),
        pv_per_kw=frame["pv_kwh"].to_numpy(dtype=float) / reference_pv_kw,
        config=config,
    )


def _evaluate_candidate_block(scenario: dict[str, Any], pv_kw: float, batteries_kwh: tuple[float, ...]) -> dict[str, Any]:
    """Evaluate one scenario/PV/BESS batch without retaining hourly dispatch rows."""
    started = time.perf_counter()
    frame = _WORKER["frame"].copy()
    availability = np.ones(len(frame), dtype=float)
    indexes = np.asarray(scenario["hour_indices"], dtype=int)
    availability[indexes] = np.asarray(scenario["availability"], dtype=float)
    if _availability_hash(availability) != scenario["outage_profile_sha256"]:
        raise ValueError("Frozen outage profile did not reconstruct exactly")
    frame["grid_availability"] = availability
    frame["outage_fraction"] = 1.0 - availability
    frame["pv_kwh"] = _WORKER["pv_per_kw"] * float(pv_kw)
    config = _WORKER["config"]
    records: list[dict[str, Any]] = []
    identifiers = {name: scenario[name] for name in (
        "case_key", "scenario_id", "outage_seed", "outage_profile_sha256",
    )}
    for battery_kwh in batteries_kwh:
        sizing = battery_sizing(float(battery_kwh), config)
        dispatch = solve_dispatch_greedy(
            frame, sizing, config, metrics_only=True, collect_totals=True, streaming_only=True,
        )
        reliability = dispatch.attrs["reliability_totals"]
        row: dict[str, Any] = {
            **identifiers,
            "pv_capacity_kw": float(pv_kw),
            **sizing,
            "analysis_years": int(config["analysis"]["years"]),
            "EUE_kwh_total": float(reliability["EUE_kwh_total"]),
            "EUE_kwh_per_year": float(reliability["EUE_kwh_total"]) / int(config["analysis"]["years"]),
            "outage_load_kwh_total": float(reliability["outage_load_kwh_total"]),
            "LOLP": float(reliability["LOLP"]),
            "no_outage_load": bool(reliability["no_outage_load"]),
            "outage_hours_total": float(reliability["outage_hours_total"]),
            "dispatch_backend": "greedy",
        }
        row.update(economic_metrics(dispatch.attrs["energy_totals"], sizing, float(pv_kw), config))
        row["EUE_threshold_difference_kwh_total"] = (
            row["unserved_energy_kwh_total"] - row["EUE_kwh_total"]
        )
        records.append(row)
    return {
        "records": records,
        "worker_pid": os.getpid(),
        "compute_seconds": time.perf_counter() - started,
    }


def _checkpoint_path(output: Path, key: tuple[int, int]) -> Path:
    return output / "checkpoints" / f"pv_{key[0]:05d}_bess_{key[1]:05d}.json"


def _read_checkpoint(
    path: Path, fingerprint: str, cache: CandidateCache,
) -> tuple[tuple[int, int], pd.DataFrame, str]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if envelope.get("fingerprint") != fingerprint or envelope.get("payload_sha256") != _digest(payload):
        raise ValueError(f"Candidate checkpoint checksum/fingerprint mismatch: {path}")
    rows = pd.DataFrame(payload["scenario_rows"])
    if rows.empty:
        raise ValueError(f"Empty candidate checkpoint: {path}")
    key = cache.key(float(rows.pv_capacity_kw.iloc[0]), float(rows.battery_energy_kwh.iloc[0]))
    if tuple(payload.get("key", [])) != key:
        raise ValueError(f"Candidate checkpoint coordinate mismatch: {path}")
    return key, rows, str(payload.get("candidate_origin", "new"))


def _write_checkpoint(
    output: Path, key: tuple[int, int], rows: pd.DataFrame, fingerprint: str, *, origin: str = "new",
) -> None:
    payload = {
        "key": list(key), "candidate_origin": origin,
        "scenario_rows": rows.to_dict(orient="records"),
    }
    _atomic_json(
        _checkpoint_path(output, key),
        {"fingerprint": fingerprint, "payload_sha256": _digest(payload), "payload": payload},
    )


class SizingRun:
    """Coordinates cache-aware, batched dispatch evaluation and adaptive search."""

    def __init__(
        self, source: dict[str, Any], output: Path, spec: SearchSpec, workers: int,
        fingerprint: str, trace: list[dict[str, Any]],
    ) -> None:
        self.source = source
        self.output = output
        self.spec = spec
        self.workers = workers
        self.fingerprint = fingerprint
        self.trace = trace
        self.pv_grid = CapacityGrid(spec.pv_step_kw, "PV")
        self.battery_grid = CapacityGrid(spec.battery_step_kwh, "BESS")
        scenarios = source["scenarios"].sort_values("scenario_id")
        self.scenario_ids = tuple(scenarios.scenario_id.astype(int).tolist())
        self.cache = CandidateCache(self.pv_grid, self.battery_grid, self.scenario_ids, spec.lolp_thresholds)
        source_rows = source["results"].copy()
        for _, group in source_rows.groupby(["pv_capacity_kw", "battery_energy_kwh"], sort=True):
            self.cache.add(group, "source")
        self.worker_pids: set[int] = set()
        self.dispatch_seconds = 0.0
        self.new_candidate_count = 0
        self.source_pv = _source_grid(source_rows.pv_capacity_kw, self.pv_grid, "PV")
        self.source_battery = _source_grid(source_rows.battery_energy_kwh, self.battery_grid, "BESS")
        self.domain = _resolve_domain(self.source_pv, self.source_battery, spec, self.pv_grid, self.battery_grid)
        self.all_cells = _cells(self.source_pv, self.source_battery, self.domain)
        self.lower_bound_context = self._make_lower_bound_context()
        self._executor: ProcessPoolExecutor | None = None
        self._serial_worker_ready = False
        self.mesh_cells: list[dict[str, Any]] = []
        self.mesh_rounds: list[dict[str, Any]] = []

    def open_workers(self) -> None:
        """Initialize each process once and retain it across adaptive rounds."""
        if self._executor is not None or self._serial_worker_ready:
            return
        case, config = self.source["case"], self.source["config"]
        reference_pv = self.pv_grid.value(self.domain[1])
        if self.workers == 1:
            _init_worker(case, config, reference_pv)
            self._serial_worker_ready = True
            return
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker, initargs=(case, config, reference_pv),
        )

    def close_workers(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        if self._serial_worker_ready:
            _WORKER.clear()
            self._serial_worker_ready = False

    def load_checkpoints(self) -> None:
        directory = self.output / "checkpoints"
        if not directory.exists():
            return
        for path in sorted(directory.glob("pv_*_bess_*.json")):
            key, rows, origin = _read_checkpoint(path, self.fingerprint, self.cache)
            if self.cache.has(key):
                # A checkpoint for a source point is harmless only if it exactly matches.
                self.cache.add(rows, self.cache.origin(key))
            else:
                saved_origin = origin if origin in {"new", "imported_checkpoint"} else "new"
                self.cache.add(rows, saved_origin)
                if saved_origin == "new":
                    self.new_candidate_count += 1

    def import_checkpoints(self, directory: Path) -> int:
        """Import compatible, completed candidate records from another run.

        The donor run may have a different final resolution, so its run
        fingerprint is deliberately not reused.  The source identity, sparse
        scenario rows, per-candidate payload checksum and final-grid alignment
        are all checked before accepting a record.
        """
        directory = directory.resolve()
        manifest_path = directory / "manifest.json"
        reference_path = directory / "source_reference.json"
        if not manifest_path.exists() or not reference_path.exists():
            raise ValueError("Checkpoint import requires manifest.json and source_reference.json")
        if json.loads(reference_path.read_text(encoding="utf-8")) != _source_identity(self.source):
            raise ValueError("Checkpoint import refused: frozen source identity differs")
        imported = 0
        skipped = 0
        for path in sorted((directory / "checkpoints").glob("pv_*_bess_*.json")):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope.get("payload")
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid checkpoint payload: {path}")
            digest_payload = payload
            if envelope.get("payload_sha256") != _digest(digest_payload):
                # Pre-mesh checkpoints did not include candidate_origin.  Verify
                # their original payload shape without weakening checksum checks.
                legacy = {name: value for name, value in payload.items() if name != "candidate_origin"}
                if envelope.get("payload_sha256") != _digest(legacy):
                    raise ValueError(f"Checkpoint checksum mismatch: {path}")
            rows = pd.DataFrame(payload.get("scenario_rows"))
            if rows.empty:
                raise ValueError(f"Empty imported checkpoint: {path}")
            try:
                key = self.cache.key(float(rows.pv_capacity_kw.iloc[0]), float(rows.battery_energy_kwh.iloc[0]))
            except ValueError:
                skipped += 1
                continue
            if not self.in_domain(key):
                skipped += 1
                continue
            if self.cache.has(key):
                self.cache.add(rows, self.cache.origin(key))
                skipped += 1
                continue
            self.cache.add(rows, "imported_checkpoint")
            _write_checkpoint(self.output, key, rows, self.fingerprint, origin="imported_checkpoint")
            imported += 1
        self.trace.append({
            "event": "checkpoint_import", "directory": str(directory),
            "imported_candidate_count": imported, "skipped_candidate_count": skipped,
        })
        return imported

    def _make_lower_bound_context(self) -> dict[str, Any]:
        """Constants for a conservative annual-cost lower bound.

        Capital costs use the longest permitted replacement life; grid imports,
        unmet energy and curtailed PV are lower bounded by zero.  Any possible
        export revenue (and negative-tariff import, if configured) is subtracted
        at its physical maximum.  This bound can only discard a point when it
        provably cannot beat an already feasible incumbent.
        """
        case, config = self.source["case"], self.source["config"]
        reference_pv = self.pv_grid.value(self.domain[1])
        frame, _ = build_case_frame(case, config, reference_pv)
        annual_years = float(config["analysis"]["years"])
        pv_per_kw = frame.pv_kwh.to_numpy(dtype=float) / reference_pv
        export_price = frame.get("pv_export_sgd_per_kwh", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
        tou = frame.tou_sgd_per_kwh.to_numpy(dtype=float)
        total_load = (
            frame.electric_load_kwh.to_numpy(dtype=float)
            + frame.cooling_load_kwh_th.to_numpy(dtype=float) / float(config["technical"]["cooling_cop"])
            + frame.heating_load_kwh_th.to_numpy(dtype=float) / float(config["technical"]["heating_cop"])
        )
        if not np.isfinite(np.r_[pv_per_kw, export_price, tou, total_load]).all():
            raise ValueError("Frozen source inputs contain nonfinite values")
        if float(config["active_scenario"]["voll_sgd_per_kwh"]) < 0:
            raise ValueError("A negative VoLL is incompatible with conservative cost pruning")
        event_bounds: list[list[tuple[float, float, float]]] = []
        # A deliberately optimistic outage-energy relaxation: each contiguous
        # outage can start with a full usable battery.  It is therefore a safe
        # lower bound on unmet energy, while remaining much cheaper than a
        # dispatch simulation.
        for payload in self.source["payloads"]:
            availability = np.ones(len(frame), dtype=float)
            positions = np.asarray(payload["hour_indices"], dtype=int)
            availability[positions] = np.asarray(payload["availability"], dtype=float)
            outage_positions = np.flatnonzero(availability < 1.0 - _EPS)
            events: list[tuple[float, float, float]] = []
            for block in np.split(outage_positions, np.where(np.diff(outage_positions) != 1)[0] + 1):
                if len(block) == 0:
                    continue
                fraction = 1.0 - availability[block]
                events.append((
                    float(np.sum(fraction * total_load[block])),
                    float(np.sum(pv_per_kw[block])),
                    float(np.sum(np.maximum(fraction, 1e-9))),
                ))
            event_bounds.append(events)
        return {
            "years": annual_years,
            "max_export_revenue_per_kw_year": float((pv_per_kw * np.maximum(export_price, 0.0)).sum() / annual_years),
            "negative_tou_load_lower_per_year": float((np.minimum(tou, 0.0) * total_load).sum() / annual_years),
            "negative_tou_charge_lower_per_kw_year": float(np.minimum(tou, 0.0).sum() / annual_years),
            "outage_event_bounds": event_bounds,
        }

    def cost_lower_bound(self, key: tuple[int, int]) -> float:
        pv_kw, battery_kwh = self.pv_grid.value(key[0]), self.battery_grid.value(key[1])
        config = self.source["config"]
        pv_fixed = float(pv_annual_cost({"pv_capacity_kw": pv_kw}, config)["pv_total_annual_cost_sgd"])
        sizing = battery_sizing(battery_kwh, config)
        battery_fixed = annualized_battery_cost(
            sizing["battery_energy_kwh"], sizing["battery_power_kw"], 0.0, config,
        )
        battery_fixed_cost = float(
            battery_fixed["annualized_battery_cost_sgd"] + battery_fixed["battery_om_cost_sgd"]
        )
        context = self.lower_bound_context
        negative_grid = context["negative_tou_load_lower_per_year"] + (
            sizing["battery_power_kw"] * context["negative_tou_charge_lower_per_kw_year"]
        )
        export = pv_kw * context["max_export_revenue_per_kw_year"]
        return float(pv_fixed + battery_fixed_cost + negative_grid - export)

    def physical_eue_lower_bound(self, pv_kw: float, battery_kwh: float) -> float:
        """Mean annual EUE lower bound from outage energy, PV and BESS limits."""
        battery = self.source["config"]["battery"]
        eta = float(battery["roundtrip_efficiency"]) ** 0.5
        usable = float(battery_kwh) * (
            float(battery["soc_max_fraction"]) - float(battery["soc_min_fraction"])
        ) * eta
        power = float(battery_kwh) * float(self.source["config"]["battery"]["max_c_rate"])
        totals = []
        for events in self.lower_bound_context["outage_event_bounds"]:
            unmet = 0.0
            for outage_load, pv_per_kw, outage_hours in events:
                residual = max(0.0, outage_load - float(pv_kw) * pv_per_kw)
                delivered = min(residual, usable, power * outage_hours)
                unmet += max(0.0, residual - delivered)
            totals.append(unmet)
        return float(np.mean(totals)) if totals else 0.0

    def cell_cost_lower_bound(self, cell: MeshCell) -> float:
        """Safe optimistic annual-cost bound for every capacity in a mesh cell."""
        corners = self.mesh_corners(cell)
        fixed = min(self.cost_lower_bound(key) for key in corners)
        pv_upper = self.pv_grid.value(cell.pv_high)
        battery_upper = self.battery_grid.value(cell.battery_high)
        voll = float(self.source["config"]["active_scenario"]["voll_sgd_per_kwh"])
        eue_cost = self.physical_eue_lower_bound(pv_upper, battery_upper) * voll / float(self.source["config"]["analysis"]["years"])
        return float(fixed + eue_cost)

    def _new_keys(self, keys: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        pv_low, pv_high, b_low, b_high = self.domain
        unique = sorted(set(keys))
        invalid = [key for key in unique if not (pv_low <= key[0] <= pv_high and b_low <= key[1] <= b_high)]
        if invalid:
            raise ValueError(f"Candidate outside requested search domain: {invalid[0]}")
        return [key for key in unique if not self.cache.has(key)]

    def in_domain(self, key: tuple[int, int]) -> bool:
        pv_low, pv_high, battery_low, battery_high = self.domain
        return pv_low <= key[0] <= pv_high and battery_low <= key[1] <= battery_high

    def evaluate(self, keys: Iterable[tuple[int, int]], *, reason: str) -> None:
        """Evaluate missing points once, batched as scenario x PV x BESS-block."""
        missing = self._new_keys(keys)
        if not missing:
            return
        by_pv: dict[int, list[int]] = {}
        for pv, battery in missing:
            by_pv.setdefault(pv, []).append(battery)
        tasks: list[tuple[dict[str, Any], float, tuple[float, ...]]] = []
        scenarios = self.source["payloads"]
        for pv_index, batteries in sorted(by_pv.items()):
            pv_kw = self.pv_grid.value(pv_index)
            for offset in range(0, len(batteries), self.spec.bess_batch_size):
                block = tuple(self.battery_grid.value(value) for value in batteries[offset: offset + self.spec.bess_batch_size])
                tasks.extend((scenario, pv_kw, block) for scenario in scenarios)
        completed: dict[tuple[int, int], list[dict[str, Any]]] = {key: [] for key in missing}

        def collect(payload: dict[str, Any]) -> None:
            self.worker_pids.add(int(payload["worker_pid"]))
            self.dispatch_seconds += float(payload["compute_seconds"])
            for row in payload["records"]:
                key = self.cache.key(float(row["pv_capacity_kw"]), float(row["battery_energy_kwh"]))
                completed[key].append(row)

        self.open_workers()
        if self.workers == 1:
            for task in tasks:
                collect(_evaluate_candidate_block(*task))
        else:
            if self._executor is None:  # defensive; open_workers establishes it.
                raise RuntimeError("Parallel worker pool was not initialized")
            futures = {self._executor.submit(_evaluate_candidate_block, *task): task for task in tasks}
            for future in as_completed(futures):
                collect(future.result())
        for key, records in completed.items():
            rows = pd.DataFrame(records)
            self.cache.add(rows, "new")
            _write_checkpoint(self.output, key, rows, self.fingerprint)
            self.new_candidate_count += 1
        self.trace.append({
            "event": "candidate_batch_evaluated", "reason": reason,
            "candidate_count": len(missing), "scenario_dispatches": len(missing) * len(self.scenario_ids),
            "task_count": len(tasks), "cached_candidate_count": len(self.cache._rows),
        })

    def source_summary(self, key: tuple[int, int]) -> dict[str, Any]:
        if self.cache.origin(key) != "source":
            raise ValueError("Coarse summary requested for a non-source candidate")
        return self.cache.summary(key)

    def coarse_incumbents(self) -> dict[tuple[float, float], float]:
        values: dict[tuple[float, float], float] = {}
        source_keys = [
            key for key in self.cache._rows if self.cache.origin(key) == "source" and self.in_domain(key)
        ]
        for threshold in self.spec.lolp_thresholds:
            for target in self.spec.ccp_targets:
                required = _target_count(target, len(self.scenario_ids))
                feasible = [
                    self.cache.summary(key)["mean_annual_cost_sgd_per_year"] for key in source_keys
                    if self.cache.count(key, threshold) >= required
                ]
                values[(threshold, target)] = min(feasible) if feasible else float("inf")
        return values

    def retained_cells(self, incumbents: dict[tuple[float, float], float]) -> set[CoarseCell]:
        retained: set[CoarseCell] = set()
        for cell in self.all_cells:
            corners = [
                (cell.pv_low, cell.battery_low), (cell.pv_low, cell.battery_high),
                (cell.pv_high, cell.battery_low), (cell.pv_high, cell.battery_high),
            ]
            reasons: list[str] = []
            for threshold in self.spec.lolp_thresholds:
                values = [self.cache.count(key, threshold) / len(self.scenario_ids) for key in corners]
                for target in self.spec.ccp_targets:
                    incumbent = incumbents[(threshold, target)]
                    if not np.isfinite(incumbent):
                        reasons.append("no_coarse_feasible_incumbent")
                    elif min(values) - _EPS <= target <= max(values) + _EPS:
                        reasons.append("coarse_reliability_boundary")
                    elif max(values) >= target - _EPS:
                        # A cost lower bound is meaningful only for a cell that
                        # already touches the coarse feasible envelope.  A cell
                        # whose four corners fail belongs in refinement only if
                        # it crosses a reliability boundary (the branch above).
                        lower = min(self.cost_lower_bound((pv, battery))
                                    for pv in self.pv_grid.inclusive(cell.pv_low, cell.pv_high)
                                    for battery in self.battery_grid.inclusive(cell.battery_low, cell.battery_high))
                        if lower < incumbent - 1e-9:
                            reasons.append("conservative_cost_lower_bound")
            if reasons:
                retained.add(cell)
                decision = "retained"
            else:
                decision = "excluded"
            self.trace.append({
                "event": "coarse_cell_screen", "decision": decision, "reason": ";".join(sorted(set(reasons))) or "none",
                "pv_low_kw": self.pv_grid.value(cell.pv_low), "pv_high_kw": self.pv_grid.value(cell.pv_high),
                "battery_low_kwh": self.battery_grid.value(cell.battery_low),
                "battery_high_kwh": self.battery_grid.value(cell.battery_high),
            })
        return retained

    def _regions(self, cells: set[CoarseCell]) -> dict[int, list[tuple[int, int]]]:
        values: dict[int, list[tuple[int, int]]] = {}
        for cell in cells:
            for pv in self.pv_grid.inclusive(cell.pv_low, cell.pv_high):
                values.setdefault(pv, []).append((cell.battery_low, cell.battery_high))
        for pv, intervals in values.items():
            values[pv] = _merge_intervals(intervals)
        return values

    def _source_bess_monotonic(self) -> bool:
        """Require monotonicity of every available coarse PV row before bisection."""
        for pv in self.source_pv:
            for threshold in self.spec.lolp_thresholds:
                counts = [self.cache.count((pv, battery), threshold) for battery in self.source_battery]
                if any(right < left for left, right in zip(counts, counts[1:])):
                    return False
        return True

    def _observed_nonmonotone_pvs(self, regions: dict[int, list[tuple[int, int]]]) -> set[int]:
        nonmonotone: set[int] = set()
        for pv, intervals in regions.items():
            batteries = sorted({battery for low, high in intervals for battery in self.battery_grid.inclusive(low, high)
                                if self.cache.has((pv, battery))})
            for threshold in self.spec.lolp_thresholds:
                counts = [self.cache.count((pv, battery), threshold) for battery in batteries]
                if any(right < left for left, right in zip(counts, counts[1:])):
                    nonmonotone.add(pv)
                    break
        return nonmonotone

    def reliability_frontiers(
        self, cells: set[CoarseCell], *, reason: str,
    ) -> dict[tuple[int, float, float], int]:
        """Find minimum feasible BESS values with discrete bisection or safe fallback."""
        regions = self._regions(cells)
        source_monotone = self._source_bess_monotonic()
        fallback_pvs: set[int] = set()
        if not source_monotone:
            fallback_pvs.update(regions)
            self.trace.append({
                "event": "bisection_fallback", "reason": "source_CCP_not_monotone_in_BESS",
                "pv_count": len(fallback_pvs),
            })

        requests: dict[tuple[int, int, int, float, float], tuple[int, int]] = {}
        initial_keys: set[tuple[int, int]] = set()
        for pv, intervals in regions.items():
            if pv in fallback_pvs:
                continue
            for low, high in intervals:
                for threshold in self.spec.lolp_thresholds:
                    for target in self.spec.ccp_targets:
                        request = (pv, low, high, threshold, target)
                        requests[request] = (low, high)
                        initial_keys.update({(pv, low), (pv, high)})
        self.evaluate(initial_keys, reason=f"{reason}: bisection endpoints")

        completed: dict[tuple[int, int, int, float, float], int | None] = {}
        while requests:
            next_requests: dict[tuple[int, int, int, float, float], tuple[int, int]] = {}
            to_evaluate: set[tuple[int, int]] = set()
            observed = self._observed_nonmonotone_pvs(regions)
            newly_fallback = observed - fallback_pvs
            if newly_fallback:
                fallback_pvs.update(newly_fallback)
                self.trace.append({
                    "event": "bisection_fallback", "reason": "observed_CCP_not_monotone_in_BESS",
                    "pv_count": len(newly_fallback),
                })
            for request, (low, high) in requests.items():
                pv, _, _, threshold, target = request
                if pv in fallback_pvs:
                    continue
                required = _target_count(target, len(self.scenario_ids))
                low_count, high_count = self.cache.count((pv, low), threshold), self.cache.count((pv, high), threshold)
                if high_count < required:
                    completed[request] = None
                elif low_count >= required:
                    completed[request] = low
                elif high - low <= 1:
                    completed[request] = high
                else:
                    middle = (low + high) // 2
                    to_evaluate.add((pv, middle))
                    # Keep the original bounds until the entire parallel wave
                    # has completed; then every request sees the same cache.
                    next_requests[request] = (low, high)
            if to_evaluate:
                self.evaluate(to_evaluate, reason=f"{reason}: bisection midpoints")
                # Re-evaluate request directions only after the whole parallel wave is in the cache.
                corrected: dict[tuple[int, int, int, float, float], tuple[int, int]] = {}
                for request, (low, high) in next_requests.items():
                    pv, _, _, threshold, target = request
                    if pv in fallback_pvs:
                        continue
                    original_low, original_high = requests[request]
                    pivot = (original_low + original_high) // 2
                    required = _target_count(target, len(self.scenario_ids))
                    corrected[request] = (original_low, pivot) if self.cache.count((pv, pivot), threshold) >= required else (pivot, original_high)
                next_requests = corrected
            requests = next_requests

        # A full local BESS scan is the safe replacement when the monotonicity
        # prerequisite did not hold (or was contradicted by evaluated points).
        if fallback_pvs:
            full_keys = {
                (pv, battery) for pv in fallback_pvs for low, high in regions[pv]
                for battery in self.battery_grid.inclusive(low, high)
            }
            self.evaluate(full_keys, reason=f"{reason}: full BESS fallback")
            for pv in fallback_pvs:
                for low, high in regions[pv]:
                    for threshold in self.spec.lolp_thresholds:
                        for target in self.spec.ccp_targets:
                            request = (pv, low, high, threshold, target)
                            required = _target_count(target, len(self.scenario_ids))
                            completed[request] = next((
                                battery for battery in self.battery_grid.inclusive(low, high)
                                if self.cache.count((pv, battery), threshold) >= required
                            ), None)

        frontiers: dict[tuple[int, float, float], int] = {}
        for (pv, _low, _high, threshold, target), battery in completed.items():
            if battery is None:
                continue
            key = (pv, threshold, target)
            frontiers[key] = min(frontiers[key], battery) if key in frontiers else battery
        return frontiers

    def evaluate_cost_tails(
        self, cells: set[CoarseCell], frontiers: dict[tuple[int, float, float], int],
        incumbents: dict[tuple[float, float], float], *, reason: str,
    ) -> None:
        """Check reliable BESS tails whenever their conservative lower bound can win."""
        regions = self._regions(cells)
        candidates: set[tuple[int, int]] = set()
        for (pv, threshold, target), minimum_battery in frontiers.items():
            incumbent = incumbents[(threshold, target)]
            for low, high in regions[pv]:
                for battery in self.battery_grid.inclusive(max(low, minimum_battery), high):
                    key = (pv, battery)
                    if not np.isfinite(incumbent) or self.cost_lower_bound(key) < incumbent - 1e-9:
                        candidates.add(key)
        self.evaluate(candidates, reason=f"{reason}: feasible cost tails")

    def best_designs(self) -> dict[tuple[float, float], tuple[tuple[int, int] | None, dict[str, Any] | None]]:
        all_keys = sorted(key for key in self.cache._rows if self.in_domain(key))
        choices: dict[tuple[float, float], tuple[tuple[int, int] | None, dict[str, Any] | None]] = {}
        for threshold in self.spec.lolp_thresholds:
            for target in self.spec.ccp_targets:
                required = _target_count(target, len(self.scenario_ids))
                feasible = [key for key in all_keys if self.cache.count(key, threshold) >= required]
                if not feasible:
                    choices[(threshold, target)] = (None, None)
                    continue
                best = min(
                    feasible,
                    key=lambda key: (
                        self.cache.summary(key)["mean_annual_cost_sgd_per_year"], key[0], key[1],
                    ),
                )
                choices[(threshold, target)] = (best, self.cache.summary(best))
        return choices

    def boundary_expansions(
        self, retained: set[CoarseCell], best: dict[tuple[float, float], tuple[tuple[int, int] | None, dict[str, Any] | None]],
    ) -> set[CoarseCell]:
        """Expand only immediately adjacent unretained cells around a winning boundary."""
        additions: set[CoarseCell] = set()
        for key, _summary in best.values():
            if key is None:
                continue
            pv, battery = key
            touching = [cell for cell in retained if cell.pv_low <= pv <= cell.pv_high and cell.battery_low <= battery <= cell.battery_high]
            for cell in touching:
                for candidate in self.all_cells:
                    if candidate in retained:
                        continue
                    p_adjacent = (
                        (candidate.pv_high == cell.pv_low and candidate.pv_high == pv)
                        or (candidate.pv_low == cell.pv_high and candidate.pv_low == pv)
                    ) and not (candidate.battery_high < battery or candidate.battery_low > battery)
                    b_adjacent = (
                        (candidate.battery_high == cell.battery_low and candidate.battery_high == battery)
                        or (candidate.battery_low == cell.battery_high and candidate.battery_low == battery)
                    ) and not (candidate.pv_high < pv or candidate.pv_low > pv)
                    if p_adjacent or b_adjacent:
                        additions.add(candidate)
        for cell in sorted(additions, key=CoarseCell.key):
            self.trace.append({
                "event": "coarse_cell_expand", "decision": "retained",
                "reason": "winner_on_retained_region_boundary",
                "pv_low_kw": self.pv_grid.value(cell.pv_low), "pv_high_kw": self.pv_grid.value(cell.pv_high),
                "battery_low_kwh": self.battery_grid.value(cell.battery_low),
                "battery_high_kwh": self.battery_grid.value(cell.battery_high),
            })
        return additions

    def mesh_corners(self, cell: MeshCell) -> tuple[tuple[int, int], ...]:
        return (
            (cell.pv_low, cell.battery_low), (cell.pv_low, cell.battery_high),
            (cell.pv_high, cell.battery_low), (cell.pv_high, cell.battery_high),
        )

    def mesh_children(
        self, parents: Iterable[MeshCell], *, pv_step_kw: float, battery_step_kwh: float, level: str,
    ) -> set[MeshCell]:
        pv_stride = self.pv_grid.index(pv_step_kw)
        battery_stride = self.battery_grid.index(battery_step_kwh)
        children: set[MeshCell] = set()
        for parent in parents:
            if (parent.pv_high - parent.pv_low) % pv_stride or (parent.battery_high - parent.battery_low) % battery_stride:
                raise ValueError(f"Mesh level {level} does not divide parent cell {parent.identifier()}")
            for pv_low in range(parent.pv_low, parent.pv_high, pv_stride):
                for battery_low in range(parent.battery_low, parent.battery_high, battery_stride):
                    children.add(MeshCell(
                        pv_low, pv_low + pv_stride, battery_low, battery_low + battery_stride,
                        level, parent.identifier(),
                    ))
        return children

    def mesh_vertices(self, cells: Iterable[MeshCell]) -> set[tuple[int, int]]:
        return {key for cell in cells for key in self.mesh_corners(cell)}

    def _pareto_keys(self) -> set[tuple[int, int]]:
        """Undominated candidates across cost and all strict-CCP measures."""
        keys = sorted(key for key in self.cache._rows if self.in_domain(key))
        values = [
            (key, self.cache.summary(key)["mean_annual_cost_sgd_per_year"],
             tuple(self.cache.count(key, threshold) for threshold in self.spec.lolp_thresholds))
            for key in keys
        ]
        frontier: set[tuple[int, int]] = set()
        for key, cost, counts in values:
            dominated = any(
                other_key != key and other_cost <= cost + _EPS
                and all(left >= right for left, right in zip(other_counts, counts))
                and (other_cost < cost - _EPS or any(left > right for left, right in zip(other_counts, counts)))
                for other_key, other_cost, other_counts in values
            )
            if not dominated:
                frontier.add(key)
        return frontier

    def _mesh_monotone(self, cell: MeshCell) -> bool:
        ll, lh, hl, hh = self.mesh_corners(cell)
        for threshold in self.spec.lolp_thresholds:
            ll_count, lh_count = self.cache.count(ll, threshold), self.cache.count(lh, threshold)
            hl_count, hh_count = self.cache.count(hl, threshold), self.cache.count(hh, threshold)
            if lh_count < ll_count or hl_count < ll_count or hh_count < lh_count or hh_count < hl_count:
                return False
        return True

    def _mesh_row(self, cell: MeshCell, status: str, reasons: list[str], *, round_name: str, priority: int = 0) -> None:
        self.mesh_cells.append({
            "round": round_name, "cell_id": cell.identifier(), "parent_cell_id": cell.parent,
            "level": cell.level, "pv_low_kw": self.pv_grid.value(cell.pv_low),
            "pv_high_kw": self.pv_grid.value(cell.pv_high),
            "battery_low_kwh": self.battery_grid.value(cell.battery_low),
            "battery_high_kwh": self.battery_grid.value(cell.battery_high),
            "status": status, "reason": ";".join(sorted(set(reasons))) or "none", "priority": priority,
        })

    def classify_mesh_cells(
        self, cells: Iterable[MeshCell], best: dict[tuple[float, float], tuple[tuple[int, int] | None, dict[str, Any] | None]],
        *, round_name: str,
    ) -> set[MeshCell]:
        """Return cells supported by current reliability and cost evidence."""
        pareto = self._pareto_keys()
        active: set[MeshCell] = set()
        for cell in sorted(set(cells), key=lambda item: item.key()):
            corners = self.mesh_corners(cell)
            if not all(self.cache.has(key) for key in corners):
                raise ValueError(f"Cannot classify unevaluated mesh cell {cell.identifier()}")
            monotone = self._mesh_monotone(cell)
            corner_cost = min(self.cache.summary(key)["mean_annual_cost_sgd_per_year"] for key in corners)
            lower_cost = self.cell_cost_lower_bound(cell)
            reasons: list[str] = []
            priority = 0
            for threshold in self.spec.lolp_thresholds:
                values = [self.cache.count(key, threshold) for key in corners]
                for target in self.spec.ccp_targets:
                    required = _target_count(target, len(self.scenario_ids))
                    incumbent_key, incumbent_summary = best[(threshold, target)]
                    incumbent_cost = (
                        float(incumbent_summary["mean_annual_cost_sgd_per_year"])
                        if incumbent_summary is not None else float("inf")
                    )
                    possible = max(values) >= required
                    if not possible:
                        continue
                    touches_incumbent = incumbent_key is not None and (
                        cell.pv_low <= incumbent_key[0] <= cell.pv_high
                        and cell.battery_low <= incumbent_key[1] <= cell.battery_high
                    )
                    crosses = min(values) < required <= max(values)
                    cost_competitive = (
                        not np.isfinite(incumbent_cost)
                        or (lower_cost < incumbent_cost - _EPS and corner_cost <= incumbent_cost * (1.0 + self.spec.cost_screen_margin))
                    )
                    pareto_touch = any(key in pareto for key in corners)
                    if not monotone:
                        reasons.append("nonmonotone_fallback")
                        priority = max(priority, 4)
                    elif touches_incumbent:
                        reasons.append("contains_incumbent")
                        priority = max(priority, 3)
                    elif crosses and cost_competitive:
                        reasons.append("reliability_boundary_cost_competitive")
                        priority = max(priority, 2)
                    elif pareto_touch and cost_competitive:
                        reasons.append("pareto_cost_competitive")
                        priority = max(priority, 1)
            if reasons:
                active.add(cell)
                self._mesh_row(cell, "active", reasons, round_name=round_name, priority=priority)
            else:
                self._mesh_row(cell, "inactive", ["no_target_cost_reliability_potential"], round_name=round_name)
        return active

    def mesh_boundary_expansions(
        self, active: set[MeshCell], best: dict[tuple[float, float], tuple[tuple[int, int] | None, dict[str, Any] | None]],
        *, pv_step_kw: float, battery_step_kwh: float, seen: set[tuple[int, int, int, int]],
    ) -> set[MeshCell]:
        """Expand only final-level neighbours touching a selected best design."""
        pv_stride = self.pv_grid.index(pv_step_kw)
        battery_stride = self.battery_grid.index(battery_step_kwh)
        pv_min, pv_max, battery_min, battery_max = self.domain
        active_rectangles = {(c.pv_low, c.pv_high, c.battery_low, c.battery_high) for c in active}
        additions: set[MeshCell] = set()
        for key, _summary in best.values():
            if key is None:
                continue
            for cell in active:
                if not (cell.pv_low <= key[0] <= cell.pv_high and cell.battery_low <= key[1] <= cell.battery_high):
                    continue
                candidates: list[tuple[int, int, int, int]] = []
                if key[0] == cell.pv_low and cell.pv_low - pv_stride >= pv_min:
                    candidates.append((cell.pv_low - pv_stride, cell.pv_low, cell.battery_low, cell.battery_high))
                if key[0] == cell.pv_high and cell.pv_high + pv_stride <= pv_max:
                    candidates.append((cell.pv_high, cell.pv_high + pv_stride, cell.battery_low, cell.battery_high))
                if key[1] == cell.battery_low and cell.battery_low - battery_stride >= battery_min:
                    candidates.append((cell.pv_low, cell.pv_high, cell.battery_low - battery_stride, cell.battery_low))
                if key[1] == cell.battery_high and cell.battery_high + battery_stride <= battery_max:
                    candidates.append((cell.pv_low, cell.pv_high, cell.battery_high, cell.battery_high + battery_stride))
                for rectangle in candidates:
                    if rectangle not in active_rectangles and rectangle not in seen:
                        seen.add(rectangle)
                        additions.add(MeshCell(*rectangle, "final_expansion", cell.identifier()))
        return additions

    def adaptive_mesh_search(self) -> tuple[dict[tuple[float, float], tuple[tuple[int, int] | None, dict[str, Any] | None]], set[MeshCell]]:
        """Run nested active-cell refinement to the requested final grid."""
        incumbents = self.coarse_incumbents()
        coarse = self.retained_cells(incumbents)
        parents = {
            MeshCell(cell.pv_low, cell.pv_high, cell.battery_low, cell.battery_high, "coarse")
            for cell in coarse
        }
        for cell in sorted(parents, key=lambda item: item.key()):
            self._mesh_row(cell, "active", ["initial_source_screen"], round_name="source")
        levels: list[tuple[str, float, float]] = []
        if self.spec.intermediate_pv_step_kw is not None:
            levels.append(("intermediate_0p5", self.spec.intermediate_pv_step_kw, self.spec.intermediate_battery_step_kwh))
        levels.append(("final_0p25", self.spec.pv_step_kw, self.spec.battery_step_kwh))
        active = parents
        final_cells: set[MeshCell] = set()
        for position, (level_name, pv_step, battery_step) in enumerate(levels):
            input_active_count = len(active)
            children = self.mesh_children(active, pv_step_kw=pv_step, battery_step_kwh=battery_step, level=level_name)
            before = len(self.cache._rows)
            self.evaluate(self.mesh_vertices(children), reason=f"{level_name}: active mesh vertices")
            best = self.best_designs()
            active = self.classify_mesh_cells(children, best, round_name=level_name)
            self.mesh_rounds.append({
                "round": level_name, "input_active_cells": input_active_count,
                "child_cells": len(children), "active_cells": len(active),
                "new_candidate_capacities": len(self.cache._rows) - before,
                "cached_candidate_capacities": len(self.cache._rows),
            })
            if position == len(levels) - 1:
                final_cells = active
            else:
                parents = active
        best = self.best_designs()
        seen: set[tuple[int, int, int, int]] = set()
        while final_cells:
            additions = self.mesh_boundary_expansions(
                final_cells, best, pv_step_kw=self.spec.pv_step_kw,
                battery_step_kwh=self.spec.battery_step_kwh, seen=seen,
            )
            if not additions:
                break
            before = len(self.cache._rows)
            self.evaluate(self.mesh_vertices(additions), reason="final_0p25: boundary expansion")
            best = self.best_designs()
            accepted = self.classify_mesh_cells(additions, best, round_name="final_expansion")
            self.mesh_rounds.append({
                "round": "final_expansion", "input_active_cells": len(final_cells),
                "child_cells": len(additions), "active_cells": len(accepted),
                "new_candidate_capacities": len(self.cache._rows) - before,
                "cached_candidate_capacities": len(self.cache._rows),
            })
            if not accepted:
                break
            final_cells.update(accepted)
        return self.best_designs(), final_cells


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(set(intervals))
    merged: list[tuple[int, int]] = []
    for low, high in ordered:
        if not merged or low > merged[-1][1] + 1:
            merged.append((low, high))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
    return merged


def _source_identity(source: dict[str, Any]) -> dict[str, Any]:
    return source["identity"]


def _settings(source: dict[str, Any], spec: SearchSpec) -> dict[str, Any]:
    return {
        "module_schema_version": 2,
        "source": _source_identity(source),
        "targets": {
            "lolp_thresholds": list(spec.lolp_thresholds), "ccp_targets": list(spec.ccp_targets),
            "comparison": "LOLP < threshold", "cost_objective": "mean annual_cost_sgd_per_year",
        },
        "search": {
            "pv_step_kw": spec.pv_step_kw, "battery_step_kwh": spec.battery_step_kwh,
            "pv_min_kw": spec.pv_min_kw, "pv_max_kw": spec.pv_max_kw,
            "battery_min_kwh": spec.battery_min_kwh, "battery_max_kwh": spec.battery_max_kwh,
            "bess_batch_size": spec.bess_batch_size,
            "intermediate_pv_step_kw": spec.intermediate_pv_step_kw,
            "intermediate_battery_step_kwh": spec.intermediate_battery_step_kwh,
            "cost_screen_margin": spec.cost_screen_margin,
            "algorithm": "adaptive_active_mesh_refinement",
            "optimality_status": "adaptive_high_confidence",
        },
        "implementation_hashes": {
            name: _file_hash(ROOT / name) for name in (
                "src/ccp_lolp_sizing.py", "src/dispatch.py", "src/sweep_economics.py", "src/costing.py",
            )
        },
    }


def _load_or_create_run(
    output: Path, source: dict[str, Any], spec: SearchSpec, resume: bool,
) -> tuple[dict[str, Any], str, float]:
    settings = _settings(source, spec)
    fingerprint = _digest(settings)
    manifest_path = output / "manifest.json"
    elapsed_before = 0.0
    if manifest_path.exists():
        if not resume:
            raise ValueError(f"Output already exists; use --resume: {output}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Resume refused: source, targets, code or search settings changed")
        status_path = output / "run_status.json"
        if status_path.exists():
            elapsed_before = float(json.loads(status_path.read_text()).get("elapsed_seconds", 0.0))
        return manifest, fingerprint, elapsed_before
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite a nonempty directory without a manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"fingerprint": fingerprint, "created_at": _utc_now(), **settings}
    _atomic_json(manifest_path, manifest)
    _atomic_json(output / "source_reference.json", _source_identity(source))
    return manifest, fingerprint, elapsed_before


def _curve_frame(
    run: SizingRun,
    best: dict[tuple[float, float], tuple[tuple[int, int] | None, dict[str, Any] | None]],
) -> pd.DataFrame:
    _pv_low, pv_high, _battery_low, battery_high = run.domain
    rows: list[dict[str, Any]] = []
    for threshold in run.spec.lolp_thresholds:
        token = _threshold_token(threshold)
        for target in run.spec.ccp_targets:
            key, summary = best[(threshold, target)]
            required = _target_count(target, len(run.scenario_ids))
            if key is None or summary is None:
                rows.append({
                    "lolp_constraint": threshold, "ccp_target": target, "comparison": "<",
                    "required_passing_scenarios": required, "actual_ccp": np.nan,
                    "passing_scenarios": 0, "feasible": False, "optimal_pv_kw": np.nan,
                    "optimal_battery_kwh": np.nan, "optimal_battery_power_kw": np.nan,
                    "mean_LOLP": np.nan, "mean_EUE_kwh_total": np.nan,
                    "mean_EUE_kwh_per_year": np.nan, "mean_annual_cost_sgd_per_year": np.nan,
                    "domain_limited": True, "boundary_status": "no_feasible_design_in_source_domain",
                })
                continue
            passing = run.cache.count(key, threshold)
            domain_limited = key[0] == pv_high or key[1] == battery_high
            rows.append({
                "lolp_constraint": threshold, "ccp_target": target, "comparison": "<",
                "required_passing_scenarios": required,
                "actual_ccp": summary[f"CCP_LOLP_lt_{token}"], "passing_scenarios": passing,
                "feasible": True, "optimal_pv_kw": summary["pv_capacity_kw"],
                "optimal_battery_kwh": summary["battery_energy_kwh"],
                "optimal_battery_power_kw": summary["battery_power_kw"],
                "mean_LOLP": summary["mean_LOLP"], "mean_EUE_kwh_total": summary["mean_EUE_kwh_total"],
                "mean_EUE_kwh_per_year": summary["mean_EUE_kwh_per_year"],
                "mean_annual_cost_sgd_per_year": summary["mean_annual_cost_sgd_per_year"],
                "domain_limited": domain_limited,
                "boundary_status": "search_domain_upper_boundary" if domain_limited else "interior_or_screened_boundary",
            })
    return pd.DataFrame(rows).sort_values(["lolp_constraint", "ccp_target"]).reset_index(drop=True)


def _validate_outputs(run: SizingRun, curves: pd.DataFrame) -> dict[str, Any]:
    expected_rows = len(run.spec.lolp_thresholds) * len(run.spec.ccp_targets)
    if len(curves) != expected_rows:
        raise ValueError("Curve output does not contain every requested target")
    if curves.duplicated(["lolp_constraint", "ccp_target"]).any():
        raise ValueError("Curve output contains duplicate targets")
    for key in run.cache._rows:
        rows = run.cache.rows(key)
        validate_economic_results(rows, run.source["config"])
        for threshold in run.spec.lolp_thresholds:
            if run.cache.count(key, threshold) != strict_ccp_count(rows.LOLP, threshold):
                raise ValueError("Strict CCP count did not reconcile with scenario LOLP values")
    for row in curves.itertuples(index=False):
        if bool(row.feasible):
            key = run.cache.key(float(row.optimal_pv_kw), float(row.optimal_battery_kwh))
            summary = run.cache.summary(key)
            if summary["mean_annual_cost_sgd_per_year"] != row.mean_annual_cost_sgd_per_year:
                raise ValueError("Curve cost does not equal the selected scenario mean")
            if row.actual_ccp + _EPS < row.ccp_target:
                raise ValueError("Reported feasible design does not meet its CCP target")
    return {
        "passed": True,
        "curve_rows": len(curves),
        "candidate_capacities": len(run.cache._rows),
        "new_candidate_capacities": sum(run.cache.origin(key) == "new" for key in run.cache._rows),
        "shared_scenarios_verified": True,
        "strict_threshold_verified": True,
        "economic_reconciliation_verified": True,
        "grid_precision_verified": True,
        "lightweight_dispatch": {
            "metrics_only": True, "collect_totals": True, "streaming_only": True,
            "hourly_or_yearly_dispatch_output_written": False,
        },
    }


def _readme(run: SizingRun) -> str:
    mesh_note = (
        "The active-mesh mode refines source cells through the configured intermediate and final "
        "nested grids.  A cell is reconsidered after every round; its source-screen result is not fixed. "
        "The result is labelled `adaptive_high_confidence`, not a mathematical all-domain certificate."
        if run.spec.intermediate_pv_step_kw is not None else
        "The legacy source-screen/bisection mode was used."
    )
    return f"""# CCP--LOLP adaptive fine capacity sizing

This run reuses the completed source sweep recorded in `source_reference.json`.
It does not regenerate the {len(run.scenario_ids)} outage scenarios or resample inputs.
Every new candidate uses the frozen {run.source['spec'].analysis_years}-year inputs and the
existing greedy dispatcher in streaming-only mode: energy, reliability and economic totals are
retained, while hourly dispatch and yearly result tables are neither created nor written.

`ccp_target_curve_source.csv` has one row per requested `(LOLP constraint, CCP target)` pair.
CCP uses the strict definition `P(LOLP < threshold)`.  The objective is the arithmetic mean of
`annual_cost_sgd_per_year` across the common scenarios, including annualized investment, O&M,
net grid cost and raw-unserved-energy VoLL cost.

{mesh_note}
`domain_limited=true` means the selected solution touches the requested PV or BESS upper bound;
no claim is made beyond that domain.

Files:

- `ccp_target_curve_source.csv`: plotting-ready 3 × 24 target data by default
  (CCP 50%, 52%, ..., 94%, 95%).
- `candidate_summary.csv`: all reused and newly evaluated capacity summaries.
- `fine_scenario_results.csv`: new fine candidates only; source candidates remain referenced.
- `refinement_trace.csv`: screening, fallback, evaluation and expansion audit trail.
- `mesh_cells.csv`, `mesh_rounds.csv`: active-cell decisions and per-level compute audit.
- `manifest.json`, `validation.json`, `run_status.json`: reproducibility and completion records.
"""


def run_sizing(
    source_results_dir: Path | str = DEFAULT_SOURCE,
    output_dir: Path | str | None = None,
    *, workers: int = min(8, os.cpu_count() or 1), resume: bool = False,
    spec: SearchSpec = SearchSpec(),
) -> dict[str, Any]:
    """Create 0.1-grid CCP--LOLP sizing data from a completed enriched sweep."""
    spec.validate()
    if workers <= 0 or int(workers) != workers:
        raise ValueError("workers must be a positive integer")
    source_path = Path(source_results_dir).resolve()
    source = _validate_source(source_path)
    output = (Path(output_dir).resolve() if output_dir else source_path.with_name(source_path.name + "_CCP_LOLP_Sizing"))
    if output == source_path:
        raise ValueError("Sizing output must not overwrite its source sweep")
    _manifest, fingerprint, elapsed_before = _load_or_create_run(output, source, spec, resume)
    trace_path = output / "refinement_trace.csv"
    if resume and trace_path.exists():
        existing_trace = pd.read_csv(trace_path)
        trace = existing_trace.where(pd.notna(existing_trace), None).to_dict(orient="records")
    else:
        trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    run = SizingRun(source, output, spec, int(workers), fingerprint, trace)
    if resume:
        run.load_checkpoints()
    if spec.import_checkpoints_from is not None:
        run.import_checkpoints(Path(spec.import_checkpoints_from))

    def status(state: str) -> None:
        _atomic_json(output / "run_status.json", {
            "state": state, "updated_at": _utc_now(), "fingerprint": fingerprint,
            "workers_requested": workers, "worker_pids": sorted(run.worker_pids),
            "candidate_capacities_cached": len(run.cache._rows),
            "new_candidate_capacities": sum(run.cache.origin(key) == "new" for key in run.cache._rows),
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
        })

    status("running")
    try:
        if spec.intermediate_pv_step_kw is not None:
            best, final_cells = run.adaptive_mesh_search()
            retained_count = sum(item["level"] == "coarse" and item["status"] == "active" for item in run.mesh_cells)
        else:
            incumbents = run.coarse_incumbents()
            retained = run.retained_cells(incumbents)
            if not retained:
                raise RuntimeError("Coarse screen retained no cells; this should be impossible with a finite source domain")
            while True:
                frontiers = run.reliability_frontiers(retained, reason="adaptive refinement")
                run.evaluate_cost_tails(retained, frontiers, incumbents, reason="adaptive refinement")
                best = run.best_designs()
                additions = run.boundary_expansions(retained, best)
                if not additions:
                    break
                retained.update(additions)
            final_cells = set()
            retained_count = len(retained)
        _check_source_still_frozen(source)
        curves = _curve_frame(run, best)
        curves["optimality_status"] = (
            "adaptive_high_confidence" if spec.intermediate_pv_step_kw is not None else "legacy_search"
        )
        validation = _validate_outputs(run, curves)
        candidates = run.cache.all_summaries()
        fine = run.cache.new_rows()
        # Checkpoints are canonical JSON (sorted keys), while live worker rows
        # retain construction order.  Reapply the source schema so a resumed run
        # writes byte-identical tabular output to a fresh run.
        fine = fine.reindex(columns=[*source["results"].columns.tolist(), "candidate_origin"])
        _atomic_csv(output / "ccp_target_curve_source.csv", curves)
        _atomic_csv(output / "candidate_summary.csv", candidates)
        _atomic_csv(output / "fine_scenario_results.csv", fine)
        _atomic_csv(output / "refinement_trace.csv", pd.DataFrame(trace))
        _atomic_csv(output / "mesh_cells.csv", pd.DataFrame(run.mesh_cells))
        _atomic_csv(output / "mesh_rounds.csv", pd.DataFrame(run.mesh_rounds))
        (output / "README.md").write_text(_readme(run), encoding="utf-8")
        validation.update({
            "source_fingerprint": source["identity"]["source_fingerprint"],
            "search_domain": {
                "pv_kw": [run.pv_grid.value(run.domain[0]), run.pv_grid.value(run.domain[1])],
                "battery_kwh": [run.battery_grid.value(run.domain[2]), run.battery_grid.value(run.domain[3])],
            },
            "fine_precision": {"pv_kw": spec.pv_step_kw, "battery_kwh": spec.battery_step_kwh},
            "workers_requested": workers, "worker_pids": sorted(run.worker_pids),
            "summed_worker_compute_seconds": run.dispatch_seconds,
            "retained_coarse_cells": retained_count,
            "active_final_cells": len(final_cells),
            "optimality_status": "adaptive_high_confidence" if spec.intermediate_pv_step_kw is not None else "legacy_search",
            "file_sha256": {
                name: _file_hash(output / name) for name in (
                    "ccp_target_curve_source.csv", "candidate_summary.csv", "fine_scenario_results.csv",
                    "refinement_trace.csv", "mesh_cells.csv", "mesh_rounds.csv", "README.md",
                )
            },
        })
        _atomic_json(output / "validation.json", validation)
        status("complete")
        return validation
    except BaseException:
        status("failed")
        raise
    finally:
        run.close_workers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lolp-thresholds", nargs="+", type=float, default=list(DEFAULT_LOLP_THRESHOLDS))
    parser.add_argument("--ccp-targets", nargs="+", type=float, default=list(DEFAULT_CCP_TARGETS))
    parser.add_argument("--pv-step-kw", type=float, default=0.25)
    parser.add_argument("--battery-step-kwh", type=float, default=0.25)
    parser.add_argument("--intermediate-pv-step-kw", type=float, default=0.5)
    parser.add_argument("--intermediate-battery-step-kwh", type=float, default=0.5)
    parser.add_argument("--cost-screen-margin", type=float, default=0.05)
    parser.add_argument("--import-checkpoints-from", type=Path)
    parser.add_argument("--pv-min-kw", type=float)
    parser.add_argument("--pv-max-kw", type=float)
    parser.add_argument("--battery-min-kwh", type=float)
    parser.add_argument("--battery-max-kwh", type=float)
    parser.add_argument("--bess-batch-size", type=int, default=12)
    args = parser.parse_args()
    spec = SearchSpec(
        lolp_thresholds=tuple(args.lolp_thresholds), ccp_targets=tuple(args.ccp_targets),
        pv_step_kw=args.pv_step_kw, battery_step_kwh=args.battery_step_kwh,
        pv_min_kw=args.pv_min_kw, pv_max_kw=args.pv_max_kw,
        battery_min_kwh=args.battery_min_kwh, battery_max_kwh=args.battery_max_kwh,
        bess_batch_size=args.bess_batch_size,
        intermediate_pv_step_kw=args.intermediate_pv_step_kw,
        intermediate_battery_step_kwh=args.intermediate_battery_step_kwh,
        cost_screen_margin=args.cost_screen_margin,
        import_checkpoints_from=args.import_checkpoints_from,
    )
    result = run_sizing(
        args.source_results_dir, args.output_dir, workers=args.workers, resume=args.resume, spec=spec,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
