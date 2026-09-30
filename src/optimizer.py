from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Callable

import pandas as pd

from .costing import pv_bess_annual_cost
from .dispatch import solve_dispatch

PvCapacityFrameBuilder = Callable[[float], tuple[pd.DataFrame, dict[str, float | str]]]
DispatchKey = tuple[str, int, float, float]
CandidateCacheKey = tuple[str, float, float]
NoBessReferenceCacheKey = tuple[str, float]
CandidatePoint = tuple[float, float]


def scenario_config(config: dict[str, Any], scenario_key: str) -> dict[str, Any]:
    resolved = deepcopy(config)
    scenario = deepcopy(config["scenarios"][scenario_key])
    resolved["active_scenario"] = scenario
    resolved["active_scenario_key"] = scenario_key
    resolved["economics"] = deepcopy(config["economics"])
    resolved["economics"].update(scenario.get("battery_economics", {}))
    return resolved


def battery_sizing(energy_kwh: float, config: dict[str, Any]) -> dict[str, float]:
    energy = float(energy_kwh)
    max_c = float(config["battery"]["max_c_rate"])
    return {
        "battery_energy_kwh": energy,
        "battery_power_kw": energy * max_c if energy > 0 else 0.0,
        "battery_c_rate": max_c if energy > 0 else 0.0,
    }


class CandidateStore:
    def __init__(self, retain_dispatch: bool = True) -> None:
        self.retain_dispatch = retain_dispatch
        self._results: dict[CandidateCacheKey, dict[str, Any]] = {}
        self._dispatches: dict[CandidateCacheKey, pd.DataFrame] = {}
        self._no_bess_references: dict[NoBessReferenceCacheKey, dict[str, Any]] = {}

    def get(
        self,
        key: CandidateCacheKey,
        require_dispatch: bool,
    ) -> tuple[dict[str, Any], pd.DataFrame | None] | None:
        if key not in self._results:
            return None
        if require_dispatch and key not in self._dispatches:
            return None
        result = dict(self._results[key])
        dispatch = self._dispatches[key].copy() if key in self._dispatches else None
        return result, dispatch

    def set(self, key: CandidateCacheKey, result: dict[str, Any], dispatch: pd.DataFrame) -> None:
        self._results[key] = dict(result)
        if self.retain_dispatch:
            self._dispatches[key] = dispatch.copy()

    def get_no_bess_reference(self, key: NoBessReferenceCacheKey) -> dict[str, Any] | None:
        if key not in self._no_bess_references:
            return None
        return dict(self._no_bess_references[key])

    def set_no_bess_reference(self, key: NoBessReferenceCacheKey, reference: dict[str, Any]) -> None:
        self._no_bess_references[key] = dict(reference)


class CandidateEvaluator:
    def __init__(self, config: dict[str, Any], store: CandidateStore | None = None) -> None:
        self.config = config
        self.store = store or CandidateStore(retain_dispatch=True)

    def evaluate(
        self,
        frame: pd.DataFrame,
        battery_energy_kwh: float,
        pv_meta: dict[str, float | str],
        require_dispatch: bool = True,
    ) -> tuple[dict[str, Any], pd.DataFrame | None]:
        key = _candidate_cache_key(frame, battery_energy_kwh, pv_meta, self.config)
        cached = self.store.get(key, require_dispatch=require_dispatch)
        if cached is not None:
            return cached
        reference_key = _no_bess_reference_cache_key(frame, pv_meta, self.config)
        no_bess_reference = self.store.get_no_bess_reference(reference_key)
        result, dispatch = evaluate_candidate(
            frame,
            battery_energy_kwh,
            pv_meta,
            self.config,
            no_bess_reference=no_bess_reference,
        )
        self.store.set(key, result, dispatch)
        self.store.set_no_bess_reference(reference_key, _no_bess_reference_from_result(result))
        return dict(result), dispatch.copy() if require_dispatch else None


def _candidate_cache_key(
    frame: pd.DataFrame,
    battery_energy_kwh: float,
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
) -> CandidateCacheKey:
    return (
        _evaluation_context_hash(frame, pv_meta, config),
        round(float(pv_meta["pv_capacity_kw"]), 6),
        round(float(battery_energy_kwh), 6),
    )


def _no_bess_reference_cache_key(
    frame: pd.DataFrame,
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
) -> NoBessReferenceCacheKey:
    return (
        _evaluation_context_hash(frame, pv_meta, config),
        round(float(pv_meta["pv_capacity_kw"]), 6),
    )


def _evaluation_context_hash(frame: pd.DataFrame, pv_meta: dict[str, float | str], config: dict[str, Any]) -> str:
    return _hash_payload(
        {
            "frame": _frame_hash(frame),
            "config": _evaluation_config_signature(config),
            "pv_meta": _jsonable(pv_meta),
        }
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64")
    digest = hashlib.sha256()
    digest.update(",".join(str(column) for column in frame.columns).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def _evaluation_config_signature(config: dict[str, Any]) -> dict[str, Any]:
    pv = config.get("pv", {})
    return {
        "active_scenario": _jsonable(config.get("active_scenario", {})),
        "active_scenario_key": config.get("active_scenario_key"),
        "analysis": _jsonable(config.get("analysis", {})),
        "battery": _jsonable(config.get("battery", {})),
        "dispatch": _jsonable(config.get("dispatch", {})),
        "economics": _jsonable(config.get("economics", {})),
        "technical": _jsonable(config.get("technical", {})),
        "pv": _jsonable(
            {
                "capex_sgd_per_kw": pv.get("capex_sgd_per_kw"),
                "om_fraction_per_year": pv.get("om_fraction_per_year"),
                "project_years": pv.get("project_years"),
                "module_power_density_kw_per_m2": pv.get("module_power_density_kw_per_m2"),
                "rooftop_area_m2": pv.get("rooftop_area_m2"),
                "rooftop_area_limit_enabled": pv.get("rooftop_area_limit_enabled", True),
                "rooftop_area_utilization_fraction": pv.get("rooftop_area_utilization_fraction"),
                "rooftop_capacity_limit_kw": pv.get("rooftop_capacity_limit_kw"),
                "export_enabled": pv.get("export_enabled", False),
            }
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(payload), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def evaluate_candidate(
    frame: pd.DataFrame,
    battery_energy_kwh: float,
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
    no_bess_reference: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    sizing = battery_sizing(battery_energy_kwh, config)
    dispatch = solve_dispatch(frame, sizing, config)
    pv_bess = pv_bess_annual_cost(frame, dispatch, sizing, pv_meta, config)
    if no_bess_reference is None:
        no_bess_reference = (
            _no_bess_reference_from_pv_bess_cost(pv_bess)
            if abs(float(battery_energy_kwh)) <= 1e-9
            else _zero_battery_reference(frame, pv_meta, config)
        )
    result = {
        **sizing,
        **no_bess_reference,
        **pv_bess,
        "dispatch_backend": str(dispatch["dispatch_backend"].iloc[0]) if "dispatch_backend" in dispatch.columns else "unknown",
        "solver_status": str(dispatch["solver_status"].iloc[0]) if "solver_status" in dispatch.columns else "",
        "milp_objective_sgd": (
            float(dispatch["milp_objective_sgd"].iloc[0]) if "milp_objective_sgd" in dispatch.columns and pd.notna(dispatch["milp_objective_sgd"].iloc[0]) else pd.NA
        ),
        "degradation_penalty_sgd_per_kwh": (
            float(dispatch["degradation_penalty_sgd_per_kwh"].iloc[0])
            if "degradation_penalty_sgd_per_kwh" in dispatch.columns
            else 0.0
        ),
        "simultaneous_charge_discharge_kwh": (
            float(dispatch["simultaneous_charge_discharge_kwh"].sum())
            if "simultaneous_charge_discharge_kwh" in dispatch.columns
            else 0.0
        ),
        "avoided_unserved_energy_kwh_per_year": (
            no_bess_reference["no_bess_reference_unserved_energy_kwh_per_year"]
            - pv_bess["unserved_energy_kwh_per_year"]
        ),
        "system_annual_cost_delta_vs_no_bess_reference_sgd": (
            pv_bess["pv_bess_system_annual_cost_sgd"]
            - no_bess_reference["no_bess_reference_system_annual_cost_sgd"]
        ),
    }
    return result, dispatch


def _zero_battery_reference(frame: pd.DataFrame, pv_meta: dict[str, float | str], config: dict[str, Any]) -> dict[str, Any]:
    zero_sizing = battery_sizing(0.0, config)
    zero_dispatch = solve_dispatch(frame, zero_sizing, config)
    zero_cost = pv_bess_annual_cost(frame, zero_dispatch, zero_sizing, pv_meta, config)
    return _no_bess_reference_from_pv_bess_cost(zero_cost)


def _no_bess_reference_from_pv_bess_cost(pv_bess: dict[str, Any]) -> dict[str, Any]:
    return {
        "no_bess_reference_grid_energy_kwh": float(pv_bess["pv_bess_grid_energy_kwh"]),
        "no_bess_reference_grid_cost_sgd": float(pv_bess["pv_bess_grid_cost_sgd"]),
        "no_bess_reference_private_annual_cost_sgd": float(pv_bess["pv_bess_private_annual_cost_sgd"]),
        "no_bess_reference_unserved_energy_kwh_per_year": float(pv_bess["unserved_energy_kwh_per_year"]),
        "no_bess_reference_unserved_energy_cost_sgd_per_year": float(
            pv_bess["unserved_energy_cost_sgd_per_year"]
        ),
        "no_bess_reference_system_annual_cost_sgd": float(pv_bess["pv_bess_system_annual_cost_sgd"]),
    }


def _no_bess_reference_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "no_bess_reference_grid_energy_kwh": float(result["no_bess_reference_grid_energy_kwh"]),
        "no_bess_reference_grid_cost_sgd": float(result["no_bess_reference_grid_cost_sgd"]),
        "no_bess_reference_private_annual_cost_sgd": float(
            result["no_bess_reference_private_annual_cost_sgd"]
        ),
        "no_bess_reference_unserved_energy_kwh_per_year": float(
            result["no_bess_reference_unserved_energy_kwh_per_year"]
        ),
        "no_bess_reference_unserved_energy_cost_sgd_per_year": float(
            result["no_bess_reference_unserved_energy_cost_sgd_per_year"]
        ),
        "no_bess_reference_system_annual_cost_sgd": float(
            result["no_bess_reference_system_annual_cost_sgd"]
        ),
    }


def _pv_capacity_upper_limit(config: dict[str, Any]) -> float | None:
    pv_cfg = config.get("pv", {})
    if not bool(pv_cfg.get("rooftop_area_limit_enabled", True)):
        return None
    value = pv_cfg.get("rooftop_capacity_limit_kw")
    if value is None:
        return None
    return max(0.0, float(value))


def _cap_pv_axis(values: list[float], config: dict[str, Any], include_upper: bool = False) -> list[float]:
    upper = _pv_capacity_upper_limit(config)
    if upper is None:
        return sorted({round(float(value), 6) for value in values})
    capped = {round(float(value), 6) for value in values if float(value) <= upper + 1e-9}
    if include_upper:
        capped.add(round(float(upper), 6))
    if len(capped) < 2 and upper > 0.0:
        capped.add(round(float(upper), 6))
    capped.add(0.0)
    return sorted(capped)


def _pv_axis_at_capacity_limit(values: list[float], config: dict[str, Any]) -> bool:
    upper = _pv_capacity_upper_limit(config)
    if upper is None:
        return False
    return max(float(value) for value in values) >= upper - 1e-6


def _refined_values(anchor: float, window: float, step: float, upper: float | None = None) -> list[float]:
    start = max(0.0, anchor - window)
    end = max(start, anchor + window)
    if upper is not None:
        end = min(end, float(upper))
    values: list[float] = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 6))
        current += step
    return sorted(set(values) | {round(float(anchor), 6)})


def _expanded_axis(values: list[float], factor: float = 1.5, capacity_upper: float | None = None) -> list[float]:
    values = sorted({float(value) for value in values})
    if len(values) < 2:
        return values
    current_upper = max(values)
    if current_upper <= 0.0:
        return values
    intervals = len(values) - 1
    target_upper = current_upper * float(factor)
    if capacity_upper is not None:
        target_upper = min(target_upper, float(capacity_upper))
    step = target_upper / intervals
    axis = [round(i * step, 6) for i in range(intervals + 1) if i * step <= target_upper + 1e-9]
    axis = sorted(set(axis) | {round(float(target_upper), 6)})
    return axis


def _nice_step(raw_step: float) -> float:
    if raw_step <= 0.0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    for multiplier in (1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        step = multiplier * magnitude
        if step >= raw_step - 1e-12:
            return float(step)
    return float(10.0 * magnitude)


def _is_upper_boundary(value: float, values: list[float]) -> bool:
    if len(values) < 2:
        return False
    upper = max(float(item) for item in values)
    if upper <= 0.0:
        return False
    return abs(float(value) - upper) <= 1e-6


def _is_axis_edge(value: float, values: list[float]) -> bool:
    if len(values) < 3:
        return False
    axis = sorted(float(item) for item in values)
    value_f = float(value)
    return abs(value_f - axis[0]) <= 1e-6 or abs(value_f - axis[-1]) <= 1e-6


def _dispatch_key(search_stage: str, search_iteration: int, pv_capacity_kw: float, battery_energy_kwh: float) -> DispatchKey:
    return (
        str(search_stage),
        int(search_iteration),
        round(float(pv_capacity_kw), 6),
        round(float(battery_energy_kwh), 6),
    )


def _evaluate_candidate_grid(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    pv_values: list[float],
    battery_values: list[float],
    config: dict[str, Any],
    search_stage: str,
    search_iteration: int,
    evaluator: CandidateEvaluator | None = None,
    retain_dispatches: bool = True,
) -> tuple[pd.DataFrame, dict[DispatchKey, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    dispatch_store: dict[DispatchKey, pd.DataFrame] = {}
    active_evaluator = evaluator or CandidateEvaluator(config)
    for pv_capacity in sorted({float(value) for value in pv_values}):
        frame, pv_meta = pv_capacity_frame_builder(float(pv_capacity))
        for battery_energy in sorted({float(value) for value in battery_values}):
            result, dispatch = active_evaluator.evaluate(frame, float(battery_energy), pv_meta, require_dispatch=True)
            if dispatch is None:
                raise ValueError("Candidate evaluator did not return dispatch for grid search")
            result.update({"search_stage": search_stage, "search_iteration": int(search_iteration)})
            rows.append(result)
            key = _dispatch_key(search_stage, int(search_iteration), result["pv_capacity_kw"], result["battery_energy_kwh"])
            if retain_dispatches:
                annotated_dispatch = dispatch.copy()
                annotated_dispatch["search_stage"] = search_stage
                annotated_dispatch["search_iteration"] = int(search_iteration)
                annotated_dispatch["pv_capacity_kw"] = float(result["pv_capacity_kw"])
                annotated_dispatch["battery_energy_kwh"] = float(result["battery_energy_kwh"])
                dispatch_store[key] = annotated_dispatch
    summary = pd.DataFrame(rows).sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    return summary, dispatch_store


def _evaluate_candidate_points(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    points: list[CandidatePoint],
    config: dict[str, Any],
    search_stage: str,
    search_iteration: int,
    evaluator: CandidateEvaluator | None = None,
    retain_dispatches: bool = True,
) -> tuple[pd.DataFrame, dict[DispatchKey, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    dispatch_store: dict[DispatchKey, pd.DataFrame] = {}
    active_evaluator = evaluator or CandidateEvaluator(config)
    frame_cache: dict[float, tuple[pd.DataFrame, dict[str, float | str]]] = {}
    unique_points = sorted({(round(float(pv), 6), round(float(battery), 6)) for pv, battery in points})
    for pv_capacity, battery_energy in unique_points:
        if pv_capacity not in frame_cache:
            frame_cache[pv_capacity] = pv_capacity_frame_builder(float(pv_capacity))
        frame, pv_meta = frame_cache[pv_capacity]
        result, dispatch = active_evaluator.evaluate(
            frame,
            float(battery_energy),
            pv_meta,
            require_dispatch=retain_dispatches,
        )
        result.update({"search_stage": search_stage, "search_iteration": int(search_iteration)})
        rows.append(result)
        key = _dispatch_key(search_stage, int(search_iteration), result["pv_capacity_kw"], result["battery_energy_kwh"])
        if retain_dispatches:
            if dispatch is None:
                raise ValueError("Candidate evaluator did not return dispatch for point search")
            annotated_dispatch = dispatch.copy()
            annotated_dispatch["search_stage"] = search_stage
            annotated_dispatch["search_iteration"] = int(search_iteration)
            annotated_dispatch["pv_capacity_kw"] = float(result["pv_capacity_kw"])
            annotated_dispatch["battery_energy_kwh"] = float(result["battery_energy_kwh"])
            dispatch_store[key] = annotated_dispatch
    if not rows:
        return pd.DataFrame(), dispatch_store
    summary = pd.DataFrame(rows).sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    return summary, dispatch_store


def _bounded_candidate(value: float, lower: float, upper: float) -> float:
    return round(min(max(float(value), float(lower)), float(upper)), 6)


def _reduced_step(step: float, minimum: float) -> float:
    if step <= minimum + 1e-9:
        return float(minimum)
    return max(float(minimum), _nice_step(float(step) / 2.0))


def _initial_local_step(window: float, minimum: float) -> float:
    if window <= 0.0:
        return float(minimum)
    return max(float(minimum), _nice_step(float(window) / 2.0))


def _is_expandable_axis_edge(
    value: float,
    lower: float,
    upper: float,
    physical_lower: float | None = None,
    physical_upper: float | None = None,
) -> bool:
    value_f = float(value)
    lower_f = float(lower)
    upper_f = float(upper)
    if abs(value_f - lower_f) <= 1e-6 and (physical_lower is None or lower_f > float(physical_lower) + 1e-6):
        return True
    if abs(value_f - upper_f) <= 1e-6 and (physical_upper is None or upper_f < float(physical_upper) - 1e-6):
        return True
    return False


def _coordinate_local_search(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    start_best: dict[str, Any],
    config: dict[str, Any],
    search_iteration: int,
    evaluator: CandidateEvaluator,
    retain_dispatches: bool,
) -> tuple[pd.DataFrame, dict[DispatchKey, pd.DataFrame], dict[str, Any], dict[str, float | int | bool]]:
    pv_window = float(config["pv"]["local_refine_window_kw"])
    battery_window = float(config["battery"]["local_refine_window_kwh"])
    pv_min_step = float(config["pv"]["local_refine_step_kw"])
    battery_min_step = float(config["battery"]["local_refine_step_kwh"])
    pv_lower = max(0.0, float(start_best["pv_capacity_kw"]) - pv_window)
    pv_upper = max(pv_lower, float(start_best["pv_capacity_kw"]) + pv_window)
    pv_capacity_limit = _pv_capacity_upper_limit(config)
    if pv_capacity_limit is not None:
        pv_upper = min(pv_upper, float(pv_capacity_limit))
    battery_lower = max(0.0, float(start_best["battery_energy_kwh"]) - battery_window)
    battery_upper = max(battery_lower, float(start_best["battery_energy_kwh"]) + battery_window)
    pv_step = _initial_local_step(pv_window, pv_min_step)
    battery_step = _initial_local_step(battery_window, battery_min_step)
    current = dict(start_best)
    current_cost = float(current["pv_bess_system_annual_cost_sgd"])
    rows: list[pd.DataFrame] = []
    dispatch_store: dict[DispatchKey, pd.DataFrame] = {}
    evaluated: set[CandidatePoint] = {
        (round(float(current["pv_capacity_kw"]), 6), round(float(current["battery_energy_kwh"]), 6))
    }
    moves = 0
    rounds = 0
    max_rounds = int(config.get("capacity_search", {}).get("local_search_max_rounds", 200))

    while rounds < max_rounds:
        rounds += 1
        pv = float(current["pv_capacity_kw"])
        battery = float(current["battery_energy_kwh"])
        neighbors = [
            (_bounded_candidate(pv + pv_step, pv_lower, pv_upper), round(battery, 6)),
            (_bounded_candidate(pv - pv_step, pv_lower, pv_upper), round(battery, 6)),
            (round(pv, 6), _bounded_candidate(battery + battery_step, battery_lower, battery_upper)),
            (round(pv, 6), _bounded_candidate(battery - battery_step, battery_lower, battery_upper)),
        ]
        points = [point for point in neighbors if point not in evaluated]
        if points:
            evaluated.update(points)
            summary, dispatches = _evaluate_candidate_points(
                pv_capacity_frame_builder,
                points,
                config,
                search_stage="local_search",
                search_iteration=search_iteration,
                evaluator=evaluator,
                retain_dispatches=retain_dispatches,
            )
            dispatch_store.update(dispatches)
            if not summary.empty:
                rows.append(summary)
                candidate = dict(summary.iloc[0])
                candidate_cost = float(candidate["pv_bess_system_annual_cost_sgd"])
                if candidate_cost < current_cost - 1e-9:
                    current = candidate
                    current_cost = candidate_cost
                    moves += 1
                    continue

        next_pv_step = _reduced_step(pv_step, pv_min_step)
        next_battery_step = _reduced_step(battery_step, battery_min_step)
        if abs(next_pv_step - pv_step) <= 1e-9 and abs(next_battery_step - battery_step) <= 1e-9:
            break
        pv_step = next_pv_step
        battery_step = next_battery_step

    local = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not local.empty:
        local = local.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    edge = _is_expandable_axis_edge(
        float(current["pv_capacity_kw"]),
        pv_lower,
        pv_upper,
        physical_lower=0.0,
        physical_upper=pv_capacity_limit,
    ) or _is_expandable_axis_edge(
        float(current["battery_energy_kwh"]),
        battery_lower,
        battery_upper,
        physical_lower=0.0,
        physical_upper=None,
    )
    metadata = {
        "local_search_rounds": int(rounds),
        "local_search_moves": int(moves),
        "local_search_evaluated_points": int(max(len(evaluated) - 1, 0)),
        "local_search_hit_edge": bool(edge),
        "local_search_final_pv_step_kw": float(pv_step),
        "local_search_final_battery_step_kwh": float(battery_step),
    }
    return local, dispatch_store, current, metadata


def _validation_points(best: dict[str, Any], config: dict[str, Any]) -> list[CandidatePoint]:
    radius = max(0, int(config.get("capacity_search", {}).get("local_search_validation_radius", 2)))
    pv_step = float(config["pv"]["local_refine_step_kw"])
    battery_step = float(config["battery"]["local_refine_step_kwh"])
    pv = float(best["pv_capacity_kw"])
    battery = float(best["battery_energy_kwh"])
    pv_capacity_limit = _pv_capacity_upper_limit(config)
    points: list[CandidatePoint] = []
    for pv_offset in range(-radius, radius + 1):
        for battery_offset in range(-radius, radius + 1):
            candidate_pv = max(0.0, pv + pv_offset * pv_step)
            if pv_capacity_limit is not None:
                candidate_pv = min(candidate_pv, float(pv_capacity_limit))
            points.append(
                (
                    round(candidate_pv, 6),
                    round(max(0.0, battery + battery_offset * battery_step), 6),
                )
            )
    return points


def _warm_start_source_point(warm_start_best: dict[str, Any], config: dict[str, Any]) -> CandidatePoint:
    pv_capacity = float(warm_start_best["pv_capacity_kw"])
    pv_limit = _pv_capacity_upper_limit(config)
    if pv_limit is not None:
        pv_capacity = min(pv_capacity, float(pv_limit))
    battery_energy = max(0.0, float(warm_start_best["battery_energy_kwh"]))
    return round(max(0.0, pv_capacity), 6), round(battery_energy, 6)


def _warm_start_fields(
    used: bool,
    source_pv: float | None = None,
    source_battery: float | None = None,
    fallback_triggered: bool = False,
    fallback_reason: str = "",
    candidate_count: int = 0,
) -> dict[str, Any]:
    return {
        "warm_start_used": bool(used),
        "warm_start_source_pv_capacity_kw": source_pv if source_pv is not None else pd.NA,
        "warm_start_source_battery_energy_kwh": source_battery if source_battery is not None else pd.NA,
        "warm_start_fallback_triggered": bool(fallback_triggered),
        "warm_start_fallback_reason": str(fallback_reason),
        "warm_start_candidate_count": int(candidate_count),
    }


def _optimize_with_warm_start(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    config: dict[str, Any],
    warm_start_best: dict[str, Any],
    evaluator: CandidateEvaluator | None = None,
    retain_candidate_dispatches: bool = True,
) -> tuple[pd.DataFrame, dict[DispatchKey, pd.DataFrame], dict[str, Any]]:
    search_cfg = config.get("capacity_search", {})
    fallback_mode = str(search_cfg.get("warm_start_fallback", "boundary")).lower()
    max_boundary_retries = max(0, int(search_cfg.get("warm_start_max_boundary_retries", 2)))
    fallback_after_hits = max(1, max_boundary_retries)
    active_evaluator = evaluator or CandidateEvaluator(config)
    source_pv, source_battery = _warm_start_source_point(warm_start_best, config)
    history: list[pd.DataFrame] = []
    dispatch_store: dict[DispatchKey, pd.DataFrame] = {}

    anchor, anchor_dispatches = _evaluate_candidate_points(
        pv_capacity_frame_builder,
        [(source_pv, source_battery)],
        config,
        search_stage="warm_start_anchor",
        search_iteration=0,
        evaluator=active_evaluator,
        retain_dispatches=retain_candidate_dispatches,
    )
    dispatch_store.update(anchor_dispatches)
    if anchor.empty:
        return _optimize_full_search(
            pv_capacity_frame_builder,
            config,
            evaluator=active_evaluator,
            retain_candidate_dispatches=retain_candidate_dispatches,
            warm_start_fields=_warm_start_fields(
                True,
                source_pv,
                source_battery,
                fallback_triggered=True,
                fallback_reason="empty_anchor",
            ),
        )
    history.append(anchor)
    current_best = dict(anchor.iloc[0])
    total_rounds = 0
    total_moves = 0
    total_points = int(len(anchor))
    boundary_hits = 0
    fallback_triggered = False
    fallback_reason = ""

    for iteration in range(max_boundary_retries + 1):
        local_refined, refined_dispatches, current_best, metadata = _coordinate_local_search(
            pv_capacity_frame_builder,
            current_best,
            config,
            search_iteration=iteration,
            evaluator=active_evaluator,
            retain_dispatches=retain_candidate_dispatches,
        )
        dispatch_store.update(refined_dispatches)
        total_rounds += int(metadata["local_search_rounds"])
        total_moves += int(metadata["local_search_moves"])
        total_points += int(metadata["local_search_evaluated_points"])
        if not local_refined.empty:
            history.append(local_refined)
        if bool(metadata["local_search_hit_edge"]):
            boundary_hits += 1
            if fallback_mode == "boundary" and boundary_hits >= fallback_after_hits:
                fallback_triggered = True
                fallback_reason = "boundary"
                break
            continue
        break

    if fallback_triggered:
        return _optimize_full_search(
            pv_capacity_frame_builder,
            config,
            evaluator=active_evaluator,
            retain_candidate_dispatches=retain_candidate_dispatches,
            warm_start_fields=_warm_start_fields(
                True,
                source_pv,
                source_battery,
                fallback_triggered=True,
                fallback_reason=fallback_reason,
                candidate_count=total_points,
            ),
        )

    validation, validation_dispatches = _evaluate_candidate_points(
        pv_capacity_frame_builder,
        _validation_points(current_best, config),
        config,
        search_stage="local_validation",
        search_iteration=0,
        evaluator=active_evaluator,
        retain_dispatches=retain_candidate_dispatches,
    )
    dispatch_store.update(validation_dispatches)
    if not validation.empty:
        history.append(validation)

    all_rows = pd.concat(history, ignore_index=True)
    all_rows = all_rows.drop_duplicates(
        subset=["pv_capacity_kw", "battery_energy_kwh", "search_stage", "search_iteration"],
        keep="last",
    )
    all_rows = all_rows.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    best = dict(all_rows.iloc[0])
    best["coarse_search_iteration"] = -1
    best["optimal_region_iteration"] = int(all_rows["search_iteration"].max())
    best.update(
        {
            "capacity_search_method": "warm_start_coordinate",
            "local_search_rounds": int(total_rounds),
            "local_search_moves": int(total_moves),
            "local_search_evaluated_points": int(max(total_points - 1, 0)),
            "local_validation_points": int(len(validation)),
        }
    )
    best.update(
        _warm_start_fields(
            True,
            source_pv,
            source_battery,
            fallback_triggered=False,
            fallback_reason="",
            candidate_count=int(len(all_rows)),
        )
    )
    return all_rows, dispatch_store, best


def optimize_pv_battery_with_refinement(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    config: dict[str, Any],
    evaluator: CandidateEvaluator | None = None,
    warm_start_best: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    all_rows, dispatch_store, best = optimize_pv_battery_with_refinement_with_dispatches(
        pv_capacity_frame_builder,
        config,
        evaluator=evaluator,
        warm_start_best=warm_start_best,
    )
    key = _dispatch_key(
        str(best["search_stage"]),
        int(best["search_iteration"]),
        float(best["pv_capacity_kw"]),
        float(best["battery_energy_kwh"]),
    )
    return all_rows, dispatch_store[key].copy(), best


def optimize_pv_battery_with_refinement_with_dispatches(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    config: dict[str, Any],
    evaluator: CandidateEvaluator | None = None,
    retain_candidate_dispatches: bool = True,
    warm_start_best: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[DispatchKey, pd.DataFrame], dict[str, Any]]:
    search_cfg = config.get("capacity_search", {})
    if warm_start_best is not None and bool(search_cfg.get("warm_start_enabled", True)):
        return _optimize_with_warm_start(
            pv_capacity_frame_builder,
            config,
            warm_start_best,
            evaluator=evaluator,
            retain_candidate_dispatches=retain_candidate_dispatches,
        )
    return _optimize_full_search(
        pv_capacity_frame_builder,
        config,
        evaluator=evaluator,
        retain_candidate_dispatches=retain_candidate_dispatches,
        warm_start_fields=_warm_start_fields(False),
    )


def _optimize_full_search(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    config: dict[str, Any],
    evaluator: CandidateEvaluator | None = None,
    retain_candidate_dispatches: bool = True,
    warm_start_fields: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[DispatchKey, pd.DataFrame], dict[str, Any]]:
    pv_candidates = _cap_pv_axis([float(value) for value in config["pv"]["capacity_kw_candidates"]], config)
    battery_candidates = [float(value) for value in config["battery"]["energy_kwh_candidates"]]
    search_cfg = config.get("capacity_search", {})
    expansion_factor = float(search_cfg.get("upper_expansion_factor", 1.5))
    max_upper_expansions = int(search_cfg.get("max_upper_expansions", 2))
    refine_divisions = max(1.0, float(search_cfg.get("local_refine_divisions", 6)))
    coarse_history: list[pd.DataFrame] = []
    dispatch_store: dict[DispatchKey, pd.DataFrame] = {}
    active_evaluator = evaluator or CandidateEvaluator(config)
    coarse_best: dict[str, Any] = {}
    final_coarse_iteration = 0
    for iteration in range(max_upper_expansions + 1):
        coarse, coarse_dispatches = _evaluate_candidate_grid(
            pv_capacity_frame_builder,
            pv_candidates,
            battery_candidates,
            config,
            search_stage="coarse",
            search_iteration=iteration,
            evaluator=active_evaluator,
            retain_dispatches=retain_candidate_dispatches,
        )
        dispatch_store.update(coarse_dispatches)
        coarse = coarse.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
        coarse_history.append(coarse)
        coarse_best = dict(coarse.iloc[0])
        expand_pv = _is_upper_boundary(float(coarse_best["pv_capacity_kw"]), pv_candidates) and not _pv_axis_at_capacity_limit(
            pv_candidates,
            config,
        )
        expand_battery = _is_upper_boundary(float(coarse_best["battery_energy_kwh"]), battery_candidates)
        final_coarse_iteration = iteration
        if iteration >= max_upper_expansions or not (expand_pv or expand_battery):
            break
        if expand_pv:
            pv_candidates = _expanded_axis(
                pv_candidates,
                factor=expansion_factor,
                capacity_upper=_pv_capacity_upper_limit(config),
            )
            pv_candidates = _cap_pv_axis(pv_candidates, config)
            config["pv"]["capacity_kw_candidates"] = pv_candidates
            pv_step = pv_candidates[1] - pv_candidates[0]
            if config["pv"].get("local_refine_window_source") != "parameter":
                config["pv"]["local_refine_window_kw"] = pv_step
            if config["pv"].get("local_refine_step_source") != "parameter":
                config["pv"]["local_refine_step_kw"] = max(_nice_step(pv_step / refine_divisions), 1e-6)
        if expand_battery:
            battery_candidates = _expanded_axis(battery_candidates, factor=expansion_factor)
            config["battery"]["energy_kwh_candidates"] = battery_candidates
            battery_step = battery_candidates[1] - battery_candidates[0]
            if config["battery"].get("local_refine_window_source") != "parameter":
                config["battery"]["local_refine_window_kwh"] = battery_step
            if config["battery"].get("local_refine_step_source") != "parameter":
                config["battery"]["local_refine_step_kwh"] = max(_nice_step(battery_step / refine_divisions), 1e-6)

    coarse = coarse_history[-1]
    best_pv_capacity = float(coarse_best["pv_capacity_kw"])
    best_battery = float(coarse_best["battery_energy_kwh"])
    refined_history: list[pd.DataFrame] = []
    refined: pd.DataFrame | None = None
    local_search_stats: dict[str, Any] = {}
    search_method = str(search_cfg.get("local_search_method", "coordinate")).lower()
    if search_method == "grid":
        for iteration in range(2):
            refined_pv_values = _refined_values(
                best_pv_capacity,
                float(config["pv"]["local_refine_window_kw"]),
                float(config["pv"]["local_refine_step_kw"]),
                upper=_pv_capacity_upper_limit(config),
            )
            refined_battery_values = _refined_values(
                best_battery,
                float(config["battery"]["local_refine_window_kwh"]),
                float(config["battery"]["local_refine_step_kwh"]),
            )

            local_refined, refined_dispatches = _evaluate_candidate_grid(
                pv_capacity_frame_builder,
                refined_pv_values,
                refined_battery_values,
                config,
                search_stage="optimal_region",
                search_iteration=iteration,
                evaluator=active_evaluator,
                retain_dispatches=retain_candidate_dispatches,
            )
            dispatch_store.update(refined_dispatches)
            refined = local_refined.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
            refined_history.append(refined)
            refined_best = dict(refined.iloc[0])
            best_pv_capacity = float(refined_best["pv_capacity_kw"])
            best_battery = float(refined_best["battery_energy_kwh"])
            if not (_is_axis_edge(best_pv_capacity, refined_pv_values) or _is_axis_edge(best_battery, refined_battery_values)):
                break
        local_search_stats = {
            "capacity_search_method": "grid",
            "local_search_rounds": 0,
            "local_search_moves": 0,
            "local_search_evaluated_points": int(sum(len(frame) for frame in refined_history)),
            "local_validation_points": 0,
        }
    else:
        current_best = dict(coarse_best)
        total_rounds = 0
        total_moves = 0
        total_points = 0
        for iteration in range(2):
            local_refined, refined_dispatches, current_best, metadata = _coordinate_local_search(
                pv_capacity_frame_builder,
                current_best,
                config,
                search_iteration=iteration,
                evaluator=active_evaluator,
                retain_dispatches=retain_candidate_dispatches,
            )
            dispatch_store.update(refined_dispatches)
            total_rounds += int(metadata["local_search_rounds"])
            total_moves += int(metadata["local_search_moves"])
            total_points += int(metadata["local_search_evaluated_points"])
            if not local_refined.empty:
                refined_history.append(local_refined)
            best_pv_capacity = float(current_best["pv_capacity_kw"])
            best_battery = float(current_best["battery_energy_kwh"])
            if not bool(metadata["local_search_hit_edge"]):
                break
        validation, validation_dispatches = _evaluate_candidate_points(
            pv_capacity_frame_builder,
            _validation_points(current_best, config),
            config,
            search_stage="local_validation",
            search_iteration=0,
            evaluator=active_evaluator,
            retain_dispatches=retain_candidate_dispatches,
        )
        dispatch_store.update(validation_dispatches)
        if not validation.empty:
            refined_history.append(validation)
        refined = pd.concat(refined_history, ignore_index=True) if refined_history else validation
        refined = refined.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
        local_search_stats = {
            "capacity_search_method": "coordinate",
            "local_search_rounds": int(total_rounds),
            "local_search_moves": int(total_moves),
            "local_search_evaluated_points": int(total_points),
            "local_validation_points": int(len(validation)),
        }

    if refined is None:
        raise ValueError("PV-BESS optimization failed to generate a refined candidate grid")
    combined = pd.concat([coarse, refined], ignore_index=True)
    combined = combined.drop_duplicates(subset=["pv_capacity_kw", "battery_energy_kwh"], keep="last")
    combined = combined.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    best = dict(combined.iloc[0])
    best["coarse_search_iteration"] = final_coarse_iteration
    best["optimal_region_iteration"] = int(refined["search_iteration"].max())
    best.update(local_search_stats)
    best.update(warm_start_fields or _warm_start_fields(False))
    all_rows = pd.concat(coarse_history + refined_history, ignore_index=True)
    all_rows = all_rows.drop_duplicates(subset=["pv_capacity_kw", "battery_energy_kwh", "search_stage", "search_iteration"], keep="last")
    all_rows = all_rows.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    return all_rows, dispatch_store, best


def surface_search(
    base_frame: pd.DataFrame,
    frame_builder,
    scenario_key: str,
    outage_profile: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    scenario_cfg = scenario_config(config, scenario_key)
    evaluator = CandidateEvaluator(scenario_cfg, CandidateStore(retain_dispatch=False))
    for pv_capacity in config["pv"]["capacity_kw_candidates"]:
        frame, pv_meta = frame_builder(base_frame, scenario_key, config, float(pv_capacity))
        frame = frame.merge(
            outage_profile[["hour_index", "grid_availability", "outage_fraction"]],
            on="hour_index",
            how="left",
        )
        for battery in config["battery"]["energy_kwh_candidates"]:
            result, _ = evaluator.evaluate(frame, float(battery), pv_meta, require_dispatch=False)
            rows.append(
                {
                    "scenario": scenario_key,
                    **result,
                }
            )
    surface = pd.DataFrame(rows)
    matrix = surface.pivot_table(
        index="pv_capacity_kw",
        columns="battery_energy_kwh",
        values="pv_bess_system_annual_cost_sgd",
        aggfunc="min",
    ).sort_index()
    return surface, matrix


def _best_dispatch_from_store(best: dict[str, Any], dispatch_store: dict[DispatchKey, pd.DataFrame]) -> pd.DataFrame:
    key = _dispatch_key(
        str(best["search_stage"]),
        int(best["search_iteration"]),
        float(best["pv_capacity_kw"]),
        float(best["battery_energy_kwh"]),
    )
    return dispatch_store[key].copy()


def resilience_design(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    config: dict[str, Any],
    scenario_key: str,
    warm_start_best: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[DispatchKey, pd.DataFrame]]:
    sizing_cfg = scenario_config(config, scenario_key)
    retain_candidate_dispatches = bool(sizing_cfg.get("outputs", {}).get("save_candidate_dispatch_workbooks", False))
    evaluator = CandidateEvaluator(sizing_cfg, CandidateStore(retain_dispatch=retain_candidate_dispatches))

    summary, dispatch_store, best = optimize_pv_battery_with_refinement_with_dispatches(
        pv_capacity_frame_builder,
        sizing_cfg,
        evaluator=evaluator,
        retain_candidate_dispatches=retain_candidate_dispatches,
        warm_start_best=warm_start_best,
    )
    if retain_candidate_dispatches:
        dispatch = _best_dispatch_from_store(best, dispatch_store)
    else:
        frame, pv_meta = pv_capacity_frame_builder(float(best["pv_capacity_kw"]))
        dispatch_result, dispatch = evaluator.evaluate(
            frame,
            float(best["battery_energy_kwh"]),
            pv_meta,
            require_dispatch=True,
        )
        if dispatch is None:
            raise ValueError("Candidate evaluator did not return dispatch for the best design")
        best.update(dispatch_result)
        dispatch["search_stage"] = str(best["search_stage"])
        dispatch["search_iteration"] = int(best["search_iteration"])
        dispatch["pv_capacity_kw"] = float(best["pv_capacity_kw"])
        dispatch["battery_energy_kwh"] = float(best["battery_energy_kwh"])
    best.update(
        {
            "scenario": scenario_key,
            "optimization_outage_mode": "city_outage",
            "optimization_voll_sgd_per_kwh": sizing_cfg["active_scenario"]["voll_sgd_per_kwh"],
            "evaluation_voll_sgd_per_kwh": sizing_cfg["active_scenario"]["voll_sgd_per_kwh"],
        }
    )
    return pd.DataFrame([best]), dispatch, summary, dispatch_store
