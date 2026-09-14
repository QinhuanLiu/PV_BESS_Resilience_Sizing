from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Callable

import pandas as pd

from .costing import pv_bess_annual_cost
from .dispatch import solve_dispatch

PvCapacityFrameBuilder = Callable[[float], tuple[pd.DataFrame, dict[str, float | str]]]
DispatchKey = tuple[str, int, float, float]
CandidateCacheKey = tuple[str, float, float]
NoBessReferenceCacheKey = tuple[str, float]


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


def _refined_values(anchor: float, window: float, step: float) -> list[float]:
    start = max(0.0, anchor - window)
    end = max(start, anchor + window)
    count = int(round((end - start) / step))
    return sorted(set(round(start + i * step, 6) for i in range(count + 1)) | {anchor})


def _expanded_axis(values: list[float], factor: float = 1.5) -> list[float]:
    values = sorted({float(value) for value in values})
    if len(values) < 2:
        return values
    upper = max(values)
    if upper <= 0.0:
        return values
    intervals = len(values) - 1
    step = upper * float(factor) / intervals
    return [round(i * step, 6) for i in range(intervals + 1)]


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
            annotated_dispatch = dispatch.copy()
            annotated_dispatch["search_stage"] = search_stage
            annotated_dispatch["search_iteration"] = int(search_iteration)
            annotated_dispatch["pv_capacity_kw"] = float(result["pv_capacity_kw"])
            annotated_dispatch["battery_energy_kwh"] = float(result["battery_energy_kwh"])
            dispatch_store[key] = annotated_dispatch
    summary = pd.DataFrame(rows).sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    return summary, dispatch_store


def optimize_pv_battery_with_refinement(
    pv_capacity_frame_builder: PvCapacityFrameBuilder,
    config: dict[str, Any],
    evaluator: CandidateEvaluator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    all_rows, dispatch_store, best = optimize_pv_battery_with_refinement_with_dispatches(
        pv_capacity_frame_builder,
        config,
        evaluator=evaluator,
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
) -> tuple[pd.DataFrame, dict[DispatchKey, pd.DataFrame], dict[str, Any]]:
    pv_candidates = [float(value) for value in config["pv"]["capacity_kw_candidates"]]
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
        )
        dispatch_store.update(coarse_dispatches)
        coarse = coarse.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
        coarse_history.append(coarse)
        coarse_best = dict(coarse.iloc[0])
        expand_pv = _is_upper_boundary(float(coarse_best["pv_capacity_kw"]), pv_candidates)
        expand_battery = _is_upper_boundary(float(coarse_best["battery_energy_kwh"]), battery_candidates)
        final_coarse_iteration = iteration
        if iteration >= max_upper_expansions or not (expand_pv or expand_battery):
            break
        if expand_pv:
            pv_candidates = _expanded_axis(pv_candidates, factor=expansion_factor)
            config["pv"]["capacity_kw_candidates"] = pv_candidates
            config["pv"]["local_refine_window_kw"] = pv_candidates[1] - pv_candidates[0]
            config["pv"]["local_refine_step_kw"] = max((pv_candidates[1] - pv_candidates[0]) / refine_divisions, 1e-6)
        if expand_battery:
            battery_candidates = _expanded_axis(battery_candidates, factor=expansion_factor)
            config["battery"]["energy_kwh_candidates"] = battery_candidates
            config["battery"]["local_refine_window_kwh"] = battery_candidates[1] - battery_candidates[0]
            config["battery"]["local_refine_step_kwh"] = max((battery_candidates[1] - battery_candidates[0]) / refine_divisions, 1e-6)

    coarse = coarse_history[-1]
    best_pv_capacity = float(coarse_best["pv_capacity_kw"])
    best_battery = float(coarse_best["battery_energy_kwh"])
    refined_history: list[pd.DataFrame] = []
    refined: pd.DataFrame | None = None
    for iteration in range(2):
        refined_pv_values = _refined_values(
            best_pv_capacity,
            float(config["pv"]["local_refine_window_kw"]),
            float(config["pv"]["local_refine_step_kw"]),
        )
        refined_battery_values = _refined_values(
            best_battery,
            float(config["battery"]["local_refine_window_kwh"]),
            float(config["battery"]["local_refine_step_kwh"]),
        )

        refined_rows: list[pd.DataFrame] = []
        local_refined, refined_dispatches = _evaluate_candidate_grid(
            pv_capacity_frame_builder,
            refined_pv_values,
            refined_battery_values,
            config,
            search_stage="optimal_region",
            search_iteration=iteration,
            evaluator=active_evaluator,
        )
        dispatch_store.update(refined_dispatches)
        refined_rows.append(local_refined)
        refined = pd.concat(refined_rows, ignore_index=True)
        refined = refined.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
        refined_history.append(refined)
        refined_best = dict(refined.iloc[0])
        best_pv_capacity = float(refined_best["pv_capacity_kw"])
        best_battery = float(refined_best["battery_energy_kwh"])
        if not (_is_axis_edge(best_pv_capacity, refined_pv_values) or _is_axis_edge(best_battery, refined_battery_values)):
            break

    if refined is None:
        raise ValueError("PV-BESS optimization failed to generate a refined candidate grid")
    combined = pd.concat([coarse, refined], ignore_index=True)
    combined = combined.drop_duplicates(subset=["pv_capacity_kw", "battery_energy_kwh"], keep="last")
    combined = combined.sort_values("pv_bess_system_annual_cost_sgd").reset_index(drop=True)
    best = dict(combined.iloc[0])
    best["coarse_search_iteration"] = final_coarse_iteration
    best["optimal_region_iteration"] = int(refined["search_iteration"].max())
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
        frame = frame.merge(outage_profile, on="hour_index", how="left")
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[DispatchKey, pd.DataFrame]]:
    sizing_cfg = scenario_config(config, scenario_key)
    evaluator = CandidateEvaluator(sizing_cfg, CandidateStore(retain_dispatch=True))

    summary, dispatch_store, best = optimize_pv_battery_with_refinement_with_dispatches(
        pv_capacity_frame_builder,
        sizing_cfg,
        evaluator=evaluator,
    )
    dispatch = _best_dispatch_from_store(best, dispatch_store)
    best.update(
        {
            "scenario": scenario_key,
            "optimization_outage_mode": "city_outage",
            "optimization_voll_sgd_per_kwh": sizing_cfg["active_scenario"]["voll_sgd_per_kwh"],
            "evaluation_voll_sgd_per_kwh": sizing_cfg["active_scenario"]["voll_sgd_per_kwh"],
        }
    )
    return pd.DataFrame([best]), dispatch, summary, dispatch_store
