from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from .resilience_metrics import DEFAULT_TOLERANCE


def _total_load(row: Any, cooling_cop: float, heating_cop: float) -> float:
    heating_load = float(getattr(row, "heating_load_kwh_th", 0.0))
    return (
        float(row.electric_load_kwh)
        + float(row.cooling_load_kwh_th) / max(cooling_cop, 1e-9)
        + heating_load / max(heating_cop, 1e-9)
    )


def _degradation_penalty_sgd_per_kwh(sizing: dict[str, float], config: dict[str, Any]) -> float:
    batt_cap = float(sizing["battery_energy_kwh"])
    if batt_cap <= 0.0 or not bool(config.get("dispatch", {}).get("include_degradation_penalty", True)):
        return 0.0
    batt_power = float(sizing["battery_power_kw"])
    econ = config["economics"]
    battery_capex = (
        batt_cap * float(econ["battery_capex_sgd_per_kwh"])
        + batt_power * float(econ["battery_power_capex_sgd_per_kw"])
    )
    cycle_life = max(float(config["battery"]["rated_cycle_life_to_eol"]), 1e-9)
    return float(battery_capex / max(2.0 * batt_cap * cycle_life, 1e-9))


def _allocate_battery_discharge_by_origin(
    delivered_kwh: float,
    eta: float,
    soc_pv_kwh: float,
    soc_grid_kwh: float,
    soc_initial_kwh: float,
) -> tuple[float, float, float, float, float, float]:
    delivered = max(0.0, float(delivered_kwh))
    dc_draw = delivered / max(float(eta), 1e-9)
    eta_value = max(float(eta), 1e-9)
    pv = max(0.0, float(soc_pv_kwh))
    grid = max(0.0, float(soc_grid_kwh))
    initial = max(0.0, float(soc_initial_kwh))
    total_soc = pv + grid + initial
    if delivered <= 0.0 or total_soc <= 1e-12:
        return (
            pv,
            grid,
            initial,
            0.0,
            0.0,
            0.0,
        )
    actual_dc_draw = min(dc_draw, total_soc)
    draw_pv = min(pv, actual_dc_draw)
    remaining = actual_dc_draw - draw_pv
    draw_grid = min(grid, remaining)
    remaining -= draw_grid
    draw_initial = min(initial, remaining)
    return (
        max(0.0, pv - draw_pv),
        max(0.0, grid - draw_grid),
        max(0.0, initial - draw_initial),
        draw_pv * eta_value,
        draw_grid * eta_value,
        draw_initial * eta_value,
    )


def _reconcile_battery_origin_soc(
    target_soc_kwh: float,
    soc_pv_kwh: float,
    soc_grid_kwh: float,
    soc_initial_kwh: float,
) -> tuple[float, float, float]:
    target = max(0.0, float(target_soc_kwh))
    pv = max(0.0, float(soc_pv_kwh))
    grid = max(0.0, float(soc_grid_kwh))
    initial = max(0.0, float(soc_initial_kwh))
    total = pv + grid + initial
    if total <= 1e-12:
        return 0.0, 0.0, target
    # Equivalent to numpy.isclose(total, target, rtol=1e-5, atol=1e-9),
    # without allocating scalar NumPy intermediates on every simulated hour.
    if total == target or (isfinite(target) and abs(total - target) <= 1e-9 + 1e-5 * abs(target)):
        return pv, grid, initial
    if total > target:
        scale = target / total
        return pv * scale, grid * scale, initial * scale
    return pv, grid, initial + (target - total)


def solve_dispatch_greedy(
    frame: pd.DataFrame,
    sizing: dict[str, float],
    config: dict[str, Any],
    *,
    metrics_only: bool = False,
    collect_totals: bool = False,
    streaming_only: bool = False,
) -> pd.DataFrame:
    """Run unchanged dispatch with optional lightweight outputs and streaming totals.

    With collect_totals, the returned frame's energy_totals attribute contains
    horizon totals; no extra hourly columns or full dispatch rows are retained.

    ``streaming_only`` is the no-hourly-output variant used by capacity-search
    workflows.  It requires ``metrics_only=True`` and records reliability
    totals in ``reliability_totals`` instead of retaining one unmet-energy row
    per timestep.  The dispatch decisions and the energy totals are otherwise
    exactly the same as the normal greedy path.
    """
    if streaming_only and not metrics_only:
        raise ValueError("streaming_only requires metrics_only=True")
    technical = config["technical"]
    cooling_cop = float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0)))
    heating_cop = float(technical.get("heating_cop", 3.5))
    batt_cfg = config["battery"]
    eta = float(batt_cfg["roundtrip_efficiency"]) ** 0.5
    batt_cap = float(sizing["battery_energy_kwh"])
    batt_power = float(sizing["battery_power_kw"])
    soc = batt_cap * float(batt_cfg["initial_soc_fraction"])
    soc_pv = 0.0
    soc_grid = 0.0
    soc_initial = soc
    soc_min = batt_cap * float(batt_cfg["soc_min_fraction"])
    soc_max = batt_cap * float(batt_cfg["soc_max_fraction"])
    reserve_fraction = batt_cfg.get("normal_operation_reserve_fraction")
    pv_export_enabled = bool(config.get("pv", {}).get("export_enabled", False))
    reserve_soc = soc_min
    if reserve_fraction is not None:
        reserve_fraction = float(reserve_fraction)
        if reserve_fraction < 0.0:
            raise ValueError("normal_operation_reserve_fraction must be non-negative")
        if reserve_fraction > float(batt_cfg["soc_max_fraction"]):
            raise ValueError("normal_operation_reserve_fraction must not exceed soc_max_fraction")
        reserve_soc = max(soc_min, batt_cap * reserve_fraction)
    low_price = float(frame["tou_sgd_per_kwh"].quantile(0.30))
    high_price = float(frame["tou_sgd_per_kwh"].quantile(0.75))
    rows: list[dict[str, float]] = []
    metric_rows: list[tuple[float, float]] = []
    sum_load = sum_pv = sum_pv_load = sum_pv_charge = sum_curtail = 0.0
    sum_grid = sum_grid_cost = sum_export = sum_export_revenue = 0.0
    sum_charge = sum_discharge = sum_soc = sum_unmet = 0.0
    sum_pv_discharge = sum_grid_discharge = sum_initial_discharge = 0.0
    sum_grid_load = sum_grid_charge = 0.0
    sum_thresholded_unmet = sum_outage_load = sum_outage_hours = 0.0

    for row in frame.itertuples(index=False):
        availability = float(row.grid_availability)
        outage_fraction = max(0.0, 1.0 - availability)
        price = float(row.tou_sgd_per_kwh)
        export_price = float(getattr(row, "pv_export_sgd_per_kwh", 0.0))
        pv = float(row.pv_kwh)
        total_load = _total_load(row, cooling_cop, heating_cop)

        outage_need = outage_fraction * total_load
        normal_need = availability * total_load
        pv_used = 0.0
        grid_import = 0.0
        batt_charge = 0.0
        batt_discharge = 0.0
        unmet = 0.0
        pv_to_load = 0.0
        pv_to_battery = 0.0
        grid_to_load = 0.0
        grid_to_battery = 0.0
        pv_export = 0.0
        discharge_from_pv = 0.0
        discharge_from_grid = 0.0
        discharge_from_initial = 0.0

        used = min(pv, outage_need)
        pv -= used
        pv_used += used
        pv_to_load += used
        outage_need -= used

        if batt_cap > 0 and batt_power > 0 and outage_need > 0:
            max_discharge = min((soc - soc_min) * eta, batt_power * max(outage_fraction, 1e-9), outage_need)
            discharge = max(0.0, max_discharge)
            (
                soc_pv,
                soc_grid,
                soc_initial,
                from_pv,
                from_grid,
                from_initial,
            ) = _allocate_battery_discharge_by_origin(discharge, eta, soc_pv, soc_grid, soc_initial)
            soc = soc_pv + soc_grid + soc_initial
            discharge_from_pv += from_pv
            discharge_from_grid += from_grid
            discharge_from_initial += from_initial
            batt_discharge += discharge
            outage_need -= discharge

        unmet += max(0.0, outage_need)

        used = min(pv, normal_need)
        pv -= used
        pv_used += used
        pv_to_load += used
        normal_need -= used

        if price > high_price and batt_cap > 0 and batt_power > 0 and normal_need > 0:
            max_discharge = min((soc - reserve_soc) * eta, batt_power * availability, normal_need)
            discharge = max(0.0, max_discharge)
            (
                soc_pv,
                soc_grid,
                soc_initial,
                from_pv,
                from_grid,
                from_initial,
            ) = _allocate_battery_discharge_by_origin(discharge, eta, soc_pv, soc_grid, soc_initial)
            soc = soc_pv + soc_grid + soc_initial
            discharge_from_pv += from_pv
            discharge_from_grid += from_grid
            discharge_from_initial += from_initial
            batt_discharge += discharge
            normal_need -= discharge

        grid_to_load = max(0.0, normal_need)
        grid_import += grid_to_load

        if batt_cap > 0 and batt_power > 0 and pv > 0:
            charge_room_ac = max(0.0, (soc_max - soc) / max(eta, 1e-9))
            charge = min(pv, batt_power, charge_room_ac)
            soc += charge * eta
            soc_pv += charge * eta
            batt_charge += charge
            pv_to_battery += charge
            pv_used += charge
            pv -= charge

        if pv_export_enabled and pv > 0:
            pv_export = max(0.0, pv)
            pv = 0.0

        if price < low_price and availability > 0 and batt_cap > 0 and batt_power > 0:
            charge_room_ac = max(0.0, (soc_max - soc) / max(eta, 1e-9))
            charge = min(batt_power * availability, charge_room_ac)
            soc += charge * eta
            soc_grid += charge * eta
            batt_charge += charge
            grid_to_battery += charge
            grid_import += charge

        soc = min(soc_max, max(soc_min, soc))
        soc_pv, soc_grid, soc_initial = _reconcile_battery_origin_soc(soc, soc_pv, soc_grid, soc_initial)
        if collect_totals:
            sum_load += total_load
            sum_pv += float(row.pv_kwh)
            sum_pv_load += pv_to_load
            sum_pv_charge += pv_to_battery
            sum_curtail += max(0.0, pv)
            sum_grid += grid_import
            sum_grid_cost += grid_import * price
            sum_export += pv_export
            sum_export_revenue += pv_export * export_price
            sum_charge += batt_charge
            sum_discharge += batt_discharge
            sum_soc += soc
            sum_unmet += unmet
            sum_pv_discharge += discharge_from_pv
            sum_grid_discharge += discharge_from_grid
            sum_initial_discharge += discharge_from_initial
            sum_grid_load += grid_to_load
            sum_grid_charge += grid_to_battery
        if streaming_only:
            # The established reliability definition discards only sub-tolerance
            # unmet energy.  Keep this separate from raw unmet energy, which is
            # intentionally used by the economic VoLL calculation above.
            sum_thresholded_unmet += unmet if unmet > DEFAULT_TOLERANCE else 0.0
            sum_outage_load += outage_fraction * total_load
            sum_outage_hours += outage_fraction
        if metrics_only:
            if not streaming_only:
                metric_rows.append((float(row.hour_index), unmet))
            continue
        rows.append(
            {
                "hour_index": float(row.hour_index),
                "grid_availability": availability,
                "outage_fraction": outage_fraction,
                "tou_sgd_per_kwh": price,
                "pv_export_sgd_per_kwh": export_price,
                "total_load_e_equiv_kwh": total_load,
                "pv_available_kwh": float(row.pv_kwh),
                "pv_used_kwh": pv_used,
                "pv_export_kwh": pv_export,
                "pv_curtailed_kwh": max(0.0, pv),
                "grid_import_kwh": grid_import,
                "battery_charge_kwh": batt_charge,
                "battery_discharge_kwh": batt_discharge,
                "pv_to_load_kwh": pv_to_load,
                "pv_to_battery_kwh": pv_to_battery,
                "grid_to_load_kwh": grid_to_load,
                "grid_to_battery_kwh": grid_to_battery,
                "battery_charge_from_pv_kwh": pv_to_battery,
                "battery_charge_from_grid_kwh": grid_to_battery,
                "battery_discharge_to_load_kwh": batt_discharge,
                "battery_discharge_from_pv_kwh": discharge_from_pv,
                "battery_discharge_from_grid_kwh": discharge_from_grid,
                "battery_discharge_from_initial_kwh": discharge_from_initial,
                "pv_self_served_load_kwh": pv_to_load + discharge_from_pv,
                "battery_soc_kwh": soc,
                "battery_capacity_kwh": batt_cap,
                "unmet_load_kwh_equiv": unmet,
            }
        )
    if metrics_only:
        dispatch = (
            pd.DataFrame(metric_rows, columns=["hour_index", "unmet_load_kwh_equiv"])
            if not streaming_only else pd.DataFrame(columns=["hour_index", "unmet_load_kwh_equiv"])
        )
    else:
        dispatch = pd.DataFrame(rows)
    if dispatch.isna().any().any():
        raise ValueError("Dispatch contains missing values")
    dispatch["dispatch_backend"] = "greedy"
    if collect_totals:
        dispatch.attrs["energy_totals"] = {
            "dispatch_timesteps": len(frame),
            "load_kwh_total": sum_load,
            "pv_available_kwh_total": sum_pv,
            "pv_to_load_kwh_total": sum_pv_load,
            "pv_to_battery_kwh_total": sum_pv_charge,
            "pv_curtailed_kwh_total": sum_curtail,
            "grid_import_kwh_total": sum_grid,
            "grid_cost_sgd_total": sum_grid_cost,
            "pv_export_kwh_total": sum_export,
            "pv_export_revenue_sgd_total": sum_export_revenue,
            "battery_charge_kwh_total": sum_charge,
            "battery_discharge_kwh_total": sum_discharge,
            "battery_soc_kwh_sum": sum_soc,
            "unserved_energy_kwh_total": sum_unmet,
            "battery_discharge_from_pv_kwh_total": sum_pv_discharge,
            "battery_discharge_from_grid_kwh_total": sum_grid_discharge,
            "battery_discharge_from_initial_kwh_total": sum_initial_discharge,
            "grid_to_load_kwh_total": sum_grid_load,
            "grid_to_battery_kwh_total": sum_grid_charge,
            "battery_soc_initial_kwh": batt_cap * float(batt_cfg["initial_soc_fraction"]),
            "battery_soc_final_kwh": soc,
            "battery_soc_pv_final_kwh": soc_pv,
            "battery_soc_grid_final_kwh": soc_grid,
            "battery_soc_initial_origin_final_kwh": soc_initial,
        }
    if streaming_only:
        dispatch.attrs["reliability_totals"] = {
            "EUE_kwh_total": sum_thresholded_unmet,
            "outage_load_kwh_total": sum_outage_load,
            "outage_hours_total": sum_outage_hours,
            "LOLP": sum_thresholded_unmet / sum_outage_load if sum_outage_load > 0 else 0.0,
            "no_outage_load": bool(sum_outage_load <= 0.0),
        }
    if metrics_only:
        return dispatch
    dispatch["solver_status"] = "heuristic"
    dispatch["milp_objective_sgd"] = np.nan
    dispatch["degradation_penalty_sgd_per_kwh"] = _degradation_penalty_sgd_per_kwh(sizing, config)
    dispatch["simultaneous_charge_discharge_kwh"] = np.minimum(
        dispatch["battery_charge_kwh"].astype(float),
        dispatch["battery_discharge_kwh"].astype(float),
    )
    return dispatch


def _pulp_solver(config: dict[str, Any]) -> Any:
    try:
        import pulp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MILP dispatch requires the 'pulp' and 'highspy' packages. "
            "Install them with: python -m pip install pulp highspy"
        ) from exc
    solver_name = str(config.get("dispatch", {}).get("solver", "HiGHS")).lower()
    if solver_name != "highs":
        raise ValueError("Only the HiGHS solver is supported for MILP dispatch in this implementation.")
    try:
        return pulp.HiGHS(msg=False)
    except Exception as exc:  # pragma: no cover - depends on solver packaging
        raise RuntimeError(
            "Could not initialize PuLP HiGHS solver. Ensure highspy is installed."
        ) from exc


def solve_dispatch_milp(
    frame: pd.DataFrame,
    sizing: dict[str, float],
    config: dict[str, Any],
    *,
    initial_soc_kwh: float | None = None,
) -> pd.DataFrame:
    try:
        import pulp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MILP dispatch requires the 'pulp' and 'highspy' packages. "
            "Install them with: python -m pip install pulp highspy"
        ) from exc

    technical = config["technical"]
    cooling_cop = float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0)))
    heating_cop = float(technical.get("heating_cop", 3.5))
    batt_cfg = config["battery"]
    eta = float(batt_cfg["roundtrip_efficiency"]) ** 0.5
    batt_cap = float(sizing["battery_energy_kwh"])
    batt_power = float(sizing["battery_power_kw"])
    soc_min = batt_cap * float(batt_cfg["soc_min_fraction"])
    soc_max = batt_cap * float(batt_cfg["soc_max_fraction"])
    initial_soc = (
        batt_cap * float(batt_cfg["initial_soc_fraction"])
        if initial_soc_kwh is None else float(initial_soc_kwh)
    )
    if not isfinite(initial_soc) or not (soc_min - 1e-9 <= initial_soc <= soc_max + 1e-9):
        raise ValueError("MILP initial SOC must be finite and within the battery SOC limits")
    reserve_fraction = batt_cfg.get("normal_operation_reserve_fraction")
    pv_export_enabled = bool(config.get("pv", {}).get("export_enabled", False))
    reserve_soc = soc_min
    if reserve_fraction is not None:
        reserve_fraction = float(reserve_fraction)
        if reserve_fraction < 0.0:
            raise ValueError("normal_operation_reserve_fraction must be non-negative")
        if reserve_fraction > float(batt_cfg["soc_max_fraction"]):
            raise ValueError("normal_operation_reserve_fraction must not exceed soc_max_fraction")
        reserve_soc = max(soc_min, batt_cap * reserve_fraction)

    records = list(frame.itertuples(index=False))
    n = len(records)
    if n == 0:
        return pd.DataFrame()

    loads = [_total_load(row, cooling_cop, heating_cop) for row in records]
    availability = [float(row.grid_availability) for row in records]
    outage_fraction = [max(0.0, 1.0 - value) for value in availability]
    normal_load = [availability[idx] * loads[idx] for idx in range(n)]
    outage_load = [outage_fraction[idx] * loads[idx] for idx in range(n)]
    pv_available = [float(row.pv_kwh) for row in records]
    prices = [float(row.tou_sgd_per_kwh) for row in records]
    export_prices = [float(getattr(row, "pv_export_sgd_per_kwh", 0.0)) for row in records]
    voll = float(config["active_scenario"]["voll_sgd_per_kwh"])
    degradation_penalty = _degradation_penalty_sgd_per_kwh(sizing, config)

    model = pulp.LpProblem("pv_bess_dispatch", pulp.LpMinimize)
    pv_to_normal = [pulp.LpVariable(f"pv_to_normal_{t}", lowBound=0) for t in range(n)]
    pv_to_outage = [pulp.LpVariable(f"pv_to_outage_{t}", lowBound=0) for t in range(n)]
    pv_to_battery = [pulp.LpVariable(f"pv_to_battery_{t}", lowBound=0) for t in range(n)]
    pv_export = [pulp.LpVariable(f"pv_export_{t}", lowBound=0) for t in range(n)]
    pv_curtailed = [pulp.LpVariable(f"pv_curtailed_{t}", lowBound=0) for t in range(n)]
    grid_to_load = [pulp.LpVariable(f"grid_to_load_{t}", lowBound=0) for t in range(n)]
    grid_to_battery = [pulp.LpVariable(f"grid_to_battery_{t}", lowBound=0) for t in range(n)]
    discharge_normal = [pulp.LpVariable(f"battery_discharge_normal_{t}", lowBound=0) for t in range(n)]
    discharge_outage = [pulp.LpVariable(f"battery_discharge_outage_{t}", lowBound=0) for t in range(n)]
    unmet_normal = [pulp.LpVariable(f"unmet_normal_{t}", lowBound=0) for t in range(n)]
    unmet_outage = [pulp.LpVariable(f"unmet_outage_{t}", lowBound=0) for t in range(n)]
    soc = [pulp.LpVariable(f"battery_soc_{t}", lowBound=soc_min, upBound=soc_max) for t in range(n + 1)]

    model += soc[0] == initial_soc
    model += soc[n] >= initial_soc
    pv_outage_available = bool(technical.get("pv_during_outage_available", True))

    for t in range(n):
        charge = pv_to_battery[t] + grid_to_battery[t]
        discharge = discharge_normal[t] + discharge_outage[t]
        model += pv_to_normal[t] + pv_to_outage[t] + pv_to_battery[t] + pv_export[t] + pv_curtailed[t] == pv_available[t]
        if not pv_outage_available:
            model += pv_to_outage[t] == 0.0
        if not pv_export_enabled:
            model += pv_export[t] == 0.0
        model += normal_load[t] == pv_to_normal[t] + discharge_normal[t] + grid_to_load[t] + unmet_normal[t]
        model += outage_load[t] == pv_to_outage[t] + discharge_outage[t] + unmet_outage[t]
        model += grid_to_load[t] <= normal_load[t]
        model += grid_to_battery[t] <= availability[t] * batt_power
        model += charge <= batt_power
        model += discharge <= batt_power
        model += discharge_normal[t] <= availability[t] * batt_power
        model += discharge_outage[t] <= outage_fraction[t] * batt_power
        model += soc[t + 1] == soc[t] + eta * charge - discharge / max(eta, 1e-9)
        model += discharge / max(eta, 1e-9) <= soc[t] - soc_min
        model += discharge_normal[t] / max(eta, 1e-9) <= soc[t] - reserve_soc
        if batt_cap <= 0.0 or batt_power <= 0.0:
            model += charge == 0.0
            model += discharge == 0.0

    model += pulp.lpSum(
        prices[t] * (grid_to_load[t] + grid_to_battery[t])
        - export_prices[t] * pv_export[t]
        + voll * (unmet_normal[t] + unmet_outage[t])
        + degradation_penalty * (pv_to_battery[t] + grid_to_battery[t] + discharge_normal[t] + discharge_outage[t])
        for t in range(n)
    )

    status_code = model.solve(_pulp_solver(config))
    status = pulp.LpStatus.get(status_code, str(status_code))
    if status != "Optimal":
        raise RuntimeError(f"MILP dispatch failed with solver status: {status}")

    objective = float(pulp.value(model.objective) or 0.0)
    rows: list[dict[str, float | str]] = []
    soc_pv = 0.0
    soc_grid = 0.0
    soc_initial = initial_soc
    for t, row in enumerate(records):
        pv_to_load_value = float(pulp.value(pv_to_normal[t]) + pulp.value(pv_to_outage[t]))
        pv_to_outage_value = float(pulp.value(pv_to_outage[t]))
        pv_to_battery_value = float(pulp.value(pv_to_battery[t]))
        pv_export_value = float(pulp.value(pv_export[t]))
        grid_to_load_value = float(pulp.value(grid_to_load[t]))
        grid_to_battery_value = float(pulp.value(grid_to_battery[t]))
        charge_value = pv_to_battery_value + grid_to_battery_value
        discharge_value = float(pulp.value(discharge_normal[t]) + pulp.value(discharge_outage[t]))
        discharge_outage_value = float(pulp.value(discharge_outage[t]))
        pv_used = pv_to_load_value + pv_to_battery_value
        grid_import = grid_to_load_value + grid_to_battery_value
        unmet = float(pulp.value(unmet_normal[t]) + pulp.value(unmet_outage[t]))
        unmet_outage_value = float(pulp.value(unmet_outage[t]))
        (
            soc_pv,
            soc_grid,
            soc_initial,
            discharge_from_pv,
            discharge_from_grid,
            discharge_from_initial,
        ) = _allocate_battery_discharge_by_origin(discharge_value, eta, soc_pv, soc_grid, soc_initial)
        soc_pv += max(0.0, pv_to_battery_value) * eta
        soc_grid += max(0.0, grid_to_battery_value) * eta
        soc_next = float(pulp.value(soc[t + 1]))
        soc_pv, soc_grid, soc_initial = _reconcile_battery_origin_soc(soc_next, soc_pv, soc_grid, soc_initial)
        rows.append(
            {
                "hour_index": float(row.hour_index),
                "grid_availability": availability[t],
                "outage_fraction": outage_fraction[t],
                "tou_sgd_per_kwh": prices[t],
                "pv_export_sgd_per_kwh": export_prices[t],
                "total_load_e_equiv_kwh": loads[t],
                "pv_available_kwh": pv_available[t],
                "pv_used_kwh": max(0.0, pv_used),
                "pv_export_kwh": max(0.0, pv_export_value),
                "pv_curtailed_kwh": max(0.0, float(pulp.value(pv_curtailed[t]))),
                "grid_import_kwh": max(0.0, grid_import),
                "battery_charge_kwh": max(0.0, charge_value),
                "battery_discharge_kwh": max(0.0, discharge_value),
                "pv_to_load_kwh": max(0.0, pv_to_load_value),
                "pv_to_outage_kwh": max(0.0, pv_to_outage_value),
                "pv_to_battery_kwh": max(0.0, pv_to_battery_value),
                "grid_to_load_kwh": max(0.0, grid_to_load_value),
                "grid_to_battery_kwh": max(0.0, grid_to_battery_value),
                "battery_charge_from_pv_kwh": max(0.0, pv_to_battery_value),
                "battery_charge_from_grid_kwh": max(0.0, grid_to_battery_value),
                "battery_discharge_to_load_kwh": max(0.0, discharge_value),
                "battery_discharge_outage_kwh": max(0.0, discharge_outage_value),
                "battery_discharge_from_pv_kwh": max(0.0, discharge_from_pv),
                "battery_discharge_from_grid_kwh": max(0.0, discharge_from_grid),
                "battery_discharge_from_initial_kwh": max(0.0, discharge_from_initial),
                "pv_self_served_load_kwh": max(0.0, pv_to_load_value + discharge_from_pv),
                "battery_soc_kwh": soc_next,
                "battery_capacity_kwh": batt_cap,
                "unmet_load_kwh_equiv": max(0.0, unmet),
                "unmet_outage_kwh": max(0.0, unmet_outage_value),
                "dispatch_backend": "milp",
                "solver_status": status,
                "milp_objective_sgd": objective,
                "degradation_penalty_sgd_per_kwh": degradation_penalty,
                "simultaneous_charge_discharge_kwh": max(0.0, min(charge_value, discharge_value)),
            }
        )
    dispatch = pd.DataFrame(rows)
    if dispatch.isna().any().any():
        raise ValueError("MILP dispatch contains missing values")
    return dispatch


def solve_dispatch(frame: pd.DataFrame, sizing: dict[str, float], config: dict[str, Any]) -> pd.DataFrame:
    backend = str(config.get("dispatch", {}).get("backend", "auto")).lower()
    if backend == "auto":
        if int(config.get("analysis", {}).get("years", 1)) > 1:
            return solve_dispatch_greedy(frame, sizing, config)
        try:
            return solve_dispatch_milp(frame, sizing, config)
        except RuntimeError:
            return solve_dispatch_greedy(frame, sizing, config)
    if backend == "greedy":
        return solve_dispatch_greedy(frame, sizing, config)
    if backend == "milp":
        return solve_dispatch_milp(frame, sizing, config)
    raise ValueError("dispatch.backend must be one of: auto, greedy, milp")


def outage_free_frame(frame: pd.DataFrame) -> pd.DataFrame:
    no_outage = frame.copy()
    no_outage["grid_availability"] = 1.0
    no_outage["outage_fraction"] = 0.0
    return no_outage
