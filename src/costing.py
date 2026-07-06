from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .degradation import annual_battery_soh_loss, annualized_battery_cost


def pv_annual_cost(pv_meta: dict[str, float | str], config: dict[str, Any]) -> dict[str, float]:
    pv_cfg = config["pv"]
    capacity = float(pv_meta["pv_capacity_kwp"])
    capex = capacity * float(pv_cfg["capex_sgd_per_kwp"])
    annualized = capex / max(float(pv_cfg["project_years"]), 1e-9)
    om = capex * float(pv_cfg["om_fraction_per_year"])
    result = {
        "pv_capacity_kwp": capacity,
        "pv_capacity_mwp": capacity / 1000.0,
        "pv_capex_sgd": float(capex),
        "annualized_pv_cost_sgd": float(annualized),
        "pv_om_cost_sgd": float(om),
        "pv_total_annual_cost_sgd": float(annualized + om),
    }
    for key in (
        "rooftop_area_m2",
        "pv_installed_area_m2",
        "pv_rooftop_utilization_fraction",
        "pv_rooftop_utilization_percent",
        "pv_module_power_density_kwp_per_m2",
    ):
        if key in pv_meta:
            result[key] = float(pv_meta[key])
    return result


def baseline_no_battery_cost(frame: pd.DataFrame, pv_meta: dict[str, float | str], config: dict[str, Any]) -> dict[str, float]:
    technical = config["technical"]
    cooling_cop = float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0)))
    heating_cop = float(technical.get("heating_cop", 3.5))
    heating = frame["heating_load_kwh_th"].to_numpy() if "heating_load_kwh_th" in frame.columns else 0.0
    load = (
        frame["electric_load_kwh"].to_numpy()
        + frame["cooling_load_kwh_th"].to_numpy() / max(cooling_cop, 1e-9)
        + heating / max(heating_cop, 1e-9)
    )
    availability = frame["grid_availability"].to_numpy()
    pv = frame["pv_kwh"].to_numpy()
    price = frame["tou_sgd_per_kwh"].to_numpy()
    voll = float(config["active_scenario"]["voll_sgd_per_kwh"])
    outage_need = (1.0 - availability) * load
    normal_need = availability * load
    pv_to_outage = np.minimum(pv, outage_need)
    remaining_pv = pv - pv_to_outage
    lost = outage_need - pv_to_outage
    pv_to_normal = np.minimum(remaining_pv, normal_need)
    grid = normal_need - pv_to_normal
    pv_cost = pv_annual_cost(pv_meta, config)
    grid_cost = float((grid * price).sum())
    lost_cost = float(lost.sum() * voll)
    private_cost = grid_cost + pv_cost["pv_total_annual_cost_sgd"]
    return {
        **pv_cost,
        "baseline_grid_energy_kwh": float(grid.sum()),
        "baseline_grid_cost_sgd": grid_cost,
        "baseline_private_annual_cost_sgd": float(private_cost),
        "baseline_unserved_energy_kwh_per_year": float(lost.sum()),
        "baseline_unserved_energy_cost_sgd_per_year": lost_cost,
        "baseline_system_annual_cost_sgd": float(private_cost + lost_cost),
    }


def pv_bess_annual_cost(
    frame: pd.DataFrame,
    dispatch: pd.DataFrame,
    sizing: dict[str, float],
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
) -> dict[str, float]:
    pv_cost = pv_annual_cost(pv_meta, config)
    degradation = annual_battery_soh_loss(dispatch, sizing["battery_energy_kwh"], config)
    battery_cost = annualized_battery_cost(
        sizing["battery_energy_kwh"],
        sizing["battery_power_kw"],
        degradation["annual_soh_loss"],
        config,
    )
    grid_cost = float((dispatch["grid_import_kwh"] * dispatch["tou_sgd_per_kwh"]).sum())
    lost = float(dispatch["unmet_load_kwh_equiv"].sum())
    lost_cost = lost * float(config["active_scenario"]["voll_sgd_per_kwh"])
    private_cost = (
        grid_cost
        + pv_cost["pv_total_annual_cost_sgd"]
        + battery_cost["annualized_battery_cost_sgd"]
        + battery_cost["battery_om_cost_sgd"]
    )
    return {
        **pv_cost,
        **degradation,
        **battery_cost,
        "pv_bess_grid_energy_kwh": float(dispatch["grid_import_kwh"].sum()),
        "pv_bess_grid_cost_sgd": grid_cost,
        "pv_bess_private_annual_cost_sgd": float(private_cost),
        "unserved_energy_kwh_per_year": lost,
        "unserved_energy_cost_sgd_per_year": lost_cost,
        "pv_bess_system_annual_cost_sgd": float(private_cost + lost_cost),
    }
