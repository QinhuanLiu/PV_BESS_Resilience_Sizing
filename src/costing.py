from __future__ import annotations

from typing import Any

import pandas as pd

from .degradation import annual_battery_soh_loss, annualized_battery_cost


def pv_annual_cost(pv_meta: dict[str, float | str], config: dict[str, Any]) -> dict[str, float]:
    pv_cfg = config["pv"]
    capacity = float(pv_meta["pv_capacity_kw"])
    capex = capacity * float(pv_cfg["capex_sgd_per_kw"])
    annualized = capex / max(float(pv_cfg["project_years"]), 1e-9)
    om = capex * float(pv_cfg["om_fraction_per_year"])
    result = {
        "pv_capacity_kw": capacity,
        "pv_capacity_mw": capacity / 1000.0,
        "pv_capex_sgd": float(capex),
        "annualized_pv_cost_sgd": float(annualized),
        "pv_om_cost_sgd": float(om),
        "pv_total_annual_cost_sgd": float(annualized + om),
    }
    for key in (
        "rooftop_area_m2",
        "pv_installed_area_m2",
        "pv_rooftop_coverage_fraction",
        "pv_rooftop_coverage_percent",
        "pv_rooftop_utilization_fraction",
        "pv_rooftop_utilization_percent",
        "pv_module_power_density_kw_per_m2",
    ):
        if key in pv_meta:
            result[key] = float(pv_meta[key])
    return result

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
