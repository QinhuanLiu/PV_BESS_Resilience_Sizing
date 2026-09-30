from __future__ import annotations

from typing import Any

import pandas as pd

from .degradation import (
    annual_battery_soh_loss, annual_battery_soh_loss_from_statistics,
    annualized_battery_cost, capital_recovery_factor,
)


def pv_annual_cost(pv_meta: dict[str, float | str], config: dict[str, Any]) -> dict[str, float]:
    pv_cfg = config["pv"]
    discount_rate = float(config.get("economics", {}).get("discount_rate", 0.0))
    capacity = float(pv_meta["pv_capacity_kw"])
    capex = capacity * float(pv_cfg["capex_sgd_per_kw"])
    project_years = float(pv_cfg["project_years"])
    annualization_factor = capital_recovery_factor(project_years, discount_rate)
    annualized = capex * annualization_factor
    om = capex * float(pv_cfg["om_fraction_per_year"])
    result = {
        "pv_capacity_kw": capacity,
        "pv_capacity_mw": capacity / 1000.0,
        "discount_rate": discount_rate,
        "pv_project_years": project_years,
        "pv_capex_sgd": float(capex),
        "pv_capital_recovery_factor": float(annualization_factor),
        "annualized_pv_cost_sgd": float(annualized),
        "pv_om_cost_sgd": float(om),
        "pv_total_annual_cost_sgd": float(annualized + om),
    }
    for key in (
        "rooftop_area_m2",
        "pv_installed_area_m2",
        "pv_rooftop_area_utilization_fraction",
        "pv_rooftop_capacity_limit_kw",
        "pv_rooftop_coverage_fraction",
        "pv_rooftop_coverage_percent",
        "pv_rooftop_utilization_fraction",
        "pv_rooftop_utilization_percent",
        "pv_module_power_density_kw_per_m2",
    ):
        if key in pv_meta:
            result[key] = float(pv_meta[key])
    if "pv_rooftop_area_limit_enabled" in pv_meta:
        result["pv_rooftop_area_limit_enabled"] = bool(pv_meta["pv_rooftop_area_limit_enabled"])
    return result

def pv_bess_annual_cost(
    frame: pd.DataFrame,
    dispatch: pd.DataFrame,
    sizing: dict[str, float],
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
) -> dict[str, float]:
    degradation = annual_battery_soh_loss(dispatch, sizing["battery_energy_kwh"], config)
    grid_energy_total = float(dispatch["grid_import_kwh"].sum())
    grid_cost_total = float((dispatch["grid_import_kwh"] * dispatch["tou_sgd_per_kwh"]).sum())
    if "pv_export_kwh" in dispatch.columns:
        export_energy_total = float(dispatch["pv_export_kwh"].sum())
        export_revenue_total = float((dispatch["pv_export_kwh"] * dispatch.get("pv_export_sgd_per_kwh", 0.0)).sum())
    else:
        export_energy_total = 0.0
        export_revenue_total = 0.0
    totals = {
        "grid_import_kwh_total": grid_energy_total,
        "grid_cost_sgd_total": grid_cost_total,
        "pv_export_kwh_total": export_energy_total,
        "pv_export_revenue_sgd_total": export_revenue_total,
        "unserved_energy_kwh_total": float(dispatch["unmet_load_kwh_equiv"].sum()),
    }
    return _annual_cost_from_statistics(totals, degradation, sizing, pv_meta, config)


def pv_bess_annual_cost_from_totals(
    totals: dict[str, float],
    sizing: dict[str, float],
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
) -> dict[str, float]:
    """Compute the same costs without retaining a full hourly dispatch table."""
    capacity = sizing["battery_energy_kwh"]
    count = int(totals["dispatch_timesteps"])
    mean_soc = totals["battery_soc_kwh_sum"] / (count * capacity) if count and capacity > 0 else 0.0
    degradation = annual_battery_soh_loss_from_statistics(
        totals["battery_charge_kwh_total"] + totals["battery_discharge_kwh_total"],
        mean_soc, capacity, config, timestep_count=count,
    )
    return _annual_cost_from_statistics(totals, degradation, sizing, pv_meta, config)


def _annual_cost_from_statistics(
    totals: dict[str, float],
    degradation: dict[str, float],
    sizing: dict[str, float],
    pv_meta: dict[str, float | str],
    config: dict[str, Any],
) -> dict[str, float]:
    analysis_years = max(1.0, float(config.get("analysis", {}).get("years", 1)))
    pv_cost = pv_annual_cost(pv_meta, config)
    battery_cost = annualized_battery_cost(
        sizing["battery_energy_kwh"], sizing["battery_power_kw"], degradation["annual_soh_loss"], config,
    )
    grid_energy_total = float(totals["grid_import_kwh_total"])
    grid_cost_total = float(totals["grid_cost_sgd_total"])
    export_energy_total = float(totals["pv_export_kwh_total"])
    export_revenue_total = float(totals["pv_export_revenue_sgd_total"])
    grid_cost = grid_cost_total / analysis_years
    export_energy = export_energy_total / analysis_years
    export_revenue = export_revenue_total / analysis_years
    net_grid_cost = grid_cost - export_revenue
    lost_total = float(totals["unserved_energy_kwh_total"])
    lost = lost_total / analysis_years
    lost_cost = lost * float(config["active_scenario"]["voll_sgd_per_kwh"])
    private_cost = (
        net_grid_cost
        + pv_cost["pv_total_annual_cost_sgd"]
        + battery_cost["annualized_battery_cost_sgd"]
        + battery_cost["battery_om_cost_sgd"]
    )
    return {
        **pv_cost,
        **degradation,
        **battery_cost,
        "analysis_years": float(analysis_years),
        "pv_bess_grid_energy_kwh_total": grid_energy_total,
        "pv_bess_grid_cost_sgd_total": grid_cost_total,
        "pv_export_energy_kwh_total": export_energy_total,
        "pv_export_revenue_sgd_total": export_revenue_total,
        "unserved_energy_kwh_total": lost_total,
        "unserved_energy_cost_sgd_total": lost_cost * analysis_years,
        "pv_bess_grid_energy_kwh": grid_energy_total / analysis_years,
        "pv_bess_grid_cost_sgd": grid_cost,
        "pv_export_energy_kwh_per_year": export_energy,
        "pv_export_revenue_sgd_per_year": export_revenue,
        "pv_bess_net_grid_cost_sgd": net_grid_cost,
        "pv_bess_private_annual_cost_sgd": float(private_cost),
        "unserved_energy_kwh_per_year": lost,
        "unserved_energy_cost_sgd_per_year": lost_cost,
        "pv_bess_system_annual_cost_sgd": float(private_cost + lost_cost),
    }
