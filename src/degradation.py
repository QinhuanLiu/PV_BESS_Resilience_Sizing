from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def capital_recovery_factor(years: float, discount_rate: float) -> float:
    """Convert a capital cost to an equivalent annual cost factor."""
    years_f = float(years)
    rate = float(discount_rate)
    if years_f <= 0.0:
        raise ValueError(f"Annualization years must be > 0, got {years}")
    if rate < 0.0:
        raise ValueError(f"discount_rate must be >= 0, got {discount_rate}")
    if rate <= 1e-12:
        return 1.0 / years_f
    growth = (1.0 + rate) ** years_f
    return rate * growth / (growth - 1.0)


def annual_battery_soh_loss(dispatch: pd.DataFrame, capacity_kwh: float, config: dict[str, Any]) -> dict[str, float]:
    if capacity_kwh <= 0 or dispatch.empty:
        return annual_battery_soh_loss_from_statistics(0.0, 0.0, capacity_kwh, config, timestep_count=0)
    throughput = float(dispatch["battery_charge_kwh"].sum() + dispatch["battery_discharge_kwh"].sum())
    soc_fraction = dispatch["battery_soc_kwh"].to_numpy() / max(capacity_kwh, 1e-9)
    return annual_battery_soh_loss_from_statistics(
        throughput, float(np.nanmean(soc_fraction)), capacity_kwh, config, timestep_count=len(dispatch),
    )


def annual_battery_soh_loss_from_statistics(
    throughput_kwh: float,
    mean_soc_fraction: float,
    capacity_kwh: float,
    config: dict[str, Any],
    *,
    timestep_count: int,
) -> dict[str, float]:
    """Shared lifetime formula for hourly dispatch and streaming statistics."""
    analysis_years = max(1.0, float(config.get("analysis", {}).get("years", 1)))
    if capacity_kwh <= 0 or timestep_count == 0:
        return {
            "annual_soh_loss": 0.0,
            "cycle_aging_loss": 0.0,
            "calendar_aging_loss": 0.0,
            "equivalent_full_cycles": 0.0,
            "equivalent_full_cycles_total": 0.0,
            "analysis_years": analysis_years,
        }
    total_equivalent_full_cycles = 0.5 * float(throughput_kwh) / max(capacity_kwh, 1e-9)
    equivalent_full_cycles = total_equivalent_full_cycles / analysis_years
    usable_window = 1.0 - float(config["battery"]["eol_soh_fraction"])
    cycle_loss = usable_window * equivalent_full_cycles / max(float(config["battery"]["rated_cycle_life_to_eol"]), 1e-9)
    mean_soc = float(np.clip(mean_soc_fraction, 0.0, 1.0))
    calendar_loss = float(config["battery"]["calendar_loss_per_year"]) * max(mean_soc / 0.5, 0.1) ** 1.2
    return {
        "annual_soh_loss": float(cycle_loss + calendar_loss),
        "cycle_aging_loss": float(cycle_loss),
        "calendar_aging_loss": float(calendar_loss),
        "equivalent_full_cycles": float(equivalent_full_cycles),
        "equivalent_full_cycles_total": float(total_equivalent_full_cycles),
        "analysis_years": analysis_years,
    }


def replacement_years(annual_soh_loss: float, config: dict[str, Any]) -> float:
    usable_window = 1.0 - float(config["battery"]["eol_soh_fraction"])
    max_years = float(config["battery"].get("max_replacement_years", 30.0))
    if annual_soh_loss <= 1e-12:
        return max_years
    return min(max_years, usable_window / annual_soh_loss)


def annualized_battery_cost(
    battery_energy_kwh: float,
    battery_power_kw: float,
    annual_soh_loss: float,
    config: dict[str, Any],
) -> dict[str, float]:
    econ = config["economics"]
    capex = (
        float(battery_energy_kwh) * float(econ["battery_capex_sgd_per_kwh"])
        + float(battery_power_kw) * float(econ["battery_power_capex_sgd_per_kw"])
    )
    years = replacement_years(annual_soh_loss, config)
    discount_rate = float(econ.get("discount_rate", 0.0))
    annualization_factor = capital_recovery_factor(years, discount_rate)
    return {
        "battery_capex_sgd_per_kwh": float(econ["battery_capex_sgd_per_kwh"]),
        "battery_power_capex_sgd_per_kw": float(econ["battery_power_capex_sgd_per_kw"]),
        "battery_om_fraction_per_year": float(econ["battery_om_fraction_per_year"]),
        "discount_rate": discount_rate,
        "battery_capex_sgd": float(capex),
        "battery_replacement_years": float(years),
        "battery_capital_recovery_factor": float(annualization_factor),
        "annualized_battery_cost_sgd": float(capex * annualization_factor),
        "battery_om_cost_sgd": float(capex * float(econ["battery_om_fraction_per_year"])),
    }
