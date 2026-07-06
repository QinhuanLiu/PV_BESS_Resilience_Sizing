from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def annual_battery_soh_loss(dispatch: pd.DataFrame, capacity_kwh: float, config: dict[str, Any]) -> dict[str, float]:
    if capacity_kwh <= 0 or dispatch.empty:
        return {
            "annual_soh_loss": 0.0,
            "cycle_aging_loss": 0.0,
            "calendar_aging_loss": 0.0,
            "equivalent_full_cycles": 0.0,
        }
    charge = dispatch["battery_charge_kwh"].to_numpy()
    discharge = dispatch["battery_discharge_kwh"].to_numpy()
    throughput = float(charge.sum() + discharge.sum())
    equivalent_full_cycles = 0.5 * throughput / max(capacity_kwh, 1e-9)
    usable_window = 1.0 - float(config["battery"]["eol_soh_fraction"])
    cycle_loss = usable_window * equivalent_full_cycles / max(float(config["battery"]["rated_cycle_life_to_eol"]), 1e-9)
    soc_fraction = dispatch["battery_soc_kwh"].to_numpy() / max(capacity_kwh, 1e-9)
    mean_soc = float(np.clip(np.nanmean(soc_fraction), 0.0, 1.0))
    calendar_loss = float(config["battery"]["calendar_loss_per_year"]) * max(mean_soc / 0.5, 0.1) ** 1.2
    return {
        "annual_soh_loss": float(cycle_loss + calendar_loss),
        "cycle_aging_loss": float(cycle_loss),
        "calendar_aging_loss": float(calendar_loss),
        "equivalent_full_cycles": float(equivalent_full_cycles),
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
    return {
        "battery_capex_sgd_per_kwh": float(econ["battery_capex_sgd_per_kwh"]),
        "battery_power_capex_sgd_per_kw": float(econ["battery_power_capex_sgd_per_kw"]),
        "battery_om_fraction_per_year": float(econ["battery_om_fraction_per_year"]),
        "battery_capex_sgd": float(capex),
        "battery_replacement_years": float(years),
        "annualized_battery_cost_sgd": float(capex / max(years, 1e-9)),
        "battery_om_cost_sgd": float(capex * float(econ["battery_om_fraction_per_year"])),
    }
