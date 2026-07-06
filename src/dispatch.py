from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def solve_dispatch_greedy(frame: pd.DataFrame, sizing: dict[str, float], config: dict[str, Any]) -> pd.DataFrame:
    technical = config["technical"]
    cooling_cop = float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0)))
    heating_cop = float(technical.get("heating_cop", 3.5))
    batt_cfg = config["battery"]
    eta = float(batt_cfg["roundtrip_efficiency"]) ** 0.5
    batt_cap = float(sizing["battery_energy_kwh"])
    batt_power = float(sizing["battery_power_kw"])
    soc = batt_cap * float(batt_cfg["initial_soc_fraction"])
    soc_min = batt_cap * float(batt_cfg["soc_min_fraction"])
    soc_max = batt_cap * float(batt_cfg["soc_max_fraction"])
    low_price = float(frame["tou_sgd_per_kwh"].quantile(0.30))
    high_price = float(frame["tou_sgd_per_kwh"].quantile(0.75))
    rows: list[dict[str, float]] = []

    for row in frame.itertuples(index=False):
        availability = float(row.grid_availability)
        outage_fraction = max(0.0, 1.0 - availability)
        price = float(row.tou_sgd_per_kwh)
        pv = float(row.pv_kwh)
        heating_load = float(getattr(row, "heating_load_kwh_th", 0.0))
        total_load = (
            float(row.electric_load_kwh)
            + float(row.cooling_load_kwh_th) / max(cooling_cop, 1e-9)
            + heating_load / max(heating_cop, 1e-9)
        )

        outage_need = outage_fraction * total_load
        normal_need = availability * total_load
        pv_used = 0.0
        grid_import = 0.0
        batt_charge = 0.0
        batt_discharge = 0.0
        unmet = 0.0

        used = min(pv, outage_need)
        pv -= used
        pv_used += used
        outage_need -= used

        if batt_cap > 0 and batt_power > 0 and outage_need > 0:
            max_discharge = min((soc - soc_min) * eta, batt_power * max(outage_fraction, 1e-9), outage_need)
            discharge = max(0.0, max_discharge)
            soc -= discharge / max(eta, 1e-9)
            batt_discharge += discharge
            outage_need -= discharge

        unmet += max(0.0, outage_need)

        used = min(pv, normal_need)
        pv -= used
        pv_used += used
        normal_need -= used

        if price >= high_price and batt_cap > 0 and batt_power > 0 and normal_need > 0:
            max_discharge = min((soc - soc_min) * eta, batt_power * availability, normal_need)
            discharge = max(0.0, max_discharge)
            soc -= discharge / max(eta, 1e-9)
            batt_discharge += discharge
            normal_need -= discharge

        grid_import += max(0.0, normal_need)

        if batt_cap > 0 and batt_power > 0 and pv > 0:
            charge_room_ac = max(0.0, (soc_max - soc) / max(eta, 1e-9))
            charge = min(pv, batt_power, charge_room_ac)
            soc += charge * eta
            batt_charge += charge
            pv_used += charge
            pv -= charge

        if price <= low_price and availability > 0 and batt_cap > 0 and batt_power > 0:
            charge_room_ac = max(0.0, (soc_max - soc) / max(eta, 1e-9))
            charge = min(batt_power * availability, charge_room_ac)
            soc += charge * eta
            batt_charge += charge
            grid_import += charge

        soc = min(soc_max, max(soc_min, soc))
        rows.append(
            {
                "hour_index": float(row.hour_index),
                "grid_availability": availability,
                "outage_fraction": outage_fraction,
                "tou_sgd_per_kwh": price,
                "total_load_e_equiv_kwh": total_load,
                "pv_available_kwh": float(row.pv_kwh),
                "pv_used_kwh": pv_used,
                "pv_curtailed_kwh": max(0.0, pv),
                "grid_import_kwh": grid_import,
                "battery_charge_kwh": batt_charge,
                "battery_discharge_kwh": batt_discharge,
                "battery_soc_kwh": soc,
                "battery_capacity_kwh": batt_cap,
                "unmet_load_kwh_equiv": unmet,
            }
        )
    dispatch = pd.DataFrame(rows)
    if dispatch.isna().any().any():
        raise ValueError("Dispatch contains missing values")
    return dispatch


def outage_free_frame(frame: pd.DataFrame) -> pd.DataFrame:
    no_outage = frame.copy()
    no_outage["grid_availability"] = 1.0
    no_outage["outage_fraction"] = 0.0
    return no_outage
