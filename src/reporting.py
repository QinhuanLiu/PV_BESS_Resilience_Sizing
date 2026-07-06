from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _effective_battery_economics(config: dict[str, Any], scenario_key: str) -> dict[str, Any]:
    econ = dict(config["economics"])
    econ.update(config["scenarios"][scenario_key].get("battery_economics", {}))
    return econ


def parameter_sources_table(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pv = config["pv"]
    rows.extend(
        [
            {
                "parameter": "PV CAPEX",
                "value": pv["capex_sgd_per_kwp"],
                "unit": "SGD/kWp",
                "source_or_assumption": "EMA Singapore rooftop PV factsheet range S$1,000-1,600/kWp; central value used.",
            },
            {
                "parameter": "PV O&M",
                "value": pv["om_fraction_per_year"],
                "unit": "fraction/year",
                "source_or_assumption": "EMA factsheet recurring rooftop PV O&M 1-2% of system cost; central value used.",
            },
            {
                "parameter": "PV lifetime",
                "value": pv["project_years"],
                "unit": "years",
                "source_or_assumption": "Common rooftop PV project-life assumption.",
            },
        ]
    )
    for key in config["scenario_order"]:
        sc = config["scenarios"][key]
        batt = _effective_battery_economics(config, key)
        batt_source = batt.get(
            "battery_economics_source",
            "City-level BESS economics assumption; replace with project-specific vendor quote when available.",
        )
        rows.extend(
            [
                {
                    "scenario": key,
                    "parameter": "SAIDI",
                    "value": sc["saidi_min_per_year"],
                    "unit": "min/year",
                    "source_or_assumption": sc["outage_source"],
                },
                {
                    "scenario": key,
                    "parameter": "SAIFI",
                    "value": sc["saifi_per_year"],
                    "unit": "events/year",
                    "source_or_assumption": sc["outage_source"],
                },
                {
                    "scenario": key,
                    "parameter": "VoLL",
                    "value": sc["voll_sgd_per_kwh"],
                    "unit": "SGD/kWh",
                    "source_or_assumption": sc["voll_source"],
                },
                {
                    "scenario": key,
                    "parameter": "TOU",
                    "value": "",
                    "unit": "SGD/kWh",
                    "source_or_assumption": sc["tou_source"],
                },
                {
                    "scenario": key,
                    "parameter": "Weather/load/PV synthesis",
                    "value": "",
                    "unit": "-",
                    "source_or_assumption": sc["weather_source"],
                },
                {
                    "scenario": key,
                    "parameter": "BESS energy CAPEX",
                    "value": batt["battery_capex_sgd_per_kwh"],
                    "unit": "SGD/kWh",
                    "source_or_assumption": batt_source,
                },
                {
                    "scenario": key,
                    "parameter": "BESS power CAPEX",
                    "value": batt["battery_power_capex_sgd_per_kw"],
                    "unit": "SGD/kW",
                    "source_or_assumption": batt_source,
                },
                {
                    "scenario": key,
                    "parameter": "BESS O&M",
                    "value": batt["battery_om_fraction_per_year"],
                    "unit": "fraction/year",
                    "source_or_assumption": batt_source,
                },
            ]
        )
    return pd.DataFrame(rows)


def make_four_city_comparison(long_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key in config["scenario_order"]:
        sc = config["scenarios"][key]
        batt = _effective_battery_economics(config, key)
        flex = long_df[(long_df["scenario"] == key) & (long_df["design_case"] == "flexibility_only")].iloc[0]
        aware = long_df[(long_df["scenario"] == key) & (long_df["design_case"] == "resilience_informed")].iloc[0]
        rows.append(
            {
                "scenario": key,
                "scenario_name": sc["name"],
                "city_or_service_area": sc["city_or_service_area"],
                "quadrant": sc["quadrant"],
                "saidi_min_per_year": sc["saidi_min_per_year"],
                "saifi_per_year": sc["saifi_per_year"],
                "voll_sgd_per_kwh": sc["voll_sgd_per_kwh"],
                "battery_capex_sgd_per_kwh": batt["battery_capex_sgd_per_kwh"],
                "battery_power_capex_sgd_per_kw": batt["battery_power_capex_sgd_per_kw"],
                "battery_om_fraction_per_year": batt["battery_om_fraction_per_year"],
                "system_annual_cost_flexibility_only_sgd": flex["pv_bess_system_annual_cost_sgd"],
                "system_annual_cost_resilience_informed_sgd": aware["pv_bess_system_annual_cost_sgd"],
                "system_annual_cost_delta_resilience_minus_flex_sgd": (
                    aware["pv_bess_system_annual_cost_sgd"] - flex["pv_bess_system_annual_cost_sgd"]
                ),
                "private_annual_cost_flexibility_only_sgd": flex["pv_bess_private_annual_cost_sgd"],
                "private_annual_cost_resilience_informed_sgd": aware["pv_bess_private_annual_cost_sgd"],
                "battery_flexibility_only_kwh": flex["battery_energy_kwh"],
                "battery_resilience_informed_kwh": aware["battery_energy_kwh"],
                "battery_delta_resilience_minus_flex_kwh": aware["battery_energy_kwh"] - flex["battery_energy_kwh"],
                "unserved_energy_flexibility_only_kwh_per_year": flex["unserved_energy_kwh_per_year"],
                "unserved_energy_resilience_informed_kwh_per_year": aware["unserved_energy_kwh_per_year"],
                "unserved_energy_cost_flexibility_only_sgd_per_year": flex["unserved_energy_cost_sgd_per_year"],
                "unserved_energy_cost_resilience_informed_sgd_per_year": aware["unserved_energy_cost_sgd_per_year"],
                "avoided_unserved_energy_flexibility_only_kwh_per_year": flex["avoided_unserved_energy_kwh_per_year"],
                "avoided_unserved_energy_resilience_informed_kwh_per_year": aware["avoided_unserved_energy_kwh_per_year"],
            }
        )
    return pd.DataFrame(rows)


def write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True if df.index.name else False)
    return path
