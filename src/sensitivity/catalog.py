from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensitivityParameter:
    name: str
    category: str
    unit: str


_PARAMETERS = (
    SensitivityParameter("pv_capex_sgd_per_kw", "pv_economics", "SGD/kW"),
    SensitivityParameter("pv_om_fraction_per_year", "pv_economics", "fraction/year"),
    SensitivityParameter("pv_project_years", "pv_economics", "years"),
    SensitivityParameter("pv_rooftop_area_utilization_fraction", "pv_design", "fraction"),
    SensitivityParameter("rooftop_area_m2", "pv_design", "m2"),
    SensitivityParameter("pv_module_power_density_kw_per_m2", "pv_design", "kW/m2"),
    SensitivityParameter("battery_capex_sgd_per_kwh", "battery_economics", "SGD/kWh"),
    SensitivityParameter("battery_power_capex_sgd_per_kw", "battery_economics", "SGD/kW"),
    SensitivityParameter("battery_om_fraction_per_year", "battery_economics", "fraction/year"),
    SensitivityParameter("battery_max_c_rate", "battery_technical", "1/h"),
    SensitivityParameter("battery_roundtrip_efficiency", "battery_technical", "fraction"),
    SensitivityParameter("battery_soc_min_fraction", "battery_technical", "fraction"),
    SensitivityParameter("battery_soc_max_fraction", "battery_technical", "fraction"),
    SensitivityParameter("battery_initial_soc_fraction", "battery_technical", "fraction"),
    SensitivityParameter("battery_eol_soh_fraction", "battery_technical", "fraction"),
    SensitivityParameter("battery_rated_cycle_life_to_eol", "battery_technical", "cycles"),
    SensitivityParameter("battery_calendar_loss_per_year", "battery_technical", "fraction/year"),
    SensitivityParameter("battery_max_replacement_years", "battery_technical", "years"),
    SensitivityParameter("discount_rate", "economics", "fraction/year"),
    SensitivityParameter("cooling_cop", "technical", "-"),
    SensitivityParameter("heating_cop", "technical", "-"),
    SensitivityParameter("saidi_min_per_year", "reliability", "min/year"),
    SensitivityParameter("saifi_per_year", "reliability", "events/year"),
    SensitivityParameter("voll_sgd_per_kwh", "reliability", "SGD/kWh"),
)

PARAMETERS_BY_NAME = {parameter.name: parameter for parameter in _PARAMETERS}


def numeric_system_parameter_names() -> tuple[str, ...]:
    return tuple(parameter.name for parameter in _PARAMETERS)


def parameter_definition(name: str) -> SensitivityParameter | None:
    return PARAMETERS_BY_NAME.get(str(name))


def is_numeric_system_parameter(name: str) -> bool:
    return str(name) in PARAMETERS_BY_NAME
