from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "input": {"directory": "Input"},
    "data": {
        "load_csv": None,
        "pv_csv": None,
        "tou_csv": None,
    },
    "outputs": {"directory": "outputs"},
    "capacity_search": {
        "coarse_intervals": 8,
        "local_refine_divisions": 6,
        "upper_expansion_factor": 1.5,
        "max_upper_expansions": 2,
    },
    "technical": {
        "cooling_cop": 5.0,
        "chiller_cop": 5.0,
        "heating_cop": 3.5,
        "timestep_hours": 1.0,
        "pv_during_outage_available": True,
    },
    "pv": {
        "capacity_kwp_candidates": [0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000],
        "local_refine_window_kwp": 3000.0,
        "local_refine_step_kwp": 500.0,
        "rooftop_area_m2": None,
        "module_power_density_kwp_per_m2": None,
        "capex_sgd_per_kwp": 1200.0,
        "om_fraction_per_year": 0.015,
        "project_years": 25.0,
    },
    "battery": {
        "energy_kwh_candidates": [0, 2500, 5000, 7500, 10000, 15000, 20000, 30000, 40000, 60000, 80000, 100000, 125000, 150000],
        "max_c_rate": 0.5,
        "roundtrip_efficiency": 0.92,
        "soc_min_fraction": 0.10,
        "soc_max_fraction": 0.95,
        "initial_soc_fraction": 0.50,
        "eol_soh_fraction": 0.80,
        "rated_cycle_life_to_eol": 9000.0,
        "calendar_loss_per_year": 0.008,
        "max_replacement_years": 30.0,
        "local_refine_window_kwh": 15000.0,
        "local_refine_step_kwh": 500.0,
    },
    "economics": {
        "battery_capex_sgd_per_kwh": 80.0,
        "battery_power_capex_sgd_per_kw": 30.0,
        "battery_om_fraction_per_year": 0.02,
        "battery_economics_source": "Global fallback; city-level battery_economics overrides are used when present.",
    },
    "outage": {
        "seed": 42,
        "forced_event_mode": "match_saidi",
    },
    "uncertainty": {
        "parallel_workers": 1,
    },
    "scenarios": {
        "singapore": {
            "name": "Singapore",
            "city_or_service_area": "Singapore national grid",
            "quadrant": "High VoLL / Low SAIDI",
            "saidi_min_per_year": 0.26,
            "saifi_per_year": 0.006,
            "voll_sgd_per_kwh": 35.0,
            "weather_source": "Original UTown Singapore 8760 load and PV output.",
            "outage_source": "EMA Annual & Sustainability Report FY2024/25.",
            "voll_source": "High-value-load research setting retained from prior study.",
            "tou_source": "Keppel Electric Weekend Saver Time-Of-Use Plan weekday structure: 9am-9pm at 32.90 c/kWh and 9pm-9am at 23.90 c/kWh, with GST; weekend rule intentionally not applied.",
            "battery_economics": {
                "battery_capex_sgd_per_kwh": 450.0,
                "battery_power_capex_sgd_per_kw": 180.0,
                "battery_om_fraction_per_year": 0.02,
                "battery_economics_source": "Singapore city-level assumption informed by high-income commercial BESS market ranges; use project quote when available.",
            },
            "load_adjustment": {
                "electric_scale": 1.0,
                "cooling_monthly_factor": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "pv_monthly_factor": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            "tou_sgd_per_kwh_24": [
                0.239, 0.239, 0.239, 0.239, 0.239, 0.239, 0.239, 0.239,
                0.239, 0.329, 0.329, 0.329, 0.329, 0.329, 0.329, 0.329,
                0.329, 0.329, 0.329, 0.329, 0.329, 0.239, 0.239, 0.239
            ],
        },
        "los_angeles": {
            "name": "Los Angeles",
            "city_or_service_area": "Southern California Edison service territory",
            "quadrant": "High VoLL / High SAIDI",
            "saidi_min_per_year": 158.0,
            "saifi_per_year": 1.0,
            "voll_sgd_per_kwh": 35.0,
            "weather_source": "Synthetic 8760 derived from public TMY/EPW relative climate factors for Los Angeles versus Singapore.",
            "outage_source": "SCE 2024 reliability report: SAIDI about 158 min/customer, SAIFI 1.0/customer, no exclusions.",
            "voll_source": "High commercial interruption-cost research setting, aligned with ICE Calculator style VoLL use.",
            "tou_source": "SCE TOU-GS-1 representative business TOU periods; fixed SGD/kWh curve.",
            "battery_economics": {
                "battery_capex_sgd_per_kwh": 500.0,
                "battery_power_capex_sgd_per_kw": 200.0,
                "battery_om_fraction_per_year": 0.02,
                "battery_economics_source": "Los Angeles/SCE city-level assumption informed by NREL ATB commercial storage cost ranges and California project cost premium.",
            },
            "load_adjustment": {
                "electric_scale": 1.0,
                "cooling_monthly_factor": [0.20, 0.24, 0.35, 0.50, 0.70, 0.95, 1.20, 1.25, 1.05, 0.70, 0.38, 0.22],
                "pv_monthly_factor": [0.72, 0.82, 1.00, 1.15, 1.25, 1.35, 1.35, 1.25, 1.12, 0.95, 0.78, 0.68],
            },
            "tou_sgd_per_kwh_24": [
                0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30,
                0.30, 0.30, 0.30, 0.30, 0.55, 0.55, 0.55, 0.55, 0.55, 0.34, 0.34, 0.30
            ],
        },
        "karachi": {
            "name": "Karachi",
            "city_or_service_area": "K-Electric service territory",
            "quadrant": "Low VoLL / High SAIDI",
            "saidi_min_per_year": 4152.59,
            "saifi_per_year": 68.64,
            "voll_sgd_per_kwh": 0.5,
            "weather_source": "Synthetic 8760 derived from public TMY/EPW relative climate factors for Karachi versus Singapore.",
            "outage_source": "NEPRA Performance Evaluation Report of Distribution Companies FY2024-25: K-Electric SAIDI 4152.59 min and SAIFI 68.64.",
            "voll_source": "Low VoLL research setting for South Asian representative case.",
            "tou_source": "Pakistan commercial tariff proxy with evening peak, converted to SGD/kWh.",
            "battery_economics": {
                "battery_capex_sgd_per_kwh": 350.0,
                "battery_power_capex_sgd_per_kw": 120.0,
                "battery_om_fraction_per_year": 0.025,
                "battery_economics_source": "Karachi/K-Electric lower-cost South Asian procurement assumption with moderate import/integration premium; replace with vendor quote when available.",
            },
            "load_adjustment": {
                "electric_scale": 0.92,
                "cooling_monthly_factor": [0.28, 0.34, 0.55, 0.85, 1.10, 1.25, 1.18, 1.14, 1.02, 0.75, 0.45, 0.32],
                "pv_monthly_factor": [0.92, 1.02, 1.16, 1.25, 1.32, 1.28, 1.12, 1.10, 1.18, 1.12, 0.98, 0.88],
            },
            "tou_sgd_per_kwh_24": [
                0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.18, 0.18, 0.19, 0.19, 0.19, 0.19,
                0.20, 0.20, 0.20, 0.20, 0.23, 0.25, 0.25, 0.25, 0.23, 0.20, 0.18, 0.17
            ],
        },
        "guangzhou": {
            "name": "Guangzhou",
            "city_or_service_area": "Guangzhou urban grid",
            "quadrant": "Low VoLL / Low SAIDI",
            "saidi_min_per_year": 30.0,
            "saifi_per_year": 0.25,
            "voll_sgd_per_kwh": 0.6,
            "weather_source": "Synthetic 8760 derived from public TMY/EPW relative climate factors for Guangzhou versus Singapore.",
            "outage_source": "China/Guangzhou urban-grid low-reliability-risk research setting from prior study notes.",
            "voll_source": "Low VoLL research setting.",
            "tou_source": "Guangdong peak-flat-valley commercial tariff proxy.",
            "battery_economics": {
                "battery_capex_sgd_per_kwh": 260.0,
                "battery_power_capex_sgd_per_kw": 100.0,
                "battery_om_fraction_per_year": 0.02,
                "battery_economics_source": "Guangzhou city-level assumption reflecting lower China domestic BESS cost ranges; replace with project quote when available.",
            },
            "load_adjustment": {
                "electric_scale": 1.0,
                "cooling_monthly_factor": [0.15, 0.20, 0.35, 0.62, 0.88, 1.10, 1.25, 1.22, 1.00, 0.68, 0.38, 0.18],
                "pv_monthly_factor": [0.68, 0.72, 0.82, 0.90, 0.92, 0.88, 0.95, 1.02, 1.02, 0.92, 0.80, 0.70],
            },
            "tou_sgd_per_kwh_24": [
                0.0684, 0.0684, 0.0684, 0.0684, 0.0684, 0.0684, 0.0684, 0.0684,
                0.1800, 0.1800, 0.3060, 0.3060, 0.1800, 0.1800, 0.3060, 0.3060,
                0.3060, 0.3060, 0.3060, 0.1800, 0.1800, 0.1800, 0.1800, 0.1800
            ],
        },
    },
    "scenario_order": ["singapore", "los_angeles", "karachi", "guangzhou"],
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if not path:
        return config
    path = Path(path)
    if not path.exists():
        return config
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return config
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _deep_update(config, loaded)
    return config


def _deep_update(base: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
