from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PARAMETER_COLUMNS = [
    "parameter",
    "base_value",
    "unit",
    "sensitivity_start",
    "sensitivity_end",
    "sensitivity_step",
    "description",
    "中文说明",
]
REMOVED_PV_PARAMETERS = {
    "base_pv_coverage_fraction",
    "pv_capacity_kw_at_base_coverage",
    "pv_coverage_candidates",
    "pv_coverage_fraction",
    "pv_capacity_kw_candidates",
    "battery_energy_kwh_candidates",
}
LOAD_COLUMN_ALIASES = {
    "Cooling load (kWh_th)": "Cooling",
    "Heating load (kWh_th)": "Heating",
    "Lighting load (kWh_e)": "Lighting",
    "Equipment load (kWh_e)": "Equipment",
}
PV_OUTPUT_COLUMN = "PV_output_kWh_per_m2"
PV_EXPORT_PRICE_COLUMN = "pv_export_sgd_per_kwh"


@dataclass(frozen=True)
class InputCase:
    city_key: str
    building_key: str
    case_key: str
    city_dir: Path
    load_csv: Path
    pv_csv: Path
    tou_csv: Path
    parameter_csv: Path


@dataclass(frozen=True)
class CaseParameters:
    base: dict[str, Any]
    sensitivity: dict[str, list[float]]
    raw: pd.DataFrame


def discover_input_cases(input_dir: str | Path) -> list[InputCase]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    cases: list[InputCase] = []
    for city_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        grouped: dict[str, dict[str, Path]] = {}
        for csv_path in sorted(city_dir.glob("*.csv")):
            if csv_path.stem.endswith("_hourly_loads_kWh"):
                prefix = city_dir.name
                grouped.setdefault(prefix, {})["Load"] = csv_path
                continue
            for suffix in ("Load", "PV", "TOU", "Parameter"):
                marker = f"_{suffix}"
                if csv_path.stem.endswith(marker):
                    prefix = csv_path.stem[: -len(marker)]
                    grouped.setdefault(prefix, {})[suffix] = csv_path
                    break

        for prefix, files in sorted(grouped.items()):
            missing = [suffix for suffix in ("Load", "PV", "TOU", "Parameter") if suffix not in files]
            if missing:
                raise ValueError(f"Input case {prefix} in {city_dir} is missing files: {', '.join(missing)}")
            if prefix == city_dir.name and "_" in prefix:
                city_key, building_key = prefix.rsplit("_", 1)
            elif prefix.startswith(f"{city_dir.name}_"):
                city_key = city_dir.name
                building_key = prefix[len(city_dir.name) + 1 :]
            else:
                raise ValueError(f"Input case prefix {prefix} must start with city folder name {city_dir.name}_")
            if not building_key:
                raise ValueError(f"Input case prefix {prefix} does not include a building name")
            cases.append(
                InputCase(
                    city_key=city_key,
                    building_key=building_key,
                    case_key=prefix,
                    city_dir=city_dir,
                    load_csv=files["Load"],
                    pv_csv=files["PV"],
                    tou_csv=files["TOU"],
                    parameter_csv=files["Parameter"],
                )
            )

    if not cases:
        raise ValueError(f"No complete input cases found under {root}")
    return cases


def read_load_csv(path: str | Path) -> pd.DataFrame:
    """Read load input and normalize supported real-data column names in memory."""
    return pd.read_csv(path).rename(columns=LOAD_COLUMN_ALIASES)


def read_pv_csv(path: str | Path) -> pd.DataFrame:
    """Read PV input, keeping the required output column when source files include extras."""
    frame = pd.read_csv(path)
    if PV_OUTPUT_COLUMN in frame.columns:
        return frame[[PV_OUTPUT_COLUMN]].copy()
    return frame


def read_tou_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_case_parameters(parameter_csv: str | Path) -> CaseParameters:
    raw = pd.read_csv(parameter_csv)
    missing_columns = [column for column in REQUIRED_PARAMETER_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Parameter file {parameter_csv} is missing columns: {', '.join(missing_columns)}")
    if raw["parameter"].duplicated().any():
        repeated = raw.loc[raw["parameter"].duplicated(), "parameter"].tolist()
        raise ValueError(f"Parameter file {parameter_csv} contains duplicated parameters: {repeated}")
    parameter_names = set(raw["parameter"].astype(str))
    if "fixed_pv_coverage_fraction" in parameter_names:
        raise ValueError(
            "fixed_pv_coverage_fraction is no longer a sensitivity parameter; "
            "PV capacity is optimized through adaptive capacity search, and rooftop coverage is "
            "a derived diagnostic field."
        )
    removed = sorted(set(raw["parameter"].astype(str)) & REMOVED_PV_PARAMETERS)
    if removed:
        raise ValueError(
            "Removed PV parameter(s) found in "
            f"{parameter_csv}: {', '.join(removed)}. "
            "PV-BESS sizing now derives capacity candidates adaptively from the 8760 load/PV inputs, "
            "rooftop_area_m2, pv_rooftop_area_utilization_fraction, and "
            "pv_module_power_density_kw_per_m2."
        )

    base: dict[str, Any] = {}
    sensitivity: dict[str, list[float]] = {}
    # Imported here to keep the loader usable by the sensitivity package without
    # introducing an import-time cycle through its planning module.
    from ..sensitivity.catalog import is_numeric_system_parameter

    for row in raw.itertuples(index=False):
        parameter = str(getattr(row, "parameter"))
        base_value = _parse_value(getattr(row, "base_value"))
        base[parameter] = base_value
        values = _parse_sensitivity_range(
            getattr(row, "sensitivity_start"),
            getattr(row, "sensitivity_end"),
            getattr(row, "sensitivity_step"),
            parameter,
        )
        if values:
            if not is_numeric_system_parameter(parameter):
                raise ValueError(
                    "Sensitivity ranges are only supported for catalogued numeric system parameters; "
                    f"{parameter!r} is not eligible."
                )
            if isinstance(base_value, bool) or base_value is None:
                raise ValueError(f"Sensitivity parameter {parameter} must have a numeric base_value")
            try:
                base_number = float(base_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Sensitivity parameter {parameter} must have a numeric base_value") from exc
            if not np.isfinite(base_number):
                raise ValueError(f"Sensitivity parameter {parameter} must have a finite base_value")
            sensitivity[parameter] = values
    return CaseParameters(base=base, sensitivity=sensitivity, raw=raw)


def load_input_case(
    case: InputCase,
    global_config: dict[str, Any],
    parameter_overrides: dict[str, Any] | None = None,
    pv_capacity_kw: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float | str], dict[str, Any], CaseParameters]:
    parameters = load_case_parameters(case.parameter_csv)
    base = dict(parameters.base)
    if parameter_overrides:
        base.update(parameter_overrides)
    case_config = case_config_from_parameters(global_config, case, base)
    frame, pv_meta = build_case_frame(case, case_config, pv_capacity_kw)
    return frame, pv_meta, case_config, parameters


def case_config_from_parameters(global_config: dict[str, Any], case: InputCase, params: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(global_config)
    scenario_key = case.case_key
    city_name = str(params.get("city_name") or case.city_key)
    building_name = str(params.get("building_name") or case.building_key)
    analysis_years = _positive_int_param(
        params,
        "analysis_years",
        int(global_config.get("analysis", {}).get("years", 1)),
    )
    config["analysis"] = deepcopy(global_config.get("analysis", {}))
    config["analysis"]["years"] = analysis_years

    config["technical"] = deepcopy(global_config.get("technical", {}))
    config["technical"]["cooling_cop"] = _float_param(params, "cooling_cop", 5.0)
    config["technical"]["chiller_cop"] = config["technical"]["cooling_cop"]
    config["technical"]["heating_cop"] = _float_param(params, "heating_cop", 3.5)
    config["technical"]["timestep_hours"] = _float_param(params, "timestep_hours", 1.0)
    config["technical"]["pv_during_outage_available"] = _bool_param(params, "pv_during_outage_available", True)

    config["pv"] = deepcopy(global_config.get("pv", {}))
    pv_refine_window = _optional_nonnegative_float_param(params, "local_refine_window_kw")
    pv_refine_step = _optional_positive_float_param(params, "local_refine_step_kw")
    config["pv"]["local_refine_window_kw"] = pv_refine_window
    config["pv"]["local_refine_step_kw"] = pv_refine_step
    config["pv"]["local_refine_window_source"] = "parameter" if pv_refine_window is not None else "adaptive"
    config["pv"]["local_refine_step_source"] = "parameter" if pv_refine_step is not None else "adaptive"
    rooftop_area_m2 = _positive_float_param(params, "rooftop_area_m2")
    module_power_density = _positive_float_param(
        params,
        "pv_module_power_density_kw_per_m2",
    )
    rooftop_limit_enabled = _bool_param(params, "pv_rooftop_area_limit_enabled", True)
    rooftop_utilization_fraction = _positive_fraction_param(
        params,
        "pv_rooftop_area_utilization_fraction",
        0.8,
    )
    config["pv"]["rooftop_area_m2"] = rooftop_area_m2
    config["pv"]["rooftop_area_limit_enabled"] = rooftop_limit_enabled
    config["pv"]["rooftop_area_utilization_fraction"] = rooftop_utilization_fraction
    config["pv"]["rooftop_capacity_limit_kw"] = float(
        rooftop_area_m2 * rooftop_utilization_fraction * module_power_density
    )
    config["pv"]["module_power_density_kw_per_m2"] = module_power_density
    config["pv"]["capex_sgd_per_kw"] = _float_param(params, "pv_capex_sgd_per_kw", 1200.0)
    config["pv"]["om_fraction_per_year"] = _float_param(params, "pv_om_fraction_per_year", 0.015)
    config["pv"]["project_years"] = _float_param(params, "pv_project_years", 25.0)
    config["pv"]["export_enabled"] = _bool_param(params, "pv_export_enabled", False)

    config["battery"] = deepcopy(global_config.get("battery", {}))
    battery_mappings = {
        "battery_max_c_rate": ("max_c_rate", 0.5),
        "battery_roundtrip_efficiency": ("roundtrip_efficiency", 0.92),
        "battery_soc_min_fraction": ("soc_min_fraction", 0.10),
        "battery_soc_max_fraction": ("soc_max_fraction", 0.95),
        "battery_initial_soc_fraction": ("initial_soc_fraction", 0.50),
        "battery_eol_soh_fraction": ("eol_soh_fraction", 0.80),
        "battery_rated_cycle_life_to_eol": ("rated_cycle_life_to_eol", 9000.0),
        "battery_calendar_loss_per_year": ("calendar_loss_per_year", 0.008),
        "battery_max_replacement_years": ("max_replacement_years", 30.0),
    }
    for parameter, (config_key, default) in battery_mappings.items():
        config["battery"][config_key] = _float_param(params, parameter, default)
    battery_refine_window = _optional_nonnegative_float_param(params, "local_refine_window_kwh")
    battery_refine_step = _optional_positive_float_param(params, "local_refine_step_kwh")
    config["battery"]["local_refine_window_kwh"] = battery_refine_window
    config["battery"]["local_refine_step_kwh"] = battery_refine_step
    config["battery"]["local_refine_window_source"] = "parameter" if battery_refine_window is not None else "adaptive"
    config["battery"]["local_refine_step_source"] = "parameter" if battery_refine_step is not None else "adaptive"

    config["economics"] = deepcopy(global_config.get("economics", {}))
    battery_economics = {
        "discount_rate": _nonnegative_float_param(
            params,
            "discount_rate",
            float(config["economics"].get("discount_rate", 0.0)),
        ),
        "battery_capex_sgd_per_kwh": _float_param(params, "battery_capex_sgd_per_kwh", 450.0),
        "battery_power_capex_sgd_per_kw": _float_param(params, "battery_power_capex_sgd_per_kw", 180.0),
        "battery_om_fraction_per_year": _float_param(params, "battery_om_fraction_per_year", 0.02),
        "battery_economics_source": "Input Parameter file.",
    }
    config["economics"].update(battery_economics)

    config["outage"] = deepcopy(global_config.get("outage", {}))
    config["outage"]["seed"] = int(_float_param(params, "outage_seed", 42.0))
    config["outage"]["forced_event_mode"] = str(params.get("forced_event_mode") or "match_saidi")
    config["outage"]["night_window_start_hour"] = _float_param(
        params,
        "night_window_start_hour",
        float(config["outage"].get("night_window_start_hour", 19.0)),
    )
    config["outage"]["night_window_end_hour"] = _float_param(
        params,
        "night_window_end_hour",
        float(config["outage"].get("night_window_end_hour", 23.0)),
    )

    config["scenarios"] = {
        scenario_key: {
            "name": f"{city_name} {building_name}",
            "city_or_service_area": city_name,
            "quadrant": str(params.get("quadrant") or ""),
            "saidi_min_per_year": _float_param(params, "saidi_min_per_year", 0.0),
            "saifi_per_year": _float_param(params, "saifi_per_year", 0.0),
            "voll_sgd_per_kwh": _float_param(params, "voll_sgd_per_kwh", 0.0),
            "weather_source": "Input Load/PV files.",
            "outage_source": "Input Parameter file.",
            "voll_source": "Input Parameter file.",
            "tou_source": "Input TOU file.",
            "battery_economics": battery_economics,
        }
    }
    config["scenario_order"] = [scenario_key]
    _apply_adaptive_capacity_search(case, config)
    return config


def _apply_adaptive_capacity_search(case: InputCase, config: dict[str, Any]) -> None:
    load = read_load_csv(case.load_csv)
    pv = read_pv_csv(case.pv_csv)
    _validate_8760(load, case.load_csv)
    _validate_8760(pv, case.pv_csv)
    _validate_required_columns(load, ["Cooling", "Heating", "Lighting", "Equipment"], case.load_csv)
    _validate_required_columns(pv, [PV_OUTPUT_COLUMN], case.pv_csv)

    cooling_cop = max(float(config["technical"].get("cooling_cop", 5.0)), 1e-9)
    heating_cop = max(float(config["technical"].get("heating_cop", 3.5)), 1e-9)
    load_e = (
        load["Lighting"].astype(float).to_numpy()
        + load["Equipment"].astype(float).to_numpy()
        + load["Cooling"].astype(float).to_numpy() / cooling_cop
        + load["Heating"].astype(float).to_numpy() / heating_cop
    )
    if np.any(load_e < 0):
        raise ValueError(f"Input case {case.case_key} generated negative electric-equivalent load")
    annual_load = float(load_e.sum())
    peak_load = float(load_e.max())
    if annual_load <= 0.0 or peak_load <= 0.0:
        raise ValueError(f"Input case {case.case_key} must have positive electric-equivalent load")

    module_power_density = max(float(config["pv"]["module_power_density_kw_per_m2"]), 1e-9)
    pv_per_kw = pv[PV_OUTPUT_COLUMN].astype(float).to_numpy() / module_power_density
    if np.any(pv_per_kw < 0):
        raise ValueError(f"Input case {case.case_key} generated negative PV output per kW")
    annual_pv_per_kw = float(pv_per_kw.sum())
    peak_pv_per_kw = float(pv_per_kw.max())
    if annual_pv_per_kw <= 0.0 or peak_pv_per_kw <= 0.0:
        raise ValueError(f"Input case {case.case_key} must have positive PV_output_kWh_per_m2")

    scenario = config["scenarios"][case.case_key]
    saidi = max(float(scenario.get("saidi_min_per_year", 0.0)), 0.0)
    saifi = max(float(scenario.get("saifi_per_year", 0.0)), 0.0)
    event_hours = saidi / 60.0 / saifi if saidi > 0.0 and saifi > 0.0 else 0.0

    pv_annual_kw = annual_load / annual_pv_per_kw
    pv_peak_kw = peak_load / peak_pv_per_kw
    unconstrained_pv_upper = max(1.10 * pv_annual_kw, 1.05 * pv_peak_kw)
    rooftop_limit_enabled = bool(config["pv"].get("rooftop_area_limit_enabled", True))
    rooftop_capacity_limit = float(config["pv"]["rooftop_capacity_limit_kw"])
    pv_upper = min(unconstrained_pv_upper, rooftop_capacity_limit) if rooftop_limit_enabled else unconstrained_pv_upper
    battery_upper = max(0.60 * annual_load / 365.0, 1.50 * peak_load * event_hours)

    search_cfg = config.get("capacity_search", {})
    intervals = max(1, int(search_cfg.get("coarse_intervals", 8)))
    refine_divisions = max(1.0, float(search_cfg.get("local_refine_divisions", 6)))

    pv_candidates, pv_step = _adaptive_axis(pv_upper, intervals=intervals)
    battery_candidates, battery_step = _adaptive_axis(battery_upper, intervals=intervals)
    config["pv"]["capacity_kw_candidates"] = pv_candidates
    if config["pv"].get("local_refine_window_kw") is None:
        config["pv"]["local_refine_window_kw"] = pv_step
        config["pv"]["local_refine_window_source"] = "adaptive"
    if config["pv"].get("local_refine_step_kw") is None:
        config["pv"]["local_refine_step_kw"] = max(_nice_step(pv_step / refine_divisions), 1e-6)
        config["pv"]["local_refine_step_source"] = "adaptive"
    config["battery"]["energy_kwh_candidates"] = battery_candidates
    if config["battery"].get("local_refine_window_kwh") is None:
        config["battery"]["local_refine_window_kwh"] = battery_step
        config["battery"]["local_refine_window_source"] = "adaptive"
    if config["battery"].get("local_refine_step_kwh") is None:
        config["battery"]["local_refine_step_kwh"] = max(_nice_step(battery_step / refine_divisions), 1e-6)
        config["battery"]["local_refine_step_source"] = "adaptive"
    config["capacity_search"] = {
        **search_cfg,
        "strategy": "adaptive_from_8760_load_and_pv",
        "coarse_intervals": intervals,
        "local_refine_divisions": refine_divisions,
        "annual_load_e_equiv_kwh": annual_load,
        "peak_load_e_equiv_kw": peak_load,
        "annual_pv_output_kwh_per_kw": annual_pv_per_kw,
        "peak_pv_output_kw_per_kw": peak_pv_per_kw,
        "pv_unconstrained_upper_kw": float(unconstrained_pv_upper),
        "pv_upper_kw": float(max(pv_candidates)),
        "pv_rooftop_area_limit_enabled": bool(rooftop_limit_enabled),
        "pv_rooftop_area_utilization_fraction": float(config["pv"]["rooftop_area_utilization_fraction"]),
        "pv_rooftop_capacity_limit_kw": float(rooftop_capacity_limit),
        "battery_upper_kwh": float(max(battery_candidates)),
        "pv_coarse_step_kw": float(pv_step),
        "battery_coarse_step_kwh": float(battery_step),
        "event_hours_per_interruption": float(event_hours),
    }


def _adaptive_axis(upper: float, intervals: int) -> tuple[list[float], float]:
    if upper <= 0.0:
        return [0.0, 1.0], 1.0
    step = _nice_step(float(upper) / max(int(intervals), 1))
    values = [round(i * step, 6) for i in range(int(intervals) + 1) if i * step <= float(upper) + 1e-9]
    values = sorted(set(values) | {round(float(upper), 6)})
    return values, step


def _nice_step(raw_step: float) -> float:
    if raw_step <= 0.0:
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    for multiplier in (1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        step = multiplier * magnitude
        if step >= raw_step - 1e-12:
            return float(step)
    return float(10.0 * magnitude)


def build_case_frame(
    case: InputCase,
    config: dict[str, Any],
    pv_capacity_kw: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    load = read_load_csv(case.load_csv)
    pv = read_pv_csv(case.pv_csv)
    tou = read_tou_csv(case.tou_csv)

    _validate_8760(load, case.load_csv)
    _validate_8760(pv, case.pv_csv)
    _validate_required_columns(load, ["Cooling", "Heating", "Lighting", "Equipment"], case.load_csv)
    _validate_required_columns(pv, [PV_OUTPUT_COLUMN], case.pv_csv)
    _validate_tou(tou, case.tou_csv)

    analysis_years = max(1, int(config.get("analysis", {}).get("years", 1)))
    total_hours = 8760 * analysis_years
    source_hour_index = np.tile(np.arange(8760, dtype=int), analysis_years)
    analysis_year = np.repeat(np.arange(1, analysis_years + 1, dtype=int), 8760)
    if "timestamp" in load.columns:
        timestamp_start = pd.to_datetime(load["timestamp"]).iloc[0]
    else:
        timestamp_start = pd.Timestamp("2023-01-01 00:00:00")
    timestamp = pd.date_range(timestamp_start, periods=total_hours, freq="h")
    hour_to_price = dict(zip(tou["hour"].astype(int), tou["tou_sgd_per_kwh"].astype(float)))
    if PV_EXPORT_PRICE_COLUMN in tou.columns:
        hour_to_export_price = dict(zip(tou["hour"].astype(int), tou[PV_EXPORT_PRICE_COLUMN].astype(float)))
    else:
        hour_to_export_price = {hour: 0.0 for hour in range(24)}

    pv_cfg = config["pv"]
    capacity_candidates = [float(value) for value in pv_cfg.get("capacity_kw_candidates", [])]
    if not capacity_candidates:
        raise ValueError(f"Input case {case.case_key} has no PV capacity candidates")
    active_capacity_kw = float(max(capacity_candidates) if pv_capacity_kw is None else pv_capacity_kw)
    _validate_pv_capacity_value(active_capacity_kw, case.case_key)
    raw_pv_per_m2 = pv[PV_OUTPUT_COLUMN].astype(float).to_numpy()
    rooftop_area_m2 = float(pv_cfg["rooftop_area_m2"])
    rooftop_limit_enabled = bool(pv_cfg.get("rooftop_area_limit_enabled", True))
    rooftop_area_utilization_fraction = float(pv_cfg.get("rooftop_area_utilization_fraction", 0.8))
    rooftop_capacity_limit_kw = float(
        pv_cfg.get(
            "rooftop_capacity_limit_kw",
            rooftop_area_m2 * rooftop_area_utilization_fraction * float(pv_cfg["module_power_density_kw_per_m2"]),
        )
    )
    if rooftop_limit_enabled and active_capacity_kw > rooftop_capacity_limit_kw + 1e-6:
        raise ValueError(
            f"PV capacity for {case.case_key} exceeds rooftop capacity limit: "
            f"{active_capacity_kw} kW > {rooftop_capacity_limit_kw} kW"
        )
    module_power_density = float(pv_cfg["module_power_density_kw_per_m2"])
    pv_per_kw = raw_pv_per_m2 / module_power_density
    installed_area_m2 = active_capacity_kw / module_power_density
    rooftop_utilization = installed_area_m2 / rooftop_area_m2

    frame = pd.DataFrame(
        {
            "hour_index": np.arange(total_hours, dtype=int),
            "analysis_year": analysis_year,
            "hour_of_year": source_hour_index,
            "source_hour_index": source_hour_index,
            "timestamp": timestamp,
            "electric_load_kwh": np.tile(
                (load["Lighting"].astype(float) + load["Equipment"].astype(float)).to_numpy(),
                analysis_years,
            ),
            "cooling_load_kwh_th": np.tile(load["Cooling"].astype(float).to_numpy(), analysis_years),
            "heating_load_kwh_th": np.tile(load["Heating"].astype(float).to_numpy(), analysis_years),
            "pv_kwh": np.tile(pv_per_kw, analysis_years) * active_capacity_kw,
        }
    )
    frame["pv_capacity_kw"] = active_capacity_kw
    frame["pv_capacity_mw"] = active_capacity_kw / 1000.0
    frame["pv_installed_area_m2"] = float(installed_area_m2)
    frame["pv_rooftop_area_limit_enabled"] = bool(rooftop_limit_enabled)
    frame["pv_rooftop_area_utilization_fraction"] = float(rooftop_area_utilization_fraction)
    frame["pv_rooftop_capacity_limit_kw"] = float(rooftop_capacity_limit_kw)
    frame["pv_rooftop_utilization_fraction"] = float(rooftop_utilization)
    frame["pv_rooftop_utilization_percent"] = float(rooftop_utilization * 100.0)
    frame["tou_sgd_per_kwh"] = frame["timestamp"].dt.hour.map(hour_to_price).astype(float)
    frame[PV_EXPORT_PRICE_COLUMN] = frame["timestamp"].dt.hour.map(hour_to_export_price).astype(float)
    if frame.isna().any().any():
        raise ValueError(f"Input case {case.case_key} generated missing values")

    pv_meta = {
        "scenario": case.case_key,
        "pv_capacity_kw": float(active_capacity_kw),
        "pv_capacity_mw": float(active_capacity_kw / 1000.0),
        "rooftop_area_m2": rooftop_area_m2,
        "pv_installed_area_m2": float(installed_area_m2),
        "pv_rooftop_area_limit_enabled": bool(rooftop_limit_enabled),
        "pv_rooftop_area_utilization_fraction": float(rooftop_area_utilization_fraction),
        "pv_rooftop_capacity_limit_kw": float(rooftop_capacity_limit_kw),
        "pv_rooftop_utilization_fraction": float(rooftop_utilization),
        "pv_rooftop_utilization_percent": float(rooftop_utilization * 100.0),
        "pv_module_power_density_kw_per_m2": module_power_density,
        "analysis_years": float(analysis_years),
        "pv_capacity_note": (
            "PV output is scaled from 1 m2 generation through module power density and PV capacity; "
            "PV capacity is capped by rooftop area when pv_rooftop_area_limit_enabled is true."
        ),
    }
    frame["pv_rooftop_coverage_fraction"] = frame["pv_rooftop_utilization_fraction"]
    frame["pv_rooftop_coverage_percent"] = frame["pv_rooftop_utilization_percent"]
    pv_meta["pv_rooftop_coverage_fraction"] = float(rooftop_utilization)
    pv_meta["pv_rooftop_coverage_percent"] = float(rooftop_utilization * 100.0)
    return frame, pv_meta


def sensitivity_runs(parameters: CaseParameters) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for parameter, values in parameters.sensitivity.items():
        for value in values:
            runs.append({"sensitivity_parameter": parameter, "sensitivity_value": value, parameter: value})
    return runs


def validate_sensitivity_config(config: dict[str, Any]) -> None:
    """Validate cross-parameter constraints for a prospective sensitivity point.

    ``case_config_from_parameters`` deliberately remains permissive enough to
    load historical parameter files.  Sensitivity runs, however, must reject a
    whole plan before executing if any selected value creates an invalid model
    configuration.
    """
    pv = config["pv"]
    battery = config["battery"]
    economics = config["economics"]
    technical = config["technical"]
    scenarios = config["scenarios"]

    _require_nonnegative_config(pv, "capex_sgd_per_kw", "PV CAPEX")
    _require_nonnegative_config(pv, "om_fraction_per_year", "PV O&M fraction")
    _require_positive_config(pv, "project_years", "PV project years")
    _require_positive_config(pv, "rooftop_area_m2", "rooftop area")
    _require_fraction_config(pv, "rooftop_area_utilization_fraction", "PV rooftop utilization")
    _require_positive_config(pv, "module_power_density_kw_per_m2", "PV module power density")

    _require_nonnegative_config(economics, "battery_capex_sgd_per_kwh", "battery energy CAPEX")
    _require_nonnegative_config(economics, "battery_power_capex_sgd_per_kw", "battery power CAPEX")
    _require_nonnegative_config(economics, "battery_om_fraction_per_year", "battery O&M fraction")
    _require_nonnegative_config(economics, "discount_rate", "discount rate")

    _require_positive_config(battery, "max_c_rate", "battery max C-rate")
    _require_fraction_config(battery, "roundtrip_efficiency", "battery round-trip efficiency")
    _require_fraction_config(battery, "soc_min_fraction", "battery minimum SOC", allow_zero=True)
    _require_fraction_config(battery, "soc_max_fraction", "battery maximum SOC")
    _require_fraction_config(battery, "initial_soc_fraction", "battery initial SOC", allow_zero=True)
    _require_fraction_config(battery, "eol_soh_fraction", "battery EOL SOH")
    _require_positive_config(battery, "rated_cycle_life_to_eol", "battery rated cycle life")
    _require_nonnegative_config(battery, "calendar_loss_per_year", "battery calendar loss")
    _require_positive_config(battery, "max_replacement_years", "battery maximum replacement years")
    soc_min = _config_number(battery, "soc_min_fraction", "battery minimum SOC")
    soc_max = _config_number(battery, "soc_max_fraction", "battery maximum SOC")
    soc_initial = _config_number(battery, "initial_soc_fraction", "battery initial SOC")
    if not soc_min < soc_max:
        raise ValueError("Battery SOC constraints require soc_min_fraction < soc_max_fraction")
    if not soc_min <= soc_initial <= soc_max:
        raise ValueError("Battery initial SOC must be within [soc_min_fraction, soc_max_fraction]")

    _require_positive_config(technical, "cooling_cop", "cooling COP")
    _require_positive_config(technical, "heating_cop", "heating COP")
    for scenario_key, scenario in scenarios.items():
        _require_nonnegative_config(scenario, "saidi_min_per_year", f"{scenario_key} SAIDI")
        _require_nonnegative_config(scenario, "saifi_per_year", f"{scenario_key} SAIFI")
        _require_nonnegative_config(scenario, "voll_sgd_per_kwh", f"{scenario_key} VoLL")


def _config_number(section: dict[str, Any], key: str, label: str) -> float:
    try:
        value = float(section[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric value") from exc
    if not np.isfinite(value):
        raise ValueError(f"{label} must be a finite numeric value")
    return value


def _require_positive_config(section: dict[str, Any], key: str, label: str) -> None:
    if _config_number(section, key, label) <= 0.0:
        raise ValueError(f"{label} must be > 0")


def _require_nonnegative_config(section: dict[str, Any], key: str, label: str) -> None:
    if _config_number(section, key, label) < 0.0:
        raise ValueError(f"{label} must be >= 0")


def _require_fraction_config(section: dict[str, Any], key: str, label: str, *, allow_zero: bool = False) -> None:
    value = _config_number(section, key, label)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value > 1.0:
        lower = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{label} must be {lower} and <= 1")


def write_scenario_timeseries(output_dir: Path, scenario_key: str, frame: pd.DataFrame) -> Path:
    out = output_dir / "scenario_inputs" / scenario_key
    out.mkdir(parents=True, exist_ok=True)
    path = out / "timeseries.csv"
    frame.to_csv(path, index=False)
    return path


def _parse_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lower = text.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if text.startswith("[") and text.endswith("]"):
            return ast.literal_eval(text)
        try:
            return float(text)
        except ValueError:
            return text
    return value


def _parse_sensitivity_range(start: Any, end: Any, step: Any, parameter: str) -> list[float]:
    values = [_parse_value(start), _parse_value(end), _parse_value(step)]
    if all(value is None for value in values):
        return []
    if any(value is None for value in values):
        raise ValueError(f"Sensitivity range for {parameter} must define start, end, and step together")
    start_f, end_f, step_f = (float(values[0]), float(values[1]), float(values[2]))
    if not np.isfinite([start_f, end_f, step_f]).all():
        raise ValueError(f"Sensitivity range for {parameter} must contain finite numeric values")
    if step_f <= 0:
        raise ValueError(f"Sensitivity step for {parameter} must be positive")
    if end_f < start_f:
        raise ValueError(f"Sensitivity end for {parameter} must be >= start")
    result: list[float] = []
    current = start_f
    while current <= end_f + step_f * 1e-9:
        result.append(round(current, 10))
        current += step_f
    return result


def _float_param(params: dict[str, Any], name: str, default: float) -> float:
    value = params.get(name)
    return float(default if value is None else value)


def _positive_float_param(params: dict[str, Any], name: str) -> float:
    value = params.get(name)
    if value is None:
        raise ValueError(f"Parameter {name} is required and must be > 0")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"Parameter {name} must be > 0, got {result}")
    return result


def _nonnegative_float_param(params: dict[str, Any], name: str, default: float) -> float:
    value = params.get(name)
    result = float(default if value is None else value)
    if result < 0.0:
        raise ValueError(f"Parameter {name} must be >= 0, got {result}")
    return result


def _positive_fraction_param(params: dict[str, Any], name: str, default: float) -> float:
    value = params.get(name)
    result = float(default if value is None else value)
    if result <= 0.0 or result > 1.0:
        raise ValueError(f"Parameter {name} must be > 0 and <= 1, got {result}")
    return result


def _optional_nonnegative_float_param(params: dict[str, Any], name: str) -> float | None:
    value = params.get(name)
    if value is None:
        return None
    result = float(value)
    if result < 0.0:
        raise ValueError(f"Parameter {name} must be >= 0, got {result}")
    return result


def _optional_positive_float_param(params: dict[str, Any], name: str) -> float | None:
    value = params.get(name)
    if value is None:
        return None
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"Parameter {name} must be > 0, got {result}")
    return result


def _positive_int_param(params: dict[str, Any], name: str, default: int) -> int:
    value = params.get(name)
    result = float(default if value is None else value)
    if result <= 0.0 or not result.is_integer():
        raise ValueError(f"Parameter {name} must be a positive integer, got {value if value is not None else default}")
    return int(result)


def _bool_param(params: dict[str, Any], name: str, default: bool) -> bool:
    value = params.get(name)
    return bool(default if value is None else value)


def _validate_pv_capacity_value(value: float, source: str) -> None:
    if value < 0.0:
        raise ValueError(f"PV capacity for {source} must be >= 0 kW, got {value}")


def _validate_8760(df: pd.DataFrame, path: Path) -> None:
    if len(df) != 8760:
        raise ValueError(f"{path} must contain 8760 rows, got {len(df)}")


def _validate_required_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def _validate_tou(tou: pd.DataFrame, path: Path) -> None:
    _validate_required_columns(tou, ["hour", "tou_sgd_per_kwh"], path)
    if len(tou) != 24:
        raise ValueError(f"{path} must contain 24 hourly TOU rows, got {len(tou)}")
    hours = sorted(tou["hour"].astype(int).tolist())
    if hours != list(range(24)):
        raise ValueError(f"{path} hour column must cover 0..23 exactly")
    if PV_EXPORT_PRICE_COLUMN in tou.columns and (tou[PV_EXPORT_PRICE_COLUMN].astype(float) < 0.0).any():
        raise ValueError(f"{path} {PV_EXPORT_PRICE_COLUMN} must be non-negative")
