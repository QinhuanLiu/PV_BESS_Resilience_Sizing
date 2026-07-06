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
SENSITIVITY_PARAMETERS = [
    "pv_capex_sgd_per_kwp",
    "battery_capex_sgd_per_kwh",
    "battery_power_capex_sgd_per_kw",
    "saidi_min_per_year",
    "voll_sgd_per_kwh",
]
REMOVED_PV_PARAMETERS = {
    "base_pv_coverage_fraction",
    "pv_capacity_kwp_at_base_coverage",
    "pv_coverage_candidates",
    "pv_coverage_fraction",
    "pv_capacity_kwp_candidates",
    "battery_energy_kwh_candidates",
}


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
            if not prefix.startswith(f"{city_dir.name}_"):
                raise ValueError(f"Input case prefix {prefix} must start with city folder name {city_dir.name}_")
            building_key = prefix[len(city_dir.name) + 1 :]
            if not building_key:
                raise ValueError(f"Input case prefix {prefix} does not include a building name")
            cases.append(
                InputCase(
                    city_key=city_dir.name,
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


def load_case_parameters(parameter_csv: str | Path) -> CaseParameters:
    raw = pd.read_csv(parameter_csv)
    missing_columns = [column for column in REQUIRED_PARAMETER_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Parameter file {parameter_csv} is missing columns: {', '.join(missing_columns)}")
    if len(raw) < len(SENSITIVITY_PARAMETERS):
        raise ValueError(f"Parameter file {parameter_csv} must include the five sensitivity parameters first")

    first_parameters = raw["parameter"].head(len(SENSITIVITY_PARAMETERS)).tolist()
    if first_parameters != SENSITIVITY_PARAMETERS:
        if first_parameters and first_parameters[0] == "fixed_pv_coverage_fraction":
            raise ValueError(
                "fixed_pv_coverage_fraction is no longer a sensitivity parameter; "
                "PV capacity is optimized through adaptive capacity search."
            )
        raise ValueError(
            "The first five parameters must be "
            + ", ".join(SENSITIVITY_PARAMETERS)
            + f"; got {', '.join(str(x) for x in first_parameters)}"
        )
    if raw["parameter"].duplicated().any():
        repeated = raw.loc[raw["parameter"].duplicated(), "parameter"].tolist()
        raise ValueError(f"Parameter file {parameter_csv} contains duplicated parameters: {repeated}")
    removed = sorted(set(raw["parameter"].astype(str)) & REMOVED_PV_PARAMETERS)
    if removed:
        raise ValueError(
            "Removed PV parameter(s) found in "
            f"{parameter_csv}: {', '.join(removed)}. "
            "PV-BESS sizing now derives capacity candidates adaptively from the 8760 load/PV inputs, "
            "rooftop_area_m2, and pv_module_power_density_kwp_per_m2."
        )

    base: dict[str, Any] = {}
    sensitivity: dict[str, list[float]] = {}
    for row in raw.itertuples(index=False):
        parameter = str(getattr(row, "parameter"))
        base[parameter] = _parse_value(getattr(row, "base_value"))
        values = _parse_sensitivity_range(
            getattr(row, "sensitivity_start"),
            getattr(row, "sensitivity_end"),
            getattr(row, "sensitivity_step"),
            parameter,
        )
        if values:
            if parameter not in SENSITIVITY_PARAMETERS:
                raise ValueError(f"Only the five approved sensitivity parameters can define a range: {parameter}")
            sensitivity[parameter] = values
    return CaseParameters(base=base, sensitivity=sensitivity, raw=raw)


def load_input_case(
    case: InputCase,
    global_config: dict[str, Any],
    parameter_overrides: dict[str, Any] | None = None,
    pv_capacity_kwp: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float | str], dict[str, Any], CaseParameters]:
    parameters = load_case_parameters(case.parameter_csv)
    base = dict(parameters.base)
    if parameter_overrides:
        base.update(parameter_overrides)
    case_config = case_config_from_parameters(global_config, case, base)
    frame, pv_meta = build_case_frame(case, case_config, pv_capacity_kwp)
    return frame, pv_meta, case_config, parameters


def case_config_from_parameters(global_config: dict[str, Any], case: InputCase, params: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(global_config)
    scenario_key = case.case_key
    city_name = str(params.get("city_name") or case.city_key)
    building_name = str(params.get("building_name") or case.building_key)

    config["technical"] = deepcopy(global_config.get("technical", {}))
    config["technical"]["cooling_cop"] = _float_param(params, "cooling_cop", 5.0)
    config["technical"]["chiller_cop"] = config["technical"]["cooling_cop"]
    config["technical"]["heating_cop"] = _float_param(params, "heating_cop", 3.5)
    config["technical"]["timestep_hours"] = _float_param(params, "timestep_hours", 1.0)
    config["technical"]["pv_during_outage_available"] = _bool_param(params, "pv_during_outage_available", True)

    config["pv"] = deepcopy(global_config.get("pv", {}))
    config["pv"]["local_refine_window_kwp"] = _nonnegative_float_param(params, "local_refine_window_kwp", 3000.0)
    config["pv"]["local_refine_step_kwp"] = _positive_float_param(params, "local_refine_step_kwp")
    config["pv"]["rooftop_area_m2"] = _positive_float_param(params, "rooftop_area_m2")
    config["pv"]["module_power_density_kwp_per_m2"] = _positive_float_param(
        params,
        "pv_module_power_density_kwp_per_m2",
    )
    config["pv"]["capex_sgd_per_kwp"] = _float_param(params, "pv_capex_sgd_per_kwp", 1200.0)
    config["pv"]["om_fraction_per_year"] = _float_param(params, "pv_om_fraction_per_year", 0.015)
    config["pv"]["project_years"] = _float_param(params, "pv_project_years", 25.0)

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
        "local_refine_window_kwh": ("local_refine_window_kwh", 15000.0),
        "local_refine_step_kwh": ("local_refine_step_kwh", 500.0),
    }
    for parameter, (config_key, default) in battery_mappings.items():
        config["battery"][config_key] = _float_param(params, parameter, default)

    config["economics"] = deepcopy(global_config.get("economics", {}))
    battery_economics = {
        "battery_capex_sgd_per_kwh": _float_param(params, "battery_capex_sgd_per_kwh", 450.0),
        "battery_power_capex_sgd_per_kw": _float_param(params, "battery_power_capex_sgd_per_kw", 180.0),
        "battery_om_fraction_per_year": _float_param(params, "battery_om_fraction_per_year", 0.02),
        "battery_economics_source": "Input Parameter file.",
    }
    config["economics"].update(battery_economics)

    config["outage"] = deepcopy(global_config.get("outage", {}))
    config["outage"]["seed"] = int(_float_param(params, "outage_seed", 42.0))
    config["outage"]["forced_event_mode"] = str(params.get("forced_event_mode") or "match_saidi")

    config["uncertainty"] = deepcopy(global_config.get("uncertainty", {}))
    config["uncertainty"]["baseline_scenario_count"] = _positive_int_param(
        params,
        "uncertainty_baseline_scenario_count",
    )
    config["uncertainty"]["sensitivity_scenario_count"] = _positive_int_param(
        params,
        "uncertainty_sensitivity_scenario_count",
    )
    config["uncertainty"]["seed_start"] = _nonnegative_int_param(params, "uncertainty_seed_start")
    config["uncertainty"]["parallel_workers"] = _positive_int_param(params, "parallel_workers", default=1)

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
    load = pd.read_csv(case.load_csv)
    pv = pd.read_csv(case.pv_csv)
    _validate_8760(load, case.load_csv)
    _validate_8760(pv, case.pv_csv)
    _validate_required_columns(load, ["Cooling", "Heating", "Lighting", "Equipment"], case.load_csv)
    _validate_required_columns(pv, ["PV_output_kWh_per_m2"], case.pv_csv)

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

    module_power_density = max(float(config["pv"]["module_power_density_kwp_per_m2"]), 1e-9)
    pv_per_kwp = pv["PV_output_kWh_per_m2"].astype(float).to_numpy() / module_power_density
    if np.any(pv_per_kwp < 0):
        raise ValueError(f"Input case {case.case_key} generated negative PV output per kWp")
    annual_pv_per_kwp = float(pv_per_kwp.sum())
    peak_pv_per_kwp = float(pv_per_kwp.max())
    if annual_pv_per_kwp <= 0.0 or peak_pv_per_kwp <= 0.0:
        raise ValueError(f"Input case {case.case_key} must have positive PV_output_kWh_per_m2")

    scenario = config["scenarios"][case.case_key]
    saidi = max(float(scenario.get("saidi_min_per_year", 0.0)), 0.0)
    saifi = max(float(scenario.get("saifi_per_year", 0.0)), 0.0)
    event_hours = saidi / 60.0 / max(saifi, 1e-9) if saidi > 0.0 else 0.0

    pv_annual_kwp = annual_load / annual_pv_per_kwp
    pv_peak_kwp = peak_load / peak_pv_per_kwp
    pv_upper = max(1.10 * pv_annual_kwp, 1.05 * pv_peak_kwp)
    battery_upper = max(0.60 * annual_load / 365.0, 1.50 * peak_load * event_hours)

    search_cfg = config.get("capacity_search", {})
    intervals = max(1, int(search_cfg.get("coarse_intervals", 8)))
    refine_divisions = max(1.0, float(search_cfg.get("local_refine_divisions", 6)))

    pv_candidates, pv_step = _adaptive_axis(pv_upper, intervals=intervals)
    battery_candidates, battery_step = _adaptive_axis(battery_upper, intervals=intervals)
    config["pv"]["capacity_kwp_candidates"] = pv_candidates
    config["pv"]["local_refine_window_kwp"] = pv_step
    config["pv"]["local_refine_step_kwp"] = max(pv_step / refine_divisions, 1e-6)
    config["battery"]["energy_kwh_candidates"] = battery_candidates
    config["battery"]["local_refine_window_kwh"] = battery_step
    config["battery"]["local_refine_step_kwh"] = max(battery_step / refine_divisions, 1e-6)
    config["capacity_search"] = {
        **search_cfg,
        "strategy": "adaptive_from_8760_load_and_pv",
        "coarse_intervals": intervals,
        "local_refine_divisions": refine_divisions,
        "annual_load_e_equiv_kwh": annual_load,
        "peak_load_e_equiv_kw": peak_load,
        "annual_pv_output_kwh_per_kwp": annual_pv_per_kwp,
        "peak_pv_output_kw_per_kwp": peak_pv_per_kwp,
        "pv_upper_kwp": float(max(pv_candidates)),
        "battery_upper_kwh": float(max(battery_candidates)),
        "pv_coarse_step_kwp": float(pv_step),
        "battery_coarse_step_kwh": float(battery_step),
        "event_hours_per_interruption": float(event_hours),
    }


def _adaptive_axis(upper: float, intervals: int) -> tuple[list[float], float]:
    if upper <= 0.0:
        return [0.0, 1.0], 1.0
    step = _nice_step(float(upper) / max(int(intervals), 1))
    return [round(i * step, 6) for i in range(int(intervals) + 1)], step


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
    pv_capacity_kwp: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    load = pd.read_csv(case.load_csv)
    pv = pd.read_csv(case.pv_csv)
    tou = pd.read_csv(case.tou_csv)

    _validate_8760(load, case.load_csv)
    _validate_8760(pv, case.pv_csv)
    _validate_required_columns(load, ["Cooling", "Heating", "Lighting", "Equipment"], case.load_csv)
    _validate_required_columns(pv, ["PV_output_kWh_per_m2"], case.pv_csv)
    _validate_tou(tou, case.tou_csv)

    if "timestamp" in load.columns:
        timestamp = pd.to_datetime(load["timestamp"])
    else:
        timestamp = pd.date_range("2023-01-01", periods=8760, freq="h")
    hour_to_price = dict(zip(tou["hour"].astype(int), tou["tou_sgd_per_kwh"].astype(float)))

    pv_cfg = config["pv"]
    capacity_candidates = [float(value) for value in pv_cfg.get("capacity_kwp_candidates", [])]
    if not capacity_candidates:
        raise ValueError(f"Input case {case.case_key} has no PV capacity candidates")
    active_capacity_kwp = float(max(capacity_candidates) if pv_capacity_kwp is None else pv_capacity_kwp)
    _validate_pv_capacity_value(active_capacity_kwp, case.case_key)
    raw_pv_per_m2 = pv["PV_output_kWh_per_m2"].astype(float)
    rooftop_area_m2 = float(pv_cfg["rooftop_area_m2"])
    module_power_density = float(pv_cfg["module_power_density_kwp_per_m2"])
    pv_per_kwp = raw_pv_per_m2 / module_power_density
    installed_area_m2 = active_capacity_kwp / module_power_density
    rooftop_utilization = installed_area_m2 / rooftop_area_m2

    frame = pd.DataFrame(
        {
            "hour_index": np.arange(8760, dtype=int),
            "timestamp": timestamp,
            "electric_load_kwh": load["Lighting"].astype(float) + load["Equipment"].astype(float),
            "cooling_load_kwh_th": load["Cooling"].astype(float),
            "heating_load_kwh_th": load["Heating"].astype(float),
            "pv_kwh": pv_per_kwp * active_capacity_kwp,
        }
    )
    frame["pv_capacity_kwp"] = active_capacity_kwp
    frame["pv_capacity_mwp"] = active_capacity_kwp / 1000.0
    frame["pv_installed_area_m2"] = float(installed_area_m2)
    frame["pv_rooftop_utilization_fraction"] = float(rooftop_utilization)
    frame["pv_rooftop_utilization_percent"] = float(rooftop_utilization * 100.0)
    frame["tou_sgd_per_kwh"] = frame["timestamp"].dt.hour.map(hour_to_price).astype(float)
    if frame.isna().any().any():
        raise ValueError(f"Input case {case.case_key} generated missing values")

    pv_meta = {
        "scenario": case.case_key,
        "pv_capacity_kwp": float(active_capacity_kwp),
        "pv_capacity_mwp": float(active_capacity_kwp / 1000.0),
        "rooftop_area_m2": rooftop_area_m2,
        "pv_installed_area_m2": float(installed_area_m2),
        "pv_rooftop_utilization_fraction": float(rooftop_utilization),
        "pv_rooftop_utilization_percent": float(rooftop_utilization * 100.0),
        "pv_module_power_density_kwp_per_m2": module_power_density,
        "pv_capacity_note": "PV output is scaled from 1 m2 generation through module power density and PV capacity.",
    }
    return frame, pv_meta


def sensitivity_runs(parameters: CaseParameters) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for parameter in SENSITIVITY_PARAMETERS:
        for value in parameters.sensitivity.get(parameter, []):
            runs.append({"sensitivity_parameter": parameter, "sensitivity_value": value, parameter: value})
    return runs


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


def _positive_int_param(params: dict[str, Any], name: str, default: int | None = None) -> int:
    value = params.get(name)
    if value is None:
        if default is None:
            raise ValueError(f"Parameter {name} is required and must be a positive integer")
        value = default
    result = int(float(value))
    if result <= 0 or float(value) != float(result):
        raise ValueError(f"Parameter {name} must be a positive integer, got {value}")
    return result


def _nonnegative_int_param(params: dict[str, Any], name: str) -> int:
    value = params.get(name)
    if value is None:
        raise ValueError(f"Parameter {name} is required and must be a non-negative integer")
    result = int(float(value))
    if result < 0 or float(value) != float(result):
        raise ValueError(f"Parameter {name} must be a non-negative integer, got {value}")
    return result


def _nonnegative_float_param(params: dict[str, Any], name: str, default: float) -> float:
    value = params.get(name)
    result = float(default if value is None else value)
    if result < 0.0:
        raise ValueError(f"Parameter {name} must be >= 0, got {result}")
    return result


def _optional_float_param(params: dict[str, Any], name: str) -> float | None:
    value = params.get(name)
    if value is None:
        return None
    return float(value)


def _bool_param(params: dict[str, Any], name: str, default: bool) -> bool:
    value = params.get(name)
    return bool(default if value is None else value)


def _list_param(params: dict[str, Any], name: str, default: Any) -> list[float]:
    value = params.get(name)
    if value is None:
        value = default
    if not isinstance(value, list):
        raise ValueError(f"Parameter {name} must be a list")
    return [float(item) for item in value]


def _pv_capacity_candidates_param(params: dict[str, Any], name: str) -> list[float]:
    if params.get(name) is None:
        raise ValueError(f"Parameter {name} is required and must be a non-empty list")
    values = _list_param(params, name, None)
    if not values:
        raise ValueError(f"Parameter {name} must not be empty")
    for value in values:
        _validate_pv_capacity_value(value, name)
    return sorted(set(values))


def _validate_pv_capacity_value(value: float, source: str) -> None:
    if value < 0.0:
        raise ValueError(f"PV capacity for {source} must be >= 0 kWp, got {value}")


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
