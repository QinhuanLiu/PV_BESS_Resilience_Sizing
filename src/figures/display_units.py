from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


SMALL_CAPACITY_MW_THRESHOLD = 1.0
SMALL_COST_SGD_THRESHOLD = 1_000_000.0

DISPLAY_PV_CAPACITY = "_display_pv_capacity"
DISPLAY_BATTERY_ENERGY = "_display_battery_energy"
DISPLAY_SYSTEM_COST = "_display_system_total_cost"
DISPLAY_UE = "_display_UE"


@dataclass(frozen=True)
class FigureDisplayUnits:
    use_kw: bool
    use_battery_kwh: bool
    use_ue_kwh: bool
    use_ksgd: bool

    @property
    def pv_axis_label(self) -> str:
        return "PV capacity (kW)" if self.use_kw else "PV capacity (MW)"

    @property
    def battery_axis_label(self) -> str:
        return "BESS energy (kWh)" if self.use_battery_kwh else "BESS energy (MWh)"

    @property
    def battery_capacity_label(self) -> str:
        return "Battery capacity (kWh)" if self.use_battery_kwh else "Battery capacity (MWh)"

    @property
    def ue_axis_label(self) -> str:
        return "UE (kWh yr$^{-1}$)" if self.use_ue_kwh else "UE (MWh yr$^{-1}$)"

    @property
    def ue_plain_label(self) -> str:
        return "kWh/year" if self.use_ue_kwh else "MWh/year"

    @property
    def cost_axis_label(self) -> str:
        return "Cost (kSGD yr$^{-1}$)" if self.use_ksgd else "Cost (million SGD yr$^{-1}$)"

    @property
    def annual_cost_axis_label(self) -> str:
        return "Annualized cost (kSGD yr$^{-1}$)" if self.use_ksgd else "Annualized cost (million SGD yr$^{-1}$)"

    @property
    def cost_plain_label(self) -> str:
        return "kSGD/yr" if self.use_ksgd else "M SGD/yr"

    @property
    def cost_divisor(self) -> float:
        return 1_000.0 if self.use_ksgd else 1_000_000.0

    @property
    def pv_unit(self) -> str:
        return "kW" if self.use_kw else "MW"

    @property
    def battery_unit(self) -> str:
        return "kWh" if self.use_battery_kwh else "MWh"


def figure_display_units(source: pd.DataFrame) -> FigureDisplayUnits:
    return FigureDisplayUnits(
        use_kw=_use_small_capacity(source, "pv_capacity_kw", "pv_capacity_mw"),
        use_battery_kwh=_use_small_capacity(source, "battery_energy_kwh", "battery_energy_mwh"),
        use_ue_kwh=_use_small_energy(source, "UE_kwh_per_year", "UE_mwh_per_year", "unserved_energy_kwh_per_year"),
        use_ksgd=_use_small_cost(source),
    )


def add_display_unit_columns(
    source: pd.DataFrame,
    units: FigureDisplayUnits | None = None,
) -> tuple[pd.DataFrame, FigureDisplayUnits]:
    prepared = source.copy()
    selected = units or figure_display_units(prepared)
    prepared[DISPLAY_PV_CAPACITY] = capacity_display_series(
        prepared,
        "pv_capacity_kw",
        "pv_capacity_mw",
        selected.use_kw,
    )
    prepared[DISPLAY_BATTERY_ENERGY] = capacity_display_series(
        prepared,
        "battery_energy_kwh",
        "battery_energy_mwh",
        selected.use_battery_kwh,
    )
    prepared[DISPLAY_SYSTEM_COST] = system_cost_display_series(prepared, selected)
    prepared[DISPLAY_UE] = ue_display_series(prepared, selected)
    return prepared, selected


def capacity_display_series(
    source: pd.DataFrame,
    k_column: str,
    m_column: str,
    use_small_unit: bool,
) -> pd.Series:
    if use_small_unit:
        if k_column in source.columns:
            return pd.to_numeric(source[k_column], errors="coerce")
        return pd.to_numeric(source[m_column], errors="coerce") * 1000.0
    if m_column in source.columns:
        return pd.to_numeric(source[m_column], errors="coerce")
    return pd.to_numeric(source[k_column], errors="coerce") / 1000.0


def system_cost_display_series(source: pd.DataFrame, units: FigureDisplayUnits) -> pd.Series:
    return _raw_system_cost_sgd(source) / units.cost_divisor


def cost_component_display_series(
    source: pd.DataFrame,
    units: FigureDisplayUnits,
    *,
    sgd_column: str,
    million_column: str | None = None,
) -> pd.Series:
    if units.use_ksgd:
        if sgd_column in source.columns:
            return pd.to_numeric(source[sgd_column], errors="coerce") / 1000.0
        if million_column and million_column in source.columns:
            return pd.to_numeric(source[million_column], errors="coerce") * 1000.0
    if million_column and million_column in source.columns:
        return pd.to_numeric(source[million_column], errors="coerce")
    return pd.to_numeric(source[sgd_column], errors="coerce") / 1_000_000.0


def ue_display_series(source: pd.DataFrame, units: FigureDisplayUnits) -> pd.Series:
    if units.use_ue_kwh:
        if "UE_kwh_per_year" in source.columns:
            return pd.to_numeric(source["UE_kwh_per_year"], errors="coerce")
        if "unserved_energy_kwh_per_year" in source.columns:
            return pd.to_numeric(source["unserved_energy_kwh_per_year"], errors="coerce")
        return pd.to_numeric(source["UE_mwh_per_year"], errors="coerce") * 1000.0
    if "UE_mwh_per_year" in source.columns:
        return pd.to_numeric(source["UE_mwh_per_year"], errors="coerce")
    if "UE_kwh_per_year" in source.columns:
        return pd.to_numeric(source["UE_kwh_per_year"], errors="coerce") / 1000.0
    return pd.to_numeric(source["unserved_energy_kwh_per_year"], errors="coerce") / 1000.0


def format_pv_capacity_from_kw(value_kw: float, units: FigureDisplayUnits) -> str:
    value = float(value_kw) if units.use_kw else float(value_kw) / 1000.0
    return f"{_format_quantity(value)} {units.pv_unit}"


def format_battery_energy_from_kwh(value_kwh: float, units: FigureDisplayUnits) -> str:
    value = float(value_kwh) if units.use_battery_kwh else float(value_kwh) / 1000.0
    return f"{_format_quantity(value)} {units.battery_unit}"


def format_design_label(row: pd.Series | dict[str, Any], units: FigureDisplayUnits) -> str:
    if _row_value(row, "design_case", "") == "no_pv_bess":
        return "No PV/BESS"
    pv_kw = _row_value(row, "pv_capacity_kw", np.nan)
    if pd.isna(pv_kw):
        pv_kw = float(_row_value(row, "pv_capacity_mw", 0.0)) * 1000.0
    battery_kwh = _row_value(row, "battery_energy_kwh", np.nan)
    if pd.isna(battery_kwh):
        battery_kwh = float(_row_value(row, "battery_energy_mwh", 0.0)) * 1000.0
    return f"{format_pv_capacity_from_kw(float(pv_kw), units)} PV / {format_battery_energy_from_kwh(float(battery_kwh), units)} Battery"


def format_energy_from_kwh(value_kwh: float, use_kwh: bool) -> str:
    if use_kwh:
        return f"{_format_quantity(float(value_kwh))} kWh"
    return f"{_format_quantity(float(value_kwh) / 1000.0)} MWh"


def nice_step(raw_step: float) -> float:
    if raw_step <= 0.0 or not np.isfinite(raw_step):
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = float(multiplier * magnitude)
        if step >= raw_step - 1e-12:
            return step
    return float(10.0 * magnitude)


def nice_levels(values: np.ndarray, *, target_intervals: int = 22) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot build display levels with no finite values.")
    low = float(finite.min())
    high = float(finite.max())
    if np.isclose(low, high):
        pad = max(abs(low) * 0.05, 1.0)
        return np.linspace(low - pad, high + pad, 3)
    step = nice_step((high - low) / max(int(target_intervals), 1))
    lower = np.floor(low / step) * step
    upper = np.ceil(high / step) * step
    if np.isclose(lower, upper):
        upper = lower + step
    return np.arange(lower, upper + step * 0.5, step, dtype=float)


def sparse_ticks_from_levels(levels: np.ndarray, *, target_count: int = 6) -> np.ndarray:
    finite = np.asarray(levels, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size <= target_count:
        return finite
    step = nice_step((float(finite[-1]) - float(finite[0])) / max(target_count - 1, 1))
    lower = np.ceil(float(finite[0]) / step) * step
    upper = np.floor(float(finite[-1]) / step) * step
    ticks = np.arange(lower, upper + step * 0.5, step, dtype=float)
    if ticks.size < 2:
        return np.linspace(float(finite[0]), float(finite[-1]), min(target_count, finite.size))
    return ticks


def _use_small_capacity(source: pd.DataFrame, k_column: str, m_column: str) -> bool:
    if m_column in source.columns:
        values_m = pd.to_numeric(source[m_column], errors="coerce")
    elif k_column in source.columns:
        values_m = pd.to_numeric(source[k_column], errors="coerce") / 1000.0
    else:
        return False
    return _finite_max(values_m) < SMALL_CAPACITY_MW_THRESHOLD


def _use_small_energy(source: pd.DataFrame, k_column: str, m_column: str, fallback_k_column: str) -> bool:
    if m_column in source.columns:
        values_m = pd.to_numeric(source[m_column], errors="coerce")
    elif k_column in source.columns:
        values_m = pd.to_numeric(source[k_column], errors="coerce") / 1000.0
    elif fallback_k_column in source.columns:
        values_m = pd.to_numeric(source[fallback_k_column], errors="coerce") / 1000.0
    else:
        return False
    return _finite_max(values_m) < SMALL_CAPACITY_MW_THRESHOLD


def _use_small_cost(source: pd.DataFrame) -> bool:
    if not any(column in source.columns for column in ("system_total_cost_sgd_per_year", "pv_bess_system_annual_cost_sgd")):
        if "system_total_cost_million_sgd_per_year" not in source.columns:
            return False
    return _finite_max(_raw_system_cost_sgd(source)) < SMALL_COST_SGD_THRESHOLD


def _raw_system_cost_sgd(source: pd.DataFrame) -> pd.Series:
    if "system_total_cost_sgd_per_year" in source.columns:
        return pd.to_numeric(source["system_total_cost_sgd_per_year"], errors="coerce")
    if "pv_bess_system_annual_cost_sgd" in source.columns:
        return pd.to_numeric(source["pv_bess_system_annual_cost_sgd"], errors="coerce")
    return pd.to_numeric(source["system_total_cost_million_sgd_per_year"], errors="coerce") * 1_000_000.0


def _finite_max(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if finite.empty:
        return float("inf")
    return float(finite.abs().max())


def _format_quantity(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value), atol=1e-9):
        return f"{int(round(value))}"
    if abs(value) >= 100.0:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs(value) >= 10.0:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _row_value(row: pd.Series | dict[str, Any], key: str, default: Any) -> Any:
    if isinstance(row, pd.Series):
        return row[key] if key in row.index else default
    return row.get(key, default)
