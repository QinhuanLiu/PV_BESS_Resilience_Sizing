from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TOU_SPREAD_K_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]


@dataclass(frozen=True)
class TouSpreadRun:
    spread_k: float
    profile: pd.DataFrame
    base_mean_sgd_per_kwh: float
    base_min_sgd_per_kwh: float
    base_max_sgd_per_kwh: float
    adjusted_mean_sgd_per_kwh: float
    adjusted_min_sgd_per_kwh: float
    adjusted_max_sgd_per_kwh: float

    @property
    def base_spread_sgd_per_kwh(self) -> float:
        return self.base_max_sgd_per_kwh - self.base_min_sgd_per_kwh

    @property
    def adjusted_spread_sgd_per_kwh(self) -> float:
        return self.adjusted_max_sgd_per_kwh - self.adjusted_min_sgd_per_kwh


def normalize_tou_spread_k_values(values: list[float] | None) -> list[float]:
    raw_values = DEFAULT_TOU_SPREAD_K_VALUES if values is None else values
    normalized: list[float] = []
    for value in raw_values:
        spread_k = float(value)
        if spread_k < 0.0:
            raise ValueError(f"TOU spread k must be >= 0, got {spread_k:g}")
        if spread_k not in normalized:
            normalized.append(spread_k)
    return normalized


def build_tou_spread_runs(tou_csv: str | Path, spread_k_values: list[float] | None = None) -> list[TouSpreadRun]:
    tou = pd.read_csv(tou_csv)
    return [
        build_tou_spread_run(tou, spread_k)
        for spread_k in normalize_tou_spread_k_values(spread_k_values)
    ]


def build_tou_spread_run(tou: pd.DataFrame, spread_k: float) -> TouSpreadRun:
    base = _validated_tou_prices(tou)
    adjusted = mean_preserving_tou_profile(base, spread_k)
    base_values = base["tou_sgd_per_kwh"].astype(float)
    adjusted_values = adjusted["tou_sgd_per_kwh"].astype(float)
    return TouSpreadRun(
        spread_k=float(spread_k),
        profile=adjusted,
        base_mean_sgd_per_kwh=float(base_values.mean()),
        base_min_sgd_per_kwh=float(base_values.min()),
        base_max_sgd_per_kwh=float(base_values.max()),
        adjusted_mean_sgd_per_kwh=float(adjusted_values.mean()),
        adjusted_min_sgd_per_kwh=float(adjusted_values.min()),
        adjusted_max_sgd_per_kwh=float(adjusted_values.max()),
    )


def mean_preserving_tou_profile(tou: pd.DataFrame, spread_k: float) -> pd.DataFrame:
    if float(spread_k) < 0.0:
        raise ValueError(f"TOU spread k must be >= 0, got {float(spread_k):g}")
    profile = _validated_tou_prices(tou)
    values = profile["tou_sgd_per_kwh"].astype(float)
    mean_value = float(values.mean())
    adjusted = mean_value + float(spread_k) * (values - mean_value)
    if float(adjusted.min()) < -1e-9:
        raise ValueError(f"TOU spread k={float(spread_k):g} creates a negative tariff.")
    profile["tou_sgd_per_kwh"] = adjusted.clip(lower=0.0)
    return profile


def apply_tou_profile(frame: pd.DataFrame, tou_profile: pd.DataFrame) -> pd.DataFrame:
    profile = _validated_tou_prices(tou_profile)
    mapping = dict(zip(profile["hour"].astype(int), profile["tou_sgd_per_kwh"].astype(float)))
    adjusted = frame.copy()
    if "timestamp" in adjusted.columns:
        hours = pd.to_datetime(adjusted["timestamp"]).dt.hour
    elif "hour_index" in adjusted.columns:
        hours = adjusted["hour_index"].astype(int) % 24
    else:
        raise ValueError("TOU profile application requires timestamp or hour_index in the frame.")
    adjusted["tou_sgd_per_kwh"] = hours.map(mapping).astype(float)
    if adjusted["tou_sgd_per_kwh"].isna().any():
        raise ValueError("Adjusted TOU profile did not cover every frame hour.")
    return adjusted


def tou_analysis_result_row(run: TouSpreadRun, design_rows: pd.DataFrame) -> dict[str, Any]:
    if design_rows.empty:
        raise ValueError(f"TOU analysis run for k={run.spread_k:g} produced no design rows")
    row = dict(design_rows.iloc[0])
    return {
        "tou_spread_k": run.spread_k,
        "tou_base_mean_sgd_per_kwh": run.base_mean_sgd_per_kwh,
        "tou_base_min_sgd_per_kwh": run.base_min_sgd_per_kwh,
        "tou_base_max_sgd_per_kwh": run.base_max_sgd_per_kwh,
        "tou_base_spread_sgd_per_kwh": run.base_spread_sgd_per_kwh,
        "tou_adjusted_mean_sgd_per_kwh": run.adjusted_mean_sgd_per_kwh,
        "tou_adjusted_min_sgd_per_kwh": run.adjusted_min_sgd_per_kwh,
        "tou_adjusted_max_sgd_per_kwh": run.adjusted_max_sgd_per_kwh,
        "tou_adjusted_spread_sgd_per_kwh": run.adjusted_spread_sgd_per_kwh,
        **row,
    }


def order_tou_analysis_summary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "case_key",
        "city_key",
        "building_key",
        "scenario_name",
        "tou_spread_k",
        "tou_base_mean_sgd_per_kwh",
        "tou_adjusted_mean_sgd_per_kwh",
        "tou_base_min_sgd_per_kwh",
        "tou_base_max_sgd_per_kwh",
        "tou_base_spread_sgd_per_kwh",
        "tou_adjusted_min_sgd_per_kwh",
        "tou_adjusted_max_sgd_per_kwh",
        "tou_adjusted_spread_sgd_per_kwh",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "battery_power_kw",
        "pv_bess_system_annual_cost_sgd",
        "system_total_cost_sgd_per_year",
        "system_total_cost_million_sgd_per_year",
        "unserved_energy_kwh_per_year",
        "UE_kwh_per_year",
        "UE_mwh_per_year",
        "unserved_energy_cost_sgd_per_year",
        "energy_based_LOLP",
        "outage_load_served_ratio",
        "outage_weighted_load_kwh",
        "saidi_min_per_year",
        "saifi_per_year",
        "voll_sgd_per_kwh",
        "simulated_saidi_min_per_year",
        "simulated_saifi_per_year",
        "search_stage",
        "search_iteration",
        "coarse_search_iteration",
        "optimal_region_iteration",
    ]
    ordered = [column for column in preferred if column in frame.columns]
    ordered.extend(column for column in frame.columns if column not in ordered)
    return frame[ordered]


def _validated_tou_prices(tou: pd.DataFrame) -> pd.DataFrame:
    required = {"hour", "tou_sgd_per_kwh"}
    missing = required - set(tou.columns)
    if missing:
        raise ValueError("TOU spread analysis requires column(s): " + ", ".join(sorted(missing)))
    profile = tou[["hour", "tou_sgd_per_kwh"]].copy()
    profile["hour"] = pd.to_numeric(profile["hour"], errors="coerce").astype("Int64")
    profile["tou_sgd_per_kwh"] = pd.to_numeric(profile["tou_sgd_per_kwh"], errors="coerce")
    if profile[["hour", "tou_sgd_per_kwh"]].isna().any().any():
        raise ValueError("TOU spread analysis received non-numeric TOU values.")
    profile = profile.sort_values("hour").reset_index(drop=True)
    if profile["hour"].astype(int).tolist() != list(range(24)):
        raise ValueError("TOU spread analysis requires hours 0-23 exactly.")
    if (profile["tou_sgd_per_kwh"].astype(float) < 0.0).any():
        raise ValueError("TOU spread analysis requires non-negative base tariffs.")
    return profile
