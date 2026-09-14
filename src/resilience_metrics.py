from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


DEFAULT_LOLP_THRESHOLD = 0.10
DEFAULT_TOLERANCE = 1e-6


def electric_equivalent_building_load(
    frame: pd.DataFrame,
    cooling_cop: float,
    heating_cop: float,
) -> pd.Series:
    """Return full-building electric-equivalent load for each timestep."""
    cooling_cop = _positive_cop(cooling_cop, "cooling_cop")
    heating_cop = _positive_cop(heating_cop, "heating_cop")
    return (
        frame["electric_load_kwh"].astype(float)
        + frame["cooling_load_kwh_th"].astype(float) / cooling_cop
        + frame["heating_load_kwh_th"].astype(float) / heating_cop
    )


def outage_weighted_load_kwh(
    total_load_e_equiv_kwh: pd.Series,
    outage_fraction: pd.Series,
) -> pd.Series:
    """Return full-building load weighted by outage fraction."""
    load = total_load_e_equiv_kwh.astype(float)
    outage = outage_fraction.astype(float)
    return pd.Series(load.to_numpy() * outage.to_numpy(), index=load.index, name="outage_weighted_load_kwh")


def ue(dispatch: pd.DataFrame, tolerance: float = DEFAULT_TOLERANCE) -> float:
    """Return unserved energy from dispatch unmet-load output."""
    unmet = dispatch["unmet_load_kwh_equiv"].astype(float)
    return float(unmet.where(unmet > float(tolerance), 0.0).sum())


def outage_load_served_ratio(
    dispatch: pd.DataFrame,
    outage_weighted_load: pd.Series,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Return the share of outage-weighted full-building load served."""
    outage_load = float(outage_weighted_load.astype(float).sum())
    if outage_load <= 0.0:
        return 1.0
    return float(1.0 - ue(dispatch, tolerance=tolerance) / outage_load)


def reliability_from_ue(
    ue_kwh_per_year: float,
    outage_weighted_load_total_kwh: float,
) -> dict[str, float]:
    """Return common outage reliability metrics from UE and outage load."""
    scenario_ue = float(ue_kwh_per_year)
    outage_load = float(outage_weighted_load_total_kwh)
    if outage_load <= 0.0:
        return {
            "outage_weighted_load_kwh": outage_load,
            "UE_kwh_per_year": scenario_ue,
            "outage_load_served_ratio": 1.0,
            "scenario_LOLP": 0.0,
        }
    scenario_lolp_value = scenario_ue / outage_load
    return {
        "outage_weighted_load_kwh": outage_load,
        "UE_kwh_per_year": scenario_ue,
        "outage_load_served_ratio": 1.0 - scenario_lolp_value,
        "scenario_LOLP": scenario_lolp_value,
    }


def scenario_lolp(
    dispatch: pd.DataFrame,
    outage_weighted_load: pd.Series,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Return unserved-energy share of outage-weighted full-building load."""
    outage_load = float(outage_weighted_load.astype(float).sum())
    if outage_load <= 0.0:
        return 0.0
    return float(ue(dispatch, tolerance=tolerance) / outage_load)


def mean_scenario_lolp(scenario_lolps: Sequence[float]) -> float:
    """Return mean scenario LOLP across outage scenarios."""
    values = _scenario_values(scenario_lolps, "scenario_lolps")
    return float(sum(values) / len(values))


def chance_constraint_probability(
    scenario_lolps: Sequence[float],
    lolp_threshold: float = DEFAULT_LOLP_THRESHOLD,
) -> float:
    """Return the proportion of scenarios satisfying the LOLP threshold."""
    values = _scenario_values(scenario_lolps, "scenario_lolps")
    threshold = float(lolp_threshold)
    return float(sum(value <= threshold for value in values) / len(values))


def _positive_cop(value: float, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _scenario_values(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must contain at least one scenario")
    return result
