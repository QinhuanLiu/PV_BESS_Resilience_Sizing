from __future__ import annotations

from pathlib import Path

import pandas as pd

from .core import write_source_data
from .display_units import DISPLAY_BATTERY_ENERGY, DISPLAY_PV_CAPACITY, DISPLAY_SYSTEM_COST
from .lolp_zero_uncertainty import (
    DistributionSpec,
    _metric_source,
    build_lolp_zero_uncertainty_figure_source,
    plot_lolp_zero_distribution,
)


FIGURE2_COST_NAME = "Figure2_LOLP_Zero_Three_Year_Uncertainty_Annualized_Cost_Distribution"
FIGURE2_PV_NAME = "Figure2_LOLP_Zero_Three_Year_Uncertainty_PV_Capacity_Distribution"
FIGURE2_BATTERY_NAME = "Figure2_LOLP_Zero_Three_Year_Uncertainty_Battery_Capacity_Distribution"


def make_lolp_zero_three_year_uncertainty_figures(
    out: Path,
    case_key: str,
    all_results: pd.DataFrame,
    x_limits_by_metric: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Path]:
    """Create Figure 2's three multi-year LOLP=0 uncertainty distribution figures."""
    source, default_specs = build_lolp_zero_uncertainty_figure_source(all_results)
    units = {spec.metric_key: spec for spec in default_specs}
    specs = (
        DistributionSpec(
            figure_name=FIGURE2_COST_NAME,
            metric_key="annualized_cost",
            source_column="system_total_cost_sgd_per_year",
            display_column=DISPLAY_SYSTEM_COST,
            title="Manila three-year LOLP=0 annualized cost",
            xlabel=units["annualized_cost"].xlabel,
            unit_label=units["annualized_cost"].unit_label,
        ),
        DistributionSpec(
            figure_name=FIGURE2_PV_NAME,
            metric_key="pv_capacity",
            source_column="pv_capacity_kw",
            display_column=DISPLAY_PV_CAPACITY,
            title="Manila three-year LOLP=0 PV capacity",
            xlabel=units["pv_capacity"].xlabel,
            unit_label=units["pv_capacity"].unit_label,
        ),
        DistributionSpec(
            figure_name=FIGURE2_BATTERY_NAME,
            metric_key="battery_capacity",
            source_column="battery_energy_kwh",
            display_column=DISPLAY_BATTERY_ENERGY,
            title="Manila three-year LOLP=0 BESS capacity",
            xlabel=units["battery_capacity"].xlabel,
            unit_label=units["battery_capacity"].unit_label,
        ),
    )
    paths: dict[str, Path] = {}
    for spec in specs:
        base = out / f"{case_key}_{spec.figure_name}"
        paths[f"figure2_{spec.metric_key}_source_data"] = write_source_data(base, _metric_source(source, spec))
        x_limits = x_limits_by_metric.get(spec.metric_key) if x_limits_by_metric else None
        paths.update(plot_lolp_zero_distribution(out, case_key, source, spec, x_limits=x_limits))
    return paths
