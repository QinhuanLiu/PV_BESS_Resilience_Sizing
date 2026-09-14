from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from ..io.data_loader import InputCase, build_case_frame
from ..optimizer import scenario_config
from ..resilience_metrics import (
    DEFAULT_LOLP_THRESHOLD,
    chance_constraint_probability,
    electric_equivalent_building_load,
)
from .core import NATURE_PALETTE, _matplotlib, _save_all, _style, _write_source_data


FIGURE_BESS_INTERPRETATION_NO = 12
FIGURE_BESS_INTERPRETATION_NAME = "Figure_12_BESS_Capacity_Physical_Interpretation"

REQUIRED_EVENT_COLUMNS = (
    "frontier_scenario_id",
    "outage_duration_hours",
    "outage_weighted_load_kwh",
    "pv_available_kwh",
    "net_deficit_kwh",
    "scenario_probability",
)

REQUIRED_SLICE_COLUMNS = (
    "pv_capacity_kw",
    "battery_energy_kwh",
    "battery_energy_mwh",
    "system_total_cost_million_sgd_per_year",
    "annualized_battery_cost_million_sgd_per_year",
    "expected_outage_loss_million_sgd_per_year",
    "mean_UE_mwh_per_year",
    "energy_based_LOLP",
    "chance_constraint_probability",
    "is_optimal_bess",
)

REQUIRED_SUMMARY_COLUMNS = (
    "selected_design_type",
    "optimal_pv_capacity_kw",
    "slice_pv_capacity_kw",
    "optimal_battery_energy_kwh",
    "usable_bess_energy_kwh",
    "net_deficit_p50_kwh",
    "net_deficit_p80_kwh",
    "net_deficit_p90_kwh",
    "net_deficit_p95_kwh",
    "net_deficit_coverage_probability",
    "support_hours",
    "soc_min_fraction",
    "soc_max_fraction",
    "roundtrip_efficiency",
)


def build_bess_capacity_interpretation_source(
    case: InputCase,
    config: dict[str, Any],
    optimal_designs: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
    scenario_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build source tables for the BESS capacity physical-interpretation figure."""
    design = _select_interpretation_design(optimal_designs)
    aware_cfg = scenario_config(config, case.case_key)
    optimal_pv = float(design["pv_capacity_kw"])
    optimal_bess = float(design["battery_energy_kwh"])
    case_frame, _ = build_case_frame(case, aware_cfg, optimal_pv)
    critical_load = _critical_load(case_frame, aware_cfg)

    event_source = _build_event_source(case_frame, critical_load, outage_scenarios)
    capacity_slice_source = _build_capacity_slice_source(scenario_results, design)
    summary_source = _build_summary_source(event_source, capacity_slice_source, design, aware_cfg)
    return event_source, capacity_slice_source, summary_source


def make_bess_capacity_interpretation_figure(
    out: Path,
    case_key: str,
    event_source: pd.DataFrame,
    capacity_slice_source: pd.DataFrame,
    summary_source: pd.DataFrame,
) -> dict[str, Path]:
    """Create the four-panel Nature-style BESS capacity interpretation figure."""
    events = validate_bess_event_source(event_source)
    capacity_slice = validate_bess_capacity_slice_source(capacity_slice_source)
    summary = validate_bess_summary_source(summary_source)

    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig = plt.figure(figsize=(7.4, 5.6), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.88,
        bottom=0.09,
        top=0.94,
        wspace=0.34,
        hspace=0.34,
        width_ratios=[1.0, 1.22],
        height_ratios=[1.04, 1.0],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    palette = _palette()
    _plot_panel_a(ax_a, events, palette)
    _plot_panel_b(ax_b, events, summary.iloc[0], palette)
    _plot_panel_c(ax_c, capacity_slice, summary.iloc[0], palette)
    _plot_panel_d(ax_d, capacity_slice, summary.iloc[0], palette)
    for label, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d), strict=True):
        _panel_label(ax, label)

    base = out / f"{case_key}_{FIGURE_BESS_INTERPRETATION_NAME}"
    paths = _save_all(fig, base)
    plt.close(fig)
    paths[f"{base.name}_event_source_data"] = _write_source_data(base.with_name(base.name + "_Event"), events)
    paths[f"{base.name}_capacity_slice_source_data"] = _write_source_data(
        base.with_name(base.name + "_Capacity_Slice"),
        capacity_slice,
    )
    paths[f"{base.name}_summary_source_data"] = _write_source_data(base.with_name(base.name + "_Summary"), summary)
    return paths


def make_demo_bess_capacity_interpretation_figure(out: Path, case_key: str = "Demo") -> dict[str, Path]:
    """Generate a self-contained demonstration figure with synthetic source data."""
    event_source, capacity_slice_source, summary_source = _demo_sources()
    demo_case_key = f"{case_key}_Figure_X_BESS_Capacity_Physical_Interpretation_Demo"
    paths = make_bess_capacity_interpretation_figure(
        out,
        demo_case_key,
        event_source,
        capacity_slice_source,
        summary_source,
    )
    desired_base = out / demo_case_key
    generated_base = out / f"{demo_case_key}_{FIGURE_BESS_INTERPRETATION_NAME}"
    for suffix in ("png", "svg", "pdf"):
        generated = generated_base.with_suffix(f".{suffix}")
        desired = desired_base.with_suffix(f".{suffix}")
        if generated.exists():
            shutil.copyfile(generated, desired)
            paths[f"{demo_case_key}_{suffix}"] = desired
    return paths


def validate_bess_event_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_EVENT_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("BESS capacity event source is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("BESS capacity event source data is empty.")
    validated = source.copy()
    for column in REQUIRED_EVENT_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(REQUIRED_EVENT_COLUMNS)].isna().any().any():
        raise ValueError("BESS capacity event source contains missing or non-numeric required values.")
    if (validated["scenario_probability"] < 0).any():
        raise ValueError("BESS capacity event source contains negative scenario probabilities.")
    prob_sum = float(validated["scenario_probability"].sum())
    if prob_sum <= 0.0:
        raise ValueError("BESS capacity event source scenario probabilities must sum to a positive value.")
    validated["scenario_probability"] = validated["scenario_probability"] / prob_sum
    if "pv_availability_ratio" not in validated.columns:
        denom = validated["outage_weighted_load_kwh"].replace(0, np.nan)
        validated["pv_availability_ratio"] = (validated["pv_available_kwh"] / denom).fillna(0.0)
    return validated.sort_values("frontier_scenario_id").reset_index(drop=True)


def validate_bess_capacity_slice_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SLICE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("BESS capacity slice source is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("BESS capacity slice source data is empty.")
    validated = source.copy()
    for column in REQUIRED_SLICE_COLUMNS:
        if column == "is_optimal_bess":
            validated[column] = validated[column].map(_truthy)
        else:
            validated[column] = pd.to_numeric(validated[column], errors="coerce")
    numeric = [column for column in REQUIRED_SLICE_COLUMNS if column != "is_optimal_bess"]
    if validated[numeric].isna().any().any():
        raise ValueError("BESS capacity slice source contains missing or non-numeric required values.")
    return validated.sort_values("battery_energy_kwh").reset_index(drop=True)


def validate_bess_summary_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SUMMARY_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("BESS capacity summary source is missing required column(s): " + ", ".join(missing))
    if len(source) != 1:
        raise ValueError("BESS capacity summary source must contain exactly one row.")
    validated = source.copy()
    for column in REQUIRED_SUMMARY_COLUMNS:
        if column == "selected_design_type":
            validated[column] = validated[column].astype(str)
        else:
            validated[column] = pd.to_numeric(validated[column], errors="coerce")
    numeric = [column for column in REQUIRED_SUMMARY_COLUMNS if column != "selected_design_type"]
    if validated[numeric].isna().any().any():
        raise ValueError("BESS capacity summary source contains missing or non-numeric required values.")
    return validated.reset_index(drop=True)


def _select_interpretation_design(optimal_designs: pd.DataFrame) -> dict[str, Any]:
    required = {"design_type", "available", "pv_capacity_kw", "battery_energy_kwh"}
    missing = sorted(required - set(optimal_designs.columns))
    if missing:
        raise ValueError("Optimal designs are missing required column(s): " + ", ".join(missing))
    for design_type in ("reliability_constrained_design", "minimum_total_cost_design"):
        matches = optimal_designs[
            (optimal_designs["design_type"].astype(str) == design_type)
            & optimal_designs["available"].map(_truthy)
        ]
        if not matches.empty:
            row = dict(matches.iloc[0])
            row["selected_design_type"] = design_type
            return row
    raise ValueError("No available design found for BESS capacity interpretation.")


def _build_event_source(
    case_frame: pd.DataFrame,
    critical_load: pd.Series,
    outage_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    if outage_scenarios.empty or "_outage_profile" not in outage_scenarios.columns:
        raise ValueError("BESS capacity interpretation requires outage_scenarios with _outage_profile objects.")
    rows: list[dict[str, Any]] = []
    scenario_count = len(outage_scenarios)
    pv = case_frame["pv_kwh"].astype(float).to_numpy()
    load = critical_load.astype(float).to_numpy()
    for scenario in outage_scenarios.sort_values("frontier_scenario_id").to_dict("records"):
        profile = scenario["_outage_profile"]
        outage_fraction = profile["outage_fraction"].astype(float).to_numpy()
        duration = float(outage_fraction.sum())
        outage_load = float(np.sum(load * outage_fraction))
        pv_available = float(np.sum(pv * outage_fraction))
        net_deficit = float(np.sum(np.maximum(load - pv, 0.0) * outage_fraction))
        max_duration = float(scenario.get("max_event_duration_min", 0.0)) / 60.0
        mean_load = outage_load / duration if duration > 0 else 0.0
        rows.append(
            {
                "case_key": scenario.get("case_key", ""),
                "city_key": scenario.get("city_key", ""),
                "building_key": scenario.get("building_key", ""),
                "frontier_scenario_id": int(scenario["frontier_scenario_id"]),
                "outage_seed": int(scenario["outage_seed"]),
                "outage_duration_hours": duration,
                "max_event_duration_hours": max_duration,
                "outage_weighted_load_kwh": outage_load,
                "pv_available_kwh": pv_available,
                "net_deficit_kwh": net_deficit,
                "mean_outage_critical_load_kw": mean_load,
                "pv_availability_ratio": pv_available / outage_load if outage_load > 0 else 0.0,
                "scenario_probability": 1.0 / scenario_count,
            }
        )
    return validate_bess_event_source(pd.DataFrame(rows))


def _build_capacity_slice_source(scenario_results: pd.DataFrame, design: dict[str, Any]) -> pd.DataFrame:
    required = {
        "pv_capacity_kw",
        "battery_energy_kwh",
        "UE_kwh_per_year",
        "scenario_LOLP",
        "unserved_energy_cost_sgd_per_year",
        "annualized_battery_cost_sgd",
        "battery_om_cost_sgd",
        "pv_total_annual_cost_sgd",
        "pv_bess_grid_cost_sgd",
        "pv_bess_private_annual_cost_sgd",
        "pv_bess_system_annual_cost_sgd",
        "lolp_threshold",
    }
    missing = sorted(required - set(scenario_results.columns))
    if missing:
        raise ValueError("Scenario results are missing required column(s): " + ", ".join(missing))
    if scenario_results.empty:
        raise ValueError("Scenario results are empty.")

    optimal_pv = float(design["pv_capacity_kw"])
    pv_values = np.array(sorted(scenario_results["pv_capacity_kw"].astype(float).unique()))
    if len(pv_values) == 0:
        raise ValueError("Scenario results contain no PV capacity values.")
    slice_pv = float(pv_values[int(np.argmin(np.abs(pv_values - optimal_pv)))])
    selected = scenario_results[np.isclose(scenario_results["pv_capacity_kw"].astype(float), slice_pv)].copy()
    if selected.empty:
        raise ValueError("Could not select a fixed-PV capacity slice for BESS capacity interpretation.")

    rows: list[dict[str, Any]] = []
    lolp_threshold = float(selected["lolp_threshold"].astype(float).iloc[0])
    for battery, group in selected.groupby("battery_energy_kwh", sort=True):
        scenario_lolps = group["scenario_LOLP"].astype(float).tolist()
        battery_energy = float(battery)
        rows.append(
            {
                "pv_capacity_kw": slice_pv,
                "pv_capacity_mw": slice_pv / 1000.0,
                "battery_energy_kwh": battery_energy,
                "battery_energy_mwh": battery_energy / 1000.0,
                "system_total_cost_sgd_per_year": float(group["pv_bess_system_annual_cost_sgd"].astype(float).mean()),
                "system_total_cost_million_sgd_per_year": float(
                    group["pv_bess_system_annual_cost_sgd"].astype(float).mean()
                )
                / 1e6,
                "annualized_battery_cost_sgd_per_year": float(
                    (
                        group["annualized_battery_cost_sgd"].astype(float)
                        + group["battery_om_cost_sgd"].astype(float)
                    ).mean()
                ),
                "annualized_battery_cost_million_sgd_per_year": float(
                    (
                        group["annualized_battery_cost_sgd"].astype(float)
                        + group["battery_om_cost_sgd"].astype(float)
                    ).mean()
                )
                / 1e6,
                "expected_outage_loss_sgd_per_year": float(
                    group["unserved_energy_cost_sgd_per_year"].astype(float).mean()
                ),
                "expected_outage_loss_million_sgd_per_year": float(
                    group["unserved_energy_cost_sgd_per_year"].astype(float).mean()
                )
                / 1e6,
                "mean_UE_kwh_per_year": float(group["UE_kwh_per_year"].astype(float).mean()),
                "mean_UE_mwh_per_year": float(group["UE_kwh_per_year"].astype(float).mean()) / 1000.0,
                "energy_based_LOLP": float(np.mean(scenario_lolps)),
                "chance_constraint_probability": chance_constraint_probability(
                    scenario_lolps,
                    lolp_threshold=lolp_threshold,
                ),
                "is_optimal_bess": np.isclose(battery_energy, float(design["battery_energy_kwh"])),
            }
        )
    return validate_bess_capacity_slice_source(pd.DataFrame(rows))


def _build_summary_source(
    event_source: pd.DataFrame,
    capacity_slice_source: pd.DataFrame,
    design: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    events = validate_bess_event_source(event_source)
    capacity_slice = validate_bess_capacity_slice_source(capacity_slice_source)
    battery = config["battery"]
    soc_min = float(battery["soc_min_fraction"])
    soc_max = float(battery["soc_max_fraction"])
    roundtrip_efficiency = float(battery["roundtrip_efficiency"])
    discharge_efficiency = roundtrip_efficiency**0.5
    optimal_bess = float(design["battery_energy_kwh"])
    usable_bess = optimal_bess * max(0.0, soc_max - soc_min) * discharge_efficiency
    total_duration = float(events["outage_duration_hours"].sum())
    mean_critical_load = (
        float(events["outage_weighted_load_kwh"].sum()) / total_duration if total_duration > 0 else np.nan
    )
    support_hours = usable_bess / mean_critical_load if mean_critical_load and mean_critical_load > 0 else 0.0
    rows = [
        {
            "selected_design_type": str(design["selected_design_type"]),
            "optimal_pv_capacity_kw": float(design["pv_capacity_kw"]),
            "slice_pv_capacity_kw": float(capacity_slice["pv_capacity_kw"].iloc[0]),
            "optimal_battery_energy_kwh": optimal_bess,
            "optimal_battery_energy_mwh": optimal_bess / 1000.0,
            "usable_bess_energy_kwh": usable_bess,
            "usable_bess_energy_mwh": usable_bess / 1000.0,
            "net_deficit_p50_kwh": _weighted_quantile(events, 0.50),
            "net_deficit_p80_kwh": _weighted_quantile(events, 0.80),
            "net_deficit_p90_kwh": _weighted_quantile(events, 0.90),
            "net_deficit_p95_kwh": _weighted_quantile(events, 0.95),
            "net_deficit_coverage_probability": _weighted_coverage(events, usable_bess),
            "support_hours": support_hours,
            "mean_outage_critical_load_kw": mean_critical_load,
            "soc_min_fraction": soc_min,
            "soc_max_fraction": soc_max,
            "roundtrip_efficiency": roundtrip_efficiency,
            "discharge_efficiency": discharge_efficiency,
        }
    ]
    return validate_bess_summary_source(pd.DataFrame(rows))


def _critical_load(case_frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    if "critical_load_e_equiv_kwh" in case_frame.columns:
        return case_frame["critical_load_e_equiv_kwh"].astype(float)
    technical = config["technical"]
    return electric_equivalent_building_load(
        case_frame,
        float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0))),
        float(technical.get("heating_cop", 3.5)),
    )


def _weighted_quantile(events: pd.DataFrame, quantile: float) -> float:
    ordered = events.sort_values("net_deficit_kwh")
    values = ordered["net_deficit_kwh"].astype(float).to_numpy()
    weights = ordered["scenario_probability"].astype(float).to_numpy()
    cumulative = np.cumsum(weights) / max(float(np.sum(weights)), 1e-12)
    return float(values[int(np.searchsorted(cumulative, quantile, side="left"))])


def _weighted_coverage(events: pd.DataFrame, usable_bess_kwh: float) -> float:
    covered = events[events["net_deficit_kwh"].astype(float) <= float(usable_bess_kwh)]
    return float(covered["scenario_probability"].astype(float).sum())


def _plot_panel_a(ax: Any, events: pd.DataFrame, palette: dict[str, str]) -> None:
    size = 28 + 210 * events["scenario_probability"].astype(float).to_numpy()
    color = events["pv_availability_ratio"].astype(float).clip(0.0, 1.25).to_numpy()
    sc = ax.scatter(
        events["outage_duration_hours"],
        events["net_deficit_kwh"] / 1000.0,
        c=color,
        s=size,
        cmap="YlGnBu",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.92,
    )
    ax.set_title("Outage severity is joint", loc="left")
    ax.set_xlabel("Outage duration (h)")
    ax.set_ylabel("Net energy deficit (MWh)")
    ax.grid(axis="both", color=palette["neutral_light"], linewidth=0.35, alpha=0.45)
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.055, pad=0.02)
    cbar.set_label("PV/load during outage", fontsize=6.3)
    cbar.ax.tick_params(labelsize=5.8, width=0.4, length=2.0)


def _plot_panel_b(ax: Any, events: pd.DataFrame, summary: pd.Series, palette: dict[str, str]) -> None:
    ordered = events.sort_values("net_deficit_kwh")
    x = ordered["net_deficit_kwh"].astype(float).to_numpy() / 1000.0
    y = np.cumsum(ordered["scenario_probability"].astype(float).to_numpy())
    ax.step(x, y, where="post", color=palette["teal"], lw=1.45)
    ax.fill_between(x, y, step="post", color=palette["teal"], alpha=0.10)

    usable = float(summary["usable_bess_energy_kwh"]) / 1000.0
    p90 = float(summary["net_deficit_p90_kwh"]) / 1000.0
    p95 = float(summary["net_deficit_p95_kwh"]) / 1000.0
    coverage = float(summary["net_deficit_coverage_probability"])
    ax.axvline(usable, color=palette["blue"], lw=1.25, ls="-")
    ax.axvline(p90, color=palette["neutral_mid"], lw=0.9, ls="--")
    ax.axvline(p95, color=palette["red"], lw=0.9, ls=":")
    ax.axhline(coverage, color=palette["blue"], lw=0.75, ls=":")
    ax.text(
        0.04,
        0.18,
        f"Net-deficit coverage\n{coverage:.0%}",
        transform=ax.transAxes,
        color=palette["blue"],
        fontsize=7.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.text(usable, 0.04, "Optimized usable BESS", color=palette["blue"], fontsize=6.2, rotation=90, va="bottom")
    ax.text(p90, 0.84, "P90", color=palette["neutral_mid"], fontsize=6.0, rotation=90, va="bottom")
    ax.text(p95, 0.74, "P95", color=palette["red"], fontsize=6.0, rotation=90, va="bottom")
    ax.set_title("Net-deficit coverage", loc="left")
    ax.set_xlabel("Outage net energy deficit (MWh)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=palette["neutral_light"], linewidth=0.35, alpha=0.45)


def _plot_panel_c(ax: Any, capacity_slice: pd.DataFrame, summary: pd.Series, palette: dict[str, str]) -> None:
    x = capacity_slice["battery_energy_mwh"].astype(float).to_numpy()
    ax.plot(
        x,
        capacity_slice["system_total_cost_million_sgd_per_year"],
        color=palette["neutral_black"],
        lw=1.25,
        label="Total cost",
    )
    ax.plot(
        x,
        capacity_slice["annualized_battery_cost_million_sgd_per_year"],
        color=palette["blue"],
        lw=1.0,
        ls="--",
        label="BESS cost",
    )
    ax.plot(
        x,
        capacity_slice["expected_outage_loss_million_sgd_per_year"],
        color=palette["red"],
        lw=1.0,
        ls=":",
        label="Outage loss",
    )
    optimal = float(summary["optimal_battery_energy_kwh"]) / 1000.0
    ax.axvline(optimal, color=palette["blue"], lw=0.95, alpha=0.9)
    ax.set_title("Cost balance at optimized PV", loc="left")
    ax.set_xlabel("BESS energy capacity (MWh)")
    ax.set_ylabel("Annualized cost (million SGD yr$^{-1}$)")
    ax.grid(axis="y", color=palette["neutral_light"], linewidth=0.35, alpha=0.45)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.50, 1.18), handlelength=1.5, columnspacing=0.9)

    ue_ax = ax.twinx()
    ue_ax.plot(
        x,
        capacity_slice["mean_UE_mwh_per_year"],
        color=palette["red"],
        marker="o",
        markersize=2.6,
        lw=0.85,
        alpha=0.65,
    )
    ue_ax.set_ylabel("Mean UE (MWh yr$^{-1}$)")
    ue_ax.spines["top"].set_visible(False)


def _plot_panel_d(ax: Any, capacity_slice: pd.DataFrame, summary: pd.Series, palette: dict[str, str]) -> None:
    source = capacity_slice.sort_values("battery_energy_kwh").reset_index(drop=True)
    x = source["battery_energy_kwh"].astype(float).to_numpy()
    ue = source["mean_UE_kwh_per_year"].astype(float).to_numpy()
    outage_loss = source["expected_outage_loss_sgd_per_year"].astype(float).to_numpy()
    battery_cost = source["annualized_battery_cost_sgd_per_year"].astype(float).to_numpy()

    delta_bess = np.diff(x)
    marginal_ue = -np.diff(ue) / np.maximum(delta_bess, 1e-12)
    marginal_avoided_loss = -np.diff(outage_loss) / np.maximum(delta_bess, 1e-12)
    marginal_battery_cost = np.diff(battery_cost) / np.maximum(delta_bess, 1e-12)
    mid_x = (x[1:] + x[:-1]) / 2.0 / 1000.0

    ax.plot(
        mid_x,
        marginal_avoided_loss,
        color=palette["teal"],
        lw=1.2,
        marker="o",
        markersize=2.8,
        label="Avoided outage loss",
    )
    if len(marginal_battery_cost):
        ax.plot(
            mid_x,
            marginal_battery_cost,
            color=palette["blue"],
            lw=1.0,
            ls="--",
            label="Annualized BESS cost",
        )
    optimal = float(summary["optimal_battery_energy_kwh"]) / 1000.0
    ax.axvline(optimal, color=palette["blue"], lw=0.95, alpha=0.9)
    ax.set_title("Marginal UE reduction value", loc="left")
    ax.set_xlabel("BESS energy capacity (MWh)")
    ax.set_ylabel("Marginal value (SGD kWh$^{-1}$ yr$^{-1}$)")
    ax.grid(axis="y", color=palette["neutral_light"], linewidth=0.35, alpha=0.45)
    ax.legend(loc="upper right", handlelength=1.5)

    ue_ax = ax.twinx()
    ue_ax.plot(mid_x, marginal_ue, color=palette["red"], lw=0.85, alpha=0.58)
    ue_ax.set_ylabel("Marginal UE reduction\n(kWh yr$^{-1}$ kWh$^{-1}$)")
    ue_ax.spines["top"].set_visible(False)


def _demo_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(17)
    scenario_count = 80
    duration = rng.gamma(shape=2.1, scale=3.4, size=scenario_count).clip(0.8, 24.0)
    pv_ratio = rng.beta(2.1, 4.2, size=scenario_count)
    critical_load = rng.normal(1250.0, 210.0, size=scenario_count).clip(740.0, 1800.0)
    outage_load = duration * critical_load
    pv_available = outage_load * pv_ratio
    net_deficit = np.maximum(outage_load - pv_available * rng.uniform(0.75, 1.10, size=scenario_count), 0.0)
    event_source = pd.DataFrame(
        {
            "frontier_scenario_id": np.arange(1, scenario_count + 1),
            "outage_seed": np.arange(1001, 1001 + scenario_count),
            "outage_duration_hours": duration,
            "max_event_duration_hours": duration * rng.uniform(0.45, 0.95, size=scenario_count),
            "outage_weighted_load_kwh": outage_load,
            "pv_available_kwh": pv_available,
            "net_deficit_kwh": net_deficit,
            "mean_outage_critical_load_kw": critical_load,
            "pv_availability_ratio": pv_available / outage_load,
            "scenario_probability": np.full(scenario_count, 1.0 / scenario_count),
        }
    )
    bess_mwh = np.arange(0.0, 62.0, 4.0)
    ue_mwh = 7.8 + 86.0 * np.exp(-bess_mwh / 13.0)
    outage_loss = ue_mwh * 42_000.0
    battery_cost = bess_mwh * 0.052
    total_cost = 6.80 + battery_cost + outage_loss / 1e6
    capacity_slice = pd.DataFrame(
        {
            "pv_capacity_kw": 20_000.0,
            "pv_capacity_mw": 20.0,
            "battery_energy_kwh": bess_mwh * 1000.0,
            "battery_energy_mwh": bess_mwh,
            "system_total_cost_sgd_per_year": total_cost * 1e6,
            "system_total_cost_million_sgd_per_year": total_cost,
            "annualized_battery_cost_sgd_per_year": battery_cost * 1e6,
            "annualized_battery_cost_million_sgd_per_year": battery_cost,
            "expected_outage_loss_sgd_per_year": outage_loss,
            "expected_outage_loss_million_sgd_per_year": outage_loss / 1e6,
            "mean_UE_kwh_per_year": ue_mwh * 1000.0,
            "mean_UE_mwh_per_year": ue_mwh,
            "energy_based_LOLP": np.clip(ue_mwh / 105.0, 0.0, 1.0),
            "chance_constraint_probability": np.clip(1.0 - ue_mwh / 120.0, 0.0, 1.0),
            "is_optimal_bess": np.isclose(bess_mwh, 28.0),
        }
    )
    summary = _build_summary_source(
        validate_bess_event_source(event_source),
        validate_bess_capacity_slice_source(capacity_slice),
        {
            "selected_design_type": "demo_reliability_constrained_design",
            "pv_capacity_kw": 20_000.0,
            "battery_energy_kwh": 28_000.0,
        },
        {
            "battery": {
                "soc_min_fraction": 0.10,
                "soc_max_fraction": 0.95,
                "roundtrip_efficiency": 0.92,
            }
        },
    )
    return event_source, capacity_slice, summary


def _palette() -> dict[str, str]:
    return {
        **NATURE_PALETTE,
        "blue": NATURE_PALETTE["blue"],
        "teal": NATURE_PALETTE["teal"],
        "red": NATURE_PALETTE["red"],
    }


def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)
