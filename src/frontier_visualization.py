from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import InputCase, build_case_frame
from .resilience_metrics import electric_equivalent_building_load
from .visualization import _matplotlib, _save_all, _style, _write_source_data


FRONTIER_COLORS = ["#0f4d92", "#3775ba", "#0f766e", "#b64342", "#767676"]


def make_frontier_figures(
    frontier_dir: Path,
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    all_candidates: pd.DataFrame,
    scenario_results: pd.DataFrame,
    optimal_designs: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    selected_pv = _selected_pv_capacities(all_candidates, optimal_designs)
    paths.update(_figure_2_input_profiles(frontier_dir, case, config, case_key, selected_pv))
    paths.update(_figure_3_outage_scenarios(frontier_dir, case_key, outage_scenarios))
    paths.update(_figure_4_pv_building_load_mismatch(frontier_dir, case, config, case_key, selected_pv))
    paths.update(_figure_5_served_ratio(frontier_dir, case_key, all_candidates, selected_pv))
    paths.update(_figure_6_eue_unserved_cost(frontier_dir, case_key, all_candidates, selected_pv))
    paths.update(_figure_7_system_total_cost(frontier_dir, case_key, all_candidates, optimal_designs, selected_pv))
    paths.update(_figure_8_achieved_lolp(frontier_dir, case_key, all_candidates, selected_pv))
    paths.update(_figure_9_chance_probability(frontier_dir, case_key, all_candidates, selected_pv))
    paths.update(_figure_10_marginal_cost_reliability(frontier_dir, case_key, all_candidates, selected_pv))
    return paths


def _figure_2_input_profiles(
    out: Path,
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    selected_pv: list[float],
) -> dict[str, Path]:
    mpl, plt = _matplotlib(out)
    _style(mpl)
    pv_capacity = max(selected_pv) if selected_pv else max(float(value) for value in config["pv"]["capacity_kwp_candidates"])
    frame, _ = build_case_frame(case, config, pv_capacity)
    load = electric_equivalent_building_load(
        frame,
        float(config["technical"].get("cooling_cop", config["technical"].get("chiller_cop", 5.0))),
        float(config["technical"].get("heating_cop", 3.5)),
    )
    week_start = _representative_week_start(load)
    week = frame.iloc[week_start : week_start + 168].copy()
    source = pd.DataFrame(
        {
            "hour_of_week": np.arange(len(week), dtype=int),
            "hour_of_day": week["timestamp"].dt.hour.to_numpy(dtype=int),
            "timestamp": week["timestamp"].astype(str).to_numpy(),
            "full_building_load_kwh": load.iloc[week_start : week_start + 168].to_numpy(dtype=float),
            "pv_generation_kwh": week["pv_kwh"].to_numpy(dtype=float),
            "pv_capacity_kwp": float(pv_capacity),
            "tou_sgd_per_kwh": week["tou_sgd_per_kwh"].to_numpy(dtype=float),
        }
    )
    base = out / f"{case_key}_Figure_2_Input_Profiles"
    _write_source_data(base, source)
    fig, axes = plt.subplots(3, 1, figsize=(150 / 25.4, 125 / 25.4), sharex=False, constrained_layout=True)
    fig.suptitle("Input profiles", y=1.04, fontsize=8.5, fontweight="bold")
    axes[0].plot(source["hour_of_week"], source["full_building_load_kwh"] / 1000.0, color="#0f4d92", linewidth=1.1)
    axes[0].set_ylabel("Load (MWh)")
    _panel_title(axes[0], "Representative-week full-building electric-equivalent load")
    axes[1].plot(source["hour_of_week"], source["pv_generation_kwh"] / 1000.0, color="#0f766e", linewidth=1.1)
    axes[1].set_ylabel("PV generation (MWh)")
    _panel_title(axes[1], f"PV generation at {pv_capacity / 1000.0:g} MWp")
    tou = source.drop_duplicates("hour_of_day").sort_values("hour_of_day")
    axes[2].step(tou["hour_of_day"], tou["tou_sgd_per_kwh"], where="mid", color="#b64342", linewidth=1.1)
    axes[2].set_xlabel("Hour")
    axes[2].set_ylabel("TOU (SGD/kWh)")
    axes[2].set_xticks(range(0, 24, 3))
    _panel_title(axes[2], "Twenty-four-hour tariff schedule")
    for ax in axes[:2]:
        ax.set_xlabel("Hour of representative week")
    _style_axes(axes)
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _figure_3_outage_scenarios(
    out: Path,
    case_key: str,
    outage_scenarios: pd.DataFrame,
) -> dict[str, Path]:
    mpl, plt = _matplotlib(out)
    _style(mpl)
    availability = _example_availability(outage_scenarios)
    events = _outage_events(outage_scenarios)
    base = out / f"{case_key}_Figure_3_Outage_Scenarios"
    source = pd.concat(
        [
            availability.assign(source_table="example_grid_availability"),
            events.assign(source_table="outage_events") if not events.empty else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    _write_source_data(base, source)
    fig, axes = plt.subplots(3, 1, figsize=(150 / 25.4, 128 / 25.4), constrained_layout=True)
    fig.suptitle("Outage scenarios", y=1.04, fontsize=8.5, fontweight="bold")
    axes[0].plot(availability["hour_index"], availability["grid_availability"], color="#0f4d92", linewidth=0.9)
    axes[0].set_ylabel("Grid availability")
    _panel_title(axes[0], "Example availability sequence")
    if events.empty:
        for ax in axes[1:]:
            ax.text(0.5, 0.5, "No outage events", ha="center", va="center", transform=ax.transAxes)
    else:
        axes[1].hist(events["start_hour"], bins=24, color="#3775ba", edgecolor="white", linewidth=0.3)
        axes[1].set_ylabel("Events")
        _panel_title(axes[1], "Outage start-time distribution")
        axes[2].hist(events["duration_minutes"], bins=min(24, max(4, len(events))), color="#0f766e", edgecolor="white", linewidth=0.3)
        axes[2].set_ylabel("Events")
        _panel_title(axes[2], "Outage duration distribution")
    axes[0].set_xlabel("Hour of year")
    axes[1].set_xlabel("Start hour of year")
    axes[2].set_xlabel("Duration (min)")
    _style_axes(axes)
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _figure_4_pv_building_load_mismatch(
    out: Path,
    case: InputCase,
    config: dict[str, Any],
    case_key: str,
    selected_pv: list[float],
) -> dict[str, Path]:
    mpl, plt = _matplotlib(out)
    _style(mpl)
    source_rows: list[pd.DataFrame] = []
    frame, _ = build_case_frame(case, config, selected_pv[0])
    load = electric_equivalent_building_load(
        frame,
        float(config["technical"].get("cooling_cop", config["technical"].get("chiller_cop", 5.0))),
        float(config["technical"].get("heating_cop", 3.5)),
    )
    source_rows.append(_cdf_source(load, "full_building_load", np.nan))
    for pv_capacity in selected_pv:
        pv_frame, _ = build_case_frame(case, config, pv_capacity)
        source_rows.append(_cdf_source(pv_frame["pv_kwh"], "pv_generation", pv_capacity))
    source = pd.concat(source_rows, ignore_index=True)
    base = out / f"{case_key}_Figure_4_PV_Building_Load_Mismatch"
    _write_source_data(base, source)
    fig, axes = plt.subplots(1, 2, figsize=(150 / 25.4, 74 / 25.4), constrained_layout=False)
    fig.subplots_adjust(top=0.78, right=0.76, wspace=0.35)
    fig.suptitle("PV-building load mismatch", y=0.98, fontsize=8.5, fontweight="bold")
    load_source = source[source["metric"] == "full_building_load"]
    axes[0].plot(load_source["value_kwh"] / 1000.0, load_source["cdf_probability"], color="#0f4d92", linewidth=1.2)
    axes[0].set_xlabel("Full-building load (MWh)")
    axes[0].set_ylabel("Cumulative probability")
    _panel_title(axes[0], "Full-building load distribution")
    for pv_capacity, color in zip(selected_pv, FRONTIER_COLORS, strict=False):
        part = source[(source["metric"] == "pv_generation") & (source["pv_capacity_kwp"].astype(float) == float(pv_capacity))]
        axes[1].plot(part["value_kwh"] / 1000.0, part["cdf_probability"], color=color, linewidth=1.05, label=f"{pv_capacity / 1000.0:g} MWp")
    axes[1].set_xlabel("PV generation (MWh)")
    axes[1].set_ylabel("Cumulative probability")
    _panel_title(axes[1], "PV generation distribution")
    _shared_legend(fig, axes[1], selected_pv)
    _style_axes(axes)
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _figure_5_served_ratio(out: Path, case_key: str, all_candidates: pd.DataFrame, selected_pv: list[float]) -> dict[str, Path]:
    return _single_metric_curve(
        out,
        case_key,
        all_candidates,
        selected_pv,
        "Figure_5_Battery_vs_Outage_Load_Served_Ratio",
        "Battery capacity vs outage load served ratio",
        "mean_outage_load_served_ratio",
        "Mean outage load served ratio",
        scale=1.0,
    )


def _figure_6_eue_unserved_cost(
    out: Path,
    case_key: str,
    all_candidates: pd.DataFrame,
    selected_pv: list[float],
) -> dict[str, Path]:
    source = _selected_candidate_source(all_candidates, selected_pv)
    source["battery_energy_mwh"] = source["battery_energy_kwh"].astype(float) / 1000.0
    source["mean_EUE_mwh_per_year"] = source["mean_EUE_kwh_per_year"].astype(float) / 1000.0
    source["mean_unserved_energy_cost_million_sgd_per_year"] = (
        source["mean_unserved_energy_cost_sgd_per_year"].astype(float) / 1e6
    )
    base = out / f"{case_key}_Figure_6_Battery_vs_EUE_Unserved_Cost"
    _write_source_data(base, source)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, axes = plt.subplots(1, 2, figsize=(158 / 25.4, 76 / 25.4), constrained_layout=True)
    fig.suptitle("Battery capacity, EUE and unserved cost", y=1.08, fontsize=8.5, fontweight="bold")
    _plot_metric_by_pv(axes[0], source, selected_pv, "mean_EUE_mwh_per_year", "Mean EUE (MWh/year)")
    _panel_title(axes[0], "a  Expected unserved energy")
    _plot_metric_by_pv(
        axes[1],
        source,
        selected_pv,
        "mean_unserved_energy_cost_million_sgd_per_year",
        "Mean unserved energy cost (million SGD/year)",
    )
    _panel_title(axes[1], "b  Unserved energy cost")
    _shared_legend(fig, axes[1], selected_pv)
    _style_axes(axes)
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _figure_7_system_total_cost(
    out: Path,
    case_key: str,
    all_candidates: pd.DataFrame,
    optimal_designs: pd.DataFrame,
    selected_pv: list[float],
) -> dict[str, Path]:
    source = _selected_candidate_source(all_candidates, selected_pv)
    source["battery_energy_mwh"] = source["battery_energy_kwh"].astype(float) / 1000.0
    source["system_total_cost_million_sgd_per_year"] = source["system_total_cost_sgd_per_year"].astype(float) / 1e6
    source["annualized_battery_cost_million_sgd_per_year"] = source["annualized_battery_cost_sgd_per_year"].astype(float) / 1e6
    source["mean_unserved_energy_cost_million_sgd_per_year"] = (
        source["mean_unserved_energy_cost_sgd_per_year"].astype(float) / 1e6
    )
    base = out / f"{case_key}_Figure_7_Battery_vs_System_Total_Cost"
    _write_source_data(base, source)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, axes = plt.subplots(2, 1, figsize=(160 / 25.4, 122 / 25.4), constrained_layout=False)
    fig.subplots_adjust(top=0.84, right=0.76, hspace=0.62)
    fig.suptitle("Battery capacity and system total cost", y=0.985, fontsize=8.5, fontweight="bold")
    _plot_metric_by_pv(axes[0], source, selected_pv, "system_total_cost_million_sgd_per_year", "System total cost (million SGD/year)")
    _mark_optimal_designs(axes[0], optimal_designs)
    _panel_title(axes[0], "System total cost frontier")
    component_pv = float(_minimum_cost_row(all_candidates)["pv_capacity_kwp"])
    component = all_candidates[np.isclose(all_candidates["pv_capacity_kwp"].astype(float), component_pv)].sort_values("battery_energy_kwh").copy()
    component["battery_energy_mwh"] = component["battery_energy_kwh"].astype(float) / 1000.0
    axes[1].plot(component["battery_energy_mwh"], component["annualized_battery_cost_sgd_per_year"] / 1e6, color="#767676", label="Annualized battery cost")
    axes[1].plot(component["battery_energy_mwh"], component["mean_unserved_energy_cost_sgd_per_year"] / 1e6, color="#b64342", label="Mean unserved energy cost")
    axes[1].set_xlabel("Battery energy capacity (MWh)")
    axes[1].set_ylabel("Cost component (million SGD/year)")
    _panel_title(axes[1], f"Cost components at {component_pv / 1000.0:g} MWp PV")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), ncols=1)
    _shared_legend(fig, axes[0], selected_pv, include_existing=True)
    _style_axes(axes)
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _figure_8_achieved_lolp(out: Path, case_key: str, all_candidates: pd.DataFrame, selected_pv: list[float]) -> dict[str, Path]:
    paths = _single_metric_curve(
        out,
        case_key,
        all_candidates,
        selected_pv,
        "Figure_8_Battery_vs_Achieved_LOLP",
        "Battery capacity vs achieved LOLP",
        "achieved_LOLP",
        "Achieved LOLP",
        scale=1.0,
        add_lolp_threshold=True,
    )
    return paths


def _figure_9_chance_probability(out: Path, case_key: str, all_candidates: pd.DataFrame, selected_pv: list[float]) -> dict[str, Path]:
    return _single_metric_curve(
        out,
        case_key,
        all_candidates,
        selected_pv,
        "Figure_9_Battery_vs_Chance_Constraint_Probability",
        "Battery capacity vs chance-constraint probability",
        "chance_constraint_probability",
        "Chance-constraint probability",
        scale=1.0,
    )


def _figure_10_marginal_cost_reliability(
    out: Path,
    case_key: str,
    all_candidates: pd.DataFrame,
    selected_pv: list[float],
) -> dict[str, Path]:
    rows: list[dict[str, float]] = []
    for pv_capacity in selected_pv:
        part = all_candidates[np.isclose(all_candidates["pv_capacity_kwp"].astype(float), float(pv_capacity))].sort_values("battery_energy_kwh")
        previous = None
        for row in part.itertuples(index=False):
            current = row._asdict()
            if previous is not None:
                delta_cost = float(current["system_total_cost_sgd_per_year"]) - float(previous["system_total_cost_sgd_per_year"])
                lolp_gain = float(previous["achieved_LOLP"]) - float(current["achieved_LOLP"])
                chance_gain = float(current["chance_constraint_probability"]) - float(previous["chance_constraint_probability"])
                rows.append(
                    {
                        "pv_capacity_kwp": float(pv_capacity),
                        "battery_energy_kwh": float(current["battery_energy_kwh"]),
                        "battery_energy_mwh": float(current["battery_energy_kwh"]) / 1000.0,
                        "delta_system_total_cost_sgd_per_year": delta_cost,
                        "achieved_LOLP_decrease": lolp_gain,
                        "chance_probability_increase": chance_gain,
                        "marginal_cost_per_lolp_pct_sgd": delta_cost / (lolp_gain * 100.0) if lolp_gain > 0 else np.nan,
                        "marginal_cost_per_chance_pct_sgd": delta_cost / (chance_gain * 100.0) if chance_gain > 0 else np.nan,
                    }
                )
            previous = current
    base = out / f"{case_key}_Figure_10_Marginal_Cost_Reliability"
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, axes = plt.subplots(1, 2, figsize=(158 / 25.4, 76 / 25.4), constrained_layout=True)
    fig.suptitle("Marginal cost of reliability gains", y=1.08, fontsize=8.5, fontweight="bold")
    source = pd.DataFrame(rows)
    if source.empty:
        _write_source_data(base, source)
        for ax in axes:
            ax.text(0.5, 0.5, "No marginal segments", ha="center", va="center", transform=ax.transAxes)
    else:
        source["marginal_cost_per_lolp_pct_million_sgd"] = source["marginal_cost_per_lolp_pct_sgd"] / 1e6
        source["marginal_cost_per_chance_pct_million_sgd"] = source["marginal_cost_per_chance_pct_sgd"] / 1e6
        _write_source_data(base, source)
        _plot_metric_by_pv(axes[0], source, selected_pv, "marginal_cost_per_lolp_pct_million_sgd", "Cost per 1% LOLP decrease (million SGD/year)")
        _plot_metric_by_pv(axes[1], source, selected_pv, "marginal_cost_per_chance_pct_million_sgd", "Cost per 1% probability increase (million SGD/year)")
    _panel_title(axes[0], "a  Achieved LOLP")
    _panel_title(axes[1], "b  Chance constraint")
    _shared_legend(fig, axes[1], selected_pv)
    _style_axes(axes)
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _single_metric_curve(
    out: Path,
    case_key: str,
    all_candidates: pd.DataFrame,
    selected_pv: list[float],
    file_label: str,
    title: str,
    metric_column: str,
    ylabel: str,
    scale: float = 1.0,
    add_lolp_threshold: bool = False,
) -> dict[str, Path]:
    source = _selected_candidate_source(all_candidates, selected_pv)
    source["battery_energy_mwh"] = source["battery_energy_kwh"].astype(float) / 1000.0
    source["plot_value"] = source[metric_column].astype(float) * float(scale)
    base = out / f"{case_key}_{file_label}"
    _write_source_data(base, source)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, ax = plt.subplots(figsize=(136 / 25.4, 82 / 25.4), constrained_layout=True)
    fig.suptitle(title, y=1.05, fontsize=8.5, fontweight="bold")
    _plot_metric_by_pv(ax, source, selected_pv, "plot_value", ylabel)
    if add_lolp_threshold and "lolp_threshold" in source.columns:
        threshold = float(source["lolp_threshold"].dropna().iloc[0])
        ax.axhline(threshold, color="#b64342", linewidth=0.85, linestyle="--", label="LOLP threshold")
    _shared_legend(fig, ax, selected_pv, include_existing=True)
    _style_axes([ax])
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _selected_candidate_source(all_candidates: pd.DataFrame, selected_pv: list[float]) -> pd.DataFrame:
    selected = all_candidates[
        all_candidates["pv_capacity_kwp"].astype(float).round(6).isin([round(value, 6) for value in selected_pv])
    ].copy()
    return selected.sort_values(["pv_capacity_kwp", "battery_energy_kwh"]).reset_index(drop=True)


def _plot_metric_by_pv(ax: Any, source: pd.DataFrame, selected_pv: list[float], column: str, ylabel: str) -> None:
    for pv_capacity, color in zip(selected_pv, FRONTIER_COLORS, strict=False):
        part = source[np.isclose(source["pv_capacity_kwp"].astype(float), float(pv_capacity))].sort_values("battery_energy_kwh")
        if part.empty:
            continue
        ax.plot(
            part["battery_energy_kwh"].astype(float) / 1000.0,
            part[column].astype(float),
            marker="o",
            markersize=2.5,
            linewidth=1.05,
            color=color,
            label=f"{pv_capacity / 1000.0:g} MWp",
        )
    ax.set_xlabel("Battery energy capacity (MWh)")
    ax.set_ylabel(ylabel)


def _mark_optimal_designs(ax: Any, optimal_designs: pd.DataFrame) -> None:
    markers = {
        "minimum_total_cost_design": ("*", "#272727", "Minimum total cost"),
        "reliability_constrained_design": ("D", "#b64342", "Reliability constrained"),
    }
    for row in optimal_designs.itertuples(index=False):
        values = row._asdict()
        if not bool(values.get("available", False)):
            continue
        marker, color, label = markers.get(str(values["design_type"]), ("o", "#272727", str(values["design_type"])))
        ax.scatter(
            float(values["battery_energy_kwh"]) / 1000.0,
            float(values["system_total_cost_sgd_per_year"]) / 1e6,
            marker=marker,
            s=36,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=5,
            label=label,
        )


def _selected_pv_capacities(all_candidates: pd.DataFrame, optimal_designs: pd.DataFrame) -> list[float]:
    pv_values = sorted(all_candidates["pv_capacity_kwp"].astype(float).unique().tolist())
    if not pv_values:
        return []
    selected = {pv_values[0], pv_values[len(pv_values) // 2], pv_values[-1]}
    available = optimal_designs[optimal_designs["available"].astype(bool, errors="ignore") == True]  # noqa: E712
    for value in available.get("pv_capacity_kwp", pd.Series(dtype=float)).dropna().astype(float).tolist():
        selected.add(value)
    return sorted(selected)


def _minimum_cost_row(all_candidates: pd.DataFrame) -> pd.Series:
    return all_candidates.sort_values("system_total_cost_sgd_per_year").iloc[0]


def _representative_week_start(load: pd.Series) -> int:
    peak_hour = int(load.astype(float).idxmax())
    start = max(0, peak_hour - 84)
    return min(start, max(0, len(load) - 168))


def _cdf_source(values: pd.Series, metric: str, pv_capacity_kwp: float) -> pd.DataFrame:
    ordered = np.sort(values.astype(float).to_numpy())
    probability = np.arange(1, len(ordered) + 1, dtype=float) / max(len(ordered), 1)
    return pd.DataFrame(
        {
            "metric": metric,
            "pv_capacity_kwp": pv_capacity_kwp,
            "value_kwh": ordered,
            "cdf_probability": probability,
        }
    )


def _example_availability(outage_scenarios: pd.DataFrame) -> pd.DataFrame:
    if outage_scenarios.empty:
        return pd.DataFrame(columns=["frontier_scenario_id", "outage_seed", "hour_index", "grid_availability", "outage_fraction"])
    row = outage_scenarios.iloc[0]
    profile = row["_outage_profile"].copy()
    profile.insert(0, "outage_seed", int(row["outage_seed"]))
    profile.insert(0, "frontier_scenario_id", int(row["frontier_scenario_id"]))
    return profile


def _outage_events(outage_scenarios: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for row in outage_scenarios.to_dict("records"):
        events = row.get("_outage_events")
        if events is None or events.empty:
            continue
        part = events.copy()
        part.insert(0, "outage_seed", int(row["outage_seed"]))
        part.insert(0, "frontier_scenario_id", int(row["frontier_scenario_id"]))
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["frontier_scenario_id", "outage_seed", "event_id", "start_hour", "end_hour", "duration_minutes"])
    return pd.concat(rows, ignore_index=True)


def _panel_title(ax: Any, title: str) -> None:
    ax.set_title(title, pad=12)


def _style_axes(axes: Any) -> None:
    for ax in list(np.ravel(axes)):
        ax.grid(axis="both", color="#e5e7eb", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.margins(x=0.03, y=0.12)


def _shared_legend(fig: Any, ax: Any, selected_pv: list[float], include_existing: bool = False) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not include_existing:
        wanted = {f"{value / 1000.0:g} MWp" for value in selected_pv}
        pairs = [(handle, label) for handle, label in zip(handles, labels, strict=False) if label in wanted]
        handles = [pair[0] for pair in pairs]
        labels = [pair[1] for pair in pairs]
    if handles:
        fig.legend(
            handles,
            labels,
            title="PV capacity",
            loc="center left",
            bbox_to_anchor=(1.01, 0.52),
            ncols=1,
            columnspacing=1.0,
            handlelength=1.6,
        )
