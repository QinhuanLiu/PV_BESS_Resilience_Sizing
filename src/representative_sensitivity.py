from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import CaseParameters, InputCase, build_case_frame
from .optimizer import evaluate_candidate, scenario_config
from .outage import simulate_outages, summarize_outages
from .reserve_sensitivity import _available
from .resilience_metrics import (
    DEFAULT_LOLP_THRESHOLD,
    eue,
    outage_load_served_ratio,
    outage_weighted_load_kwh,
    scenario_lolp,
)
from .visualization import _matplotlib, _save_all, _style, _write_source_data


CAPEX_PARAMETERS = [
    "pv_capex_sgd_per_kwp",
    "battery_capex_sgd_per_kwh",
    "battery_power_capex_sgd_per_kw",
]
CAPEX_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
OUTAGE_MULTIPLIERS = [0.5, 1.0, 2.0, 5.0, 10.0]
RESERVE_CHANGE_CASES = [
    ("Base case", 1.0, 1.0),
    ("2x SAIDI", 2.0, 1.0),
    ("2x VoLL", 1.0, 2.0),
    ("2x SAIDI + 2x VoLL", 2.0, 2.0),
    ("10x SAIDI", 10.0, 1.0),
    ("10x VoLL", 1.0, 10.0),
    ("10x SAIDI + 10x VoLL", 10.0, 10.0),
]


def run_representative_sensitivity(
    case: InputCase,
    config: dict[str, Any],
    parameters: CaseParameters,
    optimal_designs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = _select_minimum_reference(optimal_designs)
    seed = representative_outage_seed(parameters, config)
    capex_results = _run_capex_sensitivity(case, config, parameters, reference, seed)
    outage_results = _run_outage_value_sensitivity(case, config, parameters, reference, seed)
    all_results = pd.concat([capex_results, outage_results], ignore_index=True)
    cost_reliability = all_results[
        [
            "case_key",
            "city_key",
            "building_key",
            "sensitivity_family",
            "sensitivity_parameter",
            "sensitivity_label",
            "sensitivity_multiplier",
            "sensitivity_value",
            "reference_pv_capacity_kwp",
            "reference_battery_energy_kwh",
            "representative_seed",
            "saidi_min_per_year",
            "voll_sgd_per_kwh",
            "system_total_cost_sgd_per_year",
            "outage_unserved_energy_cost_sgd_per_year",
            "EUE_kwh_per_year",
            "outage_load_served_ratio",
            "achieved_LOLP",
            "chance_constraint_probability",
        ]
    ].copy()
    reserve_change = _run_reserve_change_sensitivity(case, config, parameters, reference, seed)
    return all_results, cost_reliability, reserve_change


def representative_outage_seed(parameters: CaseParameters, config: dict[str, Any]) -> int:
    base = parameters.base
    value = base.get("frontier_seed_start")
    if value is None:
        value = base.get("outage_seed")
    if value is None:
        value = config.get("outage", {}).get("seed", 42)
    result = int(float(value))
    if result < 0:
        raise ValueError("representative outage seed must be non-negative")
    return result


def make_representative_sensitivity_figures(
    sensitivity_dir: Path,
    case_key: str,
    all_results: pd.DataFrame,
    reserve_change: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths.update(_figure_1_framework(sensitivity_dir, case_key))
    paths.update(_figure_11_capex_sensitivity(sensitivity_dir, case_key, all_results))
    paths.update(_figure_12_saidi_voll_heatmap(sensitivity_dir, case_key, all_results))
    paths.update(_figure_16_reserve_change(sensitivity_dir, case_key, reserve_change))
    return paths


def _run_capex_sensitivity(
    case: InputCase,
    config: dict[str, Any],
    parameters: CaseParameters,
    reference: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_values = {
        "pv_capex_sgd_per_kwp": float(config["pv"]["capex_sgd_per_kwp"]),
        "battery_capex_sgd_per_kwh": float(config["economics"]["battery_capex_sgd_per_kwh"]),
        "battery_power_capex_sgd_per_kw": float(config["economics"]["battery_power_capex_sgd_per_kw"]),
    }
    for parameter in CAPEX_PARAMETERS:
        for multiplier in CAPEX_MULTIPLIERS:
            cfg = deepcopy(config)
            value = base_values[parameter] * multiplier
            _apply_parameter_value(cfg, parameter, value)
            row = _evaluate_fixed_design(
                case,
                cfg,
                parameters,
                reference,
                seed,
                sensitivity_family="capex",
                sensitivity_parameter=parameter,
                sensitivity_label=_parameter_label(parameter),
                sensitivity_multiplier=multiplier,
                sensitivity_value=value,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _run_outage_value_sensitivity(
    case: InputCase,
    config: dict[str, Any],
    parameters: CaseParameters,
    reference: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenario = config["scenarios"][case.case_key]
    base_saidi = float(scenario["saidi_min_per_year"])
    base_voll = float(scenario["voll_sgd_per_kwh"])
    for saidi_multiplier in OUTAGE_MULTIPLIERS:
        for voll_multiplier in OUTAGE_MULTIPLIERS:
            cfg = deepcopy(config)
            cfg["scenarios"][case.case_key]["saidi_min_per_year"] = base_saidi * saidi_multiplier
            cfg["scenarios"][case.case_key]["voll_sgd_per_kwh"] = base_voll * voll_multiplier
            row = _evaluate_fixed_design(
                case,
                cfg,
                parameters,
                reference,
                seed,
                sensitivity_family="saidi_voll",
                sensitivity_parameter="saidi_min_per_year__voll_sgd_per_kwh",
                sensitivity_label=f"{saidi_multiplier:g}x SAIDI, {voll_multiplier:g}x VoLL",
                sensitivity_multiplier=np.nan,
                sensitivity_value=np.nan,
            )
            row["saidi_multiplier"] = saidi_multiplier
            row["voll_multiplier"] = voll_multiplier
            rows.append(row)
    return pd.DataFrame(rows)


def _run_reserve_change_sensitivity(
    case: InputCase,
    config: dict[str, Any],
    parameters: CaseParameters,
    reference: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenario = config["scenarios"][case.case_key]
    base_saidi = float(scenario["saidi_min_per_year"])
    base_voll = float(scenario["voll_sgd_per_kwh"])
    for order, (label, saidi_multiplier, voll_multiplier) in enumerate(RESERVE_CHANGE_CASES, start=1):
        cfg = deepcopy(config)
        cfg["scenarios"][case.case_key]["saidi_min_per_year"] = base_saidi * saidi_multiplier
        cfg["scenarios"][case.case_key]["voll_sgd_per_kwh"] = base_voll * voll_multiplier
        low = _evaluate_fixed_design(
            case,
            cfg,
            parameters,
            reference,
            seed,
            sensitivity_family="reserve_change",
            sensitivity_parameter="reserve_20pct",
            sensitivity_label=label,
            sensitivity_multiplier=np.nan,
            sensitivity_value=0.20,
            reserve_fraction=0.20,
            include_fixed_costs=False,
        )
        high = _evaluate_fixed_design(
            case,
            cfg,
            parameters,
            reference,
            seed,
            sensitivity_family="reserve_change",
            sensitivity_parameter="reserve_80pct",
            sensitivity_label=label,
            sensitivity_multiplier=np.nan,
            sensitivity_value=0.80,
            reserve_fraction=0.80,
            include_fixed_costs=False,
        )
        delta_bill = high["normal_operation_bill_cost_sgd_per_year"] - low["normal_operation_bill_cost_sgd_per_year"]
        delta_outage = high["outage_unserved_energy_cost_sgd_per_year"] - low["outage_unserved_energy_cost_sgd_per_year"]
        delta_net = high["net_cost_sgd_per_year"] - low["net_cost_sgd_per_year"]
        rows.append(
            {
                "case_key": case.case_key,
                "city_key": case.city_key,
                "building_key": case.building_key,
                "sensitivity_case_order": order,
                "sensitivity_case": label,
                "saidi_multiplier": saidi_multiplier,
                "voll_multiplier": voll_multiplier,
                "reference_pv_capacity_kwp": float(reference["pv_capacity_kwp"]),
                "reference_battery_energy_kwh": float(reference["battery_energy_kwh"]),
                "representative_seed": seed,
                "reserve_start_fraction": 0.20,
                "reserve_end_fraction": 0.80,
                "bill_cost_20pct_sgd_per_year": low["normal_operation_bill_cost_sgd_per_year"],
                "bill_cost_80pct_sgd_per_year": high["normal_operation_bill_cost_sgd_per_year"],
                "outage_unserved_cost_20pct_sgd_per_year": low["outage_unserved_energy_cost_sgd_per_year"],
                "outage_unserved_cost_80pct_sgd_per_year": high["outage_unserved_energy_cost_sgd_per_year"],
                "net_cost_20pct_sgd_per_year": low["net_cost_sgd_per_year"],
                "net_cost_80pct_sgd_per_year": high["net_cost_sgd_per_year"],
                "delta_bill_cost_sgd_per_year": delta_bill,
                "delta_outage_unserved_cost_sgd_per_year": delta_outage,
                "delta_net_cost_sgd_per_year": delta_net,
                "net_value_gain_sgd_per_year": -delta_net,
                "delta_EUE_kwh_per_year": high["EUE_kwh_per_year"] - low["EUE_kwh_per_year"],
                "delta_achieved_LOLP": high["achieved_LOLP"] - low["achieved_LOLP"],
                "delta_chance_constraint_probability": (
                    high["chance_constraint_probability"] - low["chance_constraint_probability"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_fixed_design(
    case: InputCase,
    config: dict[str, Any],
    parameters: CaseParameters,
    reference: dict[str, Any],
    seed: int,
    sensitivity_family: str,
    sensitivity_parameter: str,
    sensitivity_label: str,
    sensitivity_multiplier: float,
    sensitivity_value: float,
    reserve_fraction: float | None = None,
    include_fixed_costs: bool = True,
) -> dict[str, Any]:
    cfg = scenario_config(config, case.case_key, resilience_informed=True)
    if reserve_fraction is not None:
        cfg["battery"] = deepcopy(cfg["battery"])
        cfg["battery"]["normal_operation_reserve_fraction"] = reserve_fraction
    scenario = cfg["active_scenario"]
    outage_profile, outage_events = simulate_outages(
        saifi_per_year=float(scenario["saifi_per_year"]),
        saidi_min_per_year=float(scenario["saidi_min_per_year"]),
        hours=8760,
        seed=seed,
        forced_event_mode=str(cfg.get("outage", {}).get("forced_event_mode", "match_saidi")),
    )
    pv_capacity = float(reference["pv_capacity_kwp"])
    battery_energy = float(reference["battery_energy_kwh"])
    frame, pv_meta = build_case_frame(case, cfg, pv_capacity)
    merged = frame.merge(outage_profile, on="hour_index", how="left")
    result, dispatch = evaluate_candidate(merged, battery_energy, pv_meta, cfg)
    outage_weighted_load = outage_weighted_load_kwh(dispatch["total_load_e_equiv_kwh"], merged["outage_fraction"])
    scenario_eue = eue(dispatch)
    scenario_lolp_value = scenario_lolp(dispatch, merged["outage_fraction"])
    outage_cost = scenario_eue * float(scenario["voll_sgd_per_kwh"])
    bill_cost = float(result["pv_bess_grid_cost_sgd"])
    net_cost = bill_cost + outage_cost
    system_total = float(result["pv_bess_system_annual_cost_sgd"]) if include_fixed_costs else net_cost
    outage_summary = summarize_outages(outage_profile, outage_events)
    threshold = float(parameters.base.get("lolp_threshold") or DEFAULT_LOLP_THRESHOLD)
    return {
        "case_key": case.case_key,
        "city_key": case.city_key,
        "building_key": case.building_key,
        "sensitivity_family": sensitivity_family,
        "sensitivity_parameter": sensitivity_parameter,
        "sensitivity_label": sensitivity_label,
        "sensitivity_multiplier": sensitivity_multiplier,
        "sensitivity_value": sensitivity_value,
        "reference_design_type": str(reference["design_type"]),
        "reference_pv_capacity_kwp": pv_capacity,
        "reference_pv_capacity_mwp": pv_capacity / 1000.0,
        "reference_battery_energy_kwh": battery_energy,
        "reference_battery_energy_mwh": battery_energy / 1000.0,
        "reference_battery_power_kw": float(result["battery_power_kw"]),
        "representative_seed": seed,
        "saifi_per_year": float(scenario["saifi_per_year"]),
        "saidi_min_per_year": float(scenario["saidi_min_per_year"]),
        "voll_sgd_per_kwh": float(scenario["voll_sgd_per_kwh"]),
        "pv_capex_sgd_per_kwp": float(cfg["pv"]["capex_sgd_per_kwp"]),
        "battery_capex_sgd_per_kwh": float(cfg["economics"]["battery_capex_sgd_per_kwh"]),
        "battery_power_capex_sgd_per_kw": float(cfg["economics"]["battery_power_capex_sgd_per_kw"]),
        "reserve_fraction": reserve_fraction if reserve_fraction is not None else np.nan,
        "normal_operation_bill_cost_sgd_per_year": bill_cost,
        "outage_unserved_energy_cost_sgd_per_year": outage_cost,
        "net_cost_sgd_per_year": net_cost,
        "system_total_cost_sgd_per_year": system_total,
        "fixed_pv_bess_cost_sgd_per_year": system_total - net_cost if include_fixed_costs else 0.0,
        "EUE_kwh_per_year": scenario_eue,
        "outage_load_served_ratio": outage_load_served_ratio(dispatch, outage_weighted_load),
        "achieved_LOLP": scenario_lolp_value,
        "chance_constraint_probability": 1.0 if scenario_lolp_value <= threshold else 0.0,
        "lolp_threshold": threshold,
        "simulated_saidi_min_per_year": float(outage_summary["simulated_saidi_min_per_year"]),
        "simulated_saifi_per_year": float(outage_summary["simulated_saifi_per_year"]),
    }


def _select_minimum_reference(optimal_designs: pd.DataFrame) -> dict[str, Any]:
    matches = optimal_designs[optimal_designs["design_type"].astype(str) == "minimum_total_cost_design"]
    if matches.empty:
        raise ValueError("Representative sensitivity requires minimum_total_cost_design from frontier outputs")
    row = matches.iloc[0].to_dict()
    if not _available(row.get("available", True)):
        raise ValueError("minimum_total_cost_design is unavailable in frontier outputs")
    return row


def _apply_parameter_value(config: dict[str, Any], parameter: str, value: float) -> None:
    if parameter == "pv_capex_sgd_per_kwp":
        config["pv"]["capex_sgd_per_kwp"] = float(value)
    elif parameter == "battery_capex_sgd_per_kwh":
        config["economics"]["battery_capex_sgd_per_kwh"] = float(value)
    elif parameter == "battery_power_capex_sgd_per_kw":
        config["economics"]["battery_power_capex_sgd_per_kw"] = float(value)
    else:
        raise ValueError(f"Unsupported representative sensitivity parameter: {parameter}")


def _figure_1_framework(out: Path, case_key: str) -> dict[str, Path]:
    base = out / f"{case_key}_Figure_1_Research_Framework"
    nodes = pd.DataFrame(
        [
            {"node_id": "inputs", "label": "Input data", "x": 0.10, "y": 0.58},
            {"node_id": "stage1", "label": "Stage 1\nCost-reliability frontier", "x": 0.37, "y": 0.58},
            {"node_id": "stage2", "label": "Stage 2\nReserve sensitivity", "x": 0.63, "y": 0.72},
            {"node_id": "repr", "label": "Representative\nsensitivity", "x": 0.63, "y": 0.40},
            {"node_id": "outputs", "label": "Decision metrics", "x": 0.89, "y": 0.58},
        ]
    )
    edges = pd.DataFrame(
        [
            {"source": "inputs", "target": "stage1"},
            {"source": "stage1", "target": "stage2"},
            {"source": "stage1", "target": "repr"},
            {"source": "stage2", "target": "outputs"},
            {"source": "repr", "target": "outputs"},
        ]
    )
    source = pd.concat([nodes.assign(source_table="nodes"), edges.assign(source_table="edges")], ignore_index=True, sort=False)
    _write_source_data(base, source)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, ax = plt.subplots(figsize=(180 / 25.4, 86 / 25.4), constrained_layout=True)
    fig.suptitle("Research framework", y=1.04, fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box_style = dict(boxstyle="round,pad=0.35,rounding_size=0.025", facecolor="white", edgecolor="#0f4d92", linewidth=1.0)
    for row in nodes.itertuples(index=False):
        ax.text(row.x, row.y, row.label, ha="center", va="center", fontsize=7.2, fontweight="bold", bbox=box_style)
    arrow_points = {
        ("inputs", "stage1"): ((0.16, 0.58), (0.29, 0.58)),
        ("stage1", "stage2"): ((0.45, 0.63), (0.56, 0.70)),
        ("stage1", "repr"): ((0.45, 0.53), (0.56, 0.42)),
        ("stage2", "outputs"): ((0.70, 0.70), (0.82, 0.60)),
        ("repr", "outputs"): ((0.70, 0.43), (0.82, 0.55)),
    }
    for row in edges.itertuples(index=False):
        start, end = arrow_points[(row.source, row.target)]
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle="->",
                color="#767676",
                linewidth=0.9,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0.0",
            ),
        )
    ax.text(0.10, 0.20, "Load, PV, TOU,\nSAIDI/SAIFI, VoLL,\nPV/BESS costs", ha="center", va="center", fontsize=6.2, color="#3f3f46")
    ax.text(0.89, 0.20, "EUE, LOLP,\nchance probability,\ncost and reserve value", ha="center", va="center", fontsize=6.2, color="#3f3f46")
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _figure_11_capex_sensitivity(out: Path, case_key: str, all_results: pd.DataFrame) -> dict[str, Path]:
    source = all_results[all_results["sensitivity_family"] == "capex"].copy()
    source["system_total_cost_million_sgd_per_year"] = source["system_total_cost_sgd_per_year"] / 1e6
    source["EUE_mwh_per_year"] = source["EUE_kwh_per_year"] / 1000.0
    base = out / f"{case_key}_Figure_11_CAPEX_Sensitivity"
    _write_source_data(base, source)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, axes = plt.subplots(2, 2, figsize=(164 / 25.4, 118 / 25.4), constrained_layout=True)
    fig.suptitle("PV and battery cost sensitivity", y=1.05, fontsize=8.5, fontweight="bold")
    metric_map = [
        ("system_total_cost_million_sgd_per_year", "System total cost\n(million SGD/year)", "a  System cost"),
        ("EUE_mwh_per_year", "EUE (MWh/year)", "b  Expected unserved energy"),
        ("achieved_LOLP", "Achieved LOLP", "c  Achieved LOLP"),
        ("chance_constraint_probability", "Chance probability", "d  Chance constraint"),
    ]
    colors = {"pv_capex_sgd_per_kwp": "#0f4d92", "battery_capex_sgd_per_kwh": "#0f766e", "battery_power_capex_sgd_per_kw": "#b64342"}
    for ax, (metric, ylabel, title) in zip(axes.ravel(), metric_map, strict=True):
        for parameter, group in source.groupby("sensitivity_parameter", sort=False):
            group = group.sort_values("sensitivity_multiplier")
            ax.plot(
                group["sensitivity_multiplier"],
                group[metric],
                marker="o",
                linewidth=1.05,
                color=colors.get(str(parameter), "#767676"),
                label=_parameter_label(str(parameter)),
            )
        ax.set_title(title, pad=12)
        ax.set_xlabel("Cost multiplier")
        ax.set_ylabel(ylabel)
        _style_axes([ax])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Parameter")
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _figure_12_saidi_voll_heatmap(out: Path, case_key: str, all_results: pd.DataFrame) -> dict[str, Path]:
    source = all_results[all_results["sensitivity_family"] == "saidi_voll"].copy()
    source["system_total_cost_million_sgd_per_year"] = source["system_total_cost_sgd_per_year"] / 1e6
    source["outage_unserved_cost_million_sgd_per_year"] = source["outage_unserved_energy_cost_sgd_per_year"] / 1e6
    source["EUE_mwh_per_year"] = source["EUE_kwh_per_year"] / 1000.0
    base = out / f"{case_key}_Figure_12_SAIDI_VoLL_Heatmap"
    _write_source_data(base, source)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, axes = plt.subplots(2, 2, figsize=(160 / 25.4, 120 / 25.4), constrained_layout=True)
    fig.suptitle("SAIDI and VoLL sensitivity", y=1.04, fontsize=8.5, fontweight="bold")
    metric_map = [
        ("system_total_cost_million_sgd_per_year", "System cost\n(million SGD/year)", "a  System cost"),
        ("outage_unserved_cost_million_sgd_per_year", "Outage cost\n(million SGD/year)", "b  Outage cost"),
        ("EUE_mwh_per_year", "EUE\n(MWh/year)", "c  Expected unserved energy"),
        ("achieved_LOLP", "Achieved LOLP", "d  Achieved LOLP"),
    ]
    for ax, (metric, cbar_label, title) in zip(axes.ravel(), metric_map, strict=True):
        matrix = source.pivot(index="voll_multiplier", columns="saidi_multiplier", values=metric).sort_index(ascending=True)
        image = ax.imshow(matrix.to_numpy(), origin="lower", aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels([f"{value:g}x" for value in matrix.columns])
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels([f"{value:g}x" for value in matrix.index])
        ax.set_xlabel("SAIDI multiplier")
        ax.set_ylabel("VoLL multiplier")
        ax.set_title(title, pad=12)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _figure_16_reserve_change(out: Path, case_key: str, reserve_change: pd.DataFrame) -> dict[str, Path]:
    source = reserve_change.sort_values("sensitivity_case_order").copy()
    source["delta_net_cost_million_sgd_per_year"] = source["delta_net_cost_sgd_per_year"] / 1e6
    source["delta_bill_cost_million_sgd_per_year"] = source["delta_bill_cost_sgd_per_year"] / 1e6
    source["delta_outage_unserved_cost_million_sgd_per_year"] = source["delta_outage_unserved_cost_sgd_per_year"] / 1e6
    base = out / f"{case_key}_Figure_16_Reserve_Change_Sensitivity"
    _write_source_data(base, source)
    mpl, plt = _matplotlib(out)
    _style(mpl)
    fig, ax = plt.subplots(figsize=(154 / 25.4, 92 / 25.4), constrained_layout=True)
    fig.suptitle("Reserve increase sensitivity", y=1.05, fontsize=8.5, fontweight="bold")
    y = np.arange(len(source))
    ax.barh(y, source["delta_net_cost_million_sgd_per_year"], color="#0f4d92", label="Net cost change")
    ax.scatter(source["delta_bill_cost_million_sgd_per_year"], y, color="#767676", marker="|", s=90, label="Bill cost change")
    ax.scatter(source["delta_outage_unserved_cost_million_sgd_per_year"], y, color="#b64342", marker="|", s=90, label="Outage cost change")
    ax.axvline(0.0, color="#272727", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(source["sensitivity_case"])
    ax.invert_yaxis()
    ax.set_xlabel("Change when reserve increases from 20% to 80% (million SGD/year)")
    ax.set_ylabel("Representative sensitivity case")
    _style_axes([ax])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _parameter_label(parameter: str) -> str:
    labels = {
        "pv_capex_sgd_per_kwp": "PV CAPEX",
        "battery_capex_sgd_per_kwh": "Battery energy CAPEX",
        "battery_power_capex_sgd_per_kw": "Battery power CAPEX",
    }
    return labels.get(parameter, parameter)


def _style_axes(axes: Any) -> None:
    for ax in list(np.ravel(axes)):
        ax.grid(axis="both", color="#e5e7eb", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.margins(x=0.05, y=0.12)
