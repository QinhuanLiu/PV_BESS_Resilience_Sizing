from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from html import escape
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.config import load_config
from src.io.data_loader import discover_input_cases, load_input_case
from src.resilience_frontier import run_resilience_frontier


TARGET_CASE_KEYS = (
    "SG_SingleFamily",
    "Bangkok_SingleFamily",
    "Brisbane_SingleFamily",
    "Jakarta_SingleFamily",
    "Manila_SingleFamily",
)
OUTPUT_DIR = ROOT / "outputs" / "Affordable_Backup_City_Comparison_Night_Window"
SUMMARY_PATH = OUTPUT_DIR / "Affordable_Backup_Night_Window_Capacity_Summary.csv"
BREAKDOWN_PATH = OUTPUT_DIR / "Affordable_Backup_Night_Window_Cost_Breakdown_Source.csv"
SVG_PATH = OUTPUT_DIR / "Affordable_Backup_Night_Window_Cost_Breakdown.svg"
OUTAGE_WINDOW = "night_19_23"
NIGHT_WINDOW_START_HOUR = 19.0
NIGHT_WINDOW_END_HOUR = 23.0

DESIGN_ORDER = (
    ("no_pv_no_bess", "0 PV / 0 BESS"),
    ("minimum_total_cost_design", "Cost-optimal"),
    ("affordable_backup_design", "Affordable backup"),
)
COMPONENTS = (
    ("grid_bill", "Grid bill", "#767676"),
    ("pv_annual_cost", "PV annual cost", "#7EBC67"),
    ("battery_annual_cost", "Battery annual cost", "#0F4D92"),
    ("expected_outage_loss", "Expected outage loss", "#B64342"),
)
DESIGN_SHORT_LABELS = {
    "no_pv_no_bess": "0/0",
    "minimum_total_cost_design": "Cost-opt",
    "affordable_backup_design": "Affordable",
}
CITY_LABEL_OVERRIDES = {
    "SG": "Singapore",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_config = load_config(ROOT / "config.yaml")
    cases = {case.case_key: case for case in discover_input_cases(ROOT / "Input")}
    missing = [case_key for case_key in TARGET_CASE_KEYS if case_key not in cases]
    if missing:
        raise ValueError(f"Missing target input case(s): {', '.join(missing)}")

    summary_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    for case_key in TARGET_CASE_KEYS:
        case_summary, case_breakdown = _run_case(cases[case_key], global_config)
        summary_rows.extend(case_summary)
        breakdown_rows.extend(case_breakdown)

    summary = pd.DataFrame(summary_rows)
    breakdown = pd.DataFrame(breakdown_rows)
    _validate_outputs(summary, breakdown)
    summary.to_csv(SUMMARY_PATH, index=False)
    breakdown.to_csv(BREAKDOWN_PATH, index=False)
    SVG_PATH.write_text(_build_svg(summary, breakdown), encoding="utf-8")
    print(f"Capacity summary: {SUMMARY_PATH}")
    print(f"Cost breakdown source: {BREAKDOWN_PATH}")
    print(f"SVG figure: {SVG_PATH}")


def _run_case(case: Any, global_config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, _, case_config, parameters = load_input_case(case, global_config)
    case_config = deepcopy(case_config)
    outage_seed = int(case_config.get("outage", {}).get("seed", parameters.base.get("outage_seed", 42)))
    case_config.setdefault("outage", {})
    case_config["outage"]["forced_event_mode"] = "night_window_random"
    case_config["outage"]["night_window_start_hour"] = NIGHT_WINDOW_START_HOUR
    case_config["outage"]["night_window_end_hour"] = NIGHT_WINDOW_END_HOUR
    one_scenario_parameters = replace(
        parameters,
        base={
            **parameters.base,
            "frontier_outage_scenario_count": 1,
            "frontier_seed_start": outage_seed,
        },
    )
    all_candidates, _, optimal_designs, outage_scenarios = run_resilience_frontier(
        case,
        case_config,
        one_scenario_parameters,
    )
    design_rows = _selected_design_rows(all_candidates, optimal_designs)
    scenario = case_config["scenarios"][case.case_key]
    scenario_meta = outage_scenarios.iloc[0].to_dict()
    event_meta = _outage_event_meta(scenario_meta.get("_outage_events"))
    summary_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    for design_order, (design_case, design_label) in enumerate(DESIGN_ORDER, start=1):
        row = design_rows[design_case]
        summary_row = _summary_row(
            case=case,
            row=row,
            scenario=scenario,
            scenario_meta=scenario_meta,
            design_case=design_case,
            design_label=design_label,
            design_order=design_order,
            outage_seed=outage_seed,
            event_meta=event_meta,
        )
        summary_rows.append(summary_row)
        breakdown_rows.extend(_cost_component_rows(summary_row, row))
    return summary_rows, breakdown_rows


def _selected_design_rows(all_candidates: pd.DataFrame, optimal_designs: pd.DataFrame) -> dict[str, pd.Series]:
    zero = all_candidates[
        np.isclose(all_candidates["pv_capacity_kw"].astype(float), 0.0, atol=1e-9)
        & np.isclose(all_candidates["battery_energy_kwh"].astype(float), 0.0, atol=1e-9)
    ].copy()
    if zero.empty:
        raise ValueError("Frontier output does not contain the required 0 PV / 0 BESS baseline candidate.")
    rows = {"no_pv_no_bess": zero.sort_values("system_total_cost_sgd_per_year").iloc[0]}
    for design_case in ("minimum_total_cost_design", "affordable_backup_design"):
        matches = optimal_designs[
            (optimal_designs["design_type"].astype(str) == design_case)
            & (optimal_designs["available"].astype(str).str.lower().isin({"true", "1", "yes"}))
        ]
        if matches.empty:
            raise ValueError(f"Optimal designs do not contain an available {design_case}.")
        rows[design_case] = matches.iloc[0]
    return rows


def _outage_event_meta(events: Any) -> dict[str, float]:
    if not isinstance(events, pd.DataFrame) or events.empty:
        return {
            "event_count": 0.0,
            "first_event_start_hour_of_day": float("nan"),
            "last_event_start_hour_of_day": float("nan"),
            "total_outage_duration_min": 0.0,
        }
    starts = events["start_hour"].astype(float) % 24.0
    durations = events["duration_minutes"].astype(float)
    return {
        "event_count": float(len(events)),
        "first_event_start_hour_of_day": float(starts.min()),
        "last_event_start_hour_of_day": float(starts.max()),
        "total_outage_duration_min": float(durations.sum()),
    }


def _summary_row(
    case: Any,
    row: pd.Series,
    scenario: dict[str, Any],
    scenario_meta: dict[str, Any],
    design_case: str,
    design_label: str,
    design_order: int,
    outage_seed: int,
    event_meta: dict[str, float],
) -> dict[str, Any]:
    city_label = str(scenario.get("city_or_service_area") or case.city_key)
    return {
        "case_key": case.case_key,
        "city_key": case.city_key,
        "city_label": CITY_LABEL_OVERRIDES.get(city_label, city_label),
        "building_key": case.building_key,
        "design_case": design_case,
        "design_case_label": design_label,
        "design_case_order": int(design_order),
        "single_outage_seed": int(outage_seed),
        "frontier_outage_scenario_count": 1,
        "outage_window": OUTAGE_WINDOW,
        "night_window_start_hour": NIGHT_WINDOW_START_HOUR,
        "night_window_end_hour": NIGHT_WINDOW_END_HOUR,
        "event_count": int(event_meta["event_count"]),
        "first_event_start_hour_of_day": float(event_meta["first_event_start_hour_of_day"]),
        "last_event_start_hour_of_day": float(event_meta["last_event_start_hour_of_day"]),
        "total_outage_duration_min": float(event_meta["total_outage_duration_min"]),
        "simulated_saidi_min_per_year": float(scenario_meta["simulated_saidi_min_per_year"]),
        "simulated_saifi_per_year": float(scenario_meta["simulated_saifi_per_year"]),
        "max_event_duration_min": float(scenario_meta["max_event_duration_min"]),
        "input_saidi_min_per_year": float(scenario["saidi_min_per_year"]),
        "input_saifi_per_year": float(scenario["saifi_per_year"]),
        "voll_sgd_per_kwh": float(scenario["voll_sgd_per_kwh"]),
        "pv_capacity_kw": float(row["pv_capacity_kw"]),
        "pv_capacity_mw": float(row["pv_capacity_kw"]) / 1000.0,
        "battery_energy_kwh": float(row["battery_energy_kwh"]),
        "battery_energy_mwh": float(row["battery_energy_kwh"]) / 1000.0,
        "battery_power_kw": float(row["battery_power_kw"]),
        "mean_outage_load_served_ratio": float(row["mean_outage_load_served_ratio"]),
        "mean_scenario_LOLP": float(row["mean_scenario_LOLP"]),
        "mean_UE_kwh_per_year": float(row["mean_UE_kwh_per_year"]),
        "mean_grid_cost_sgd_per_year": float(row["mean_grid_cost_sgd_per_year"]),
        "pv_total_annual_cost_sgd_per_year": float(row["pv_total_annual_cost_sgd_per_year"]),
        "annualized_battery_cost_sgd_per_year": float(row["annualized_battery_cost_sgd_per_year"]),
        "battery_om_cost_sgd_per_year": float(row["battery_om_cost_sgd_per_year"]),
        "battery_total_annual_cost_sgd_per_year": float(row["annualized_battery_cost_sgd_per_year"])
        + float(row["battery_om_cost_sgd_per_year"]),
        "mean_unserved_energy_cost_sgd_per_year": float(row["mean_unserved_energy_cost_sgd_per_year"]),
        "system_total_cost_sgd_per_year": float(row["system_total_cost_sgd_per_year"]),
        "baseline_generalized_cost_sgd_per_year": float(row["baseline_generalized_cost_sgd_per_year"]),
        "affordability_margin_sgd_per_year": float(row["affordability_margin_sgd_per_year"]),
        "cost_ratio_vs_baseline": float(row["cost_ratio_vs_baseline"]),
        "is_affordable": bool(row["is_affordable"]),
    }


def _cost_component_rows(summary_row: dict[str, Any], row: pd.Series) -> list[dict[str, Any]]:
    component_values = {
        "grid_bill": float(row["mean_grid_cost_sgd_per_year"]),
        "pv_annual_cost": float(row["pv_total_annual_cost_sgd_per_year"]),
        "battery_annual_cost": float(row["annualized_battery_cost_sgd_per_year"])
        + float(row["battery_om_cost_sgd_per_year"]),
        "expected_outage_loss": float(row["mean_unserved_energy_cost_sgd_per_year"]),
    }
    total = max(float(summary_row["system_total_cost_sgd_per_year"]), 1e-9)
    rows = []
    for stack_order, (component_key, component_label, color) in enumerate(COMPONENTS, start=1):
        value = component_values[component_key]
        rows.append(
            {
                "case_key": summary_row["case_key"],
                "city_key": summary_row["city_key"],
                "city_label": summary_row["city_label"],
                "building_key": summary_row["building_key"],
                "design_case": summary_row["design_case"],
                "design_case_label": summary_row["design_case_label"],
                "design_case_order": summary_row["design_case_order"],
                "component_key": component_key,
                "component_label": component_label,
                "component_color": color,
                "stack_order": stack_order,
                "cost_sgd": value,
                "cost_ksgd": value / 1000.0,
                "share_of_total_percent": 100.0 * value / total,
                "system_total_cost_sgd_per_year": summary_row["system_total_cost_sgd_per_year"],
                "pv_capacity_kw": summary_row["pv_capacity_kw"],
                "battery_energy_kwh": summary_row["battery_energy_kwh"],
                "mean_outage_load_served_ratio": summary_row["mean_outage_load_served_ratio"],
                "mean_UE_kwh_per_year": summary_row["mean_UE_kwh_per_year"],
                "is_affordable": summary_row["is_affordable"],
            }
        )
    return rows


def _validate_outputs(summary: pd.DataFrame, breakdown: pd.DataFrame) -> None:
    expected_designs = {case for case, _ in DESIGN_ORDER}
    for city_key, group in summary.groupby("city_key", sort=False):
        observed = set(group["design_case"].astype(str))
        if observed != expected_designs:
            raise ValueError(f"{city_key} missing design case(s): {sorted(expected_designs - observed)}")
        affordable = group[group["design_case"].astype(str) == "affordable_backup_design"].iloc[0]
        if not bool(affordable["is_affordable"]):
            raise ValueError(f"{city_key} affordable_backup_design is not affordable.")
        starts = group["first_event_start_hour_of_day"].astype(float)
        ends = group["last_event_start_hour_of_day"].astype(float)
        if not ((starts >= NIGHT_WINDOW_START_HOUR).all() and (ends < NIGHT_WINDOW_END_HOUR).all()):
            raise ValueError(f"{city_key} outage event start hour is outside the night window.")
    component_counts = breakdown.groupby(["city_key", "design_case"]).size()
    if set(component_counts.astype(int).tolist()) != {len(COMPONENTS)}:
        raise ValueError("Each city/design must have exactly four cost components.")
    sums = breakdown.groupby(["city_key", "design_case"], as_index=False).agg(
        component_sum=("cost_sgd", "sum"),
        total=("system_total_cost_sgd_per_year", "first"),
    )
    if not np.allclose(sums["component_sum"].astype(float), sums["total"].astype(float), rtol=1e-7, atol=1e-5):
        raise ValueError("Cost component sums do not match system total costs.")


def _build_svg(summary: pd.DataFrame, breakdown: pd.DataFrame) -> str:
    width, height = 1580, 760
    plot_w, plot_h = 300, 145
    panel_positions = [(72, 124), (575, 124), (1078, 124), (72, 442), (575, 442)]
    city_order = [case.split("_", 1)[0] for case in TARGET_CASE_KEYS]
    max_ksgd = float(summary["system_total_cost_sgd_per_year"].max()) / 1000.0
    y_max = _nice_ceiling(max_ksgd)
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>'
        'text{font-family:Arial,Helvetica,sans-serif;fill:#272727;}'
        '.title{font-size:22px;font-weight:700;}'
        '.subtitle{font-size:12px;fill:#4D4D4D;}'
        '.panel-title{font-size:15px;font-weight:700;}'
        '.axis{stroke:#4D4D4D;stroke-width:1;}'
        '.grid{stroke:#E7E7E7;stroke-width:1;}'
        '.tick{font-size:10px;fill:#4D4D4D;}'
        '.label{font-size:10.5px;fill:#272727;}'
        '.small{font-size:9px;fill:#4D4D4D;}'
        '</style>',
        '<text x="72" y="42" class="title">Night-window affordable-backup cost breakdown</text>',
        '<text x="72" y="65" class="subtitle">Five single-family city cases; one outage scenario per city; event starts sampled from 19:00-23:00; costs are annualized generalized costs.</text>',
    ]
    legend_x = 72
    for idx, (_, label, color) in enumerate(COMPONENTS):
        x = legend_x + idx * 225
        svg.append(f'<rect x="{x}" y="84" width="14" height="14" fill="{color}" stroke="#FFFFFF" stroke-width="0.5"/>')
        svg.append(f'<text x="{x + 20}" y="95" class="small">{escape(label)}</text>')

    for (panel_x, panel_y), city_key in zip(panel_positions, city_order, strict=True):
        city_summary = summary[summary["city_key"].astype(str) == city_key].sort_values("design_case_order")
        city_breakdown = breakdown[breakdown["city_key"].astype(str) == city_key].copy()
        city_label = str(city_summary["city_label"].iloc[0])
        svg.append(f'<text x="{panel_x}" y="{panel_y - 20}" class="panel-title">{escape(city_label)}</text>')
        plot_x = panel_x + 70
        plot_y = panel_y + 18
        bottom_y = plot_y + plot_h
        svg.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{bottom_y}" class="axis"/>')
        svg.append(f'<line x1="{plot_x}" y1="{bottom_y}" x2="{plot_x + plot_w}" y2="{bottom_y}" class="axis"/>')
        for tick in (0.0, y_max / 2.0, y_max):
            y = bottom_y - (tick / y_max) * plot_h
            svg.append(f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_w}" y2="{y:.2f}" class="grid"/>')
            svg.append(f'<text x="{plot_x - 8}" y="{y + 3:.2f}" text-anchor="end" class="tick">{_fmt_tick(tick)}</text>')
        svg.append(
            f'<text x="{plot_x - 50}" y="{plot_y + plot_h / 2}" text-anchor="middle" class="small" '
            'transform="rotate(-90 '
            f'{plot_x - 50} {plot_y + plot_h / 2})">kSGD/yr</text>'
        )

        bar_w = 54
        gap = 48
        first_x = plot_x + 33
        for idx, design_case in enumerate([case for case, _ in DESIGN_ORDER]):
            bar_x = first_x + idx * (bar_w + gap)
            base_y = bottom_y
            subset = city_breakdown[city_breakdown["design_case"].astype(str) == design_case].sort_values("stack_order")
            for component in subset.to_dict("records"):
                h = (float(component["cost_ksgd"]) / y_max) * plot_h
                y = base_y - h
                svg.append(
                    f'<rect x="{bar_x}" y="{y:.2f}" width="{bar_w}" height="{max(h, 0.0):.2f}" '
                    f'fill="{component["component_color"]}" stroke="#FFFFFF" stroke-width="0.5"/>'
                )
                base_y = y
            design = city_summary[city_summary["design_case"].astype(str) == design_case].iloc[0]
            total = float(design["system_total_cost_sgd_per_year"]) / 1000.0
            svg.append(f'<text x="{bar_x + bar_w / 2}" y="{base_y - 5:.2f}" text-anchor="middle" class="tick">{_fmt_tick(total)}</text>')
            svg.append(f'<text x="{bar_x + bar_w / 2}" y="{bottom_y + 17}" text-anchor="middle" class="label">{DESIGN_SHORT_LABELS[design_case]}</text>')
            svg.append(
                f'<text x="{bar_x + bar_w / 2}" y="{bottom_y + 31}" text-anchor="middle" class="small">'
                f'{_fmt_capacity(float(design["pv_capacity_kw"]))} kW PV</text>'
            )
            svg.append(
                f'<text x="{bar_x + bar_w / 2}" y="{bottom_y + 44}" text-anchor="middle" class="small">'
                f'{_fmt_capacity(float(design["battery_energy_kwh"]))} kWh BESS</text>'
            )
        affordable = city_summary[city_summary["design_case"].astype(str) == "affordable_backup_design"].iloc[0]
        backup_percent = 100.0 * float(affordable["mean_outage_load_served_ratio"])
        margin = float(affordable["affordability_margin_sgd_per_year"]) / 1000.0
        svg.append(
            f'<text x="{plot_x}" y="{bottom_y + 67}" class="small">'
            f'Affordable backup: {backup_percent:.1f}% served, margin {_fmt_signed(margin)} kSGD/yr</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(value))
    for multiplier in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        candidate = multiplier * magnitude
        if candidate >= value * 1.05:
            return float(candidate)
    return float(10.0 * magnitude)


def _fmt_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _fmt_capacity(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _fmt_signed(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{_fmt_tick(abs(value))}"


if __name__ == "__main__":
    main()
