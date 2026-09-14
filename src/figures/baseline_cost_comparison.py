from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    matplotlib_context,
    save_publication_figure,
    write_source_data as _write_source_data,
)
from .display_units import (
    FigureDisplayUnits,
    cost_component_display_series,
    figure_display_units,
    format_battery_energy_from_kwh,
    format_design_label as format_display_design_label,
    format_pv_capacity_from_kw,
    system_cost_display_series,
)


FIGURE_4_NAME = "Figure_4_Baseline_Cost_Comparison"
FIGURE_4A_NAME = "Figure_4A_Baseline_Cost_Comparison"
FIGURE_4B_NAME = "Figure_4B_Baseline_Cost_Comparison"
FIGURE_4A_CASES = ("no_pv_bess", "pv20_bess10")
FIGURE_4B_CASES = ("no_pv_bess", "pv20_no_bess", "pv20_bess10")
LEGACY_FIGURE_4_SUFFIXES = (".png", ".svg", ".pdf", "_Source_Data.csv")
REQUIRED_COLUMNS = (
    "pv_capacity_kw",
    "battery_energy_kwh",
    "pv_bess_system_annual_cost_sgd",
    "pv_bess_grid_cost_sgd",
    "annualized_pv_cost_sgd",
    "pv_om_cost_sgd",
    "annualized_battery_cost_sgd",
    "battery_om_cost_sgd",
    "unserved_energy_kwh_per_year",
)
SOURCE_COLUMNS = (
    "design_case",
    "design_case_label",
    "pv_capacity_kw",
    "pv_capacity_mw",
    "battery_energy_kwh",
    "battery_energy_mwh",
    "grid_cost_sgd_per_year",
    "pv_annual_cost_sgd_per_year",
    "battery_annual_cost_sgd_per_year",
    "lost_load_cost_sgd_per_year",
    "system_total_cost_sgd_per_year",
    "grid_cost_million_sgd_per_year",
    "pv_annual_cost_million_sgd_per_year",
    "battery_annual_cost_million_sgd_per_year",
    "lost_load_cost_million_sgd_per_year",
    "system_total_cost_million_sgd_per_year",
    "UE_kwh_per_year",
    "UE_mwh_per_year",
)


def build_baseline_cost_comparison_source(
    candidate_summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build Figure 4 source rows for fixed 20 MW PV comparison designs."""
    missing = [column for column in REQUIRED_COLUMNS if column not in candidate_summary.columns]
    if missing:
        raise ValueError("Baseline cost comparison source is missing required column(s): " + ", ".join(missing))
    if candidate_summary.empty:
        raise ValueError("Baseline cost comparison source data is empty.")

    prepared = candidate_summary.copy()
    for column in REQUIRED_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if prepared[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Baseline cost comparison source contains missing or non-numeric required values.")

    no_system = prepared[
        np.isclose(prepared["pv_capacity_kw"].astype(float), 0.0)
        & np.isclose(prepared["battery_energy_kwh"].astype(float), 0.0)
    ].copy()
    if no_system.empty:
        raise ValueError("Baseline cost comparison requires a no-PV/no-BESS reference row in candidate summary.")
    no_system_row = no_system.sort_values("pv_bess_system_annual_cost_sgd").iloc[0]
    pv_only_row = _select_reference_pv_only_row(prepared)
    pv_bess_row = _select_reference_pv_bess_row(prepared)

    source = pd.DataFrame(
        [
            _source_row("no_pv_bess", "No PV/BESS", no_system_row, config),
            _source_row("pv20_no_bess", _design_label(pv_only_row), pv_only_row, config),
            _source_row("pv20_bess10", _design_label(pv_bess_row), pv_bess_row, config),
        ]
    )
    return validate_baseline_cost_comparison_source(source)


def validate_baseline_cost_comparison_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SOURCE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Baseline cost comparison source is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Baseline cost comparison source data is empty.")
    validated = source.copy()
    for column in SOURCE_COLUMNS:
        if column in {"design_case", "design_case_label"}:
            validated[column] = validated[column].astype(str)
        else:
            validated[column] = pd.to_numeric(validated[column], errors="coerce")
    numeric = [column for column in SOURCE_COLUMNS if column not in {"design_case", "design_case_label"}]
    if validated[numeric].isna().any().any():
        raise ValueError("Baseline cost comparison source contains missing or non-numeric values.")
    cost_columns = [
        "grid_cost_sgd_per_year",
        "pv_annual_cost_sgd_per_year",
        "battery_annual_cost_sgd_per_year",
        "lost_load_cost_sgd_per_year",
        "system_total_cost_sgd_per_year",
    ]
    if (validated[cost_columns] < 0.0).any().any():
        raise ValueError("Baseline cost comparison costs must be non-negative.")
    expected_cases = ["no_pv_bess", "pv20_no_bess", "pv20_bess10"]
    if validated["design_case"].tolist() != expected_cases:
        raise ValueError(
            "Baseline cost comparison source must contain no_pv_bess, pv20_no_bess, "
            "and pv20_bess10 rows."
        )
    return validated.reset_index(drop=True)


def make_baseline_cost_comparison_figure(
    out: Path,
    case_key: str,
    candidate_summary: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Path]:
    source = build_baseline_cost_comparison_source(candidate_summary, config)
    _remove_legacy_combined_figure(out, case_key)
    source_4a = _panel_source(source, FIGURE_4A_CASES)
    source_4b = _panel_source(source, FIGURE_4B_CASES)
    paths: dict[str, Path] = {
        "figure_4a_source_data": _write_source_data(out / f"{case_key}_{FIGURE_4A_NAME}", source_4a),
        "figure_4b_source_data": _write_source_data(out / f"{case_key}_{FIGURE_4B_NAME}", source_4b),
    }
    paths.update(plot_baseline_cost_comparison(out, case_key, source))
    return paths


def plot_baseline_cost_comparison(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    source = validate_baseline_cost_comparison_source(source)
    paths: dict[str, Path] = {}
    paths.update(
        _plot_cost_comparison_panel(
            out,
            case_key,
            _panel_source(source, FIGURE_4A_CASES),
            FIGURE_4A_NAME,
            "Baseline annual cost comparison",
        )
    )
    paths.update(
        _plot_cost_comparison_panel(
            out,
            case_key,
            _panel_source(source, FIGURE_4B_CASES),
            FIGURE_4B_NAME,
            "PV-BESS reference comparison",
        )
    )
    return paths


def _plot_cost_comparison_panel(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
    figure_name: str,
    title: str,
) -> dict[str, Path]:
    source = _validate_panel_source(source)
    units = figure_display_units(source)
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)

    fig_width = 4.4 if len(source) == 2 else 5.2
    fig, ax = plt.subplots(figsize=(fig_width, 3.35), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.22, top=0.68)
    components = [
        ("Annual grid bill", "grid_cost_sgd_per_year", "grid_cost_million_sgd_per_year", NATURE_PALETTE["neutral_light"]),
        ("PV depreciation + O&M", "pv_annual_cost_sgd_per_year", "pv_annual_cost_million_sgd_per_year", NATURE_PALETTE["teal"]),
        ("Battery depreciation + O&M", "battery_annual_cost_sgd_per_year", "battery_annual_cost_million_sgd_per_year", NATURE_PALETTE["blue"]),
        ("Lost load cost", "lost_load_cost_sgd_per_year", "lost_load_cost_million_sgd_per_year", NATURE_PALETTE["red"]),
    ]
    x = np.arange(len(source), dtype=float)
    bottom = np.zeros(len(source), dtype=float)
    for label, sgd_column, million_column, color in components:
        values = cost_component_display_series(
            source,
            units,
            sgd_column=sgd_column,
            million_column=million_column,
        ).to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, width=0.56, color=color, edgecolor="white", linewidth=0.35, label=label)
        bottom += values
    total_cost = system_cost_display_series(source, units).to_numpy(dtype=float)
    ax.scatter(
        x,
        total_cost,
        marker="_",
        s=190,
        color=NATURE_PALETTE["neutral_black"],
        linewidth=1.3,
        label="System total",
        zorder=5,
    )
    _annotate_design(ax, x, source, units, total_cost)
    ax.set_title(title, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(_display_case_labels(source, units))
    ax.set_ylabel(units.annual_cost_axis_label)
    ax.set_ylim(0.0, max(float(bottom.max()) * 1.22, 0.05))
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        handlelength=1.2,
        columnspacing=1.0,
    )

    base = out / f"{case_key}_{figure_name}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _select_exact_design_row(candidate_summary: pd.DataFrame, pv_capacity_kw: float, battery_energy_kwh: float) -> pd.Series:
    matches = candidate_summary[
        np.isclose(candidate_summary["pv_capacity_kw"].astype(float), float(pv_capacity_kw), atol=1e-9)
        & np.isclose(candidate_summary["battery_energy_kwh"].astype(float), float(battery_energy_kwh), atol=1e-9)
    ].copy()
    if matches.empty:
        raise ValueError(
            "Baseline cost comparison requires an existing candidate summary row for "
            f"PV {pv_capacity_kw / 1000.0:g} MW and BESS {battery_energy_kwh / 1000.0:g} MWh."
        )
    return matches.sort_values("pv_bess_system_annual_cost_sgd").iloc[0]


def _select_reference_pv_only_row(candidate_summary: pd.DataFrame) -> pd.Series:
    try:
        return _select_exact_design_row(candidate_summary, 20_000.0, 0.0)
    except ValueError:
        no_bess = candidate_summary[np.isclose(candidate_summary["battery_energy_kwh"].astype(float), 0.0)].copy()
        no_bess = no_bess[no_bess["pv_capacity_kw"].astype(float) > 0.0]
        if no_bess.empty:
            raise ValueError("Baseline cost comparison requires at least one PV-only candidate row.") from None
        max_pv = float(no_bess["pv_capacity_kw"].astype(float).max())
        matches = no_bess[np.isclose(no_bess["pv_capacity_kw"].astype(float), max_pv)].copy()
        return matches.sort_values("pv_bess_system_annual_cost_sgd").iloc[0]


def _select_reference_pv_bess_row(candidate_summary: pd.DataFrame) -> pd.Series:
    try:
        return _select_exact_design_row(candidate_summary, 20_000.0, 10_000.0)
    except ValueError:
        pv_bess = candidate_summary[
            (candidate_summary["pv_capacity_kw"].astype(float) > 0.0)
            & (candidate_summary["battery_energy_kwh"].astype(float) > 0.0)
        ].copy()
        if pv_bess.empty:
            raise ValueError("Baseline cost comparison requires at least one PV-BESS candidate row.") from None
        return pv_bess.sort_values("pv_bess_system_annual_cost_sgd").iloc[0]


def _panel_source(source: pd.DataFrame, cases: tuple[str, ...]) -> pd.DataFrame:
    panel = source[source["design_case"].astype(str).isin(cases)].copy()
    panel["design_case"] = pd.Categorical(panel["design_case"].astype(str), categories=list(cases), ordered=True)
    panel = panel.sort_values("design_case").reset_index(drop=True)
    panel["design_case"] = panel["design_case"].astype(str)
    return _validate_panel_source(panel)


def _validate_panel_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SOURCE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Baseline cost comparison panel source is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Baseline cost comparison panel source data is empty.")
    validated = source.copy()
    for column in SOURCE_COLUMNS:
        if column in {"design_case", "design_case_label"}:
            validated[column] = validated[column].astype(str)
        else:
            validated[column] = pd.to_numeric(validated[column], errors="coerce")
    numeric = [column for column in SOURCE_COLUMNS if column not in {"design_case", "design_case_label"}]
    if validated[numeric].isna().any().any():
        raise ValueError("Baseline cost comparison panel source contains missing or non-numeric values.")
    return validated.reset_index(drop=True)


def _remove_legacy_combined_figure(out: Path, case_key: str) -> None:
    base = out / f"{case_key}_{FIGURE_4_NAME}"
    for suffix in LEGACY_FIGURE_4_SUFFIXES:
        path = base.with_suffix(suffix) if suffix.startswith(".") else base.with_name(base.name + suffix)
        path.unlink(missing_ok=True)


def _source_row(design_case: str, label: str, row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    pv_cost = float(row["annualized_pv_cost_sgd"]) + float(row["pv_om_cost_sgd"])
    battery_cost = float(row["annualized_battery_cost_sgd"]) + float(row["battery_om_cost_sgd"])
    lost_load_cost = _lost_load_cost(row, config)
    grid_cost = float(row["pv_bess_grid_cost_sgd"])
    system_total = float(row["pv_bess_system_annual_cost_sgd"])
    ue = float(row["unserved_energy_kwh_per_year"])
    pv_capacity = float(row["pv_capacity_kw"])
    battery_energy = float(row["battery_energy_kwh"])
    return {
        "design_case": design_case,
        "design_case_label": label,
        "pv_capacity_kw": pv_capacity,
        "pv_capacity_mw": pv_capacity / 1000.0,
        "battery_energy_kwh": battery_energy,
        "battery_energy_mwh": battery_energy / 1000.0,
        "grid_cost_sgd_per_year": grid_cost,
        "pv_annual_cost_sgd_per_year": pv_cost,
        "battery_annual_cost_sgd_per_year": battery_cost,
        "lost_load_cost_sgd_per_year": lost_load_cost,
        "system_total_cost_sgd_per_year": system_total,
        "grid_cost_million_sgd_per_year": grid_cost / 1e6,
        "pv_annual_cost_million_sgd_per_year": pv_cost / 1e6,
        "battery_annual_cost_million_sgd_per_year": battery_cost / 1e6,
        "lost_load_cost_million_sgd_per_year": lost_load_cost / 1e6,
        "system_total_cost_million_sgd_per_year": system_total / 1e6,
        "UE_kwh_per_year": ue,
        "UE_mwh_per_year": ue / 1000.0,
    }


def _design_label(row: pd.Series) -> str:
    return f"{float(row['pv_capacity_kw']) / 1000.0:g} MW PV / {float(row['battery_energy_kwh']) / 1000.0:g} MWh Battery"


def _display_case_labels(source: pd.DataFrame, units: FigureDisplayUnits) -> list[str]:
    labels = []
    for _, row in source.iterrows():
        label = format_display_design_label(row, units)
        labels.append(label.replace("No PV/BESS", "No\nPV/BESS").replace(" PV / ", " PV /\n"))
    return labels


def _format_mw(pv_capacity_kw: float) -> str:
    return format_pv_capacity_from_kw(pv_capacity_kw, FigureDisplayUnits(True, True, True, True))


def _format_mwh(battery_energy_kwh: float) -> str:
    return format_battery_energy_from_kwh(battery_energy_kwh, FigureDisplayUnits(True, True, True, True))


def _lost_load_cost(row: pd.Series, config: dict[str, Any]) -> float:
    if "unserved_energy_cost_sgd_per_year" in row.index and pd.notna(row["unserved_energy_cost_sgd_per_year"]):
        return float(row["unserved_energy_cost_sgd_per_year"])
    voll = _voll_from_config(config)
    return float(row["unserved_energy_kwh_per_year"]) * voll


def _voll_from_config(config: dict[str, Any]) -> float:
    active = config.get("active_scenario", {})
    if "voll_sgd_per_kwh" in active:
        return float(active["voll_sgd_per_kwh"])
    scenarios = config.get("scenarios", {})
    if isinstance(scenarios, dict) and scenarios:
        first = next(iter(scenarios.values()))
        if "voll_sgd_per_kwh" in first:
            return float(first["voll_sgd_per_kwh"])
    return 0.0


def _annotate_design(
    ax: Any,
    x: np.ndarray,
    source: pd.DataFrame,
    units: FigureDisplayUnits,
    total_cost: np.ndarray,
) -> None:
    for xpos, row, cost in zip(x, source.to_dict("records"), total_cost, strict=True):
        label = (
            f"PV {format_pv_capacity_from_kw(float(row['pv_capacity_kw']), units)}\n"
            f"BESS {format_battery_energy_from_kwh(float(row['battery_energy_kwh']), units)}"
        )
        ax.text(
            float(xpos),
            float(cost) * 1.03,
            label,
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=NATURE_PALETTE["neutral_dark"],
        )
