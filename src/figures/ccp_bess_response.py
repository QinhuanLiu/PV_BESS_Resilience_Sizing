from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    FigureDependencyError,
    matplotlib_context,
    plot_smooth_line,
    save_publication_figure,
    write_source_data as _write_source_data,
)
from ..resilience_metrics import DEFAULT_LOLP_THRESHOLD


FIGURE_12_NAME = "Figure_12_CCP_BESS_Capacity_Response"
FIGURE_12_PV_SLICES_MW = (
    ("PV = 8 MW", 1, 8.0),
    ("PV = 20 MW", 2, 20.0),
    ("PV = 32 MW", 3, 32.0),
)
REQUIRED_CANDIDATE_COLUMNS = (
    "pv_capacity_kw",
    "battery_energy_kwh",
    "chance_constraint_probability",
    "mean_scenario_LOLP",
    "lolp_threshold",
    "frontier_outage_scenario_count",
)
REQUIRED_SOURCE_COLUMNS = (
    "pv_slice_label",
    "pv_slice_rank",
    "pv_slice_target_mw",
    "pv_slice_selected_mw",
    "pv_slice_source",
    "pv_capacity_kw",
    "pv_capacity_mw",
    "battery_energy_kwh",
    "battery_energy_mwh",
    "chance_constraint_probability",
    "chance_constraint_probability_percent",
    "mean_scenario_LOLP",
    "lolp_threshold",
    "lolp_threshold_percent",
    "frontier_outage_scenario_count",
)


def build_ccp_bess_capacity_response_source(candidates: pd.DataFrame) -> pd.DataFrame:
    """Build Figure 12 source data from resilience-frontier candidate aggregates."""
    missing = [column for column in REQUIRED_CANDIDATE_COLUMNS if column not in candidates.columns]
    if missing:
        raise ValueError("Figure 12 CCP-BESS source is missing required column(s): " + ", ".join(missing))
    if candidates.empty:
        raise ValueError("Figure 12 CCP-BESS source data is empty.")

    source = candidates.copy()
    for column in REQUIRED_CANDIDATE_COLUMNS:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source[list(REQUIRED_CANDIDATE_COLUMNS)].isna().any().any():
        raise ValueError("Figure 12 CCP-BESS source contains missing or non-numeric required values.")
    if "pv_capacity_mw" in source.columns:
        source["pv_capacity_mw"] = pd.to_numeric(source["pv_capacity_mw"], errors="coerce")
    else:
        source["pv_capacity_mw"] = source["pv_capacity_kw"].astype(float) / 1000.0
    if source["pv_capacity_mw"].isna().any():
        raise ValueError("Figure 12 CCP-BESS source contains invalid PV capacity values.")
    source["battery_energy_mwh"] = source["battery_energy_kwh"].astype(float) / 1000.0

    selected: list[pd.DataFrame] = []
    missing_targets: list[float] = []
    for label, rank, target_mw in FIGURE_12_PV_SLICES_MW:
        rows = source[np.isclose(source["pv_capacity_mw"].astype(float), target_mw, atol=1e-9)].copy()
        if rows.empty:
            missing_targets.append(target_mw)
            continue
        rows["pv_slice_label"] = label
        rows["pv_slice_rank"] = rank
        rows["pv_slice_target_mw"] = target_mw
        rows["pv_slice_selected_mw"] = target_mw
        rows["pv_slice_source"] = "fixed_8_20_32_mw"
        selected.append(rows)
    if missing_targets:
        available = ", ".join(_format_capacity_label(value) for value in sorted(source["pv_capacity_mw"].unique()))
        required = ", ".join(_format_capacity_label(value) for value in missing_targets)
        raise ValueError(
            "Figure 12 requires fixed PV slices from Figure 3C. "
            f"Missing: {required}. Available PV capacities: {available or 'none'}."
        )

    result = pd.concat(selected, ignore_index=True)
    result["chance_constraint_probability_percent"] = result["chance_constraint_probability"].astype(float) * 100.0
    result["lolp_threshold_percent"] = result["lolp_threshold"].astype(float) * 100.0
    return validate_ccp_bess_capacity_response_source(result)


def validate_ccp_bess_capacity_response_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Figure 12 CCP-BESS source is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Figure 12 CCP-BESS source data is empty.")
    validated = source.copy()
    for column in REQUIRED_SOURCE_COLUMNS:
        if column in {"pv_slice_label", "pv_slice_source"}:
            validated[column] = validated[column].astype(str)
        else:
            validated[column] = pd.to_numeric(validated[column], errors="coerce")
    numeric = [column for column in REQUIRED_SOURCE_COLUMNS if column not in {"pv_slice_label", "pv_slice_source"}]
    if validated[numeric].isna().any().any():
        raise ValueError("Figure 12 CCP-BESS source contains missing or non-numeric values.")
    if not validated["chance_constraint_probability"].between(0.0, 1.0).all():
        raise ValueError("Figure 12 CCP-BESS CCP values must be between 0 and 1.")
    if not np.isclose(validated["lolp_threshold"].astype(float), DEFAULT_LOLP_THRESHOLD, atol=1e-9).all():
        raise ValueError("Figure 12 CCP-BESS source must use lolp_threshold = 0.10.")
    labels = validated[["pv_slice_label", "pv_slice_rank", "pv_capacity_mw"]].drop_duplicates()
    if labels["pv_slice_label"].nunique() != 3 or labels["pv_capacity_mw"].nunique() != 3:
        raise ValueError("Figure 12 CCP-BESS source requires exactly three distinct PV slices.")
    bess_counts = validated.groupby("pv_slice_label")["battery_energy_mwh"].nunique()
    if (bess_counts < 3).any():
        raise ValueError("Figure 12 CCP-BESS source requires at least three BESS points per PV slice.")
    scenario_counts = validated["frontier_outage_scenario_count"].astype(float)
    if (scenario_counts <= 0).any():
        raise ValueError("Figure 12 CCP-BESS source requires a positive frontier scenario count.")
    return validated.sort_values(["pv_slice_rank", "battery_energy_mwh"]).reset_index(drop=True)


def make_ccp_bess_capacity_response_figure(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    validated = validate_ccp_bess_capacity_response_source(source)
    base = out / f"{case_key}_{FIGURE_12_NAME}"
    paths: dict[str, Path] = {"source_data": _write_source_data(base, validated)}
    paths.update(plot_ccp_bess_capacity_response(out, case_key, validated))
    return paths


def plot_ccp_bess_capacity_response(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    source = validate_ccp_bess_capacity_response_source(source)
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.18, top=0.78)
    colors = {
        "PV = 8 MW": NATURE_PALETTE["blue"],
        "PV = 20 MW": NATURE_PALETTE["teal"],
        "PV = 32 MW": NATURE_PALETTE["red"],
    }
    handles: list[Any] = []
    grouped = list(source.groupby(["pv_slice_rank", "pv_slice_label", "pv_capacity_mw"], sort=True))
    for _rank, label, pv_mw in [key for key, _group in grouped]:
        group = source[
            (source["pv_slice_label"].astype(str) == str(label))
            & np.isclose(source["pv_capacity_mw"].astype(float), float(pv_mw), atol=1e-9)
        ].sort_values("battery_energy_mwh")
        line = _plot_response_line(ax, group, str(label), colors[str(label)])
        handles.append(line)

    ax.set_title("CCP response to BESS capacity", loc="left")
    ax.set_xlabel("BESS energy capacity (MWh)")
    ax.set_ylabel("CCP at LOLP <= 10%")
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0.0, right=float(source["battery_energy_mwh"].max()))
    ax.margins(x=0.0)
    ax.grid(True, color="#E4E4E4", linewidth=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        handlelength=2.5,
        columnspacing=1.2,
    )

    base = out / f"{case_key}_{FIGURE_12_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _plot_response_line(ax: Any, group: pd.DataFrame, label: str, color: str) -> Any:
    x = group["battery_energy_mwh"].to_numpy(dtype=float)
    y = group["chance_constraint_probability"].to_numpy(dtype=float)
    try:
        return plot_smooth_line(ax, x, y, color=color, linewidth=1.35, label=label)
    except (FigureDependencyError, ValueError):
        (line,) = ax.plot(x, y, color=color, linewidth=1.35, marker=None, label=label)
        return line


def _format_capacity_label(value_mw: float) -> str:
    if np.isclose(value_mw, round(value_mw), atol=1e-9):
        return f"{int(round(value_mw))} MW"
    return f"{value_mw:.1f} MW"
