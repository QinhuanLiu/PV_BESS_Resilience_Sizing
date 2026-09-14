from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .baseline_design_space import build_baseline_design_space_source
from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    matplotlib_context,
    nice_display_ticks,
    save_publication_figure,
    write_source_data as _write_source_data,
)


FIGURE_X_NAME = "Figure_X_Baseline_LOLP_Constraint_Scan"
FIGURE_X_BESS_SLICE_NAME = "Figure_X_Baseline_LOLP_BESS_Slices"
DEFAULT_BASELINE_LOLP_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
BASELINE_LOLP_BESS_PV_SLICES_MW = (
    ("Low PV", 1, 8.0),
    ("Medium PV", 2, 20.0),
    ("High PV", 3, 32.0),
)
REQUIRED_SCAN_COLUMNS = (
    "design_lolp_constraint",
    "design_lolp_constraint_percent",
    "available",
    "actual_LOLP",
    "actual_LOLP_percent",
    "system_total_cost_sgd_per_year",
    "system_total_cost_million_sgd_per_year",
    "pv_capacity_kw",
    "pv_capacity_mw",
    "battery_energy_kwh",
    "battery_energy_mwh",
    "UE_kwh_per_year",
    "UE_mwh_per_year",
)
REQUIRED_BESS_SLICE_COLUMNS = (
    "pv_slice_label",
    "pv_slice_rank",
    "pv_slice_target_mw",
    "pv_slice_selected_mw",
    "pv_slice_source",
    "pv_capacity_kw",
    "pv_capacity_mw",
    "battery_energy_kwh",
    "battery_energy_mwh",
    "energy_based_LOLP",
    "UE_kwh_per_year",
    "UE_mwh_per_year",
    "system_total_cost_sgd_per_year",
    "system_total_cost_million_sgd_per_year",
)


def build_baseline_lolp_bess_slice_source(
    candidate_summary: pd.DataFrame,
    case_frame: pd.DataFrame,
    outage_profile: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Return fixed-PV baseline LOLP slices versus BESS energy capacity."""
    design_space = build_baseline_design_space_source(candidate_summary, case_frame, outage_profile, config)
    return select_baseline_lolp_bess_slice_source(design_space)


def select_baseline_lolp_bess_slice_source(design_space_source: pd.DataFrame) -> pd.DataFrame:
    source = design_space_source.copy()
    required = [column for column in REQUIRED_BESS_SLICE_COLUMNS if not column.startswith("pv_slice_")]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError("Baseline LOLP-BESS slice source is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Baseline LOLP-BESS slice source data is empty.")
    for column in required:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source[required].isna().any().any():
        raise ValueError("Baseline LOLP-BESS slice source contains missing or non-numeric required values.")
    if not source["energy_based_LOLP"].between(0.0, 1.0).all():
        raise ValueError("Baseline LOLP-BESS slice LOLP values must be between 0 and 1.")

    selected: list[pd.DataFrame] = []
    bess_counts = source.groupby("pv_capacity_mw")["battery_energy_mwh"].nunique()
    available_pv = sorted(bess_counts[bess_counts >= 3].index.astype(float).tolist())
    if len(available_pv) < 3:
        available_text = ", ".join(_format_capacity_label(value) for value in available_pv)
        raise ValueError(
            "Baseline LOLP-BESS slices require at least three PV capacities with at least three BESS points each. "
            f"Available: {available_text or 'none'}."
        )
    missing_targets: list[float] = []
    for label, rank, target_mw in BASELINE_LOLP_BESS_PV_SLICES_MW:
        mask = np.isclose(source["pv_capacity_mw"].astype(float), float(target_mw), atol=1e-9)
        rows = source[mask].copy()
        if rows.empty:
            missing_targets.append(float(target_mw))
            continue
        rows["pv_slice_label"] = label
        rows["pv_slice_rank"] = int(rank)
        rows["pv_slice_target_mw"] = float(target_mw)
        rows["pv_slice_selected_mw"] = float(target_mw)
        rows["pv_slice_source"] = "fixed_8_20_32_mw"
        selected.append(rows)
    if missing_targets:
        selected = _select_adaptive_lolp_bess_slices(source, available_pv)
    return validate_baseline_lolp_bess_slice_source(pd.concat(selected, ignore_index=True))


def validate_baseline_lolp_bess_slice_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_BESS_SLICE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Baseline LOLP-BESS slice source is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Baseline LOLP-BESS slice source data is empty.")
    validated = source.copy()
    numeric_columns = [column for column in REQUIRED_BESS_SLICE_COLUMNS if column not in {"pv_slice_label", "pv_slice_source"}]
    for column in numeric_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[numeric_columns].isna().any().any():
        raise ValueError("Baseline LOLP-BESS slice source contains missing or non-numeric required values.")
    if not validated["energy_based_LOLP"].between(0.0, 1.0).all():
        raise ValueError("Baseline LOLP-BESS slice LOLP values must be between 0 and 1.")
    for column in ("pv_slice_label", "pv_slice_source"):
        validated[column] = validated[column].astype(str)
        if (validated[column].str.strip() == "").any():
            raise ValueError(f"Baseline LOLP-BESS slice source contains empty {column} values.")
    labels = validated[["pv_slice_label", "pv_slice_rank", "pv_capacity_mw"]].drop_duplicates()
    if labels["pv_slice_label"].nunique() != 3 or labels["pv_capacity_mw"].nunique() != 3:
        raise ValueError("Baseline LOLP-BESS slice source requires exactly three distinct PV slices.")
    bess_counts = validated.groupby("pv_slice_label")["battery_energy_mwh"].nunique()
    if (bess_counts < 3).any():
        raise ValueError("Baseline LOLP-BESS slice source requires at least three BESS points per PV slice.")
    return validated.sort_values(["pv_slice_rank", "battery_energy_mwh"]).reset_index(drop=True)


def make_baseline_lolp_bess_slice_figure(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    validated = validate_baseline_lolp_bess_slice_source(source)
    base = out / f"{case_key}_{FIGURE_X_BESS_SLICE_NAME}"
    paths: dict[str, Path] = {"source_data": _write_source_data(base, validated)}
    paths.update(plot_baseline_lolp_bess_slices(out, case_key, validated))
    return paths


def plot_baseline_lolp_bess_slices(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    source = validate_baseline_lolp_bess_slice_source(source)
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.18, top=0.76)
    colors = {
        "Low PV": NATURE_PALETTE["blue"],
        "Medium PV": NATURE_PALETTE["teal"],
        "High PV": NATURE_PALETTE["red"],
    }
    handles: list[Any] = []
    grouped = list(source.groupby(["pv_slice_rank", "pv_slice_label", "pv_capacity_mw"], sort=True))
    for _rank, label, pv_mw in [key for key, _group in grouped]:
        group = source[
            (source["pv_slice_label"].astype(str) == str(label))
            & np.isclose(source["pv_capacity_mw"].astype(float), float(pv_mw), atol=1e-9)
        ].sort_values("battery_energy_mwh")
        line_label = f"{label} ({_format_capacity_label(float(pv_mw))})"
        (line,) = ax.plot(
            group["battery_energy_mwh"].to_numpy(dtype=float),
            group["energy_based_LOLP"].to_numpy(dtype=float),
            color=colors[str(label)],
            linewidth=1.25,
            marker="o",
            markersize=2.8,
            label=line_label,
        )
        handles.append(line)

    ax.set_title("Energy-based LOLP by BESS capacity", loc="left")
    ax.set_xlabel("BESS energy (MWh)")
    ax.set_ylabel("Energy-based LOLP")
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")
    ticks = nice_display_ticks(source["energy_based_LOLP"].astype(float), unit_type="lolp")
    if len(ticks) >= 2:
        ax.set_yticks(ticks)
    ax.set_ylim(bottom=0.0)
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

    base = out / f"{case_key}_{FIGURE_X_BESS_SLICE_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _select_adaptive_lolp_bess_slices(source: pd.DataFrame, available_pv: list[float]) -> list[pd.DataFrame]:
    selected: list[pd.DataFrame] = []
    indices = (0, len(available_pv) // 2, len(available_pv) - 1)
    for (label, rank, _fixed_target), index in zip(BASELINE_LOLP_BESS_PV_SLICES_MW, indices, strict=True):
        selected_mw = float(available_pv[int(index)])
        rows = source[np.isclose(source["pv_capacity_mw"].astype(float), selected_mw, atol=1e-9)].copy()
        rows["pv_slice_label"] = label
        rows["pv_slice_rank"] = int(rank)
        rows["pv_slice_target_mw"] = selected_mw
        rows["pv_slice_selected_mw"] = selected_mw
        rows["pv_slice_source"] = "adaptive_low_medium_high_mw"
        selected.append(rows)
    return selected


def build_baseline_lolp_constraint_scan_source(
    candidate_summary: pd.DataFrame,
    case_frame: pd.DataFrame,
    outage_profile: pd.DataFrame,
    config: dict[str, Any],
    thresholds: Sequence[float] = DEFAULT_BASELINE_LOLP_THRESHOLDS,
) -> pd.DataFrame:
    """Return cheapest baseline single-outage designs satisfying each LOLP threshold."""
    design_space = build_baseline_design_space_source(candidate_summary, case_frame, outage_profile, config)
    return select_baseline_lolp_constraint_designs(design_space, thresholds=thresholds)


def select_baseline_lolp_constraint_designs(
    design_space_source: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_BASELINE_LOLP_THRESHOLDS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = design_space_source.copy()
    for column in (
        "energy_based_LOLP",
        "system_total_cost_sgd_per_year",
        "system_total_cost_million_sgd_per_year",
        "pv_capacity_kw",
        "pv_capacity_mw",
        "battery_energy_kwh",
        "battery_energy_mwh",
        "UE_kwh_per_year",
        "UE_mwh_per_year",
    ):
        if column not in source.columns:
            raise ValueError(f"Baseline LOLP scan source is missing required column: {column}")
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.empty:
        raise ValueError("Baseline LOLP scan source data is empty.")
    if source[
        [
            "energy_based_LOLP",
            "system_total_cost_sgd_per_year",
            "pv_capacity_kw",
            "battery_energy_kwh",
        ]
    ].isna().any().any():
        raise ValueError("Baseline LOLP scan source contains missing or non-numeric required values.")

    for threshold in _validated_thresholds(thresholds):
        feasible = source[source["energy_based_LOLP"].astype(float) <= float(threshold)].copy()
        if feasible.empty:
            rows.append(_unavailable_row(threshold))
            continue
        selected = feasible.sort_values(
            ["system_total_cost_sgd_per_year", "pv_capacity_kw", "battery_energy_kwh"],
            ascending=[True, True, True],
        ).iloc[0]
        rows.append(
            {
                "design_lolp_constraint": float(threshold),
                "design_lolp_constraint_percent": float(threshold) * 100.0,
                "available": True,
                "actual_LOLP": float(selected["energy_based_LOLP"]),
                "actual_LOLP_percent": float(selected["energy_based_LOLP"]) * 100.0,
                "system_total_cost_sgd_per_year": float(selected["system_total_cost_sgd_per_year"]),
                "system_total_cost_million_sgd_per_year": float(selected["system_total_cost_million_sgd_per_year"]),
                "pv_capacity_kw": float(selected["pv_capacity_kw"]),
                "pv_capacity_mw": float(selected["pv_capacity_mw"]),
                "battery_energy_kwh": float(selected["battery_energy_kwh"]),
                "battery_energy_mwh": float(selected["battery_energy_mwh"]),
                "UE_kwh_per_year": float(selected["UE_kwh_per_year"]),
                "UE_mwh_per_year": float(selected["UE_mwh_per_year"]),
            }
        )
    return validate_baseline_lolp_constraint_scan_source(pd.DataFrame(rows))


def validate_baseline_lolp_constraint_scan_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SCAN_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Baseline LOLP scan source data is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Baseline LOLP scan source data is empty.")
    validated = source.copy()
    validated["available"] = validated["available"].map(_truthy)
    for column in ("design_lolp_constraint", "design_lolp_constraint_percent"):
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[["design_lolp_constraint", "design_lolp_constraint_percent"]].isna().any().any():
        raise ValueError("Baseline LOLP scan constraints must be numeric.")
    if not validated["design_lolp_constraint"].between(0.0, 1.0).all():
        raise ValueError("Baseline LOLP scan constraints must be between 0 and 1.")

    metric_columns = [column for column in REQUIRED_SCAN_COLUMNS if column not in {"available"}]
    selected_metrics = [column for column in metric_columns if not column.startswith("design_lolp_constraint")]
    available = validated["available"]
    if available.any():
        converted = validated.loc[available, selected_metrics].apply(pd.to_numeric, errors="coerce")
        if converted.isna().any().any():
            raise ValueError("Available baseline LOLP scan rows must contain numeric selected-design metrics.")
        for column in selected_metrics:
            validated.loc[available, column] = converted[column]
        if not validated.loc[available, "actual_LOLP"].between(0.0, 1.0).all():
            raise ValueError("Actual LOLP values must be between 0 and 1.")
    return validated.sort_values("design_lolp_constraint").reset_index(drop=True)


def make_baseline_lolp_constraint_scan_figure(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    validated = validate_baseline_lolp_constraint_scan_source(source)
    base = out / f"{case_key}_{FIGURE_X_NAME}"
    paths: dict[str, Path] = {"source_data": _write_source_data(base, validated)}
    paths.update(plot_baseline_lolp_constraint_scan(out, case_key, validated))
    return paths


def plot_baseline_lolp_constraint_scan(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    source = validate_baseline_lolp_constraint_scan_source(source)
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.7), constrained_layout=True)
    flat_axes = axes.ravel()
    available = source[source["available"]].copy()
    x_all = source["design_lolp_constraint"].astype(float).to_numpy()
    x_available = available["design_lolp_constraint"].astype(float).to_numpy()

    _plot_scan_metric(
        flat_axes[0],
        x_available,
        available["system_total_cost_million_sgd_per_year"].astype(float).to_numpy(),
        "a  Optimal annual cost",
        "Cost (million SGD yr$^{-1}$)",
        NATURE_PALETTE["blue"],
    )
    _plot_scan_metric(
        flat_axes[1],
        x_available,
        available["actual_LOLP"].astype(float).to_numpy(),
        "b  Actual LOLP",
        "Actual LOLP",
        NATURE_PALETTE["red"],
        percent_y=True,
    )
    flat_axes[1].plot(
        x_all,
        x_all,
        color=NATURE_PALETTE["neutral_mid"],
        linestyle="--",
        linewidth=0.8,
    )
    _plot_scan_metric(
        flat_axes[2],
        x_available,
        available["pv_capacity_mw"].astype(float).to_numpy(),
        "c  Selected PV capacity",
        "PV capacity (MW)",
        NATURE_PALETTE["teal"],
    )
    _plot_scan_metric(
        flat_axes[3],
        x_available,
        available["battery_energy_mwh"].astype(float).to_numpy(),
        "d  Selected BESS capacity",
        "BESS energy (MWh)",
        NATURE_PALETTE["gold"],
    )
    for ax in flat_axes:
        ax.set_xlabel("Design LOLP constraint")
        ax.set_xticks(x_all)
        ax.set_xticklabels([f"{value * 100:.0f}%" for value in x_all])
        ax.grid(True, color="#E4E4E4", linewidth=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    base = out / f"{case_key}_{FIGURE_X_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _plot_scan_metric(
    ax: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    ylabel: str,
    color: str,
    *,
    percent_y: bool = False,
) -> None:
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    if x_values.size:
        ax.plot(x_values, y_values, color=color, marker="o", linewidth=1.2, markersize=3.0)
    if percent_y:
        ax.yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")
        ax.set_ylim(bottom=0.0)


def _validated_thresholds(thresholds: Sequence[float]) -> list[float]:
    values = sorted(dict.fromkeys(float(value) for value in thresholds))
    if not values:
        raise ValueError("Baseline LOLP scan requires at least one threshold.")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Baseline LOLP thresholds must be between 0 and 1.")
    return values


def _unavailable_row(threshold: float) -> dict[str, Any]:
    return {
        "design_lolp_constraint": float(threshold),
        "design_lolp_constraint_percent": float(threshold) * 100.0,
        "available": False,
        "actual_LOLP": pd.NA,
        "actual_LOLP_percent": pd.NA,
        "system_total_cost_sgd_per_year": pd.NA,
        "system_total_cost_million_sgd_per_year": pd.NA,
        "pv_capacity_kw": pd.NA,
        "pv_capacity_mw": pd.NA,
        "battery_energy_kwh": pd.NA,
        "battery_energy_mwh": pd.NA,
        "UE_kwh_per_year": pd.NA,
        "UE_mwh_per_year": pd.NA,
    }


def _format_capacity_label(value_mw: float) -> str:
    if np.isclose(value_mw, round(value_mw), atol=1e-9):
        return f"{int(round(value_mw))} MW"
    return f"{value_mw:.1f} MW"


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "available"}
    return bool(value)
