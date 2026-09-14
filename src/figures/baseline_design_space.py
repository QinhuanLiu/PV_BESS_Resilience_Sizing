from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    matplotlib_context,
    nice_display_ticks,
    plot_smooth_line,
    save_publication_figure,
    write_source_data as _write_source_data,
)
from .display_units import (
    DISPLAY_BATTERY_ENERGY,
    DISPLAY_UE,
    DISPLAY_PV_CAPACITY,
    DISPLAY_SYSTEM_COST,
    FigureDisplayUnits,
    add_display_unit_columns,
    format_pv_capacity_from_kw,
    nice_levels,
    sparse_ticks_from_levels,
)
from ..resilience_metrics import electric_equivalent_building_load, outage_weighted_load_kwh


FIGURE_3_SOURCE_NAME = "Figure_3_Baseline_PV_BESS_Design_Space"
FIGURE_3A_NAME = "Figure_3A_Baseline_PV_BESS_3D_Response_Surfaces"
FIGURE_3B_NAME = "Figure_3B_Baseline_PV_BESS_Contour_Maps"
FIGURE_3C_NAME = "Figure_3C_Baseline_PV_Capacity_BESS_Slices"
FIGURE_3B_REFERENCE_PV_MW = 20.0
FIGURE_3B_REFERENCE_BESS_MWH = 10.0
CONTOUR_INTERVALS = 10
COLORBAR_LABEL_DIVISIONS = 5
FINE_CONTOUR_TARGET_INTERVALS = 22
FIGURE_3C_FIXED_PV_SLICES_MW = (
    ("Low PV", 1, 8.0),
    ("Medium PV", 2, 20.0),
    ("High PV", 3, 32.0),
)

REQUIRED_SUMMARY_COLUMNS = (
    "pv_capacity_kw",
    "battery_energy_kwh",
    "pv_bess_system_annual_cost_sgd",
    "unserved_energy_kwh_per_year",
)
REQUIRED_SOURCE_COLUMNS = (
    "pv_capacity_kw",
    "pv_capacity_mw",
    "battery_energy_kwh",
    "battery_energy_mwh",
    "system_total_cost_sgd_per_year",
    "system_total_cost_million_sgd_per_year",
    "UE_kwh_per_year",
    "UE_mwh_per_year",
    "outage_weighted_load_kwh",
    "energy_based_LOLP",
)
REQUIRED_SLICE_SOURCE_COLUMNS = REQUIRED_SOURCE_COLUMNS + (
    "pv_slice_label",
    "pv_slice_rank",
    "pv_slice_source",
)


@dataclass(frozen=True)
class BaselineMetricSpec:
    key: str
    title: str
    z_label: str
    colorbar_label: str
    cmap: str
    unit_type: str


@dataclass(frozen=True)
class BaselinePVSliceSpec:
    label: str
    rank: int
    target_mw: float
    selected_mw: float
    source: str


BASELINE_METRICS = (
    BaselineMetricSpec(
        key="system_total_cost_million_sgd_per_year",
        title="System total cost",
        z_label="Cost (million SGD yr$^{-1}$)",
        colorbar_label="Cost (million SGD yr$^{-1}$)",
        cmap="Blues",
        unit_type="cost",
    ),
    BaselineMetricSpec(
        key="UE_mwh_per_year",
        title="Unserved energy",
        z_label="UE (MWh yr$^{-1}$)",
        colorbar_label="UE (MWh yr$^{-1}$)",
        cmap="Reds",
        unit_type="ue",
    ),
    BaselineMetricSpec(
        key="energy_based_LOLP",
        title="Energy-based LOLP",
        z_label="LOLP",
        colorbar_label="Energy-based LOLP",
        cmap="YlGnBu",
        unit_type="lolp",
    ),
)


def _display_baseline_metrics(units: FigureDisplayUnits) -> tuple[BaselineMetricSpec, ...]:
    return (
        BaselineMetricSpec(
            key=DISPLAY_SYSTEM_COST,
            title="System total cost",
            z_label=units.cost_axis_label,
            colorbar_label=units.cost_axis_label,
            cmap="Blues",
            unit_type="cost",
        ),
        BaselineMetricSpec(
            key=DISPLAY_UE,
            title="Unserved energy",
            z_label=units.ue_axis_label,
            colorbar_label=units.ue_axis_label,
            cmap="Reds",
            unit_type="ue",
        ),
        BaselineMetricSpec(
            key="energy_based_LOLP",
            title="Energy-based LOLP",
            z_label="LOLP",
            colorbar_label="Energy-based LOLP",
            cmap="YlGnBu",
            unit_type="lolp",
        ),
    )


def build_baseline_design_space_source(
    candidate_summary: pd.DataFrame,
    case_frame: pd.DataFrame,
    outage_profile: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build single-outage-scenario source data for baseline PV-BESS Figure 3."""
    missing = [column for column in REQUIRED_SUMMARY_COLUMNS if column not in candidate_summary.columns]
    if missing:
        raise ValueError("Baseline candidate summary is missing required column(s): " + ", ".join(missing))
    if "outage_fraction" not in outage_profile.columns or "hour_index" not in outage_profile.columns:
        raise ValueError("Baseline outage profile must include hour_index and outage_fraction")
    if "hour_index" not in case_frame.columns:
        raise ValueError("Baseline case frame must include hour_index")

    technical = config["technical"]
    total_load = electric_equivalent_building_load(
        case_frame,
        float(technical.get("cooling_cop", technical.get("chiller_cop", 5.0))),
        float(technical.get("heating_cop", 3.5)),
    )
    outage = case_frame[["hour_index"]].merge(
        outage_profile[["hour_index", "outage_fraction"]],
        on="hour_index",
        how="left",
    )["outage_fraction"].fillna(0.0)
    outage_load = outage_weighted_load_kwh(total_load, outage)
    outage_weighted_total = float(outage_load.sum())

    source = candidate_summary.copy()
    for column in REQUIRED_SUMMARY_COLUMNS:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source[list(REQUIRED_SUMMARY_COLUMNS)].isna().any().any():
        raise ValueError("Baseline candidate summary contains missing or non-numeric required values.")

    source = (
        source.groupby(["pv_capacity_kw", "battery_energy_kwh"], as_index=False)
        .agg(
            {
                "pv_bess_system_annual_cost_sgd": "mean",
                "unserved_energy_kwh_per_year": "mean",
            }
        )
        .reset_index(drop=True)
    )
    source["pv_capacity_mw"] = source["pv_capacity_kw"].astype(float) / 1000.0
    source["battery_energy_mwh"] = source["battery_energy_kwh"].astype(float) / 1000.0
    source["system_total_cost_sgd_per_year"] = source["pv_bess_system_annual_cost_sgd"].astype(float)
    source["system_total_cost_million_sgd_per_year"] = source["system_total_cost_sgd_per_year"] / 1e6
    source["UE_kwh_per_year"] = source["unserved_energy_kwh_per_year"].astype(float)
    source["UE_mwh_per_year"] = source["UE_kwh_per_year"] / 1000.0
    source["outage_weighted_load_kwh"] = outage_weighted_total
    if outage_weighted_total <= 0.0:
        source["energy_based_LOLP"] = 0.0
    else:
        source["energy_based_LOLP"] = source["UE_kwh_per_year"] / outage_weighted_total
    rooftop_reference = _rooftop_pv_reference_kw(config)
    if rooftop_reference is not None:
        rooftop_kw, twice_rooftop_kw = rooftop_reference
        source["rooftop_pv_limit_kw"] = rooftop_kw
        source["rooftop_pv_limit_mw"] = rooftop_kw / 1000.0
        source["twice_rooftop_pv_limit_kw"] = twice_rooftop_kw
        source["twice_rooftop_pv_limit_mw"] = twice_rooftop_kw / 1000.0
    return validate_baseline_design_space_source(source)


def validate_baseline_design_space_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Baseline design-space source data is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Baseline design-space source data is empty.")
    validated = source.copy()
    for column in REQUIRED_SOURCE_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(REQUIRED_SOURCE_COLUMNS)].isna().any().any():
        raise ValueError("Baseline design-space source data contains missing or non-numeric required values.")
    if (validated["system_total_cost_sgd_per_year"] < 0).any():
        raise ValueError("Baseline design-space costs must be non-negative.")
    if (validated["UE_kwh_per_year"] < 0).any():
        raise ValueError("Baseline design-space UE values must be non-negative.")
    if not validated["energy_based_LOLP"].between(0.0, 1.0).all():
        raise ValueError("Baseline design-space LOLP values must be between 0 and 1.")
    return validated.sort_values(["pv_capacity_mw", "battery_energy_mwh"]).reset_index(drop=True)


def make_baseline_design_space_figures(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    """Create baseline-only Figure 3 source data, 3D surfaces, contour maps, and PV slices."""
    validated = validate_baseline_design_space_source(source)
    base = out / f"{case_key}_{FIGURE_3_SOURCE_NAME}"
    paths: dict[str, Path] = {"source_data": _write_source_data(base, validated)}
    paths.update(plot_baseline_design_space_3d(out, case_key, validated))
    paths.update(plot_baseline_design_space_contours(out, case_key, validated))
    slice_source = build_baseline_pv_capacity_slice_source(validated)
    slice_base = out / f"{case_key}_{FIGURE_3C_NAME}"
    paths["figure_3c_source_data"] = _write_source_data(slice_base, slice_source)
    paths.update(plot_baseline_pv_capacity_bess_slices(out, case_key, slice_source))
    return paths


def build_baseline_pv_capacity_slice_source(source: pd.DataFrame) -> pd.DataFrame:
    """Select low, medium, and high PV slices for Figure 3C from baseline source data."""
    validated = validate_baseline_design_space_source(source)
    slices = select_baseline_pv_capacity_slices(validated)
    selected = []
    for spec in slices:
        rows = validated[np.isclose(validated["pv_capacity_mw"].astype(float), spec.selected_mw)].copy()
        rows["pv_slice_label"] = spec.label
        rows["pv_slice_rank"] = spec.rank
        rows["pv_slice_target_mw"] = spec.target_mw
        rows["pv_slice_selected_mw"] = spec.selected_mw
        rows["pv_slice_source"] = spec.source
        selected.append(rows)
    if not selected:
        raise ValueError("Figure 3C requires at least one PV slice.")
    slice_source = pd.concat(selected, ignore_index=True)
    return validate_baseline_pv_capacity_slice_source(slice_source)


def select_baseline_pv_capacity_slices(source: pd.DataFrame) -> tuple[BaselinePVSliceSpec, ...]:
    """Return fixed 8/20/32 MW PV slices, or adaptive low/mid/high slices for smaller cases."""
    validated = validate_baseline_design_space_source(source)
    bess_counts = validated.groupby("pv_capacity_mw")["battery_energy_mwh"].nunique()
    eligible_pv = bess_counts[bess_counts >= 3].index.to_numpy(dtype=float)
    pv_grid = np.sort(eligible_pv)
    if pv_grid.size < 3:
        available = ", ".join(_format_capacity_label(value) for value in pv_grid)
        raise ValueError(
            "Figure 3C requires at least three PV capacities with at least three BESS points each. "
            f"Available eligible PV capacities: {available or 'none'}."
        )
    selected: list[BaselinePVSliceSpec] = []
    missing: list[float] = []
    for label, rank, target_mw in FIGURE_3C_FIXED_PV_SLICES_MW:
        matches = pv_grid[np.isclose(pv_grid, target_mw, atol=1e-9)]
        if matches.size == 0:
            missing.append(target_mw)
            continue
        selected.append(
            BaselinePVSliceSpec(
                label=label,
                rank=rank,
                target_mw=target_mw,
                selected_mw=float(matches[0]),
                source="fixed_8_20_32_mw",
            )
        )
    if not missing:
        return tuple(selected)
    return _adaptive_low_medium_high_pv_slices(pv_grid)


def validate_baseline_pv_capacity_slice_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SLICE_SOURCE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Baseline Figure 3C source data is missing required column(s): " + ", ".join(missing))
    validated = validate_baseline_design_space_source(source)
    for column in ("pv_slice_label", "pv_slice_source"):
        if validated[column].isna().any() or (validated[column].astype(str).str.strip() == "").any():
            raise ValueError(f"Baseline Figure 3C source data contains empty {column} values.")
        validated[column] = validated[column].astype(str)
    validated["pv_slice_rank"] = pd.to_numeric(validated["pv_slice_rank"], errors="coerce")
    if validated["pv_slice_rank"].isna().any():
        raise ValueError("Baseline Figure 3C source data contains non-numeric pv_slice_rank values.")
    labels = validated[["pv_slice_label", "pv_slice_rank", "pv_capacity_mw"]].drop_duplicates()
    if labels["pv_slice_label"].nunique() != 3:
        raise ValueError("Baseline Figure 3C requires exactly three PV slice labels.")
    if labels["pv_capacity_mw"].nunique() != 3:
        raise ValueError("Baseline Figure 3C PV slice capacities must not repeat.")
    for label in ("Low PV", "Medium PV", "High PV"):
        if label not in set(labels["pv_slice_label"]):
            raise ValueError(f"Baseline Figure 3C is missing the {label} slice.")
    bess_counts = validated.groupby("pv_slice_label")["battery_energy_mwh"].nunique()
    if (bess_counts < 3).any():
        raise ValueError("Baseline Figure 3C requires at least three BESS capacity points per PV slice.")
    return validated.sort_values(["pv_slice_rank", "battery_energy_mwh"]).reset_index(drop=True)


def plot_baseline_design_space_3d(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    source = validate_baseline_design_space_source(source)
    display_source, units = add_display_unit_columns(source)
    metrics = _display_baseline_metrics(units)
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    fig = plt.figure(figsize=(9.2, 3.4), constrained_layout=False)
    fig.subplots_adjust(left=0.045, right=0.98, bottom=0.04, top=0.88, wspace=0.20)
    xlim, ylim = _capacity_limits(display_source)

    for index, spec in enumerate(metrics, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        _plot_metric_surface(ax, display_source, spec, xlim, ylim, units)
        ax.text2D(0.0, 1.02, f"{chr(96 + index)}  {spec.title}", transform=ax.transAxes, fontsize=8, fontweight="bold")

    base = out / f"{case_key}_{FIGURE_3A_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_baseline_design_space_contours(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    source = validate_baseline_design_space_source(source)
    display_source, units = add_display_unit_columns(source)
    metrics = _display_baseline_metrics(units)
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.6), constrained_layout=True)
    xlim, ylim = _capacity_limits(display_source)

    for index, (ax, spec) in enumerate(zip(axes, metrics, strict=True), start=1):
        levels = _contour_levels(display_source[spec.key].to_numpy(dtype=float), spec)
        artist = _plot_metric_contour(ax, display_source, spec, xlim, ylim, levels, units)
        _add_figure_3b_reference_marker(ax, display_source)
        cbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.025)
        _format_contour_colorbar(cbar, levels, spec)
        ax.set_title(f"{chr(96 + index)}  {spec.title}", loc="left")
        ax.set_box_aspect(1.0)

    base = out / f"{case_key}_{FIGURE_3B_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_baseline_pv_capacity_bess_slices(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    """Plot Figure 3C BESS-capacity slices for low, medium, and high PV capacities."""
    source = validate_baseline_pv_capacity_slice_source(source)
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.1), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.72, wspace=0.34)
    handles = _plot_slice_panels(axes, source)
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        handlelength=2.5,
        columnspacing=1.3,
    )

    base = out / f"{case_key}_{FIGURE_3C_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _plot_slice_panels(axes: Any, source: pd.DataFrame) -> list[Any]:
    source, units = add_display_unit_columns(source)
    metrics = _display_baseline_metrics(units)
    colors = {
        "Low PV": NATURE_PALETTE["blue"],
        "Medium PV": NATURE_PALETTE["teal"],
        "High PV": NATURE_PALETTE["red"],
    }
    handles: list[Any] = []
    grouped = list(source.groupby(["pv_slice_rank", "pv_slice_label", "pv_capacity_kw"], sort=True))

    for index, (ax, spec) in enumerate(zip(axes, metrics, strict=True), start=1):
        metric_values = source[spec.key].to_numpy(dtype=float)
        for (_, label, pv_kw), group in grouped:
            ordered = group.sort_values(DISPLAY_BATTERY_ENERGY)
            line_label = f"{label} ({format_pv_capacity_from_kw(float(pv_kw), units)})"
            line = plot_smooth_line(
                ax,
                ordered[DISPLAY_BATTERY_ENERGY].to_numpy(dtype=float),
                ordered[spec.key].to_numpy(dtype=float),
                color=colors[str(label)],
                linewidth=1.25,
                label=line_label,
            )
            if index == 1:
                handles.append(line)
        ax.set_title(f"{chr(96 + index)}  {spec.title}", loc="left")
        ax.set_xlabel(units.battery_axis_label)
        ax.set_ylabel(spec.z_label)
        ax.grid(True, color="#E4E4E4", linewidth=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.0)
        ax.set_xlim(left=0.0, right=float(source[DISPLAY_BATTERY_ENERGY].max()))
        _apply_metric_axis_format(ax, spec, metric_values)
    return handles


def _apply_metric_axis_format(ax: Any, spec: BaselineMetricSpec, values: np.ndarray) -> None:
    if spec.unit_type == "lolp":
        ax.yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")
        ticks = nice_display_ticks(values, unit_type="lolp")
        if len(ticks) >= 2:
            ax.set_yticks(ticks)
        ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0]))
    else:
        ticks = nice_display_ticks(values, unit_type=spec.unit_type)
        if len(ticks) >= 2:
            ax.set_yticks(ticks)


def _format_capacity_label(value_mw: float) -> str:
    if np.isclose(value_mw, round(value_mw), atol=1e-9):
        return f"{int(round(value_mw))} MW"
    return f"{value_mw:.1f} MW"


def _capacity_limits(source: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x_column = DISPLAY_PV_CAPACITY if DISPLAY_PV_CAPACITY in source.columns else "pv_capacity_mw"
    y_column = DISPLAY_BATTERY_ENERGY if DISPLAY_BATTERY_ENERGY in source.columns else "battery_energy_mwh"
    x = source[x_column].astype(float)
    y = source[y_column].astype(float)
    return (float(x.min()), float(x.max())), (float(y.min()), float(y.max()))


def _rooftop_pv_reference_kw(config: dict[str, Any]) -> tuple[float, float] | None:
    pv_config = config.get("pv", {}) if isinstance(config, dict) else {}
    try:
        rooftop_area = float(pv_config["rooftop_area_m2"])
        power_density = float(pv_config["pv_module_power_density_kw_per_m2"])
    except (KeyError, TypeError, ValueError):
        return None
    if rooftop_area <= 0.0 or power_density <= 0.0:
        return None
    rooftop_limit = rooftop_area * power_density
    return rooftop_limit, rooftop_limit * 2.0


def _rooftop_reference_from_source(source: pd.DataFrame) -> tuple[float, float] | None:
    if "rooftop_pv_limit_mw" not in source.columns or "twice_rooftop_pv_limit_mw" not in source.columns:
        return None
    rooftop = pd.to_numeric(source["rooftop_pv_limit_mw"], errors="coerce").dropna()
    twice_rooftop = pd.to_numeric(source["twice_rooftop_pv_limit_mw"], errors="coerce").dropna()
    if rooftop.empty or twice_rooftop.empty:
        return None
    rooftop_value = float(rooftop.iloc[0])
    twice_rooftop_value = float(twice_rooftop.iloc[0])
    if rooftop_value <= 0.0 or twice_rooftop_value <= 0.0:
        return None
    return rooftop_value, twice_rooftop_value


def _snap_pv_targets(
    pv_grid: np.ndarray,
    targets: tuple[tuple[str, int, float], ...],
    *,
    source: str,
) -> tuple[BaselinePVSliceSpec, ...]:
    selected: list[BaselinePVSliceSpec] = []
    used: set[int] = set()
    for label, rank, target in targets:
        candidate_index = _nearest_unused_index(pv_grid, float(target), used)
        used.add(candidate_index)
        selected.append(
            BaselinePVSliceSpec(
                label=label,
                rank=rank,
                target_mw=float(target),
                selected_mw=float(pv_grid[candidate_index]),
                source=source,
            )
        )
    return tuple(selected)


def _adaptive_low_medium_high_pv_slices(pv_grid: np.ndarray) -> tuple[BaselinePVSliceSpec, ...]:
    if pv_grid.size < 3:
        raise ValueError("Figure 3C requires at least three eligible PV capacities.")
    indices = (0, int(pv_grid.size // 2), int(pv_grid.size - 1))
    selected: list[BaselinePVSliceSpec] = []
    for (label, rank, _fixed_target), index in zip(FIGURE_3C_FIXED_PV_SLICES_MW, indices, strict=True):
        value = float(pv_grid[int(index)])
        selected.append(
            BaselinePVSliceSpec(
                label=label,
                rank=rank,
                target_mw=value,
                selected_mw=value,
                source="adaptive_low_medium_high_mw",
            )
        )
    return tuple(selected)


def _nearest_unused_index(values: np.ndarray, target: float, used: set[int]) -> int:
    ordered = np.argsort(np.abs(values - target), kind="mergesort")
    for index in ordered:
        candidate = int(index)
        if candidate not in used:
            return candidate
    raise ValueError("Unable to select a unique PV capacity for Figure 3C.")


def _nearest_target_count(pv_grid: np.ndarray, targets: tuple[tuple[str, int, float], ...]) -> int:
    nearest = []
    for _label, _rank, target in targets:
        index = int(np.argmin(np.abs(pv_grid - float(target))))
        nearest.append(round(float(pv_grid[index]), 9))
    return len(set(nearest))


def _grid_values(source: pd.DataFrame, value_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    x_column = DISPLAY_PV_CAPACITY if DISPLAY_PV_CAPACITY in source.columns else "pv_capacity_mw"
    y_column = DISPLAY_BATTERY_ENERGY if DISPLAY_BATTERY_ENERGY in source.columns else "battery_energy_mwh"
    pivot = source.pivot_table(
        index=y_column,
        columns=x_column,
        values=value_column,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    if pivot.shape[0] < 2 or pivot.shape[1] < 2 or pivot.isna().any().any():
        return None
    x, y = np.meshgrid(pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float))
    z = pivot.to_numpy(dtype=float)
    return x, y, z


def _plot_metric_surface(
    ax: Any,
    source: pd.DataFrame,
    spec: BaselineMetricSpec,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    units: FigureDisplayUnits,
) -> Any:
    grid = _grid_values(source, spec.key)
    if grid is not None:
        x, y, z = grid
        artist = ax.plot_surface(x, y, z, cmap=spec.cmap, linewidth=0, antialiased=True, alpha=0.94)
        ax.contour(x, y, z, zdir="z", offset=float(np.nanmin(z)), cmap=spec.cmap, alpha=0.25, linewidths=0.35)
    else:
        artist = ax.plot_trisurf(
            source[DISPLAY_PV_CAPACITY].to_numpy(dtype=float),
            source[DISPLAY_BATTERY_ENERGY].to_numpy(dtype=float),
            source[spec.key].to_numpy(dtype=float),
            cmap=spec.cmap,
            linewidth=0,
            antialiased=True,
            alpha=0.94,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(units.pv_axis_label, labelpad=1)
    ax.set_ylabel(units.battery_axis_label, labelpad=1)
    ax.set_zlabel(spec.z_label, labelpad=1)
    ax.view_init(elev=25, azim=-135)
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
    return artist


def _plot_metric_contour(
    ax: Any,
    source: pd.DataFrame,
    spec: BaselineMetricSpec,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    levels: np.ndarray,
    units: FigureDisplayUnits,
) -> Any:
    grid = _grid_values(source, spec.key)
    if grid is not None:
        x, y, z = grid
        artist = ax.contourf(x, y, z, levels=levels, cmap=spec.cmap, alpha=0.88)
        ax.contour(x, y, z, levels=_sparse_levels_for_contour(z, levels, spec), colors="white", linewidths=0.35, alpha=0.65)
    else:
        artist = ax.tricontourf(
            source[DISPLAY_PV_CAPACITY].to_numpy(dtype=float),
            source[DISPLAY_BATTERY_ENERGY].to_numpy(dtype=float),
            source[spec.key].to_numpy(dtype=float),
            levels=levels,
            cmap=spec.cmap,
            alpha=0.88,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(units.pv_axis_label)
    ax.set_ylabel(units.battery_axis_label)
    return artist


def _add_figure_3b_reference_marker(ax: Any, source: pd.DataFrame) -> None:
    x_value, y_value = _figure_3b_reference_marker_coordinates(source)
    ax.scatter(
        [x_value],
        [y_value],
        marker="*",
        s=95,
        facecolor=NATURE_PALETTE["neutral_black"],
        edgecolor="white",
        linewidth=0.45,
        zorder=8,
    )


def _figure_3b_reference_marker_coordinates(source: pd.DataFrame) -> tuple[float, float]:
    x_limits, y_limits = _capacity_limits(source)
    fixed_x = FIGURE_3B_REFERENCE_PV_MW
    fixed_y = FIGURE_3B_REFERENCE_BESS_MWH
    if DISPLAY_PV_CAPACITY in source.columns and source[DISPLAY_PV_CAPACITY].max() > 1.0:
        if np.isclose(float(source["pv_capacity_mw"].max()), float(source[DISPLAY_PV_CAPACITY].max()), atol=1e-9):
            fixed_x = FIGURE_3B_REFERENCE_PV_MW
        else:
            fixed_x = FIGURE_3B_REFERENCE_PV_MW * 1000.0
    if DISPLAY_BATTERY_ENERGY in source.columns and source[DISPLAY_BATTERY_ENERGY].max() > 1.0:
        if np.isclose(float(source["battery_energy_mwh"].max()), float(source[DISPLAY_BATTERY_ENERGY].max()), atol=1e-9):
            fixed_y = FIGURE_3B_REFERENCE_BESS_MWH
        else:
            fixed_y = FIGURE_3B_REFERENCE_BESS_MWH * 1000.0
    if x_limits[0] <= fixed_x <= x_limits[1] and y_limits[0] <= fixed_y <= y_limits[1]:
        return float(fixed_x), float(fixed_y)
    best = source.sort_values(DISPLAY_SYSTEM_COST if DISPLAY_SYSTEM_COST in source.columns else "system_total_cost_million_sgd_per_year").iloc[0]
    return float(best[DISPLAY_PV_CAPACITY]), float(best[DISPLAY_BATTERY_ENERGY])


def _format_contour_colorbar(colorbar: Any, levels: np.ndarray, spec: BaselineMetricSpec) -> None:
    ticks = _contour_colorbar_ticks(levels, spec)
    colorbar.set_ticks(ticks)
    colorbar.set_label(spec.colorbar_label, fontsize=6.2)
    if spec.unit_type == "lolp":
        colorbar.set_ticklabels([f"{tick * 100:.0f}%" for tick in ticks])
    else:
        colorbar.set_ticklabels([_format_tick(tick) for tick in ticks])
    colorbar.ax.tick_params(labelsize=6.0, length=2.2, width=0.6)


def _levels(values: np.ndarray, count: int = 12) -> np.ndarray:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot plot a metric with no finite values.")
    low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if np.isclose(low, high):
        return np.linspace(low - 0.5, high + 0.5, count)
    return np.linspace(low, high, count)


def _contour_levels(values: np.ndarray, spec: BaselineMetricSpec) -> np.ndarray:
    if spec.unit_type in {"cost", "ue"}:
        return nice_levels(values, target_intervals=FINE_CONTOUR_TARGET_INTERVALS)
    if spec.unit_type == "lolp":
        return np.linspace(0.0, 1.0, 21)
    return _levels(values, count=CONTOUR_INTERVALS + 1)


def _fine_contour_levels(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot plot a metric with no finite values.")
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if np.isclose(low, high):
        anchor = round(low)
        return np.array([anchor - 2.5, anchor, anchor + 2.5], dtype=float)
    step = 2.5 if (high - low) <= 50.0 else 5.0
    lower = np.floor(low / step) * step
    upper = np.ceil(high / step) * step
    if np.isclose(lower, upper):
        upper = lower + step
    return np.arange(lower, upper + step * 0.5, step, dtype=float)


def _contour_colorbar_ticks(levels: np.ndarray, spec: BaselineMetricSpec) -> np.ndarray:
    if spec.unit_type in {"cost", "ue"}:
        return sparse_ticks_from_levels(levels, target_count=6)
    return np.linspace(float(levels[0]), float(levels[-1]), COLORBAR_LABEL_DIVISIONS + 1)


def _sparse_levels_for_contour(values: np.ndarray, levels: np.ndarray, spec: BaselineMetricSpec) -> np.ndarray:
    if spec.unit_type in {"cost", "ue"}:
        return levels[1:-1]
    return _sparse_levels(values)


def _sparse_levels(values: np.ndarray) -> np.ndarray:
    return _levels(values, count=6)[1:-1]


def _format_tick(value: float) -> str:
    if np.isclose(value, round(value), atol=1e-9):
        return f"{int(round(value))}"
    return f"{value:.2f}".rstrip("0").rstrip(".")
