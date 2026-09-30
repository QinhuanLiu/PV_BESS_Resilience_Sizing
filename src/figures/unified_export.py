"""Re-render frozen Brisbane result tables with one paper and slide visual grammar.

This entrypoint reads only completed figure source tables. It never invokes a
simulation, and it refuses to write outside its dedicated Unified_Figures_v1 root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pv_bess_unified_mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .core import NATURE_PALETTE


ROOT = Path(__file__).resolve().parents[2]
BRISBANE = ROOT / "outputs" / "Brisbane"
DEFAULT_OUTPUT_ROOT = BRISBANE / "Unified_Figures_v1"
FORMATS = ("png", "svg", "pdf")
CATEGORY_COUNTS = {
    "01_capacity_sweep": 17,
    "02_target_sizing": 9,
    "03_fixed_design": 18,
    "04_outage_process": 2,
    "05_sensitivity": 48,
}
METRIC_COLORS = {
    "annual_cost": NATURE_PALETTE["blue"],
    "pv": NATURE_PALETTE["teal"],
    "bess": NATURE_PALETTE["purple"],
    "eue": NATURE_PALETTE["red"],
    "lolp": "#2878A8",
    "ssr": "#2D8D58",
    "scr": NATURE_PALETTE["gold"],
}
THRESHOLD_COLORS = {0.1: NATURE_PALETTE["blue"], 0.2: NATURE_PALETTE["purple"], 0.3: NATURE_PALETTE["gold"]}
LEVEL_COLORS = (NATURE_PALETTE["blue"], NATURE_PALETTE["teal"], NATURE_PALETTE["purple"])
DESIGN_COLORS = {
    (3.25, 8.5): NATURE_PALETTE["teal"],
    (3.25, 11.5): NATURE_PALETTE["blue"],
    (3.25, 14.5): NATURE_PALETTE["purple"],
    (5.0, 11.5): NATURE_PALETTE["gold"],
    (6.5, 11.5): NATURE_PALETTE["red"],
    (6.5, 14.5): NATURE_PALETTE["neutral_dark"],
}
METRICS = {
    "Annual_System_Cost": ("annual_system_cost_ksgd_per_year", "Annual system cost", "kSGD yr$^{-1}$", "annual_cost", "Blues"),
    "Optimal_PV_Capacity": ("pv_capacity_kw", "Optimal PV capacity", "kW", "pv", "PVTeal"),
    "Optimal_BESS_Energy_Capacity": ("battery_energy_kwh", "Optimal BESS energy", "kWh", "bess", "Purples"),
    "EUE": ("EUE_kwh_per_year", "EUE", "kWh yr$^{-1}$", "eue", "Reds"),
    "LOLP": ("LOLP", "LOLP", "%", "lolp", "YlGnBu"),
    "SSR": ("SSR", "SSR", "%", "ssr", "Greens"),
    "SCR": ("SCR", "SCR", "%", "scr", "SCRGold"),
}
SENSITIVITY_X_LABELS = {
    "PV_CAPEX": "PV CAPEX (SGD kW$^{-1}$)",
    "Battery_CAPEX": "Battery energy CAPEX (SGD kWh$^{-1}$)",
    "SAIDI": "SAIDI (min yr$^{-1}$)",
    "VoLL": "VoLL (SGD kWh$^{-1}$)",
}
PROFILES = {
    "paper": {"width": 180 / 25.4, "dpi": 600, "font": 8.0, "axis": 9.0, "tick": 8.0, "legend": 8.0, "title": 10.5, "subtitle": 8.0, "line": 1.4, "marker": 3.5},
    "presentation": {"width": 40 / 3, "dpi": 300, "font": 15.0, "axis": 16.0, "tick": 14.0, "legend": 14.0, "title": 23.0, "subtitle": 14.0, "line": 2.3, "marker": 5.5},
}


@dataclass(frozen=True)
class FigureEntry:
    original: Path
    sources: tuple[Path, ...]
    category: str
    family: str

    @property
    def run_name(self) -> str:
        return self.original.relative_to(BRISBANE).parts[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sibling(path: Path, stem: str) -> Path:
    return path.parent / f"{stem}_Source_Data.csv"


def identify(path: Path) -> FigureEntry:
    name = path.stem
    run = path.relative_to(BRISBANE).parts[0]
    if "Typical_Outage" in name:
        category, family = "04_outage_process", "outage_72h" if "72h" in name else "outage_short"
        sources = (_sibling(path, name), path.with_name(f"{name}_Event_Summary.json"))
    elif "CCP_LOLP_Cost_Sensitivity_Figures" in run:
        category, family = "05_sensitivity", "sensitivity_overview"
        sources = (_sibling(path, name),)
    elif "Sensitivity_Split_Line_Figures" in run:
        category, family = "05_sensitivity", "sensitivity_single"
        parameter = path.parent.name
        source_stem = f"Brisbane_SingleFamily_{parameter}_Split_Sensitivity_Line"
        sources = (_sibling(path, source_stem),)
    elif "Sensitivity_Line_Figures" in run:
        category, family = "05_sensitivity", "sensitivity_overview"
        sources = (_sibling(path, name),)
    elif "Sensitivity_Figures" in run:
        category, family = "05_sensitivity", "sensitivity_contour"
        group = path.parent.name
        sources = (_sibling(path, f"Brisbane_SingleFamily_{group}_Contour"),)
    elif "FixedDesign" in run:
        category = "03_fixed_design"
        if "CDF_Comparison" in name:
            family = "cdf_comparison"
            sources = (_sibling(path, name),)
        elif "Histogram_CDF" in name:
            family = "histogram_cdf"
            sources = (_sibling(path, name + "_Bins"), _sibling(path, name + "_CDF"))
        elif name.endswith("Histogram"):
            family = "histogram"
            sources = (_sibling(path, name + "_Bins"),)
        elif name.endswith("CDF"):
            family = "cdf"
            sources = (_sibling(path, name),)
        else:
            raise ValueError(f"Unknown fixed-design chart: {path}")
    elif "CCP_LOLP_Sizing" in run:
        category = "02_target_sizing"
        if "Incremental_Cost_Reliability_Frontier" in name:
            family = "frontier_fit" if "Power_Law_Fit" in name else "frontier"
            sources = (_sibling(path, "Brisbane_SingleFamily_Incremental_Cost_Reliability_Frontier"),)
            if family == "frontier_fit":
                sources += (path.with_name(f"{name}_Summary.csv"),)
        elif "CCP2pct_Scatter" in name:
            family = "target_scatter"
            sources = (_sibling(path, "Brisbane_SingleFamily_CCP2pct_Scatter"),)
            if "Optimal_BESS" in name or "Mean_Annual_Cost" in name:
                sources += (path.with_name("Brisbane_SingleFamily_CCP2pct_Quadratic_Fit_Summary.csv"),)
        elif "CCP_Target_PV_BESS_Annual_Cost" in name:
            family = "target_triptych"
            sources = (_sibling(path, name),)
        else:
            family = "target_line"
            sources = (_sibling(path, "Brisbane_SingleFamily_CCP_Target_Line_Plots"),)
    elif "Capacity_Sweep" in run:
        category = "01_capacity_sweep"
        if "3D_Surface" in name:
            family = "surface"
            sources = (_sibling(path, name),)
        elif "Response" in name:
            family = "response"
            sources = (_sibling(path, name),)
        elif "CCP_Target_PV_BESS_Annual_Cost" in name:
            family = "target_triptych"
            sources = (_sibling(path, name),)
        else:
            family = "capacity_contour"
            if "Q90_LOLP" in name or "LOLP_at_90pct_CCP" in name:
                sources = (_sibling(path, name),)
            else:
                sources = (_sibling(path, "Brisbane_SingleFamily_Mean_Resilience_Contours"),)
    else:
        raise ValueError(f"Unclassified result figure: {path}")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Missing frozen figure source: {source}")
    return FigureEntry(path, sources, category, family)


def inventory(output_root: Path = DEFAULT_OUTPUT_ROOT) -> list[FigureEntry]:
    originals = sorted(path for path in BRISBANE.rglob("*.png") if output_root not in path.parents)
    entries = [identify(path) for path in originals]
    counts = {key: sum(entry.category == key for entry in entries) for key in CATEGORY_COUNTS}
    if counts != CATEGORY_COUNTS:
        raise ValueError(f"Expected 94 original figures in five groups; found {counts}")
    return entries


def _read(entry: FigureEntry, index: int = 0) -> pd.DataFrame:
    return pd.read_csv(entry.sources[index], float_precision="round_trip")


def _style(profile: str) -> dict[str, float]:
    cfg = PROFILES[profile]
    matplotlib.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": cfg["font"],
        "axes.labelsize": cfg["axis"], "xtick.labelsize": cfg["tick"], "ytick.labelsize": cfg["tick"],
        "legend.fontsize": cfg["legend"], "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.85,
        "lines.linewidth": cfg["line"], "lines.markersize": cfg["marker"],
    })
    return cfg


def _canvas(profile: str, family: str, panels: int = 1) -> tuple[Any, Any]:
    cfg = _style(profile)
    if profile == "presentation":
        height = 7.5
    elif panels == 7:
        height = 8.15
    elif panels == 3:
        height = 7.0
    elif family == "outage_short":
        height = 5.2
    elif family in {"capacity_contour", "sensitivity_contour", "surface"}:
        height = 5.3
    else:
        height = 4.55
    fig = plt.figure(figsize=(cfg["width"], height), facecolor="white")
    return fig, cfg


def _header(fig: Any, profile: str, title: str, subtitle: str = "") -> None:
    cfg = PROFILES[profile]
    fig.text(0.055, 0.975, title, ha="left", va="top", fontsize=cfg["title"], fontweight="bold", color=NATURE_PALETTE["neutral_black"])
    if subtitle:
        fig.text(0.055, 0.918, subtitle, ha="left", va="top", fontsize=cfg["subtitle"], color=NATURE_PALETTE["neutral_dark"])


def _axis(ax: Any, profile: str, x_label: str, y_label: str) -> None:
    cfg = PROFILES[profile]
    ax.set_xlabel(x_label, fontsize=cfg["axis"])
    ax.set_ylabel(y_label, fontsize=cfg["axis"])
    ax.tick_params(direction="out", length=3.4 if profile == "paper" else 5, width=0.8, labelsize=cfg["tick"])
    ax.grid(False)


def _x_limits(values: Any, padding_fraction: float = 0.025) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    low, high = float(np.nanmin(data)), float(np.nanmax(data))
    padding = (high - low) * padding_fraction if high > low else max(abs(low) * padding_fraction, 1.0)
    return low - padding, high + padding


def _trace(entry: FigureEntry, x: str, y: str, *, z: str | None = None, rows: int | None = None) -> dict[str, Any]:
    return {"x": x, "y": y, "z": z, "source_rows": rows, "source_sha256": {str(p.relative_to(BRISBANE)): sha256(p) for p in entry.sources}}


def _threshold_label(value: float) -> str:
    return f"LOLP < {value:.0%}"


def _line_data(ax: Any, x: np.ndarray, y: np.ndarray, *, color: str, profile: str, label: str | None = None, smooth: bool = False, marker: bool = True, fit: bool = False) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if not len(x):
        return
    order = np.argsort(x)
    x, y = x[order], y[order]
    if smooth and len(x) >= 3 and np.all(np.diff(x) > 0):
        dense = np.linspace(float(x.min()), float(x.max()), 200)
        ax.plot(dense, PchipInterpolator(x, y)(dense), color=color, label=label, linestyle="--" if fit else "-")
        if marker:
            ax.plot(x, y, linestyle="none", marker="o", color=color, markersize=PROFILES[profile]["marker"])
    else:
        ax.plot(x, y, color=color, label=label, linestyle="--" if fit else "-", marker="o" if marker else None, markersize=PROFILES[profile]["marker"])


def _legend(fig: Any, ax: Any, profile: str, *, ncol: int = 3, y: float = 0.875) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=min(ncol, len(handles)), frameon=False, fontsize=PROFILES[profile]["legend"])


def _metric_from_name(stem: str) -> str:
    for key in sorted(METRICS, key=len, reverse=True):
        if f"_{key}_" in stem or stem.endswith(f"_{key}"):
            return key
    raise ValueError(f"Unrecognized metric in {stem}")


def _contour(entry: FigureEntry, profile: str) -> tuple[Any, dict[str, Any]]:
    source = _read(entry)
    stem = entry.original.stem
    is_sensitivity = entry.family == "sensitivity_contour"
    if is_sensitivity:
        x_col, y_col = "column_value", "row_value"
        metric = _metric_from_name(stem)
        z_col, title, unit, _, cmap = METRICS[metric]
        group = entry.original.parent.name
        if group == "Cost_CAPEX":
            x_label, y_label = SENSITIVITY_X_LABELS["Battery_CAPEX"], SENSITIVITY_X_LABELS["PV_CAPEX"]
        else:
            x_label, y_label = SENSITIVITY_X_LABELS["VoLL"], SENSITIVITY_X_LABELS["SAIDI"]
    else:
        x_col, y_col = "battery_energy_kwh", "pv_capacity_kw"
        x_label, y_label = "BESS energy (kWh)", "PV capacity (kW)"
        if "Mean_EUE" in stem:
            z_col, title, unit, cmap = "EUE_kwh_per_year_mean", "Mean EUE", "kWh yr$^{-1}$", "Reds"
        elif "Mean_LOLP" in stem:
            z_col, title, unit, cmap = "LOLP_mean", "Mean LOLP", "%", "YlGnBu"
        elif "Q90_LOLP" in stem:
            z_col, title, unit, cmap = "Q90_LOLP", "Q90 LOLP", "%", "YlGnBu"
        elif "LOLP_at_90pct_CCP" in stem:
            z_col, title, unit, cmap = "LOLP_at_CCP90", "LOLP at 90% CCP", "%", "YlGnBu"
        elif "CCP_LOLP_LT20" in stem:
            z_col, title, unit, cmap = "CCP_LOLP_lt20", "CCP for LOLP < 20%", "%", "Purples"
        elif "Mean_Annual_Cost" in stem:
            z_col, title, unit, cmap = "annual_cost_sgd_per_year_mean", "Mean annual cost", "kSGD yr$^{-1}$", "Blues"
        elif "Mean_SSR" in stem:
            z_col, title, unit, cmap = "SSR_mean", "Mean SSR", "%", "Greens"
        elif "Mean_SCR" in stem:
            z_col, title, unit, cmap = "SCR_display_mean", "Mean SCR", "%", "Oranges"
        else:
            raise ValueError(f"Unknown capacity contour: {stem}")
    if z_col not in source.columns:
        raise ValueError(f"Missing {z_col} in {entry.sources[0]}")
    if title == "Optimal PV capacity":
        cmap = colors.LinearSegmentedColormap.from_list("PVTeal", ["#FFFFFF", "#B6D8DB", NATURE_PALETTE["teal"], "#1F6269"])
    elif title in {"SCR", "Mean SCR"}:
        cmap = colors.LinearSegmentedColormap.from_list("SCRGold", ["#FFFFFF", "#F5E5AD", NATURE_PALETTE["gold"], "#8B650E"])
    data = source[[x_col, y_col, z_col]].copy()
    data[z_col] = pd.to_numeric(data[z_col], errors="coerce")
    if z_col == "annual_cost_sgd_per_year_mean":
        data[z_col] /= 1000.0
    elif unit == "%":
        data[z_col] *= 100.0
    grid = data.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc="mean").sort_index().sort_index(axis=1)
    xx, yy = np.meshgrid(grid.columns.to_numpy(dtype=float), grid.index.to_numpy(dtype=float))
    zz = np.ma.masked_invalid(grid.to_numpy(dtype=float))
    finite = zz.compressed()
    if not len(finite):
        raise ValueError(f"No finite contour values: {entry.original}")
    if title.startswith("CCP") or (unit == "%" and not is_sensitivity and title in {"Mean SSR", "Mean SCR"}):
        low, high = 0.0, 100.0
    elif title in {"Mean LOLP", "Q90 LOLP", "LOLP at 90% CCP"} and not is_sensitivity:
        low, high = 0.0, 100.0
    else:
        low, high = float(finite.min()), float(finite.max())
        if math.isclose(low, high):
            pad = max(abs(low) * 0.05, 0.01)
            low, high = low - pad, high + pad
    levels = np.linspace(low, high, 13)
    fig, cfg = _canvas(profile, entry.family)
    _header(fig, profile, f"Brisbane · {title}", "15-year results · frozen source data")
    ax = fig.add_axes([0.14, 0.15, 0.63, 0.68])
    artist = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, extend="both" if finite.min() < low or finite.max() > high else "neither")
    if len(finite) > 4:
        try:
            ax.contour(xx, yy, zz, levels=levels[2:-1:3], colors="white", linewidths=0.45, alpha=0.55)
        except ValueError:
            pass
    _axis(ax, profile, x_label, y_label)
    ax.set_xlim(float(np.nanmin(xx)), float(np.nanmax(xx)))
    ax.set_ylim(float(np.nanmin(yy)), float(np.nanmax(yy)))
    cax = fig.add_axes([0.84, 0.2, 0.035, 0.57])
    bar = fig.colorbar(artist, cax=cax)
    bar.set_label(f"{title} ({unit})", fontsize=cfg["axis"])
    bar.ax.tick_params(labelsize=cfg["tick"])
    if unit == "%" and low == 0.0 and high == 100.0:
        bar.set_ticks(np.arange(0, 101, 20))
        bar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    return fig, _trace(entry, x_col, y_col, z=z_col, rows=len(source))


def _surface(entry: FigureEntry, profile: str) -> tuple[Any, dict[str, Any]]:
    source = _read(entry)
    stem = entry.original.stem
    pair = (("CCP_LOLP_lt10", "LOLP < 10%", THRESHOLD_COLORS[0.1]), ("CCP_LOLP_lt20", "LOLP < 20%", THRESHOLD_COLORS[0.2])) if "LT10_LT20" in stem else (("CCP_LOLP_lt20", "LOLP < 20%", THRESHOLD_COLORS[0.2]), ("CCP_LOLP_lt30", "LOLP < 30%", NATURE_PALETTE["gold"]))
    fig, cfg = _canvas(profile, "surface")
    _header(fig, profile, "Brisbane · CCP capacity surfaces", "Original capacity grid · no interpolation")
    ax = fig.add_axes([0.1, 0.09, 0.78, 0.73], projection="3d")
    for key, label, color in pair:
        grid = source.pivot_table(index="pv_capacity_kw", columns="battery_energy_kwh", values=key).sort_index().sort_index(axis=1)
        xx, yy = np.meshgrid(grid.columns.to_numpy(dtype=float), grid.index.to_numpy(dtype=float))
        ax.plot_surface(xx, yy, grid.to_numpy(dtype=float) * 100.0, color=color, alpha=0.58, linewidth=0.2, edgecolor=(1, 1, 1, 0.45), shade=False, label=label)
    ax.set_xlim(float(source.battery_energy_kwh.min()), float(source.battery_energy_kwh.max()))
    ax.set_ylim(float(source.pv_capacity_kw.min()), float(source.pv_capacity_kw.max()))
    ax.set_xticks(np.linspace(float(source.battery_energy_kwh.min()), float(source.battery_energy_kwh.max()), 5))
    ax.set_yticks(np.linspace(float(source.pv_capacity_kw.min()), float(source.pv_capacity_kw.max()), 6))
    ax.set_xlabel("BESS energy (kWh)", labelpad=10)
    ax.set_ylabel("PV capacity (kW)", labelpad=10)
    ax.set_zlabel("CCP (%)", labelpad=8)
    ax.set_zlim(0, 100)
    ax.view_init(elev=28, azim=-135)
    ax.tick_params(labelsize=cfg["tick"] * 0.85)
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=color, alpha=0.65, label=label) for _, label, color in pair], [label for _, label, _ in pair], loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=2, frameon=False, fontsize=cfg["legend"])
    return fig, _trace(entry, "battery_energy_kwh", "pv_capacity_kw", z=" / ".join(key for key, _, _ in pair), rows=len(source))


def _response(entry: FigureEntry, profile: str) -> tuple[Any, dict[str, Any]]:
    source = _read(entry)
    is_battery_x = "Battery_Response" in entry.original.stem
    x_col = "battery_energy_kwh" if is_battery_x else "pv_capacity_kw"
    group_col = "pv_capacity_kw" if is_battery_x else "battery_energy_kwh"
    x_label = "BESS energy (kWh)" if is_battery_x else "PV capacity (kW)"
    title = "CCP response to BESS energy" if is_battery_x else "CCP response to PV capacity"
    fig, cfg = _canvas(profile, "response")
    _header(fig, profile, f"Brisbane · {title}", "CCP = P(LOLP < 20%) · points show evaluated capacities")
    ax = fig.add_axes([0.13, 0.18, 0.78, 0.61])
    values = sorted(source[group_col].dropna().unique())
    for index, group_value in enumerate(values):
        frame = source.loc[np.isclose(source[group_col], group_value)].sort_values(x_col)
        x = frame[x_col].to_numpy(dtype=float)
        y = frame["CCP_LOLP_lt20"].to_numpy(dtype=float) * 100.0
        label = f"{'PV' if is_battery_x else 'BESS'} {group_value:g} {'kW' if is_battery_x else 'kWh'}"
        _line_data(ax, x, y, color=LEVEL_COLORS[index], profile=profile, label=label, smooth=True)
    _axis(ax, profile, x_label, "CCP (%)")
    ax.set_xlim(*_x_limits(source[x_col]))
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    _legend(fig, ax, profile)
    if "PV3_PV6_PV9" in entry.original.stem:
        frame = source.loc[np.isclose(source.pv_capacity_kw, 3.0) & np.isclose(source.battery_energy_kwh, 15.0)]
        if len(frame) == 1:
            ax.scatter([15], [float(frame.CCP_LOLP_lt20.iloc[0]) * 100], marker="*", color=NATURE_PALETTE["neutral_black"], s=cfg["marker"] ** 2 * 2, zorder=5)
    return fig, _trace(entry, x_col, "CCP_LOLP_lt20", rows=len(source))


def _target_columns(stem: str) -> tuple[str, str, str]:
    if "Optimal_PV" in stem:
        return "optimal_pv_kw", "Optimal PV capacity", "kW"
    if "Optimal_BESS" in stem:
        return "optimal_battery_kwh", "Optimal BESS energy", "kWh"
    return "mean_annual_cost_ksgd_per_year", "Mean annual system cost", "kSGD yr$^{-1}$"


def _target_panel(ax: Any, source: pd.DataFrame, profile: str, column: str, *, smooth: bool, fit_source: pd.DataFrame | None = None, scatter_only: bool = False) -> None:
    for threshold in sorted(source.lolp_constraint.unique()):
        frame = source.loc[np.isclose(source.lolp_constraint, threshold)].sort_values("ccp_target_percent")
        x = frame.ccp_target_percent.to_numpy(dtype=float)
        y = frame[column].to_numpy(dtype=float)
        color = THRESHOLD_COLORS[round(float(threshold), 1)]
        if fit_source is None:
            if scatter_only:
                ax.scatter(x, y, color=color, s=PROFILES[profile]["marker"] ** 2, label=_threshold_label(float(threshold)), zorder=3)
            else:
                _line_data(ax, x, y, color=color, profile=profile, label=_threshold_label(float(threshold)), smooth=smooth)
        else:
            ax.scatter(x, y, color=color, s=PROFILES[profile]["marker"] ** 2, label=_threshold_label(float(threshold)), zorder=3)
            key = "bess" if column == "optimal_battery_kwh" else "annual_cost"
            fit = fit_source.loc[(fit_source.metric == key) & np.isclose(fit_source.lolp_constraint, threshold)]
            if len(fit) == 1:
                row = fit.iloc[0]
                dense = np.linspace(float(x.min()), float(x.max()), 200)
                ax.plot(dense, row.a * dense ** 2 + row.b * dense + row.c, color=color, linestyle="--")


def _target_line(entry: FigureEntry, profile: str) -> tuple[Any, dict[str, Any]]:
    source = _read(entry)
    column, title, unit = _target_columns(entry.original.stem)
    fig, _ = _canvas(profile, entry.family)
    _header(fig, profile, f"Brisbane · {title}", "Fixed LOLP thresholds · evaluated CCP targets")
    ax = fig.add_axes([0.13, 0.18, 0.78, 0.61])
    fit = _read(entry, 1) if entry.family == "target_scatter" and len(entry.sources) > 1 else None
    _target_panel(ax, source, profile, column, smooth=entry.family == "target_line", fit_source=fit, scatter_only=entry.family == "target_scatter")
    _axis(ax, profile, "CCP target (%)", f"{title} ({unit})")
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_xlim(*_x_limits(source.ccp_target_percent))
    ax.set_xticks(np.arange(50, 96, 10))
    _legend(fig, ax, profile)
    return fig, _trace(entry, "ccp_target_percent", column, rows=len(source))


def _target_triptych(entry: FigureEntry, profile: str) -> tuple[Any, dict[str, Any]]:
    source = _read(entry)
    fig, cfg = _canvas(profile, entry.family, panels=3)
    _header(fig, profile, "Brisbane · CCP-target PV–BESS sizing", "Fixed LOLP thresholds · original evaluated designs")
    if profile == "paper":
        axes = fig.subplots(3, 1, sharex=True)
        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.08, top=0.82, hspace=0.34)
    else:
        axes = fig.subplots(1, 3, sharex=True)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.78, wspace=0.37)
    columns = ("optimal_pv_kw", "optimal_battery_kwh", "mean_annual_cost_ksgd_per_year")
    labels = ("Optimal PV (kW)", "Optimal BESS (kWh)", "Annual cost (kSGD yr$^{-1}$)")
    for ax, column, label, letter in zip(axes, columns, labels, "abc", strict=True):
        _target_panel(ax, source, profile, column, smooth=False)
        _axis(ax, profile, "CCP target (%)", label)
        ax.set_title(f"{letter}  {label}", loc="left", fontsize=cfg["axis"], fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.set_xlim(*_x_limits(source.ccp_target_percent))
        ax.set_xticks(np.arange(50, 96, 10))
    _legend(fig, axes[0], profile)
    return fig, _trace(entry, "ccp_target_percent", ", ".join(columns), rows=len(source))


def _frontier(entry: FigureEntry, profile: str) -> tuple[Any, dict[str, Any]]:
    source = _read(entry)
    x_col, y_col = "incremental_annual_cost_sgd_per_year", "incremental_ccp_percentage_points"
    fig, cfg = _canvas(profile, entry.family)
    _header(fig, profile, "Brisbane · Incremental cost–reliability frontier", "Evaluated non-dominated PV–BESS designs · LOLP < 20%")
    ax = fig.add_axes([0.13, 0.19, 0.78, 0.60])
    points = source.loc[source.is_non_dominated.astype(str).str.lower().eq("true")]
    if points.empty:
        points = source.loc[source.point_type.astype(str).ne("baseline")]
    ax.scatter(points[x_col], points[y_col], color=NATURE_PALETTE["blue"], s=cfg["marker"] ** 2, zorder=3, label="Evaluated design")
    baseline = source.loc[source.point_type.astype(str).eq("baseline")]
    if not baseline.empty:
        ax.scatter(baseline[x_col], baseline[y_col], facecolors="white", edgecolors=NATURE_PALETTE["neutral_black"], s=cfg["marker"] ** 2 * 1.7, zorder=4, label="Baseline", clip_on=False)
    if entry.family == "frontier_fit":
        fit = _read(entry, 1).iloc[0]
        dense = np.linspace(0, float(points[x_col].max()), 200)
        ax.plot(dense, fit.coefficient_a * dense ** fit.exponent_b, color=NATURE_PALETTE["red"], linestyle="--", label="Power-law fit")
    _axis(ax, profile, "Incremental annual cost (SGD yr$^{-1}$)", "Incremental CCP (percentage points)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    _legend(fig, ax, profile)
    return fig, _trace(entry, x_col, y_col, rows=len(source))


def _design(entry: FigureEntry) -> tuple[float, float]:
    text = "/".join(entry.original.relative_to(BRISBANE).parts)
    matches = re.findall(r"PV(\d+(?:p\d+)?)_Battery(\d+(?:p\d+)?)", text)
    if not matches:
        raise ValueError(f"Cannot identify fixed-design capacities: {entry.original}")
    pv, battery = matches[-1]
    return float(pv.replace("p", ".")), float(battery.replace("p", "."))


def _fixed_design(entry: FigureEntry, profile: str) -> tuple[Any, dict[str, Any]]:
    family = entry.family
    if family == "cdf_comparison":
        source = _read(entry)
        fig, _ = _canvas(profile, family)
        _header(fig, profile, "Brisbane · Fixed-design LOLP comparison", "Empirical CDF · 1,000 paired scenarios per design")
        ax = fig.add_axes([0.13, 0.18, 0.78, 0.61])
        for (pv, battery), frame in source.groupby(["pv_capacity_kw", "battery_energy_kwh"], sort=True):
            frame = frame.sort_values("LOLP_value_percent")
            color = DESIGN_COLORS.get((float(pv), float(battery)), NATURE_PALETTE["neutral_dark"])
            ax.step(frame.LOLP_value_percent, frame.CDF * 100, where="post", color=color, label=f"PV {pv:g} kW · BESS {battery:g} kWh")
        _axis(ax, profile, "LOLP (%)", "CDF (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        _legend(fig, ax, profile, ncol=2, y=0.88)
        return fig, _trace(entry, "LOLP_value_percent", "CDF", rows=len(source))

    pv, battery = _design(entry)
    design_label = f"PV {pv:g} kW · BESS {battery:g} kWh"
    if family == "histogram_cdf":
        bins, cdf = _read(entry, 0), _read(entry, 1)
        fig, cfg = _canvas(profile, family)
        _header(fig, profile, "Brisbane · LOLP distribution", design_label + " · histogram and empirical CDF")
        ax = fig.add_axes([0.13, 0.18, 0.75, 0.61])
        other = ax.twinx()
        widths = bins.LOLP_upper_percent.to_numpy(dtype=float) - bins.LOLP_lower_percent.to_numpy(dtype=float)
        ax.bar(bins.LOLP_lower_percent, bins.frequency_percent, width=widths, align="edge", color=NATURE_PALETTE["teal"], alpha=0.83, edgecolor="white", linewidth=0.4, label="Frequency")
        other.step(cdf.LOLP_value_percent, cdf.CDF * 100, where="post", color=NATURE_PALETTE["blue"], linewidth=cfg["line"], label="CDF")
        _axis(ax, profile, "LOLP (%)", "Frequency (%)")
        other.set_ylabel("CDF (%)", fontsize=cfg["axis"], color=NATURE_PALETTE["blue"])
        other.tick_params(axis="y", labelsize=cfg["tick"], colors=NATURE_PALETTE["blue"])
        other.spines["right"].set_visible(True)
        other.spines["right"].set_color(NATURE_PALETTE["blue"])
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        other.set_ylim(0, 100)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        other.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        fig.legend([Patch(color=NATURE_PALETTE["teal"]), Line2D([], [], color=NATURE_PALETTE["blue"])], ["Frequency", "CDF"], loc="upper center", bbox_to_anchor=(0.5, 0.875), ncol=2, frameon=False, fontsize=cfg["legend"])
        return fig, _trace(entry, "LOLP_lower_percent / LOLP_value_percent", "frequency_percent / CDF", rows=len(bins) + len(cdf))

    source = _read(entry)
    fig, _ = _canvas(profile, family)
    title = "LOLP histogram" if family == "histogram" else "Empirical LOLP CDF"
    _header(fig, profile, f"Brisbane · {title}", design_label + " · 1,000 scenarios")
    ax = fig.add_axes([0.13, 0.18, 0.78, 0.61])
    if family == "histogram":
        widths = source.LOLP_upper_percent.to_numpy(dtype=float) - source.LOLP_lower_percent.to_numpy(dtype=float)
        ax.bar(source.LOLP_lower_percent, source.frequency, width=widths, align="edge", color=NATURE_PALETTE["teal"], alpha=0.83, edgecolor="white", linewidth=0.4)
        y_col, y_label = "frequency", "Frequency (scenarios)"
    else:
        ax.step(source.LOLP_value_percent, source.CDF * 100, where="post", color=NATURE_PALETTE["blue"])
        y_col, y_label = "CDF", "CDF (%)"
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    _axis(ax, profile, "LOLP (%)", y_label)
    ax.set_xlim(left=0, right=max(80.0, float(source.LOLP_upper_percent.max()) if family == "histogram" else float(source.LOLP_value_percent.max())))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    if family == "histogram":
        ax.set_ylim(bottom=0)
    return fig, _trace(entry, "LOLP_lower_percent" if family == "histogram" else "LOLP_value_percent", y_col, rows=len(source))


def _metric_values(source: pd.DataFrame, metric_key: str) -> tuple[np.ndarray, str, str, str]:
    column, title, unit, color_key, _ = METRICS[metric_key]
    if column not in source.columns:
        raise ValueError(f"Missing sensitivity metric column {column}")
    values = pd.to_numeric(source[column], errors="coerce").to_numpy(dtype=float)
    if unit == "%":
        values = values * 100.0
    return values, title, unit, color_key


def _sensitivity(entry: FigureEntry, profile: str) -> tuple[Any, dict[str, Any]]:
    source = _read(entry).sort_values("sweep_value")
    sweep = entry.original.parent.name
    x = source.sweep_value.to_numpy(dtype=float)
    x_label = SENSITIVITY_X_LABELS[sweep]
    constrained = "CCP_LOLP_Cost_Sensitivity_Figures" in entry.run_name
    subtitle = "P(LOLP < 20%) ≥ 90% · frozen scenario results" if constrained else "15-year results · fixed background parameters"
    if entry.family == "sensitivity_single":
        metric = _metric_from_name(entry.original.stem)
        values, title, unit, color_key = _metric_values(source, metric)
        fig, _ = _canvas(profile, entry.family)
        _header(fig, profile, f"Brisbane · {title} vs {sweep.replace('_', ' ')}", subtitle)
        ax = fig.add_axes([0.13, 0.18, 0.78, 0.61])
        _line_data(ax, x, values, color=METRIC_COLORS[color_key], profile=profile, smooth=True)
        _axis(ax, profile, x_label, f"{title} ({unit})")
        ax.set_xlim(*_x_limits(x))
        ax.set_xticks(np.linspace(float(x.min()), float(x.max()), 5))
        if unit == "%":
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=None))
        return fig, _trace(entry, "sweep_value", METRICS[metric][0], rows=len(source))

    fig, cfg = _canvas(profile, entry.family, panels=7)
    _header(fig, profile, f"Brisbane · {sweep.replace('_', ' ')} sensitivity", subtitle)
    if profile == "paper":
        axes = fig.subplots(4, 2).flat
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.08, top=0.82, wspace=0.35, hspace=0.62)
    else:
        axes = fig.subplots(2, 4).flat
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.77, wspace=0.5, hspace=0.6)
    axes = list(axes)
    for index, (ax, (metric, _)) in enumerate(zip(axes, METRICS.items()), start=1):
        values, title, unit, color_key = _metric_values(source, metric)
        _line_data(ax, x, values, color=METRIC_COLORS[color_key], profile=profile, smooth=True)
        if profile == "paper":
            _axis(ax, profile, x_label, f"{title} ({unit})")
            ax.set_title(f"{chr(96 + index)}  {title}", loc="left", fontsize=cfg["axis"], fontweight="bold")
        else:
            _axis(ax, profile, "", "")
            ax.set_title(f"{chr(96 + index)}  {title} ({unit})", loc="left", fontsize=12.2, fontweight="bold")
            ax.tick_params(labelsize=11.5)
        ax.set_xlim(*_x_limits(x))
        ax.set_xticks(np.linspace(float(x.min()), float(x.max()), 5))
        if unit == "%":
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=None))
    axes[-1].set_visible(False)
    if profile == "presentation":
        fig.text(0.53, 0.055, x_label, ha="center", va="center", fontsize=cfg["axis"])
    return fig, _trace(entry, "sweep_value", ", ".join(spec[0] for spec in METRICS.values()), rows=len(source))


def _outage(entry: FigureEntry, profile: str) -> tuple[Any, dict[str, Any]]:
    source = _read(entry)
    summary = json.loads(entry.sources[1].read_text(encoding="utf-8"))
    cfg = PROFILES[profile]
    if entry.family == "outage_short":
        fig, _ = _canvas(profile, entry.family)
        _header(fig, profile, "Brisbane · Typical outage energy balance", "PV 3.25 kW · BESS 11.5 kWh · seed 999, event 6")
        axes = fig.subplots(2, 1)
        fig.subplots_adjust(left=0.13, right=0.91, bottom=0.13, top=0.78, hspace=0.55)
        top, bottom = axes
        event = source.loc[source.is_event_hour.astype(str).str.lower().eq("true")].sort_values("hour_index")
        x = np.arange(1, len(event) + 1, dtype=float)
        pv = event.pv_to_outage_kwh.to_numpy(dtype=float)
        battery = event.battery_to_outage_kwh.to_numpy(dtype=float)
        top.bar(x, pv, width=0.6, color=NATURE_PALETTE["teal"], label="PV to load")
        top.bar(x, battery, bottom=pv, width=0.6, color=NATURE_PALETTE["purple"], label="BESS to load")
        top.plot(x, event.outage_load_kwh.to_numpy(dtype=float), color=NATURE_PALETTE["neutral_black"], marker="o", label="Interrupted load")
        _axis(top, profile, "Affected model hour", "Energy (kWh)")
        top.set_xlim(0.5, len(event) + 0.5)
        relative = source.hours_from_outage_start_at_hour_end.to_numpy(dtype=float)
        bottom.plot(relative, source.battery_soc_percent.to_numpy(dtype=float), color=NATURE_PALETTE["purple"], marker="o")
        bottom.axhline(float(source.minimum_soc_percent.iloc[0]), color=NATURE_PALETTE["red"], linestyle="--")
        _axis(bottom, profile, "Hours relative to outage start", "BESS SOC (%)")
        bottom.set_ylim(0, 100)
        bottom.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        _legend(fig, top, profile)
        return fig, _trace(entry, "hour_index / hours_from_outage_start_at_hour_end", "pv_to_outage_kwh, battery_to_outage_kwh, outage_load_kwh / battery_soc_percent", rows=len(source))

    fig, _ = _canvas(profile, entry.family)
    _header(fig, profile, "Brisbane · 72-hour outage energy balance", "PV 3.25 kW · BESS 11.5 kWh · seed 999, event 6")
    ax = fig.add_axes([0.1, 0.19, 0.78, 0.59])
    x = source.plot_hour.to_numpy(dtype=float)
    outage_start = float(summary["event_start_hour"]) - float(summary["window_start_hour"])
    outage_end = float(summary["event_end_hour"]) - float(summary["window_start_hour"])
    ax.axvspan(outage_start, outage_end, color=NATURE_PALETTE["neutral_light"], alpha=0.45)
    positive = np.zeros(len(source))
    flow_pos = (("pv_available_kwh", "PV available", NATURE_PALETTE["teal"]), ("grid_import_kwh", "Grid import", NATURE_PALETTE["neutral_mid"]), ("battery_discharge_kwh", "BESS discharge", NATURE_PALETTE["purple"]))
    for column, label, color in flow_pos:
        values = source[column].to_numpy(dtype=float)
        ax.bar(x, values, bottom=positive, width=0.92, align="edge", color=color, label=label)
        positive += values
    negative = np.zeros(len(source))
    flow_neg = (("served_load_kwh", "Served load", NATURE_PALETTE["neutral_black"]), ("battery_charge_kwh", "BESS charge", NATURE_PALETTE["red_light"]), ("pv_curtailed_kwh", "PV curtailed", NATURE_PALETTE["teal_light"]), ("pv_export_kwh", "PV export", NATURE_PALETTE["neutral_light"]))
    for column, label, color in flow_neg:
        values = source[column].to_numpy(dtype=float)
        if column == "pv_export_kwh" and not np.any(values > 1e-8):
            continue
        ax.bar(x, -values, bottom=negative, width=0.92, align="edge", color=color, label=label)
        negative -= values
    unmet = source.unmet_load_kwh_equiv.to_numpy(dtype=float)
    if np.any(unmet > 1e-8):
        ax.bar(x, -unmet, bottom=-source.served_load_kwh.to_numpy(dtype=float), width=0.92, align="edge", facecolor="none", edgecolor=NATURE_PALETTE["red"], hatch="////", linewidth=0.75, label="Unserved load")
    ax.axhline(0, color=NATURE_PALETTE["neutral_black"], linewidth=0.8)
    for bound in (24, 48):
        ax.axvline(bound, color=NATURE_PALETTE["neutral_mid"], linestyle=":", linewidth=0.8)
    soc_ax = ax.twinx()
    soc_ax.plot(x + 0.5, source.battery_soc_fraction, color=NATURE_PALETTE["purple"], linestyle="-.", label="BESS SOC")
    soc_ax.plot(x + 0.5, source.minimum_soc_fraction, color=NATURE_PALETTE["red"], linestyle=":", label="Minimum SOC")
    soc_ax.set_ylabel("BESS SOC", fontsize=cfg["axis"])
    soc_ax.set_ylim(0, 1.05)
    soc_ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    soc_ax.tick_params(labelsize=cfg["tick"])
    soc_ax.spines["right"].set_visible(True)
    _axis(ax, profile, "Hour in 72 h window", "Hourly energy balance (kWh)")
    ax.set_xlim(0, 72)
    ax.set_xticks(np.arange(0, 73, 12))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = soc_ax.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=4, frameon=False, fontsize=cfg["legend"])
    return fig, _trace(entry, "plot_hour", "AC flow columns / battery_soc_fraction", rows=len(source))


def render(entry: FigureEntry, profile: str) -> tuple[Any, dict[str, Any]]:
    if entry.family in {"capacity_contour", "sensitivity_contour"}:
        return _contour(entry, profile)
    if entry.family == "surface":
        return _surface(entry, profile)
    if entry.family == "response":
        return _response(entry, profile)
    if entry.family in {"target_line", "target_scatter"}:
        return _target_line(entry, profile)
    if entry.family == "target_triptych":
        return _target_triptych(entry, profile)
    if entry.family in {"frontier", "frontier_fit"}:
        return _frontier(entry, profile)
    if entry.family in {"histogram", "cdf", "histogram_cdf", "cdf_comparison"}:
        return _fixed_design(entry, profile)
    if entry.family in {"sensitivity_single", "sensitivity_overview"}:
        return _sensitivity(entry, profile)
    if entry.family in {"outage_short", "outage_72h"}:
        return _outage(entry, profile)
    raise ValueError(f"No renderer for {entry.family}")


def _protected_paths(entries: list[FigureEntry]) -> dict[Path, str]:
    paths: set[Path] = set()
    for entry in entries:
        paths.update(entry.sources)
        paths.update(entry.original.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf"))
    return {path: sha256(path) for path in sorted(paths)}


def _validate_output_root(output_root: Path) -> Path:
    root = output_root.resolve()
    if root != DEFAULT_OUTPUT_ROOT.resolve():
        raise ValueError(f"Output root must be {DEFAULT_OUTPUT_ROOT}")
    return root


def _output_base(entry: FigureEntry, output_root: Path, profile: str) -> Path:
    relative = entry.original.relative_to(BRISBANE)
    return output_root / profile / entry.category / relative.with_suffix("")


def export(profile: str = "both", output_root: Path = DEFAULT_OUTPUT_ROOT, *, dry_run: bool = False) -> dict[str, Any]:
    output_root = _validate_output_root(Path(output_root))
    entries = inventory(output_root)
    selected = ("paper", "presentation") if profile == "both" else (profile,)
    if any(item not in PROFILES for item in selected):
        raise ValueError(f"Unknown profile: {profile}")
    original_hashes = _protected_paths(entries)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "output_root": str(output_root),
        "style_profiles": selected,
        "original_figure_count": len(entries),
        "category_counts": CATEGORY_COUNTS,
        "source_policy": "Read frozen existing figure source CSV/JSON; no simulation or data mutation",
        "entries": [],
    }
    for number, entry in enumerate(entries, start=1):
        record: dict[str, Any] = {
            "old_png": str(entry.original.relative_to(ROOT)),
            "category": entry.category,
            "family": entry.family,
            "original_sha256": {suffix: original_hashes[entry.original.with_suffix(suffix)] for suffix in (".png", ".svg", ".pdf")},
            "sources": [{"path": str(path.relative_to(ROOT)), "sha256": original_hashes[path]} for path in entry.sources],
            "exports": {},
        }
        for current_profile in selected:
            base = _output_base(entry, output_root, current_profile)
            if dry_run:
                record["exports"][current_profile] = {suffix: str(base.with_suffix("." + suffix).relative_to(ROOT)) for suffix in FORMATS}
                continue
            base.parent.mkdir(parents=True, exist_ok=True)
            fig, trace = render(entry, current_profile)
            try:
                fig.canvas.draw()
                paths: dict[str, str] = {}
                for suffix in FORMATS:
                    path = base.with_suffix("." + suffix)
                    fig.savefig(path, dpi=PROFILES[current_profile]["dpi"], facecolor="white")
                    if not path.is_file() or path.stat().st_size == 0:
                        raise RuntimeError(f"Empty figure export: {path}")
                    paths[suffix] = str(path.relative_to(ROOT))
                record["exports"][current_profile] = {"files": paths, "sha256": {suffix: sha256(base.with_suffix("." + suffix)) for suffix in FORMATS}, "plot_trace": trace}
            finally:
                plt.close(fig)
        manifest["entries"].append(record)
        if not dry_run and (number % 10 == 0 or number == len(entries)):
            print(f"Rendered {number}/{len(entries)} source figures for {', '.join(selected)}", flush=True)
    if not dry_run:
        for path, before in original_hashes.items():
            if sha256(path) != before:
                raise RuntimeError(f"Protected original changed during export: {path}")
        manifest["protected_original_count"] = len(original_hashes)
        manifest["old_files_unchanged"] = True
        manifest["figure_count_by_profile"] = {item: sum(item in record["exports"] for record in manifest["entries"]) for item in selected}
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Manifest: {manifest_path}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("paper", "presentation", "both"), default="both")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Check the 94 figure/source mappings without writing files")
    args = parser.parse_args()
    result = export(args.profile, args.output_root, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Validated {result['original_figure_count']} original figures and their frozen sources")


if __name__ == "__main__":
    main()
