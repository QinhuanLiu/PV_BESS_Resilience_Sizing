from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    NATURE_PALETTE,
    apply_nature_style,
    format_colorbar,
    matplotlib_context,
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
)


FIGURE_7_SOURCE_NAME = "Figure_7_SAIFI_SAIDI_Sensitivity"
FIGURE_7A_NAME = "Figure_7A_SAIFI_SAIDI_Response_Surfaces"
FIGURE_7B_NAME = "Figure_7B_SAIFI_SAIDI_Contour_Maps"
CONTOUR_INTERVALS = 10
REQUIRED_INPUT_COLUMNS = (
    "saifi_per_year",
    "saidi_min_per_year",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "pv_bess_system_annual_cost_sgd",
    "unserved_energy_kwh_per_year",
)
REQUIRED_SOURCE_COLUMNS = (
    "saifi_per_year",
    "saidi_min_per_year",
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


@dataclass(frozen=True)
class SaifiSaidiMetricSpec:
    key: str
    title: str
    z_label: str
    colorbar_label: str
    cmap: str
    unit_type: str


SAIFI_SAIDI_METRICS = (
    SaifiSaidiMetricSpec(
        key="system_total_cost_million_sgd_per_year",
        title="System total cost",
        z_label="Cost (million SGD yr$^{-1}$)",
        colorbar_label="Cost (million SGD yr$^{-1}$)",
        cmap="Blues",
        unit_type="cost",
    ),
    SaifiSaidiMetricSpec(
        key="pv_capacity_mw",
        title="PV capacity",
        z_label="PV capacity (MW)",
        colorbar_label="PV capacity (MW)",
        cmap="YlOrBr",
        unit_type="capacity",
    ),
    SaifiSaidiMetricSpec(
        key="battery_energy_mwh",
        title="Battery capacity",
        z_label="Battery capacity (MWh)",
        colorbar_label="Battery capacity (MWh)",
        cmap="Purples",
        unit_type="energy",
    ),
    SaifiSaidiMetricSpec(
        key="UE_mwh_per_year",
        title="Unserved energy",
        z_label="UE (MWh yr$^{-1}$)",
        colorbar_label="UE (MWh yr$^{-1}$)",
        cmap="Reds",
        unit_type="ue",
    ),
    SaifiSaidiMetricSpec(
        key="energy_based_LOLP",
        title="Energy-based LOLP",
        z_label="LOLP",
        colorbar_label="Energy-based LOLP",
        cmap="YlGnBu",
        unit_type="lolp",
    ),
)


def _display_saifi_saidi_metrics(units: FigureDisplayUnits) -> tuple[SaifiSaidiMetricSpec, ...]:
    return (
        SaifiSaidiMetricSpec(
            key=DISPLAY_SYSTEM_COST,
            title="System total cost",
            z_label=units.cost_axis_label,
            colorbar_label=units.cost_axis_label,
            cmap="Blues",
            unit_type="cost",
        ),
        SaifiSaidiMetricSpec(
            key=DISPLAY_PV_CAPACITY,
            title="PV capacity",
            z_label=units.pv_axis_label,
            colorbar_label=units.pv_axis_label,
            cmap="YlOrBr",
            unit_type="capacity",
        ),
        SaifiSaidiMetricSpec(
            key=DISPLAY_BATTERY_ENERGY,
            title="Battery capacity",
            z_label=units.battery_capacity_label,
            colorbar_label=units.battery_capacity_label,
            cmap="Purples",
            unit_type="energy",
        ),
        SaifiSaidiMetricSpec(
            key=DISPLAY_UE,
            title="Unserved energy",
            z_label=units.ue_axis_label,
            colorbar_label=units.ue_axis_label,
            cmap="Reds",
            unit_type="ue",
        ),
        SaifiSaidiMetricSpec(
            key="energy_based_LOLP",
            title="Energy-based LOLP",
            z_label="LOLP",
            colorbar_label="Energy-based LOLP",
            cmap="YlGnBu",
            unit_type="lolp",
        ),
    )


def build_saifi_saidi_sensitivity_source(rows: pd.DataFrame) -> pd.DataFrame:
    """Build Figure 7 source data from deterministic SAIFI x SAIDI grid runs."""
    if rows.empty:
        raise ValueError("Figure 7 SAIFI-SAIDI source data is empty.")
    if set(REQUIRED_SOURCE_COLUMNS).issubset(rows.columns) and not set(REQUIRED_INPUT_COLUMNS).issubset(rows.columns):
        return validate_saifi_saidi_sensitivity_source(rows)
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError("Figure 7 SAIFI-SAIDI source is missing required column(s): " + ", ".join(missing))
    source = rows.copy()
    for column in REQUIRED_INPUT_COLUMNS:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source[list(REQUIRED_INPUT_COLUMNS)].isna().any().any():
        raise ValueError("Figure 7 SAIFI-SAIDI source contains missing or non-numeric required values.")
    if "outage_weighted_load_kwh" in source.columns:
        source["outage_weighted_load_kwh"] = pd.to_numeric(source["outage_weighted_load_kwh"], errors="coerce")
    else:
        source["outage_weighted_load_kwh"] = 0.0
    if source["outage_weighted_load_kwh"].isna().any():
        raise ValueError("Figure 7 SAIFI-SAIDI source contains non-numeric outage_weighted_load_kwh values.")

    source["pv_capacity_mw"] = source["pv_capacity_kw"].astype(float) / 1000.0
    source["battery_energy_mwh"] = source["battery_energy_kwh"].astype(float) / 1000.0
    source["system_total_cost_sgd_per_year"] = source["pv_bess_system_annual_cost_sgd"].astype(float)
    source["system_total_cost_million_sgd_per_year"] = source["system_total_cost_sgd_per_year"] / 1e6
    source["UE_kwh_per_year"] = source["unserved_energy_kwh_per_year"].astype(float)
    source["UE_mwh_per_year"] = source["UE_kwh_per_year"] / 1000.0
    if "energy_based_LOLP" in source.columns:
        source["energy_based_LOLP"] = pd.to_numeric(source["energy_based_LOLP"], errors="coerce")
    else:
        source["energy_based_LOLP"] = source.apply(_energy_based_lolp, axis=1)

    grouped = source.groupby(["saifi_per_year", "saidi_min_per_year"], as_index=False)
    mean_columns = {
        column: "mean"
        for column in REQUIRED_SOURCE_COLUMNS
        if column not in {"saifi_per_year", "saidi_min_per_year"}
    }
    aggregated = grouped.agg(mean_columns).reset_index(drop=True)
    scenario_counts = source.groupby(["saifi_per_year", "saidi_min_per_year"]).size().reset_index(name="scenario_count")
    aggregated = aggregated.merge(scenario_counts, on=["saifi_per_year", "saidi_min_per_year"], how="left")
    if "outage_seed" in source.columns:
        seeds = (
            source.assign(outage_seed=pd.to_numeric(source["outage_seed"], errors="coerce"))
            .groupby(["saifi_per_year", "saidi_min_per_year"], as_index=False)
            .agg(outage_seed_start=("outage_seed", "min"), outage_seed_end=("outage_seed", "max"))
        )
        aggregated = aggregated.merge(seeds, on=["saifi_per_year", "saidi_min_per_year"], how="left")
    if "dispatch_backend" in source.columns:
        backend = (
            source.groupby(["saifi_per_year", "saidi_min_per_year"])["dispatch_backend"]
            .agg(_backend_summary)
            .reset_index(name="dispatch_backend")
        )
        aggregated = aggregated.merge(backend, on=["saifi_per_year", "saidi_min_per_year"], how="left")
    for column in (
        "pv_capacity_kw",
        "battery_energy_kwh",
        "system_total_cost_sgd_per_year",
        "UE_kwh_per_year",
        "energy_based_LOLP",
    ):
        stats = (
            source.groupby(["saifi_per_year", "saidi_min_per_year"])[column]
            .agg(
                **{
                    f"{column}_std": "std",
                    f"{column}_p10": lambda values: values.quantile(0.10),
                    f"{column}_p90": lambda values: values.quantile(0.90),
                }
            )
            .reset_index()
        )
        aggregated = aggregated.merge(stats, on=["saifi_per_year", "saidi_min_per_year"], how="left")
    return validate_saifi_saidi_sensitivity_source(aggregated)


def validate_saifi_saidi_sensitivity_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Figure 7 SAIFI-SAIDI source data is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Figure 7 SAIFI-SAIDI source data is empty.")
    validated = source.copy()
    for column in REQUIRED_SOURCE_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(REQUIRED_SOURCE_COLUMNS)].isna().any().any():
        raise ValueError("Figure 7 SAIFI-SAIDI source data contains missing or non-numeric values.")
    if (validated[["saifi_per_year", "saidi_min_per_year"]] < 0.0).any().any():
        raise ValueError("Figure 7 SAIFI and SAIDI values must be non-negative.")
    if (validated["system_total_cost_sgd_per_year"] < 0.0).any():
        raise ValueError("Figure 7 costs must be non-negative.")
    if (validated["UE_kwh_per_year"] < 0.0).any():
        raise ValueError("Figure 7 UE values must be non-negative.")
    if not validated["energy_based_LOLP"].between(0.0, 1.0).all():
        raise ValueError("Figure 7 energy-based LOLP values must be between 0 and 1.")
    _require_complete_grid(validated)
    return validated.sort_values(["saidi_min_per_year", "saifi_per_year"]).reset_index(drop=True)


def _backend_summary(values: pd.Series) -> str:
    backends = sorted(set(values.dropna().astype(str).str.lower().tolist()))
    if len(backends) == 1:
        return backends[0]
    return "mixed" if backends else ""


def make_saifi_saidi_sensitivity_figures(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    prepared = build_saifi_saidi_sensitivity_source(source)
    base = out / f"{case_key}_{FIGURE_7_SOURCE_NAME}"
    paths: dict[str, Path] = {"source_data": _write_source_data(base, prepared)}
    paths.update(plot_saifi_saidi_response_surfaces(out, case_key, prepared))
    paths.update(plot_saifi_saidi_contour_maps(out, case_key, prepared))
    return paths


def plot_saifi_saidi_response_surfaces(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    source = validate_saifi_saidi_sensitivity_source(source)
    source, units = add_display_unit_columns(source)
    metrics = _display_saifi_saidi_metrics(units)
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    fig = plt.figure(figsize=(9.7, 6.35), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.07, top=0.91, wspace=0.22, hspace=0.22)

    for index, spec in enumerate(metrics, start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        _plot_metric_surface(ax, source, spec)
        ax.text2D(0.0, 1.02, f"{chr(96 + index)}  {spec.title}", transform=ax.transAxes, fontsize=8, fontweight="bold")
    blank = fig.add_subplot(2, 3, 6)
    blank.axis("off")

    base = out / f"{case_key}_{FIGURE_7A_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def plot_saifi_saidi_contour_maps(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    source = validate_saifi_saidi_sensitivity_source(source)
    source, units = add_display_unit_columns(source)
    metrics = _display_saifi_saidi_metrics(units)
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2), constrained_layout=True)
    flat_axes = list(axes.ravel())
    for index, (ax, spec) in enumerate(zip(flat_axes, metrics, strict=False), start=1):
        artist = _plot_metric_contour(ax, source, spec)
        cbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.025)
        format_colorbar(cbar, _levels(source[spec.key].to_numpy(dtype=float)), spec.colorbar_label, spec.unit_type)
        ax.set_title(f"{chr(96 + index)}  {spec.title}", loc="left")
        ax.set_xlabel("SAIFI (events yr$^{-1}$)")
        ax.set_ylabel("SAIDI (min yr$^{-1}$)")
        ax.set_box_aspect(1.0)
    flat_axes[-1].axis("off")

    base = out / f"{case_key}_{FIGURE_7B_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _plot_metric_surface(ax: Any, source: pd.DataFrame, spec: SaifiSaidiMetricSpec) -> None:
    x, y, z = _metric_grid(source, spec.key)
    ax.plot_surface(
        x,
        y,
        z,
        cmap=spec.cmap,
        edgecolor="white",
        linewidth=0.28,
        antialiased=True,
        alpha=0.96,
    )
    ax.set_xlabel("SAIFI")
    ax.set_ylabel("SAIDI")
    ax.set_zlabel(spec.z_label)
    ax.view_init(elev=25, azim=-135)
    ax.tick_params(axis="both", pad=0)
    ax.tick_params(axis="z", pad=1)


def _plot_metric_contour(ax: Any, source: pd.DataFrame, spec: SaifiSaidiMetricSpec) -> Any:
    x, y, z = _metric_grid(source, spec.key)
    levels = _levels(z.ravel(), count=CONTOUR_INTERVALS + 1)
    artist = ax.contourf(x, y, z, levels=levels, cmap=spec.cmap, extend="neither")
    ax.contour(x, y, z, levels=levels, colors="white", linewidths=0.35, alpha=0.70)
    return artist


def _metric_grid(source: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = source.pivot_table(
        index="saidi_min_per_year",
        columns="saifi_per_year",
        values=column,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    if pivot.isna().any().any():
        raise ValueError("Figure 7 SAIFI-SAIDI source requires a complete rectangular grid.")
    x_values = pivot.columns.to_numpy(dtype=float)
    y_values = pivot.index.to_numpy(dtype=float)
    x, y = np.meshgrid(x_values, y_values)
    z = pivot.to_numpy(dtype=float)
    return x, y, z


def _levels(values: np.ndarray, count: int = CONTOUR_INTERVALS + 1) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Figure 7 contour levels require finite values.")
    lower = float(finite.min())
    upper = float(finite.max())
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.05, 1e-3)
        lower -= pad
        upper += pad
    return np.linspace(lower, upper, int(count))


def _require_complete_grid(source: pd.DataFrame) -> None:
    saifi_count = source["saifi_per_year"].nunique()
    saidi_count = source["saidi_min_per_year"].nunique()
    if saifi_count < 2 or saidi_count < 2:
        raise ValueError("Figure 7 requires at least two SAIFI and two SAIDI values.")
    expected = int(saifi_count * saidi_count)
    actual = int(source[["saifi_per_year", "saidi_min_per_year"]].drop_duplicates().shape[0])
    if actual != expected:
        raise ValueError("Figure 7 SAIFI-SAIDI source requires a complete rectangular grid.")


def _energy_based_lolp(row: pd.Series) -> float:
    outage_load = float(row["outage_weighted_load_kwh"])
    if outage_load <= 0.0:
        return 0.0
    return float(row["unserved_energy_kwh_per_year"]) / outage_load
