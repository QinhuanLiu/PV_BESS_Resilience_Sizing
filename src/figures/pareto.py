from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import NATURE_PALETTE, _matplotlib, _save_all, _style, _write_source_data


REQUIRED_PARETO_COLUMNS = (
    "pv_capacity_mw",
    "battery_energy_mwh",
    "system_total_cost_million_sgd_per_year",
    "mean_UE_mwh_per_year",
    "energy_based_LOLP",
    "chance_constraint_probability",
    "lolp_threshold",
)
KNEE_POINT_COLOR = "#D6A21E"


def make_pareto_frontier_publication_figure(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
) -> dict[str, Path]:
    """Create the publication-style cost-resilience Pareto frontier figure."""
    prepared = build_pareto_plot_source(source)
    base = out / f"{case_key}_Figure_5_Cost_Resilience_Pareto_Frontier"
    paths: dict[str, Path] = {"source_data": _write_source_data(base, prepared)}
    paths.update(plot_pareto_frontier(out, case_key, prepared))
    return paths


def validate_pareto_source(source: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize Fig. 5 source data into plotting units."""
    if source.empty:
        raise ValueError("Pareto frontier source data is empty.")
    prepared = source.copy()
    _derive_pareto_columns(prepared)
    missing = [column for column in REQUIRED_PARETO_COLUMNS if column not in prepared.columns]
    if missing:
        raise ValueError("Pareto frontier source data is missing required column(s): " + ", ".join(missing))

    for column in REQUIRED_PARETO_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if prepared[list(REQUIRED_PARETO_COLUMNS)].isna().any().any():
        raise ValueError("Pareto frontier source data contains missing or non-numeric required values.")
    if (prepared["system_total_cost_million_sgd_per_year"].astype(float) < 0).any():
        raise ValueError("Pareto frontier costs must be non-negative.")
    if (prepared["mean_UE_mwh_per_year"].astype(float) < 0).any():
        raise ValueError("Pareto frontier UE values must be non-negative.")
    if not prepared["chance_constraint_probability"].astype(float).between(0.0, 1.0).all():
        raise ValueError("Pareto frontier CCP values must be between 0 and 1.")
    if not prepared["energy_based_LOLP"].astype(float).between(0.0, 1.0).all():
        raise ValueError("Pareto frontier LOLP values must be between 0 and 1.")
    return prepared.reset_index(drop=True)


def build_pareto_plot_source(source: pd.DataFrame) -> pd.DataFrame:
    """Return source data augmented with Pareto, knee, and marginal-cost fields."""
    prepared = validate_pareto_source(source)
    prepared["is_pareto_efficient"] = identify_pareto_efficient(prepared).astype(bool)
    prepared["is_minimum_total_cost_design"] = False
    prepared["is_resilience_optimum"] = False
    prepared["is_knee_point"] = False

    economic_index = _economic_optimum_index(prepared)
    resilience_index = _resilience_optimum_index(prepared)
    prepared.loc[economic_index, "is_minimum_total_cost_design"] = True
    prepared.loc[resilience_index, "is_resilience_optimum"] = True

    knee_index = identify_knee_point(prepared)
    if knee_index is not None:
        prepared.loc[knee_index, "is_knee_point"] = True

    prepared = compute_marginal_resilience_cost(prepared)
    return prepared.sort_values(
        [
            "is_pareto_efficient",
            "system_total_cost_million_sgd_per_year",
            "mean_UE_mwh_per_year",
            "pv_capacity_mw",
            "battery_energy_mwh",
        ],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)


def identify_pareto_efficient(source: pd.DataFrame) -> pd.Series:
    """Identify non-dominated designs minimizing annualized cost and UE."""
    prepared = validate_pareto_source(source)
    values = prepared[["system_total_cost_million_sgd_per_year", "mean_UE_mwh_per_year"]].astype(float)
    efficient: list[bool] = []
    for row in values.itertuples(index=False):
        dominated = (
            (
                (values["system_total_cost_million_sgd_per_year"] <= float(row.system_total_cost_million_sgd_per_year))
                & (values["mean_UE_mwh_per_year"] <= float(row.mean_UE_mwh_per_year))
            )
            & (
                (values["system_total_cost_million_sgd_per_year"] < float(row.system_total_cost_million_sgd_per_year))
                | (values["mean_UE_mwh_per_year"] < float(row.mean_UE_mwh_per_year))
            )
        ).any()
        efficient.append(not bool(dominated))
    return pd.Series(efficient, index=prepared.index)


def identify_knee_point(source: pd.DataFrame) -> int | None:
    """Return the index of the normalized maximum-distance knee point, if defined."""
    prepared = validate_pareto_source(source)
    if "is_pareto_efficient" in prepared.columns:
        pareto = prepared[prepared["is_pareto_efficient"].map(_truthy)].copy()
    else:
        pareto = prepared[identify_pareto_efficient(prepared)].copy()
    pareto = pareto.sort_values(["system_total_cost_million_sgd_per_year", "mean_UE_mwh_per_year"])
    if len(pareto) < 3:
        return None

    x = pareto["system_total_cost_million_sgd_per_year"].to_numpy(dtype=float)
    y = pareto["mean_UE_mwh_per_year"].to_numpy(dtype=float)
    points = np.column_stack([_normalize(x), _normalize(y)])
    start = points[0]
    end = points[-1]
    span = end - start
    denom = float(np.linalg.norm(span))
    if np.isclose(denom, 0.0):
        return int(pareto.index[len(pareto) // 2])
    shifted = points - start
    distances = np.abs(span[0] * shifted[:, 1] - span[1] * shifted[:, 0]) / denom
    return int(pareto.index[int(np.argmax(distances))])


def compute_marginal_resilience_cost(source: pd.DataFrame) -> pd.DataFrame:
    """Add Pareto-step UE reduction and marginal cost per MWh UE reduction."""
    prepared = validate_pareto_source(source)
    if "is_pareto_efficient" not in prepared.columns:
        prepared["is_pareto_efficient"] = identify_pareto_efficient(prepared)

    prepared["pareto_step"] = pd.NA
    prepared["ue_reduction_from_economic_mwh_per_year"] = np.nan
    prepared["marginal_cost_per_mwh_ue_reduction_million_sgd_per_mwh"] = np.nan

    pareto = prepared[prepared["is_pareto_efficient"].map(_truthy)].copy()
    pareto = pareto.sort_values(["system_total_cost_million_sgd_per_year", "mean_UE_mwh_per_year"])
    if pareto.empty:
        return prepared

    base_ue = float(pareto["mean_UE_mwh_per_year"].iloc[0])
    previous_cost: float | None = None
    previous_ue: float | None = None
    for step, (index, row) in enumerate(pareto.iterrows(), start=1):
        cost = float(row["system_total_cost_million_sgd_per_year"])
        ue_value = float(row["mean_UE_mwh_per_year"])
        prepared.loc[index, "pareto_step"] = step
        prepared.loc[index, "ue_reduction_from_economic_mwh_per_year"] = max(0.0, base_ue - ue_value)
        if previous_cost is None or previous_ue is None:
            marginal = np.nan
        else:
            delta_cost = max(0.0, cost - previous_cost)
            delta_ue_reduction = max(0.0, previous_ue - ue_value)
            marginal = delta_cost / delta_ue_reduction if delta_ue_reduction > 0 else np.nan
        prepared.loc[index, "marginal_cost_per_mwh_ue_reduction_million_sgd_per_mwh"] = marginal
        previous_cost = cost
        previous_ue = ue_value
    return prepared


def plot_pareto_frontier(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    """Plot the 2x2 Nature-style Pareto frontier figure."""
    source = build_pareto_plot_source(source)
    mpl, plt = _matplotlib(out)
    _style(mpl)

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.35), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    _plot_cost_ue_panel(ax_a, source)
    _plot_lolp_ccp_panel(ax_b, source)
    _plot_marginal_cost_panel(ax_c, source)
    _plot_capacity_map_panel(ax_d, source)

    base = out / f"{case_key}_Figure_5_Cost_Resilience_Pareto_Frontier"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _derive_pareto_columns(frame: pd.DataFrame) -> None:
    if "pv_capacity_mw" not in frame.columns and "pv_capacity_kw" in frame.columns:
        frame["pv_capacity_mw"] = pd.to_numeric(frame["pv_capacity_kw"], errors="coerce") / 1000.0
    if "battery_energy_mwh" not in frame.columns and "battery_energy_kwh" in frame.columns:
        frame["battery_energy_mwh"] = pd.to_numeric(frame["battery_energy_kwh"], errors="coerce") / 1000.0
    if "system_total_cost_million_sgd_per_year" not in frame.columns and "system_total_cost_sgd_per_year" in frame.columns:
        frame["system_total_cost_million_sgd_per_year"] = (
            pd.to_numeric(frame["system_total_cost_sgd_per_year"], errors="coerce") / 1e6
        )
    if "mean_UE_mwh_per_year" not in frame.columns and "mean_UE_kwh_per_year" in frame.columns:
        frame["mean_UE_mwh_per_year"] = pd.to_numeric(frame["mean_UE_kwh_per_year"], errors="coerce") / 1000.0
    if "energy_based_LOLP" not in frame.columns and "mean_scenario_LOLP" in frame.columns:
        frame["energy_based_LOLP"] = pd.to_numeric(frame["mean_scenario_LOLP"], errors="coerce")


def _plot_cost_ue_panel(ax: Any, source: pd.DataFrame) -> None:
    pareto = _pareto_source(source)
    _scatter_all_candidates(ax, source)
    sizes = _marker_sizes(pareto)
    sc = ax.scatter(
        pareto["system_total_cost_million_sgd_per_year"],
        pareto["mean_UE_mwh_per_year"],
        c=pareto["energy_based_LOLP"],
        cmap="Reds",
        s=sizes,
        edgecolor=_ccp_edges(pareto),
        linewidth=0.8,
        zorder=4,
    )
    ax.plot(
        pareto["system_total_cost_million_sgd_per_year"],
        pareto["mean_UE_mwh_per_year"],
        color=NATURE_PALETTE["neutral_dark"],
        linewidth=0.9,
        zorder=3,
    )
    _mark_special_designs(ax, source, x_column="system_total_cost_million_sgd_per_year", y_column="mean_UE_mwh_per_year")
    ax.set_title("a  Cost-UE Pareto frontier", loc="left")
    ax.set_xlabel("Annualized system cost (million SGD yr$^{-1}$)")
    ax.set_ylabel("Mean UE (MWh yr$^{-1}$)")
    ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.025).set_label("Energy-based LOLP", fontsize=6.2)
    _add_panel_note(ax, "size = BESS\nedge = CCP >= 0.90")


def _plot_lolp_ccp_panel(ax: Any, source: pd.DataFrame) -> None:
    pareto = _pareto_source(source)
    sc = ax.scatter(
        pareto["system_total_cost_million_sgd_per_year"],
        pareto["energy_based_LOLP"],
        c=pareto["chance_constraint_probability"],
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        s=32,
        edgecolor=NATURE_PALETTE["neutral_black"],
        linewidth=0.45,
        zorder=3,
    )
    ax.plot(
        pareto["system_total_cost_million_sgd_per_year"],
        pareto["energy_based_LOLP"],
        color=NATURE_PALETTE["neutral_mid"],
        linewidth=0.85,
        zorder=2,
    )
    threshold = float(source["lolp_threshold"].dropna().astype(float).iloc[0])
    ax.axhline(threshold, color=NATURE_PALETTE["red"], linestyle="--", linewidth=0.9)
    ax.text(
        0.98,
        threshold,
        "LOLP target",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=NATURE_PALETTE["red"],
    )
    _mark_special_designs(ax, source, x_column="system_total_cost_million_sgd_per_year", y_column="energy_based_LOLP")
    ax.set_title("b  LOLP and CCP screen", loc="left")
    ax.set_xlabel("Annualized system cost (million SGD yr$^{-1}$)")
    ax.set_ylabel("Energy-based LOLP")
    ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.025).set_label("CCP", fontsize=6.2)


def _plot_marginal_cost_panel(ax: Any, source: pd.DataFrame) -> None:
    pareto = _pareto_source(source)
    steps = pareto["pareto_step"].astype(float).to_numpy()
    reduction = pareto["ue_reduction_from_economic_mwh_per_year"].astype(float).to_numpy()
    marginal = pareto["marginal_cost_per_mwh_ue_reduction_million_sgd_per_mwh"].astype(float).to_numpy()

    ax.bar(
        steps,
        reduction,
        color=NATURE_PALETTE["teal"],
        alpha=0.72,
        width=0.72,
        edgecolor="white",
        linewidth=0.35,
        label="Cumulative UE reduction",
    )
    ax.set_xlabel("Pareto step ordered by cost")
    ax.set_ylabel("UE reduction (MWh yr$^{-1}$)")
    ax.set_title("c  Marginal resilience cost", loc="left")
    ax2 = ax.twinx()
    finite = np.isfinite(marginal)
    if finite.any():
        ax2.plot(
            steps[finite],
            marginal[finite],
            color=NATURE_PALETTE["red"],
            marker="o",
            markersize=2.8,
            linewidth=0.9,
            label="Marginal cost",
        )
    ax2.set_ylabel("Marginal cost (million SGD MWh$^{-1}$)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.8)
    _mark_knee_step(ax, source)
    _combined_legend(ax, ax2)


def _plot_capacity_map_panel(ax: Any, source: pd.DataFrame) -> None:
    pareto = _pareto_source(source)
    sc = ax.scatter(
        source["pv_capacity_mw"],
        source["battery_energy_mwh"],
        c=source["mean_UE_mwh_per_year"],
        cmap="Reds",
        s=14,
        alpha=0.30,
        edgecolor="none",
        zorder=1,
    )
    ax.plot(
        pareto["pv_capacity_mw"],
        pareto["battery_energy_mwh"],
        color=NATURE_PALETTE["neutral_black"],
        linewidth=1.0,
        zorder=3,
    )
    ax.scatter(
        pareto["pv_capacity_mw"],
        pareto["battery_energy_mwh"],
        color=NATURE_PALETTE["blue"],
        s=24,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )
    _mark_special_designs(ax, source, x_column="pv_capacity_mw", y_column="battery_energy_mwh")
    ax.set_title("d  Pareto map in PV-BESS space", loc="left")
    ax.set_xlabel("PV capacity (MW)")
    ax.set_ylabel("BESS energy (MWh)")
    ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.025).set_label("Mean UE (MWh yr$^{-1}$)", fontsize=6.2)


def _scatter_all_candidates(ax: Any, source: pd.DataFrame) -> None:
    ax.scatter(
        source["system_total_cost_million_sgd_per_year"],
        source["mean_UE_mwh_per_year"],
        color=NATURE_PALETTE["neutral_light"],
        s=12,
        alpha=0.55,
        edgecolor="none",
        zorder=1,
        label="All candidates",
    )


def _mark_special_designs(ax: Any, source: pd.DataFrame, x_column: str, y_column: str) -> None:
    specs = (
        ("is_minimum_total_cost_design", "Economic optimum", "o", NATURE_PALETTE["neutral_black"], "white"),
        ("is_knee_point", "Knee point", "*", KNEE_POINT_COLOR, NATURE_PALETTE["neutral_black"]),
        ("is_resilience_optimum", "Resilience optimum", "D", NATURE_PALETTE["blue"], "white"),
    )
    for column, label, marker, facecolor, edgecolor in specs:
        if column not in source.columns:
            continue
        selected = source[source[column].map(_truthy)]
        if selected.empty:
            continue
        row = selected.iloc[0]
        ax.scatter(
            [float(row[x_column])],
            [float(row[y_column])],
            marker=marker,
            s=78 if marker == "*" else 48,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.85,
            zorder=8,
            label=label,
        )


def _mark_knee_step(ax: Any, source: pd.DataFrame) -> None:
    selected = source[source.get("is_knee_point", pd.Series(False, index=source.index)).map(_truthy)]
    if selected.empty or pd.isna(selected.iloc[0].get("pareto_step")):
        return
    step = float(selected.iloc[0]["pareto_step"])
    ax.axvline(step, color=NATURE_PALETTE["neutral_black"], linestyle="--", linewidth=0.85)
    ax.text(
        step,
        0.96,
        "knee",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=5.8,
        color=NATURE_PALETTE["neutral_black"],
    )


def _combined_legend(ax: Any, ax2: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles or handles2:
        ax.legend(handles + handles2, labels + labels2, loc="upper left", fontsize=5.8)


def _add_panel_note(ax: Any, text: str) -> None:
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=5.6,
        color=NATURE_PALETTE["neutral_dark"],
        va="top",
        ha="right",
    )


def _pareto_source(source: pd.DataFrame) -> pd.DataFrame:
    pareto = source[source["is_pareto_efficient"].map(_truthy)].copy()
    return pareto.sort_values(["system_total_cost_million_sgd_per_year", "mean_UE_mwh_per_year"])


def _marker_sizes(source: pd.DataFrame) -> np.ndarray:
    values = source["battery_energy_mwh"].astype(float).to_numpy()
    if values.size == 0 or np.isclose(float(np.nanmin(values)), float(np.nanmax(values))):
        return np.full(values.shape, 36.0)
    scaled = (values - float(np.nanmin(values))) / (float(np.nanmax(values)) - float(np.nanmin(values)))
    return 26.0 + scaled * 58.0


def _ccp_edges(source: pd.DataFrame, threshold: float = 0.90) -> list[str]:
    return [
        NATURE_PALETTE["neutral_black"] if float(value) >= threshold else "white"
        for value in source["chance_constraint_probability"].astype(float)
    ]


def _economic_optimum_index(source: pd.DataFrame) -> int:
    return int(
        source.sort_values(
            ["system_total_cost_million_sgd_per_year", "mean_UE_mwh_per_year", "pv_capacity_mw", "battery_energy_mwh"]
        ).index[0]
    )


def _resilience_optimum_index(source: pd.DataFrame) -> int:
    return int(
        source.sort_values(
            ["mean_UE_mwh_per_year", "system_total_cost_million_sgd_per_year", "pv_capacity_mw", "battery_energy_mwh"]
        ).index[0]
    )


def _normalize(values: np.ndarray) -> np.ndarray:
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if np.isclose(low, high):
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "available"}
    return bool(value)
