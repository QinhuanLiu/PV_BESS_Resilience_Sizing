from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import NATURE_PALETTE, _matplotlib, _save_all, _style, _write_source_data


RELIABILITY_TARGETS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99)
REQUIRED_DESIGN_SPACE_COLUMNS = (
    "pv_capacity_mw",
    "battery_energy_mwh",
    "system_total_cost_million_sgd_per_year",
)

REQUIRED_CANDIDATE_COLUMNS = (
    "pv_capacity_kw",
    "battery_energy_kwh",
    "system_total_cost_sgd_per_year",
    "annualized_pv_cost_sgd_per_year",
    "annualized_battery_cost_sgd_per_year",
    "battery_om_cost_sgd_per_year",
    "mean_grid_cost_sgd_per_year",
    "mean_unserved_energy_cost_sgd_per_year",
    "mean_UE_kwh_per_year",
    "mean_scenario_LOLP",
    "chance_constraint_probability",
)

REQUIRED_TRAJECTORY_COLUMNS = (
    "target_CCP",
    "available",
    "pv_capacity_kw",
    "pv_capacity_mw",
    "battery_energy_kwh",
    "battery_energy_mwh",
    "system_total_cost_million_sgd_per_year",
    "annualized_pv_cost_million_sgd_per_year",
    "annualized_battery_cost_million_sgd_per_year",
    "battery_om_cost_million_sgd_per_year",
    "grid_cost_million_sgd_per_year",
    "expected_outage_loss_million_sgd_per_year",
    "mean_UE_mwh_per_year",
    "energy_based_LOLP",
    "achieved_CCP",
    "is_knee_point",
)


def make_reliability_trajectory_publication_figure(
    out: Path,
    case_key: str,
    candidates: pd.DataFrame,
    design_space: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Create the four-panel reliability-driven PV-BESS sizing trajectory figure."""
    if _looks_like_trajectory_source(candidates):
        source = validate_reliability_trajectory_source(candidates)
    else:
        source = build_reliability_trajectory_source(candidates)
    base = out / f"{case_key}_Figure_4_Reliability_Driven_Sizing_Trajectory"
    paths: dict[str, Path] = {"source_data": _write_source_data(base, source)}
    background = design_space if design_space is not None else pd.DataFrame()
    paths.update(plot_reliability_trajectory(out, case_key, source, background))
    return paths


def build_reliability_trajectory_source(
    candidates: pd.DataFrame,
    targets: tuple[float, ...] = RELIABILITY_TARGETS,
) -> pd.DataFrame:
    """Select the minimum-cost feasible design for each target CCP."""
    missing = [column for column in REQUIRED_CANDIDATE_COLUMNS if column not in candidates.columns]
    if missing:
        raise ValueError("Reliability trajectory candidates are missing required column(s): " + ", ".join(missing))
    if not targets:
        raise ValueError("At least one reliability target is required.")

    prepared = candidates.copy()
    for column in REQUIRED_CANDIDATE_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if prepared[list(REQUIRED_CANDIDATE_COLUMNS)].isna().any().any():
        raise ValueError("Reliability trajectory candidates contain missing or non-numeric required values.")

    rows: list[dict[str, Any]] = []
    for target in targets:
        target_value = float(target)
        if not 0.0 <= target_value <= 1.0:
            raise ValueError("Reliability targets must be between 0 and 1.")
        feasible = prepared[prepared["chance_constraint_probability"].astype(float) >= target_value].copy()
        if feasible.empty:
            rows.append(_unavailable_target_row(target_value))
            continue
        selected = feasible.sort_values(
            ["system_total_cost_sgd_per_year", "pv_capacity_kw", "battery_energy_kwh"]
        ).iloc[0]
        rows.append(_available_target_row(target_value, selected))

    source = pd.DataFrame(rows)
    source["is_knee_point"] = False
    knee_index = _knee_point_index(source)
    if knee_index is not None:
        source.loc[knee_index, "is_knee_point"] = True
    return validate_reliability_trajectory_source(source)


def validate_reliability_trajectory_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_TRAJECTORY_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Reliability trajectory source is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Reliability trajectory source data is empty.")

    validated = source.copy()
    validated["available"] = validated["available"].map(_truthy)
    validated["is_knee_point"] = validated["is_knee_point"].map(_truthy)
    numeric = [column for column in REQUIRED_TRAJECTORY_COLUMNS if column not in {"available", "is_knee_point"}]
    for column in numeric:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated["target_CCP"].isna().any():
        raise ValueError("Reliability trajectory source contains missing target_CCP values.")
    available = validated["available"]
    value_columns = [column for column in numeric if column != "target_CCP"]
    if validated.loc[available, value_columns].isna().any().any():
        raise ValueError("Available reliability trajectory rows contain missing numeric values.")
    return validated.sort_values("target_CCP").reset_index(drop=True)


def plot_reliability_trajectory(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
    design_space: pd.DataFrame | None = None,
) -> dict[str, Path]:
    source = validate_reliability_trajectory_source(source)
    design_source = _validated_design_space_or_empty(design_space)
    mpl, plt = _matplotlib(out)
    _style(mpl)

    fig = plt.figure(figsize=(7.3, 5.4), constrained_layout=False)
    outer = fig.add_gridspec(2, 2, left=0.08, right=0.98, bottom=0.08, top=0.95, wspace=0.27, hspace=0.34)
    ax_a = fig.add_subplot(outer[0, 0])
    sub_b = outer[0, 1].subgridspec(2, 1, hspace=0.12)
    ax_b1 = fig.add_subplot(sub_b[0, 0])
    ax_b2 = fig.add_subplot(sub_b[1, 0], sharex=ax_b1)
    ax_c = fig.add_subplot(outer[1, 0])
    sub_d = outer[1, 1].subgridspec(3, 1, hspace=0.20)
    ax_d1 = fig.add_subplot(sub_d[0, 0])
    ax_d2 = fig.add_subplot(sub_d[1, 0], sharex=ax_d1)
    ax_d3 = fig.add_subplot(sub_d[2, 0], sharex=ax_d1)

    _plot_panel_a(ax_a, source, design_source)
    _plot_panel_b(ax_b1, ax_b2, source)
    _plot_panel_c(ax_c, source)
    _plot_panel_d(ax_d1, ax_d2, ax_d3, source)

    base = out / f"{case_key}_Figure_4_Reliability_Driven_Sizing_Trajectory"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _available_target_row(target: float, row: pd.Series) -> dict[str, Any]:
    return {
        "target_CCP": target,
        "available": True,
        "pv_capacity_kw": float(row["pv_capacity_kw"]),
        "pv_capacity_mw": float(row["pv_capacity_kw"]) / 1000.0,
        "battery_energy_kwh": float(row["battery_energy_kwh"]),
        "battery_energy_mwh": float(row["battery_energy_kwh"]) / 1000.0,
        "system_total_cost_million_sgd_per_year": float(row["system_total_cost_sgd_per_year"]) / 1e6,
        "annualized_pv_cost_million_sgd_per_year": float(row["annualized_pv_cost_sgd_per_year"]) / 1e6,
        "annualized_battery_cost_million_sgd_per_year": float(row["annualized_battery_cost_sgd_per_year"]) / 1e6,
        "battery_om_cost_million_sgd_per_year": float(row["battery_om_cost_sgd_per_year"]) / 1e6,
        "grid_cost_million_sgd_per_year": float(row["mean_grid_cost_sgd_per_year"]) / 1e6,
        "expected_outage_loss_million_sgd_per_year": float(row["mean_unserved_energy_cost_sgd_per_year"]) / 1e6,
        "mean_UE_mwh_per_year": float(row["mean_UE_kwh_per_year"]) / 1000.0,
        "energy_based_LOLP": float(row["mean_scenario_LOLP"]),
        "achieved_CCP": float(row["chance_constraint_probability"]),
    }


def _unavailable_target_row(target: float) -> dict[str, Any]:
    row = {column: np.nan for column in REQUIRED_TRAJECTORY_COLUMNS}
    row["target_CCP"] = target
    row["available"] = False
    row["is_knee_point"] = False
    return row


def _knee_point_index(source: pd.DataFrame) -> int | None:
    available = source[source["available"].map(_truthy)].copy()
    if len(available) < 3:
        return None
    x = available["system_total_cost_million_sgd_per_year"].to_numpy(dtype=float)
    y = 1.0 - available["achieved_CCP"].to_numpy(dtype=float)
    points = np.column_stack([_normalize(x), _normalize(y)])
    start = points[0]
    end = points[-1]
    span = end - start
    denom = float(np.linalg.norm(span))
    if np.isclose(denom, 0.0):
        return int(available.index[len(available) // 2])
    shifted = points - start
    distances = np.abs(span[0] * shifted[:, 1] - span[1] * shifted[:, 0]) / denom
    return int(available.index[int(np.argmax(distances))])


def _normalize(values: np.ndarray) -> np.ndarray:
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if np.isclose(low, high):
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def _plot_panel_a(ax: Any, source: pd.DataFrame, design_space: pd.DataFrame) -> None:
    if not design_space.empty:
        grid = _grid_values(design_space, "system_total_cost_million_sgd_per_year")
        if grid is not None:
            x, y, z = grid
            ax.contourf(x, y, z, levels=_levels(z, count=12), cmap="Blues", alpha=0.72)
            ax.contour(x, y, z, levels=_levels(z, count=6)[1:-1], colors="white", linewidths=0.35, alpha=0.72)
            ax.text(
                0.02,
                0.96,
                "background: annualized cost",
                transform=ax.transAxes,
                fontsize=5.8,
                color=NATURE_PALETTE["neutral_dark"],
                va="top",
            )

    available = source[source["available"]].copy()
    unavailable = source[~source["available"]].copy()
    if not available.empty:
        ax.plot(
            available["pv_capacity_mw"],
            available["battery_energy_mwh"],
            color=NATURE_PALETTE["neutral_dark"],
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            available["pv_capacity_mw"],
            available["battery_energy_mwh"],
            c=available["target_CCP"],
            cmap="YlGnBu",
            vmin=0.50,
            vmax=0.99,
            s=26,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
    if not unavailable.empty:
        x_fallback = _fallback_unavailable_x(source)
        y_fallback = _fallback_unavailable_y(source)
        ax.scatter(
            [x_fallback] * len(unavailable),
            [y_fallback] * len(unavailable),
            marker="x",
            color=NATURE_PALETTE["neutral_mid"],
            s=20,
            linewidth=0.8,
            zorder=5,
        )
    _mark_knee(ax, source, x_column="pv_capacity_mw", y_column="battery_energy_mwh")
    _label_key_targets(ax, source)
    ax.set_title("a  PV-BESS sizing trajectory", loc="left")
    ax.set_xlabel("PV capacity (MW)")
    ax.set_ylabel("BESS energy (MWh)")


def _plot_panel_b(ax_pv: Any, ax_bess: Any, source: pd.DataFrame) -> None:
    available = source["available"]
    x = source["target_CCP"] * 100.0
    ax_pv.plot(x[available], source.loc[available, "pv_capacity_mw"], color=NATURE_PALETTE["blue"], marker="o")
    ax_bess.plot(x[available], source.loc[available, "battery_energy_mwh"], color=NATURE_PALETTE["teal"], marker="o")
    _mark_knee(ax_pv, source.assign(target_percent=x), x_column="target_percent", y_column="pv_capacity_mw")
    _mark_knee(ax_bess, source.assign(target_percent=x), x_column="target_percent", y_column="battery_energy_mwh")
    ax_pv.set_title("b  Optimal capacity vs CCP", loc="left")
    ax_pv.set_ylabel("PV (MW)")
    ax_bess.set_ylabel("BESS (MWh)")
    ax_bess.set_xlabel("Target CCP (%)")
    ax_pv.tick_params(labelbottom=False)
    for ax in (ax_pv, ax_bess):
        ax.set_xlim(float(x.min()) - 2.0, float(x.max()) + 2.0)
        ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.55)


def _plot_panel_c(ax: Any, source: pd.DataFrame) -> None:
    available = source[source["available"]].copy()
    x = available["target_CCP"].to_numpy(dtype=float) * 100.0
    width = _bar_width(x)
    components = [
        ("PV", "annualized_pv_cost_million_sgd_per_year", NATURE_PALETTE["blue"]),
        ("BESS", "annualized_battery_cost_million_sgd_per_year", NATURE_PALETTE["teal"]),
        ("BESS O&M", "battery_om_cost_million_sgd_per_year", "#9CCED3"),
        ("Grid", "grid_cost_million_sgd_per_year", NATURE_PALETTE["neutral_light"]),
        ("Outage loss", "expected_outage_loss_million_sgd_per_year", NATURE_PALETTE["red"]),
    ]
    bottom = np.zeros(len(available), dtype=float)
    for label, column, color in components:
        values = available[column].to_numpy(dtype=float)
        ax.bar(x, values, width=width, bottom=bottom, color=color, edgecolor="white", linewidth=0.25, label=label)
        bottom += values
    ax.plot(
        x,
        available["system_total_cost_million_sgd_per_year"],
        color=NATURE_PALETTE["neutral_black"],
        marker="o",
        linewidth=0.9,
        label="Total",
    )
    ax.set_title("c  Cost composition vs CCP", loc="left")
    ax.set_xlabel("Target CCP (%)")
    ax.set_ylabel("Annualized cost\n(million SGD yr$^{-1}$)")
    ax.set_xlim(float(source["target_CCP"].min() * 100.0) - 2.0, float(source["target_CCP"].max() * 100.0) + 2.0)
    ax.legend(loc="upper left", ncol=2, fontsize=5.7, handlelength=1.0, columnspacing=0.8)


def _plot_panel_d(ax_ue: Any, ax_lolp: Any, ax_ccp: Any, source: pd.DataFrame) -> None:
    available = source["available"]
    x = source["target_CCP"] * 100.0
    ax_ue.plot(x[available], source.loc[available, "mean_UE_mwh_per_year"], color=NATURE_PALETTE["red"], marker="o")
    ax_lolp.plot(x[available], source.loc[available, "energy_based_LOLP"], color=NATURE_PALETTE["neutral_dark"], marker="o")
    ax_ccp.plot(x[available], source.loc[available, "achieved_CCP"], color=NATURE_PALETTE["teal"], marker="o")
    ax_ccp.plot(x, source["target_CCP"], color=NATURE_PALETTE["neutral_mid"], linestyle="--", linewidth=0.8)

    ax_ue.set_title("d  Achieved resilience vs CCP", loc="left")
    ax_ue.set_ylabel("UE\n(MWh yr$^{-1}$)")
    ax_lolp.set_ylabel("LOLP")
    ax_ccp.set_ylabel("CCP")
    ax_ccp.set_xlabel("Target CCP (%)")
    ax_ue.tick_params(labelbottom=False)
    ax_lolp.tick_params(labelbottom=False)
    for ax in (ax_ue, ax_lolp, ax_ccp):
        ax.set_xlim(float(x.min()) - 2.0, float(x.max()) + 2.0)
        ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.55)
    ax_ccp.text(
        0.02,
        0.12,
        "dashed: target",
        transform=ax_ccp.transAxes,
        fontsize=5.7,
        color=NATURE_PALETTE["neutral_mid"],
    )


def _mark_knee(ax: Any, source: pd.DataFrame, x_column: str, y_column: str) -> None:
    knee = source[source["is_knee_point"] & source["available"]]
    if knee.empty:
        return
    ax.scatter(
        knee[x_column],
        knee[y_column],
        s=42,
        facecolor="none",
        edgecolor=NATURE_PALETTE["red"],
        linewidth=1.2,
        zorder=6,
    )


def _label_key_targets(ax: Any, source: pd.DataFrame) -> None:
    labels = {0.50: "50%", 0.90: "90%", 0.99: "99%"}
    for target, label in labels.items():
        rows = source[np.isclose(source["target_CCP"], target) & source["available"]]
        if rows.empty:
            continue
        row = rows.iloc[0]
        ax.annotate(
            label,
            (float(row["pv_capacity_mw"]), float(row["battery_energy_mwh"])),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.8,
            color=NATURE_PALETTE["neutral_black"],
        )
    knee = source[source["is_knee_point"] & source["available"]]
    if not knee.empty:
        row = knee.iloc[0]
        ax.annotate(
            "knee",
            (float(row["pv_capacity_mw"]), float(row["battery_energy_mwh"])),
            xytext=(4, -8),
            textcoords="offset points",
            fontsize=5.8,
            color=NATURE_PALETTE["red"],
        )


def _bar_width(x: np.ndarray) -> float:
    if len(x) < 2:
        return 4.0
    diffs = np.diff(np.sort(x))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 4.0
    return float(np.min(positive) * 0.72)


def _fallback_unavailable_x(source: pd.DataFrame) -> float:
    available = source[source["available"]]
    if available.empty:
        return 0.0
    return float(available["pv_capacity_mw"].max())


def _fallback_unavailable_y(source: pd.DataFrame) -> float:
    available = source[source["available"]]
    if available.empty:
        return 0.0
    return float(available["battery_energy_mwh"].max())


def _validated_design_space_or_empty(design_space: pd.DataFrame | None) -> pd.DataFrame:
    if design_space is None or design_space.empty:
        return pd.DataFrame()
    try:
        return validate_design_space_source(design_space)
    except ValueError:
        return pd.DataFrame()


def validate_design_space_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_DESIGN_SPACE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Reliability trajectory background is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Reliability trajectory background source data is empty.")
    validated = source.copy()
    for column in REQUIRED_DESIGN_SPACE_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(REQUIRED_DESIGN_SPACE_COLUMNS)].isna().any().any():
        raise ValueError("Reliability trajectory background contains missing or non-numeric values.")
    return validated.sort_values(["pv_capacity_mw", "battery_energy_mwh"]).reset_index(drop=True)


def _grid_values(source: pd.DataFrame, value_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pivot = source.pivot_table(
        index="battery_energy_mwh",
        columns="pv_capacity_mw",
        values=value_column,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    if pivot.shape[0] < 2 or pivot.shape[1] < 2 or pivot.isna().any().any():
        return None
    x, y = np.meshgrid(pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float))
    z = pivot.to_numpy(dtype=float)
    return x, y, z


def _levels(values: np.ndarray, count: int = 12) -> np.ndarray:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot plot a metric with no finite values.")
    low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if np.isclose(low, high):
        return np.linspace(low - 0.5, high + 0.5, count)
    return np.linspace(low, high, count)


def _looks_like_trajectory_source(source: pd.DataFrame) -> bool:
    return bool(set(REQUIRED_TRAJECTORY_COLUMNS).issubset(set(source.columns)))


def _truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "available"}
    return bool(value)
