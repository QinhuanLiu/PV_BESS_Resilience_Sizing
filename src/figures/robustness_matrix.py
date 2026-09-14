from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .core import NATURE_PALETTE, matplotlib_context, apply_nature_style, save_publication_figure, write_source_data


DEFAULT_DESIGN_TARGETS = (0.50, 0.70, 0.80, 0.90, 0.95)
DEFAULT_ACTUAL_QUANTILES = (0.50, 0.70, 0.80, 0.90, 0.95)
DEFAULT_LOLP_THRESHOLD = 0.10
FIGURE_ROBUSTNESS_NAME = "Figure_9_Design_Realization_Robustness_Matrix"
REQUIRED_CANDIDATE_COLUMNS = (
    "pv_capacity_kw",
    "battery_energy_kwh",
    "system_total_cost_sgd_per_year",
    "chance_constraint_probability",
)
REQUIRED_SCENARIO_COLUMNS = (
    "frontier_scenario_id",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "outage_weighted_load_kwh",
    "UE_kwh_per_year",
    "scenario_LOLP",
)
REQUIRED_MATRIX_COLUMNS = (
    "design_target_CCP",
    "actual_severity_quantile",
    "pv_capacity_mw",
    "battery_energy_mwh",
    "mean_UE_mwh_per_year",
    "achieved_CCP",
    "failure_probability",
    "scenario_count",
)


def make_robustness_matrix_publication_figure(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
) -> dict[str, Path]:
    validated = validate_robustness_matrix_source(source)
    base = out / f"{case_key}_{FIGURE_ROBUSTNESS_NAME}"
    paths: dict[str, Path] = {"source_data": write_source_data(base, validated)}
    paths.update(plot_robustness_matrix(out, case_key, validated))
    return paths


def build_robustness_matrix_source(
    candidates: pd.DataFrame,
    scenario_results: pd.DataFrame,
    design_targets: Sequence[float] = DEFAULT_DESIGN_TARGETS,
    actual_quantiles: Sequence[float] = DEFAULT_ACTUAL_QUANTILES,
    lolp_threshold: float = DEFAULT_LOLP_THRESHOLD,
) -> pd.DataFrame:
    _require_columns(candidates, REQUIRED_CANDIDATE_COLUMNS, "robustness candidates")
    _require_columns(scenario_results, REQUIRED_SCENARIO_COLUMNS, "robustness scenario results")
    _validate_probabilities(tuple(float(value) for value in design_targets), "design_targets")
    _validate_probabilities(tuple(float(value) for value in actual_quantiles), "actual_quantiles")
    prepared_candidates = _numeric_copy(candidates, REQUIRED_CANDIDATE_COLUMNS)
    prepared_scenarios = _numeric_copy(scenario_results, REQUIRED_SCENARIO_COLUMNS)

    rows: list[dict[str, Any]] = []
    for target in design_targets:
        design = _minimum_cost_design_for_target(prepared_candidates, float(target))
        matches = prepared_scenarios[
            np.isclose(prepared_scenarios["pv_capacity_kw"], float(design["pv_capacity_kw"]))
            & np.isclose(prepared_scenarios["battery_energy_kwh"], float(design["battery_energy_kwh"]))
        ].copy()
        if matches.empty:
            continue
        for actual_quantile in actual_quantiles:
            severity_limit = float(matches["outage_weighted_load_kwh"].quantile(float(actual_quantile)))
            actual = matches[matches["outage_weighted_load_kwh"].astype(float) <= severity_limit].copy()
            if actual.empty:
                actual = matches.nsmallest(1, "outage_weighted_load_kwh").copy()
            scenario_lolp = actual["scenario_LOLP"].astype(float)
            rows.append(
                {
                    "design_target_CCP": float(target),
                    "actual_severity_quantile": float(actual_quantile),
                    "lolp_threshold": float(lolp_threshold),
                    "pv_capacity_kw": float(design["pv_capacity_kw"]),
                    "pv_capacity_mw": float(design["pv_capacity_kw"]) / 1000.0,
                    "battery_energy_kwh": float(design["battery_energy_kwh"]),
                    "battery_energy_mwh": float(design["battery_energy_kwh"]) / 1000.0,
                    "system_total_cost_sgd_per_year": float(design["system_total_cost_sgd_per_year"]),
                    "system_total_cost_million_sgd_per_year": float(design["system_total_cost_sgd_per_year"]) / 1e6,
                    "mean_UE_kwh_per_year": float(actual["UE_kwh_per_year"].astype(float).mean()),
                    "mean_UE_mwh_per_year": float(actual["UE_kwh_per_year"].astype(float).mean()) / 1000.0,
                    "achieved_CCP": float((scenario_lolp <= float(lolp_threshold)).mean()),
                    "failure_probability": float((scenario_lolp > float(lolp_threshold)).mean()),
                    "scenario_count": int(len(actual)),
                    "actual_severity_limit_kwh": severity_limit,
                }
            )
    return validate_robustness_matrix_source(pd.DataFrame(rows))


def example_robustness_matrix_source() -> pd.DataFrame:
    candidates = pd.DataFrame(
        [
            {"pv_capacity_kw": 1000.0, "battery_energy_kwh": 1000.0, "system_total_cost_sgd_per_year": 1_000_000.0, "chance_constraint_probability": 0.50},
            {"pv_capacity_kw": 1500.0, "battery_energy_kwh": 2000.0, "system_total_cost_sgd_per_year": 1_150_000.0, "chance_constraint_probability": 0.75},
            {"pv_capacity_kw": 2200.0, "battery_energy_kwh": 3400.0, "system_total_cost_sgd_per_year": 1_420_000.0, "chance_constraint_probability": 0.92},
            {"pv_capacity_kw": 3000.0, "battery_energy_kwh": 5000.0, "system_total_cost_sgd_per_year": 1_800_000.0, "chance_constraint_probability": 0.97},
        ]
    )
    rows: list[dict[str, float]] = []
    designs = [(1000.0, 1000.0, 1.00), (1500.0, 2000.0, 0.72), (2200.0, 3400.0, 0.40), (3000.0, 5000.0, 0.18)]
    for scenario_id, load in enumerate([1000.0, 1200.0, 1500.0, 1800.0, 2200.0, 2600.0], start=1):
        for pv_kw, bess_kwh, risk_factor in designs:
            ue = max(0.0, load * risk_factor - bess_kwh * 0.14 - pv_kw * 0.035)
            rows.append(
                {
                    "frontier_scenario_id": float(scenario_id),
                    "pv_capacity_kw": pv_kw,
                    "battery_energy_kwh": bess_kwh,
                    "outage_weighted_load_kwh": load,
                    "UE_kwh_per_year": ue,
                    "scenario_LOLP": ue / load,
                }
            )
    return build_robustness_matrix_source(candidates, pd.DataFrame(rows))


def validate_robustness_matrix_source(source: pd.DataFrame) -> pd.DataFrame:
    _require_columns(source, REQUIRED_MATRIX_COLUMNS, "robustness matrix source")
    if source.empty:
        raise ValueError("Robustness matrix source data is empty.")
    validated = source.copy()
    for column in REQUIRED_MATRIX_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[list(REQUIRED_MATRIX_COLUMNS)].isna().any().any():
        raise ValueError("Robustness matrix source contains missing or non-numeric required values.")
    for column in ("design_target_CCP", "actual_severity_quantile", "achieved_CCP", "failure_probability"):
        if not validated[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} must be between 0 and 1.")
    if (validated["scenario_count"] < 1).any():
        raise ValueError("Robustness matrix source scenario_count must be at least 1.")
    return validated.sort_values(["design_target_CCP", "actual_severity_quantile"]).reset_index(drop=True)


def plot_robustness_matrix(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    source = validate_robustness_matrix_source(source)
    mpl, plt = matplotlib_context(out)
    apply_nature_style(mpl)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), constrained_layout=True)
    _plot_capacity_panel(axes[0], source)
    _plot_matrix_panel(axes[1], source, "achieved_CCP", "Achieved CCP", "YlGnBu")
    _plot_matrix_panel(axes[2], source, "failure_probability", "Failure region map", "Reds")
    base = out / f"{case_key}_{FIGURE_ROBUSTNESS_NAME}"
    paths = save_publication_figure(fig, base)
    plt.close(fig)
    return paths


def _plot_capacity_panel(ax: Any, source: pd.DataFrame) -> None:
    designs = source.drop_duplicates("design_target_CCP").sort_values("design_target_CCP")
    x = designs["design_target_CCP"].astype(float).to_numpy()
    ax.plot(x, designs["pv_capacity_mw"], color=NATURE_PALETTE["blue"], marker="o", label="PV")
    ax.plot(x, designs["battery_energy_mwh"], color=NATURE_PALETTE["teal"], marker="o", label="BESS")
    ax.set_title("a  Design capacity", loc="left")
    ax.set_xlabel("Design target CCP")
    ax.set_ylabel("Capacity (MW / MWh)")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.60)


def _plot_matrix_panel(ax: Any, source: pd.DataFrame, value_column: str, title: str, cmap: str) -> None:
    matrix, x_labels, y_labels = _matrix_table(source, value_column)
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
    ax.set_title(("b  " if value_column == "achieved_CCP" else "c  ") + title, loc="left")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Actual severity quantile")
    ax.set_ylabel("Design target CCP")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = float(matrix[y, x])
            ax.text(x, y, f"{value:.0%}", ha="center", va="center", fontsize=6.0, color="white" if value > 0.55 else NATURE_PALETTE["neutral_black"])
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["0%", "50%", "100%"])


def _matrix_table(source: pd.DataFrame, value_column: str) -> tuple[np.ndarray, list[str], list[str]]:
    table = source.pivot_table(
        index="design_target_CCP",
        columns="actual_severity_quantile",
        values=value_column,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    return (
        table.to_numpy(dtype=float),
        [f"{value:.0%}" for value in table.columns.to_numpy(dtype=float)],
        [f"{value:.0%}" for value in table.index.to_numpy(dtype=float)],
    )


def _minimum_cost_design_for_target(candidates: pd.DataFrame, design_target: float) -> pd.Series:
    feasible = candidates[candidates["chance_constraint_probability"].astype(float) >= float(design_target)].copy()
    if feasible.empty:
        feasible = candidates.copy()
    return feasible.sort_values(["system_total_cost_sgd_per_year", "pv_capacity_kw", "battery_energy_kwh"]).iloc[0]


def _require_columns(source: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s): " + ", ".join(missing))


def _numeric_copy(source: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = source.copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(columns)].isna().any().any():
        raise ValueError("Robustness input contains missing or non-numeric required values.")
    return frame


def _validate_probabilities(values: tuple[float, ...], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must contain at least one value")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{label} values must be between 0 and 1")
