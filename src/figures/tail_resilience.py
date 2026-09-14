from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import NATURE_PALETTE, _matplotlib, _save_all, _style, _write_source_data


FIGURE_TAIL_RESILIENCE_NAME = "Figure_11_Mean_Tail_Resilience_Distribution"
DEFAULT_TAIL_DESIGN_TYPES = ("minimum_total_cost_design", "reliability_constrained_design")

DESIGN_TYPE_LABELS = {
    "minimum_total_cost_design": "Economic optimum",
    "reliability_constrained_design": "Reliability-constrained",
    "reserve_aware_design": "Reserve-aware",
}

REQUIRED_SCENARIO_COLUMNS = (
    "frontier_scenario_id",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "outage_weighted_load_kwh",
    "UE_kwh_per_year",
    "scenario_LOLP",
    "outage_load_served_ratio",
)

REQUIRED_DESIGN_COLUMNS = (
    "design_type",
    "pv_capacity_kw",
    "battery_energy_kwh",
)

REQUIRED_TAIL_SOURCE_COLUMNS = (
    "frontier_scenario_id",
    "design_case",
    "design_case_label",
    "design_order",
    "pv_capacity_kw",
    "battery_energy_kwh",
    "outage_weighted_load_kwh",
    "UE_kwh_per_year",
    "scenario_LOLP",
    "outage_load_served_ratio",
    "lolp_threshold",
    "scenario_probability",
)


def build_tail_resilience_source(
    scenario_results: pd.DataFrame,
    optimal_designs: pd.DataFrame,
    lolp_threshold: float | None = None,
) -> pd.DataFrame:
    """Return per-scenario source rows for the selected result-design tail-risk comparison."""
    _require_columns(scenario_results, REQUIRED_SCENARIO_COLUMNS, "tail resilience scenario results")
    _require_columns(optimal_designs, REQUIRED_DESIGN_COLUMNS, "tail resilience optimal designs")
    if scenario_results.empty:
        raise ValueError("Tail resilience scenario results are empty.")
    if optimal_designs.empty:
        raise ValueError("Tail resilience optimal designs are empty.")

    designs = _selected_designs(optimal_designs)
    rows: list[pd.DataFrame] = []
    for order, design in enumerate(designs.to_dict("records"), start=1):
        matches = _matching_scenario_rows(scenario_results, design)
        if matches.empty:
            continue
        part = matches.copy()
        design_case = str(design["design_type"])
        part["design_case"] = design_case
        part["design_case_label"] = DESIGN_TYPE_LABELS.get(design_case, _humanize_design_label(design_case))
        part["design_order"] = order
        threshold = lolp_threshold
        if threshold is None:
            threshold = _design_lolp_threshold(design, part)
        part["lolp_threshold"] = float(threshold)
        if "scenario_probability" not in part.columns:
            part["scenario_probability"] = 1.0 / len(part)
        rows.append(part)

    if not rows:
        raise ValueError("No matching scenario rows were found for the selected tail resilience designs.")
    return validate_tail_resilience_source(pd.concat(rows, ignore_index=True))


def validate_tail_resilience_source(source: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize Fig. 11 per-scenario source data."""
    missing = [column for column in REQUIRED_TAIL_SOURCE_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError("Tail resilience source is missing required column(s): " + ", ".join(missing))
    if source.empty:
        raise ValueError("Tail resilience source data is empty.")

    validated = source.copy()
    numeric = [
        "frontier_scenario_id",
        "design_order",
        "pv_capacity_kw",
        "battery_energy_kwh",
        "outage_weighted_load_kwh",
        "UE_kwh_per_year",
        "scenario_LOLP",
        "outage_load_served_ratio",
        "lolp_threshold",
        "scenario_probability",
    ]
    for column in numeric:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[numeric].isna().any().any():
        raise ValueError("Tail resilience source contains missing or non-numeric required values.")
    if (validated["outage_weighted_load_kwh"] < 0).any():
        raise ValueError("Tail resilience source contains negative outage-weighted load.")
    if (validated["UE_kwh_per_year"] < 0).any():
        raise ValueError("Tail resilience source contains negative UE.")
    for column in ("scenario_LOLP", "outage_load_served_ratio", "lolp_threshold"):
        if not validated[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} must be between 0 and 1.")
    if (validated["scenario_probability"] < 0).any():
        raise ValueError("Tail resilience source contains negative scenario probabilities.")

    normalized_groups: list[pd.DataFrame] = []
    for _, group in validated.groupby("design_case", sort=False):
        probability_sum = float(group["scenario_probability"].sum())
        if probability_sum <= 0.0:
            raise ValueError("Scenario probabilities must sum to a positive value for each design.")
        adjusted = group.copy()
        adjusted["scenario_probability"] = adjusted["scenario_probability"] / probability_sum
        normalized_groups.append(adjusted)
    validated = pd.concat(normalized_groups, ignore_index=True)
    validated["design_case"] = validated["design_case"].astype(str)
    validated["design_case_label"] = validated["design_case_label"].astype(str)
    return validated.sort_values(["design_order", "frontier_scenario_id"]).reset_index(drop=True)


def build_tail_resilience_plot_source(source: pd.DataFrame) -> pd.DataFrame:
    """Add plotting, tail-statistic, and failure-classification fields to per-scenario rows."""
    prepared = validate_tail_resilience_source(source)
    derived_columns = [
        "UE_mwh_per_year",
        "normalized_UE_fraction",
        "load_cover_ratio",
        "target_failure_flag",
        "any_unserved_flag",
        "mean_UE_mwh_per_year",
        "median_UE_mwh_per_year",
        "p90_UE_mwh_per_year",
        "p95_UE_mwh_per_year",
        "worst_UE_mwh_per_year",
        "mean_load_cover_ratio",
        "p10_load_cover_ratio",
        "p05_load_cover_ratio",
        "worst_load_cover_ratio",
        "target_failure_probability",
        "any_unserved_probability",
    ]
    prepared = prepared.drop(columns=[column for column in derived_columns if column in prepared.columns])
    load = prepared["outage_weighted_load_kwh"].replace(0.0, np.nan)
    prepared["UE_mwh_per_year"] = prepared["UE_kwh_per_year"] / 1000.0
    prepared["normalized_UE_fraction"] = (prepared["UE_kwh_per_year"] / load).fillna(0.0)
    prepared["load_cover_ratio"] = prepared["outage_load_served_ratio"].fillna(1.0 - prepared["normalized_UE_fraction"])
    prepared["load_cover_ratio"] = prepared["load_cover_ratio"].clip(0.0, 1.0)
    prepared["target_failure_flag"] = prepared["scenario_LOLP"] > prepared["lolp_threshold"]
    prepared["any_unserved_flag"] = prepared["UE_kwh_per_year"] > 1e-9

    stats = _tail_statistics(prepared)
    prepared = prepared.merge(stats, on=["design_case", "design_case_label", "design_order"], how="left")
    return prepared.sort_values(["design_order", "frontier_scenario_id"]).reset_index(drop=True)


def example_tail_resilience_source() -> pd.DataFrame:
    """Return deterministic synthetic data for visual style review and tests."""
    rng = np.random.default_rng(1107)
    designs = [
        ("minimum_total_cost_design", "Economic optimum", 10.0, 0.52),
        ("reliability_constrained_design", "Reliability-constrained", 4.0, 0.26),
        ("reserve_aware_design", "Reserve-aware", 2.2, 0.18),
    ]
    severity = np.linspace(0.18, 1.0, 40)
    rows: list[dict[str, Any]] = []
    for order, (design_case, label, scale, lolp_scale) in enumerate(designs, start=1):
        noise = rng.normal(0.0, 0.28, size=len(severity))
        tail_bump = np.clip(severity - 0.78, 0.0, None) ** 2 * scale * 6.0
        ue_mwh = np.maximum(0.0, scale * severity**2.3 + tail_bump + noise)
        if design_case == "reserve_aware_design":
            ue_mwh *= np.where(severity > 0.88, 0.56, 0.74)
        outage_load_mwh = 22.0 + severity * 36.0
        for scenario_id, (ue, load) in enumerate(zip(ue_mwh, outage_load_mwh, strict=True), start=1):
            scenario_lolp = min(1.0, max(0.0, ue / load * (1.0 + lolp_scale)))
            rows.append(
                {
                    "frontier_scenario_id": scenario_id,
                    "design_case": design_case,
                    "design_case_label": label,
                    "design_order": order,
                    "pv_capacity_kw": 1800.0 + 400.0 * order,
                    "battery_energy_kwh": 1600.0 + 1200.0 * order,
                    "outage_weighted_load_kwh": load * 1000.0,
                    "UE_kwh_per_year": ue * 1000.0,
                    "scenario_LOLP": scenario_lolp,
                    "outage_load_served_ratio": 1.0 - min(1.0, max(0.0, ue / load)),
                    "lolp_threshold": 0.10,
                    "scenario_probability": 1.0 / len(severity),
                }
            )
    return validate_tail_resilience_source(pd.DataFrame(rows))


def make_tail_resilience_publication_figure(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
) -> dict[str, Path]:
    """Create the Nature-style mean and tail resilience distribution figure."""
    prepared = build_tail_resilience_plot_source(source)
    base = out / f"{case_key}_{FIGURE_TAIL_RESILIENCE_NAME}"
    paths: dict[str, Path] = {"source_data": _write_source_data(base, prepared)}
    paths.update(plot_tail_resilience_distribution(out, case_key, prepared))
    return paths


def make_demo_tail_resilience_figure(out: Path, case_key: str = "Demo") -> dict[str, Path]:
    """Generate a self-contained style mockup from deterministic synthetic data."""
    return make_tail_resilience_publication_figure(out, case_key, example_tail_resilience_source())


def plot_tail_resilience_distribution(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    source = build_tail_resilience_plot_source(source)
    mpl, plt = _matplotlib(out)
    _style(mpl)

    fig = plt.figure(figsize=(7.3, 5.6), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.08,
        right=0.97,
        bottom=0.10,
        top=0.95,
        wspace=0.46,
        hspace=0.42,
        width_ratios=[1.35, 1.0, 1.0],
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1:])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])

    palette = _design_palette(source)
    _plot_ue_distribution(ax_a, source, palette)
    _plot_exceedance_curve(ax_b, source, palette)
    _plot_tail_stat_lollipop(ax_c, source, palette)
    _plot_lcr_low_tail(ax_d, source, palette)
    for label, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d), strict=True):
        _panel_label(ax, label)

    base = out / f"{case_key}_{FIGURE_TAIL_RESILIENCE_NAME}"
    paths = _save_all(fig, base)
    plt.close(fig)
    return paths


def _selected_designs(optimal_designs: pd.DataFrame) -> pd.DataFrame:
    designs = optimal_designs.copy()
    if "available" in designs.columns:
        designs = designs[designs["available"].map(_truthy)].copy()
    selected = designs[designs["design_type"].astype(str).isin(DEFAULT_TAIL_DESIGN_TYPES)].copy()
    if selected.empty:
        selected = designs.copy()
    if selected.empty:
        raise ValueError("No available optimal designs were found for tail resilience plotting.")
    selected["_order"] = selected["design_type"].astype(str).map(
        {design_type: index for index, design_type in enumerate(DEFAULT_TAIL_DESIGN_TYPES)}
    ).fillna(len(DEFAULT_TAIL_DESIGN_TYPES))
    return selected.sort_values(["_order", "design_type"]).drop(columns="_order").reset_index(drop=True)


def _matching_scenario_rows(scenario_results: pd.DataFrame, design: dict[str, Any]) -> pd.DataFrame:
    pv = float(design["pv_capacity_kw"])
    bess = float(design["battery_energy_kwh"])
    mask = (
        np.isclose(pd.to_numeric(scenario_results["pv_capacity_kw"], errors="coerce"), pv)
        & np.isclose(pd.to_numeric(scenario_results["battery_energy_kwh"], errors="coerce"), bess)
    )
    return scenario_results.loc[mask].copy()


def _design_lolp_threshold(design: dict[str, Any], matches: pd.DataFrame) -> float:
    if "lolp_threshold" in design and pd.notna(design["lolp_threshold"]):
        return float(design["lolp_threshold"])
    if "lolp_threshold" in matches.columns and matches["lolp_threshold"].notna().any():
        return float(pd.to_numeric(matches["lolp_threshold"], errors="coerce").dropna().iloc[0])
    return 0.10


def _tail_statistics(prepared: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in prepared.groupby(["design_case", "design_case_label", "design_order"], sort=True):
        design_case, label, order = keys
        ue = group["UE_mwh_per_year"].to_numpy(dtype=float)
        lcr = group["load_cover_ratio"].to_numpy(dtype=float)
        weights = group["scenario_probability"].to_numpy(dtype=float)
        rows.append(
            {
                "design_case": design_case,
                "design_case_label": label,
                "design_order": int(order),
                "mean_UE_mwh_per_year": _weighted_mean(ue, weights),
                "median_UE_mwh_per_year": _weighted_quantile(ue, weights, 0.50),
                "p90_UE_mwh_per_year": _weighted_quantile(ue, weights, 0.90),
                "p95_UE_mwh_per_year": _weighted_quantile(ue, weights, 0.95),
                "worst_UE_mwh_per_year": float(np.max(ue)),
                "mean_load_cover_ratio": _weighted_mean(lcr, weights),
                "p10_load_cover_ratio": _weighted_quantile(lcr, weights, 0.10),
                "p05_load_cover_ratio": _weighted_quantile(lcr, weights, 0.05),
                "worst_load_cover_ratio": float(np.min(lcr)),
                "target_failure_probability": _weighted_mean(
                    group["target_failure_flag"].astype(float).to_numpy(),
                    weights,
                ),
                "any_unserved_probability": _weighted_mean(group["any_unserved_flag"].astype(float).to_numpy(), weights),
            }
        )
    return pd.DataFrame(rows)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value.")
    return float(np.sum(values * weights) / total)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be between 0 and 1.")
    sorter = np.argsort(values)
    sorted_values = values[sorter]
    sorted_weights = weights[sorter]
    cumulative = np.cumsum(sorted_weights)
    total = float(cumulative[-1])
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value.")
    return float(sorted_values[np.searchsorted(cumulative / total, float(quantile), side="left")])


def _plot_ue_distribution(ax: Any, source: pd.DataFrame, palette: dict[str, str]) -> None:
    groups = _ordered_groups(source)
    positions = np.arange(1, len(groups) + 1)
    data = [group["UE_mwh_per_year"].to_numpy(dtype=float) for _, group in groups]
    violins = ax.violinplot(data, positions=positions, showmeans=False, showmedians=False, widths=0.72)
    for body, (design_case, _) in zip(violins["bodies"], groups, strict=True):
        body.set_facecolor(palette[design_case])
        body.set_edgecolor(NATURE_PALETTE["neutral_black"])
        body.set_alpha(0.30)
        body.set_linewidth(0.65)
    for key in ("cbars", "cmins", "cmaxes"):
        violins[key].set_color(NATURE_PALETTE["neutral_mid"])
        violins[key].set_linewidth(0.55)

    rng = np.random.default_rng(11)
    for position, (design_case, group) in zip(positions, groups, strict=True):
        y = group["UE_mwh_per_year"].to_numpy(dtype=float)
        x = position + rng.uniform(-0.18, 0.18, size=len(group))
        colors = np.where(group["target_failure_flag"].to_numpy(dtype=bool), NATURE_PALETTE["red"], palette[design_case])
        ax.scatter(x, y, s=8, c=colors, alpha=0.42, linewidth=0, zorder=3)
        stats = _stats_row(group)
        ax.scatter(position, stats["mean_UE_mwh_per_year"], marker="D", s=23, color=NATURE_PALETTE["neutral_black"], zorder=5)
        ax.hlines(
            [stats["median_UE_mwh_per_year"], stats["p90_UE_mwh_per_year"], stats["p95_UE_mwh_per_year"]],
            position - 0.28,
            position + 0.28,
            colors=[NATURE_PALETTE["neutral_black"], NATURE_PALETTE["red"], NATURE_PALETTE["red"]],
            linestyles=["-", "--", ":"],
            linewidths=[0.9, 0.8, 0.8],
            zorder=4,
        )
    ax.set_title("Scenario-level UE distribution", loc="left")
    ax.set_ylabel("UE (MWh yr$^{-1}$)")
    ax.set_xticks(positions)
    ax.set_xticklabels([_wrap_label(group["design_case_label"].iloc[0]) for _, group in groups], rotation=25, ha="right")
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.65)
    ax.text(
        0.02,
        0.98,
        "diamond, mean\nsolid, median\ndashed/dotted, P90/P95",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color=NATURE_PALETTE["neutral_dark"],
    )


def _plot_exceedance_curve(ax: Any, source: pd.DataFrame, palette: dict[str, str]) -> None:
    for design_case, group in _ordered_groups(source):
        x, y = _exceedance_points(group["UE_mwh_per_year"].to_numpy(dtype=float), group["scenario_probability"].to_numpy(dtype=float))
        ax.step(x, y, where="post", color=palette[design_case], linewidth=1.25, label=str(group["design_case_label"].iloc[0]))
    ax.set_title("Tail exceedance probability", loc="left")
    ax.set_xlabel("UE threshold (MWh yr$^{-1}$)")
    ax.set_ylabel("P(UE > threshold)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.65)
    ax.legend(loc="upper right", handlelength=1.8)


def _plot_tail_stat_lollipop(ax: Any, source: pd.DataFrame, palette: dict[str, str]) -> None:
    stats = _unique_stats(source)
    metrics = [
        ("mean_UE_mwh_per_year", "Mean"),
        ("p90_UE_mwh_per_year", "P90"),
        ("p95_UE_mwh_per_year", "P95"),
        ("worst_UE_mwh_per_year", "Worst"),
    ]
    y_positions = np.arange(len(metrics))[::-1]
    max_value = float(stats[[metric for metric, _ in metrics]].to_numpy(dtype=float).max())
    for design_index, row in enumerate(stats.to_dict("records")):
        offset = (design_index - (len(stats) - 1) / 2.0) * 0.16
        color = palette[str(row["design_case"])]
        values = [float(row[metric]) for metric, _ in metrics]
        ax.hlines(y_positions + offset, 0.0, values, color=color, alpha=0.72, linewidth=1.0)
        ax.scatter(values, y_positions + offset, color=color, edgecolor=NATURE_PALETTE["neutral_black"], linewidth=0.25, s=18, zorder=3)
    ax.set_title("Mean versus tail UE", loc="left")
    ax.set_xlabel("UE (MWh yr$^{-1}$)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in metrics])
    ax.set_xlim(0.0, max_value * 1.12 if max_value > 0 else 1.0)
    ax.grid(axis="x", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.65)


def _plot_lcr_low_tail(ax: Any, source: pd.DataFrame, palette: dict[str, str]) -> None:
    stats = _unique_stats(source)
    metrics = [
        ("mean_load_cover_ratio", "Mean"),
        ("p10_load_cover_ratio", "P10"),
        ("p05_load_cover_ratio", "P5"),
        ("worst_load_cover_ratio", "Worst"),
    ]
    x = np.arange(len(stats))
    width = 0.17
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(metrics))
    colors = [NATURE_PALETTE["neutral_mid"], NATURE_PALETTE["blue"], NATURE_PALETTE["teal"], NATURE_PALETTE["red"]]
    for offset, (metric, label), color in zip(offsets, metrics, colors, strict=True):
        ax.bar(x + offset, stats[metric].to_numpy(dtype=float) * 100.0, width=width, color=color, alpha=0.86, label=label)
    ax.set_title("Low-tail load cover ratio", loc="left")
    ax.set_ylabel("Load covered (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(label) for label in stats["design_case_label"]], rotation=25, ha="right")
    ax.set_ylim(0.0, 104.0)
    ax.grid(axis="y", color=NATURE_PALETTE["neutral_light"], linewidth=0.35, alpha=0.65)
    ax.legend(loc="lower right", ncol=1, handlelength=1.2)


def _exceedance_points(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = float(np.sum(sorted_weights))
    exceedance = 1.0 - np.cumsum(sorted_weights) / total
    x = np.r_[0.0, sorted_values]
    y = np.r_[1.0, exceedance]
    return x, y


def _ordered_groups(source: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups: list[tuple[str, pd.DataFrame]] = []
    for design_case, group in source.sort_values("design_order").groupby("design_case", sort=False):
        groups.append((str(design_case), group.copy()))
    return groups


def _unique_stats(source: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "design_case",
        "design_case_label",
        "design_order",
        "mean_UE_mwh_per_year",
        "median_UE_mwh_per_year",
        "p90_UE_mwh_per_year",
        "p95_UE_mwh_per_year",
        "worst_UE_mwh_per_year",
        "mean_load_cover_ratio",
        "p10_load_cover_ratio",
        "p05_load_cover_ratio",
        "worst_load_cover_ratio",
        "target_failure_probability",
    ]
    return source[columns].drop_duplicates("design_case").sort_values("design_order").reset_index(drop=True)


def _stats_row(group: pd.DataFrame) -> pd.Series:
    return _unique_stats(group).iloc[0]


def _design_palette(source: pd.DataFrame) -> dict[str, str]:
    default_colors = [
        "#5B4B76",
        NATURE_PALETTE["blue"],
        NATURE_PALETTE["teal"],
        NATURE_PALETTE["neutral_dark"],
    ]
    palette: dict[str, str] = {}
    for index, design_case in enumerate(_unique_stats(source)["design_case"].astype(str).tolist()):
        palette[design_case] = default_colors[index % len(default_colors)]
    return palette


def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _wrap_label(label: str) -> str:
    return str(label).replace(" ", "\n", 1)


def _short_label(label: str) -> str:
    words = str(label).replace("-", " ").split()
    if len(words) <= 1:
        return str(label)
    return "".join(word[0].upper() for word in words[:2])


def _humanize_design_label(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "available"}
    return bool(value)


def _require_columns(source: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s): " + ", ".join(missing))
