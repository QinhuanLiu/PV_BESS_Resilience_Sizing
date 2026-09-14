from __future__ import annotations

from pathlib import Path

import pandas as pd

from .names import (
    PARAMETER_LABELS,
    RESULT_FIGURE_NUMBERS,
    RESULT_FIGURE_STATUS,
    RESULT_SENSITIVITY_PARAMETERS,
)


def make_result_figures(
    result_dir: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
    figures: list[int] | None = None,
) -> dict[str, Path]:
    selected = set(figures or RESULT_FIGURE_NUMBERS)
    paths: dict[str, Path] = {}
    for figure_no in RESULT_FIGURE_NUMBERS:
        if figure_no not in selected:
            continue
        paths.update(make_result_figure(result_dir, case_key, sources, figure_no))
    return paths


def make_result_figure(
    result_dir: Path,
    case_key: str,
    sources: dict[str, pd.DataFrame],
    figure_no: int,
) -> dict[str, Path]:
    if figure_no == 4:
        return plot_figure_4_reliability_trajectory(
            result_dir,
            case_key,
            sources.get("reliability_trajectory", pd.DataFrame()),
            sources.get("design_space", pd.DataFrame()),
        )
    if figure_no == 5:
        return plot_figure_5_pareto(result_dir, case_key, sources.get("pareto", pd.DataFrame()))
    if figure_no == 6:
        return plot_figure_6_sensitivity(result_dir, case_key, sources.get("sensitivity", pd.DataFrame()))
    if figure_no == 7:
        from .saifi_saidi import make_saifi_saidi_sensitivity_figures

        return make_saifi_saidi_sensitivity_figures(
            result_dir,
            case_key,
            sources.get("saifi_saidi", pd.DataFrame()),
        )
    if figure_no == 8:
        from .robust_uncertainty import make_figure_8_uncertainty_baseline

        return make_figure_8_uncertainty_baseline(
            result_dir,
            case_key,
            sources.get("uncertainty_baseline", pd.DataFrame()),
        )
    if figure_no == 9:
        from .robust_uncertainty import make_figure_9_robust_sensitivity_bands

        return make_figure_9_robust_sensitivity_bands(
            result_dir,
            case_key,
            sources.get("robust_sensitivity_bands", pd.DataFrame()),
        )
    if figure_no == 10:
        from .robust_uncertainty import make_figure_10_robust_sensitivity_mechanism

        return make_figure_10_robust_sensitivity_mechanism(
            result_dir,
            case_key,
            sources.get("robust_sensitivity_mechanism", pd.DataFrame()),
        )
    if figure_no == 11:
        from .robust_uncertainty import make_figure_11_tail_robustness

        return make_figure_11_tail_robustness(
            result_dir,
            case_key,
            sources.get("tail_robustness", pd.DataFrame()),
        )
    if figure_no == 12:
        from .ccp_bess_response import make_ccp_bess_capacity_response_figure

        return make_ccp_bess_capacity_response_figure(
            result_dir,
            case_key,
            sources.get("ccp_bess_capacity_response", pd.DataFrame()),
        )
    raise ValueError(f"Unsupported result figure number: {figure_no}")


def build_result_sensitivity_source(sensitivity_rows: pd.DataFrame) -> pd.DataFrame:
    from .sensitivity import build_sensitivity_comparison_source

    return build_sensitivity_comparison_source(sensitivity_rows)


def plot_figure_4_reliability_trajectory(
    out: Path,
    case_key: str,
    source: pd.DataFrame,
    design_space: pd.DataFrame,
) -> dict[str, Path]:
    from .reliability_trajectory import make_reliability_trajectory_publication_figure

    return make_reliability_trajectory_publication_figure(out, case_key, source, design_space)


def plot_figure_5_pareto(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    from .pareto import make_pareto_frontier_publication_figure

    return make_pareto_frontier_publication_figure(out, case_key, source)


def plot_figure_6_sensitivity(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    from .sensitivity import make_sensitivity_comparison_publication_figure

    return make_sensitivity_comparison_publication_figure(out, case_key, source)


def plot_figure_11_tail_resilience(out: Path, case_key: str, source: pd.DataFrame) -> dict[str, Path]:
    from .tail_resilience import make_tail_resilience_publication_figure

    return make_tail_resilience_publication_figure(out, case_key, source)
