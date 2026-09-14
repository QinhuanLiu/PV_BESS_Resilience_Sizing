from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from .names import RESULT_SENSITIVITY_PARAMETERS
from .registry import build_result_sensitivity_source
from .ccp_bess_response import build_ccp_bess_capacity_response_source
from .robust_uncertainty import (
    build_figure_8_uncertainty_baseline_source,
    build_robust_sensitivity_band_source,
    build_robust_sensitivity_mechanism_source,
    build_tail_robustness_source,
)
from .saifi_saidi import build_saifi_saidi_sensitivity_source
from ..resilience_frontier import result_design_space_source, result_pareto_source


MissingSensitivityRunner = Callable[[list[str]], pd.DataFrame]
SaifiSaidiGridRunner = Callable[[], pd.DataFrame]


def build_result_sources(
    case: Any,
    case_config: dict[str, Any],
    parameters: Any,
    all_candidates: pd.DataFrame,
    scenario_results: pd.DataFrame,
    optimal_designs: pd.DataFrame,
    outage_scenarios: pd.DataFrame,
    case_sensitivity: pd.DataFrame,
    uncertainty_baseline: pd.DataFrame | None = None,
    uncertainty_sensitivity: pd.DataFrame | None = None,
    figures: list[int] | None = None,
    compute_missing_sensitivity_rows: MissingSensitivityRunner | None = None,
    compute_saifi_saidi_grid: SaifiSaidiGridRunner | None = None,
) -> dict[str, pd.DataFrame]:
    selected = set(figures or [4, 5, 6, 7, 8, 9, 10, 11, 12])
    sources: dict[str, pd.DataFrame] = {}
    if selected.intersection({4, 5}):
        if 4 in selected:
            sources["design_space"] = result_design_space_source(all_candidates, optimal_designs)
            sources["reliability_trajectory"] = all_candidates.copy()
        if 5 in selected:
            sources["pareto"] = result_pareto_source(all_candidates, optimal_designs)
    if 6 in selected:
        sensitivity = case_sensitivity
        missing = missing_result_sensitivity_parameters(sensitivity)
        if missing:
            if compute_missing_sensitivity_rows is None:
                raise ValueError(
                    "Figure 6 source generation requires missing sensitivity rows for: "
                    + ", ".join(missing)
                )
            computed = compute_missing_sensitivity_rows(missing)
            sensitivity = computed if case_sensitivity.empty else pd.concat([case_sensitivity, computed], ignore_index=True)
        sources["sensitivity"] = build_result_sensitivity_source(sensitivity)
    if 7 in selected:
        if compute_saifi_saidi_grid is None:
            raise ValueError("Figure 7 source generation requires SAIFI-SAIDI grid rows.")
        sources["saifi_saidi"] = build_saifi_saidi_sensitivity_source(compute_saifi_saidi_grid())
    uncertainty_baseline_rows = uncertainty_baseline if uncertainty_baseline is not None else pd.DataFrame()
    uncertainty_sensitivity_rows = uncertainty_sensitivity if uncertainty_sensitivity is not None else pd.DataFrame()
    if 8 in selected:
        sources["uncertainty_baseline"] = build_figure_8_uncertainty_baseline_source(uncertainty_baseline_rows)
    if 9 in selected:
        sources["robust_sensitivity_bands"] = build_robust_sensitivity_band_source(uncertainty_sensitivity_rows)
    if 10 in selected:
        sources["robust_sensitivity_mechanism"] = build_robust_sensitivity_mechanism_source(uncertainty_sensitivity_rows)
    if 11 in selected:
        sources["tail_robustness"] = build_tail_robustness_source(
            uncertainty_baseline_rows,
            uncertainty_sensitivity_rows,
        )
    if 12 in selected:
        sources["ccp_bess_capacity_response"] = build_ccp_bess_capacity_response_source(all_candidates)
    return sources


def read_result_sources(result_dir: Path, case_key: str, figures: list[int] | None = None) -> dict[str, pd.DataFrame]:
    selected = set(figures or [4, 5, 6, 7, 8, 9, 10, 11, 12])
    paths = {
        4: (
            "reliability_trajectory",
            result_dir / f"{case_key}_Figure_4_Reliability_Driven_Sizing_Trajectory_Source_Data.csv",
        ),
        5: ("pareto", result_dir / f"{case_key}_Figure_5_Cost_Resilience_Pareto_Frontier_Source_Data.csv"),
        6: ("sensitivity", result_dir / f"{case_key}_Figure_6_CAPEX_SAIDI_SAIFI_VoLL_Sensitivity_Source_Data.csv"),
        7: ("saifi_saidi", result_dir / f"{case_key}_Figure_7_SAIFI_SAIDI_Sensitivity_Source_Data.csv"),
        8: ("uncertainty_baseline", result_dir / f"{case_key}_Figure_8_Uncertainty_Baseline_Source_Data.csv"),
        9: ("robust_sensitivity_bands", result_dir / f"{case_key}_Figure_9_Robust_Sensitivity_Bands_Source_Data.csv"),
        10: (
            "robust_sensitivity_mechanism",
            result_dir / f"{case_key}_Figure_10_Robust_Sensitivity_Mechanism_Source_Data.csv",
        ),
        11: ("tail_robustness", result_dir / f"{case_key}_Figure_11_Tail_Robustness_Resilience_Source_Data.csv"),
        12: (
            "ccp_bess_capacity_response",
            result_dir / f"{case_key}_Figure_12_CCP_BESS_Capacity_Response_Source_Data.csv",
        ),
    }
    unsupported = sorted(selected - set(paths))
    if unsupported:
        raise ValueError(f"Unsupported result figure number(s): {unsupported}")
    sources: dict[str, pd.DataFrame] = {}
    for figure_no in selected:
        key, path = paths[figure_no]
        if not path.exists():
            raise FileNotFoundError(f"Missing source data for Figure {figure_no}: {path}")
        sources[key] = pd.read_csv(path)
    return sources


def missing_result_sensitivity_parameters(sensitivity: pd.DataFrame) -> list[str]:
    if sensitivity.empty or "sensitivity_parameter" not in sensitivity.columns:
        return list(RESULT_SENSITIVITY_PARAMETERS)
    present = set(sensitivity["sensitivity_parameter"].astype(str).unique().tolist())
    return [parameter for parameter in RESULT_SENSITIVITY_PARAMETERS if parameter not in present]
