from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseOutputPaths:
    baseline: Path
    sensitivity: Path
    uncertainty: Path
    uncertainty_baseline: Path
    uncertainty_sensitivity: Path
    lolp_zero_uncertainty: Path
    lolp_zero_three_year_uncertainty: Path
    variable_uncertainty: Path
    frontier: Path
    representative_sensitivity: Path
    result_figures: Path
    baseline_result_figures: Path
    figure_versions: Path


def input_dir(config: dict[str, Any], override: str | None) -> Path:
    return Path(override or config.get("input", {}).get("directory", "Input")).resolve()


def output_dir(config: dict[str, Any], override: str | None) -> Path:
    return Path(override or config.get("outputs", {}).get("directory", "outputs")).resolve()


def case_output_paths(output_root: Path, city_key: str, case_key: str) -> CaseOutputPaths:
    uncertainty = output_root / city_key / f"{case_key}_Uncertainty_Analysis"
    baseline = output_root / city_key / f"{case_key}_Baseline"
    return CaseOutputPaths(
        baseline=baseline,
        sensitivity=output_root / city_key / f"{case_key}_Sensitive_Analysis",
        uncertainty=uncertainty,
        uncertainty_baseline=uncertainty / f"{case_key}_Uncertainty_Baseline",
        uncertainty_sensitivity=uncertainty / f"{case_key}_Uncertainty_Sensitive_Analysis",
        lolp_zero_uncertainty=uncertainty / f"{case_key}_LOLP_Zero_Uncertainty",
        lolp_zero_three_year_uncertainty=uncertainty / f"{case_key}_Figure2_LOLP_Zero_Three_Year_Uncertainty",
        variable_uncertainty=uncertainty / f"{case_key}_Variable_Uncertainty",
        frontier=output_root / city_key / f"{case_key}_Resilience_Frontier",
        representative_sensitivity=output_root / city_key / f"{case_key}_Representative_Sensitivity",
        result_figures=output_root / city_key / f"{case_key}_Result_Figures",
        baseline_result_figures=baseline / f"{case_key}_Baseline_Result_Figures",
        figure_versions=output_root / city_key / f"{case_key}_Figure_Versions",
    )
