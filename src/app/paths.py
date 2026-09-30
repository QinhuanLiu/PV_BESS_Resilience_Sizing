from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseOutputPaths:
    baseline: Path
    tou_analysis: Path


def input_dir(config: dict[str, Any], override: str | None) -> Path:
    return Path(override or config.get("input", {}).get("directory", "Input")).resolve()


def output_dir(config: dict[str, Any], override: str | None) -> Path:
    return Path(override or config.get("outputs", {}).get("directory", "outputs")).resolve()


def case_output_paths(output_root: Path, city_key: str, case_key: str) -> CaseOutputPaths:
    return CaseOutputPaths(
        baseline=output_root / city_key / f"{case_key}_Baseline",
        tou_analysis=output_root / city_key / f"{case_key}_TOU_Analysis",
    )
