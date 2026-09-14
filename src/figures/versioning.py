from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import pandas as pd


FIGURE_VERSION_1 = "Figure_Version_1"
FIGURE_VERSION_2 = "Figure_Version_2"
FIGURE_VERSION_MANIFEST = "Figure_Version_Manifest.csv"
FIGURE_EXPORT_SUFFIXES = {".png", ".svg", ".pdf"}
SOURCE_DATA_TOKEN = "source_data"

VERSION_2_PATTERNS = (
    "Figure_3_Baseline_PV_BESS_Design_Space",
    "Figure_3A_Baseline_PV_BESS_3D_Response_Surfaces",
    "Figure_3B_Baseline_PV_BESS_Contour_Maps",
    "Figure_3C_Baseline_PV_Capacity_BESS_Slices",
    "Figure_4A_Baseline_Cost_Comparison",
    "Figure_4B_Baseline_Cost_Comparison",
    "Figure_5_Baseline_Outage_Energy_Balance_Comparison",
    "Figure_6_CAPEX_SAIDI_SAIFI_VoLL_Sensitivity",
    "Figure_7_SAIFI_SAIDI_Sensitivity",
    "Figure_7A_SAIFI_SAIDI_Response_Surfaces",
    "Figure_7B_SAIFI_SAIDI_Contour_Maps",
    "Figure_8_Uncertainty_Baseline",
    "Figure_8A_Uncertainty_Baseline_Distributions",
    "Figure_8B_Uncertainty_Design_Cost_Bubble",
    "Figure_9_Robust_Sensitivity_Bands",
    "Figure_10_Robust_Sensitivity_Mechanism",
    "Figure_10A_Robust_Design_Trajectories",
    "Figure_10B_Robust_Cost_Composition",
    "Figure_11_Tail_Robustness_Resilience",
    "Figure_12_CCP_BESS_Capacity_Response",
)

MANIFEST_COLUMNS = (
    "case_key",
    "figure_version",
    "figure_id",
    "file_type",
    "original_path",
    "archived_path",
    "classification_rule",
)


@dataclass(frozen=True)
class VersionedFigureFile:
    case_key: str
    figure_version: str
    figure_id: str
    file_type: str
    original_path: Path
    archived_path: Path
    classification_rule: str

    def as_manifest_row(self) -> dict[str, str]:
        return {
            "case_key": self.case_key,
            "figure_version": self.figure_version,
            "figure_id": self.figure_id,
            "file_type": self.file_type,
            "original_path": str(self.original_path),
            "archived_path": str(self.archived_path),
            "classification_rule": self.classification_rule,
        }


def archive_case_figures(
    case_root: Path,
    case_key: str,
    archive_root: Path,
) -> dict[str, Path]:
    """Copy already-generated figure files into Figure Version 1/2 folders."""
    case_root = Path(case_root)
    archive_root = Path(archive_root)
    version_1_dir = archive_root / FIGURE_VERSION_1
    version_2_dir = archive_root / FIGURE_VERSION_2
    for version_dir in (version_1_dir, version_2_dir):
        if version_dir.exists():
            shutil.rmtree(version_dir)
    version_1_dir.mkdir(parents=True, exist_ok=True)
    version_2_dir.mkdir(parents=True, exist_ok=True)

    records: list[VersionedFigureFile] = []
    for path in iter_case_figure_files(case_root, case_key, archive_root):
        version, rule = classify_figure_file(path)
        target_dir = version_2_dir if version == FIGURE_VERSION_2 else version_1_dir
        archived_path = _unique_archive_path(target_dir / path.name)
        shutil.copy2(path, archived_path)
        records.append(
            VersionedFigureFile(
                case_key=case_key,
                figure_version=version,
                figure_id=figure_id_from_name(path.name),
                file_type=file_type_from_path(path),
                original_path=path,
                archived_path=archived_path,
                classification_rule=rule,
            )
        )

    manifest = pd.DataFrame([record.as_manifest_row() for record in records], columns=MANIFEST_COLUMNS)
    manifest_path = archive_root / FIGURE_VERSION_MANIFEST
    archive_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    paths: dict[str, Path] = {"figure_version_manifest": manifest_path}
    for index, record in enumerate(records, start=1):
        paths[f"figure_version_{index:03d}_{record.figure_version.lower()}"] = record.archived_path
    return paths


def iter_case_figure_files(case_root: Path, case_key: str, archive_root: Path) -> list[Path]:
    root = Path(case_root)
    archive = Path(archive_root).resolve()
    paths: list[Path] = []
    if not root.exists():
        return paths
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if archive in path.resolve().parents or path.resolve() == archive:
            continue
        if not path.name.startswith(f"{case_key}_"):
            continue
        if is_figure_like_file(path):
            paths.append(path)
    return sorted(paths, key=lambda item: str(item))


def is_figure_like_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in FIGURE_EXPORT_SUFFIXES:
        return True
    if suffix == ".csv" and SOURCE_DATA_TOKEN in path.name.lower():
        return True
    return False


def classify_figure_file(path: Path) -> tuple[str, str]:
    name = path.name
    for pattern in VERSION_2_PATTERNS:
        if pattern in name:
            return FIGURE_VERSION_2, pattern
    return FIGURE_VERSION_1, "legacy_or_auxiliary_figure"


def figure_id_from_name(name: str) -> str:
    stem = Path(name).stem
    if stem.endswith("_Source_Data"):
        stem = stem[: -len("_Source_Data")]
    return stem


def file_type_from_path(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "csv":
        return "source_data"
    return suffix


def _unique_archive_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10_000):
        candidate = path.with_name(f"{stem}__{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique archive path for {path}")
