from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.figures.versioning import (
    FIGURE_VERSION_1,
    FIGURE_VERSION_2,
    archive_case_figures,
    classify_figure_file,
    figure_id_from_name,
    is_figure_like_file,
)


class FigureVersioningTests(unittest.TestCase):
    def test_classifies_new_mainline_figures_as_version_2(self) -> None:
        examples = [
            "SG_UTown_Figure_3A_Baseline_PV_BESS_3D_Response_Surfaces.png",
            "SG_UTown_Figure_4A_Baseline_Cost_Comparison_Source_Data.csv",
            "SG_UTown_Figure_4B_Baseline_Cost_Comparison.pdf",
            "SG_UTown_Figure_7B_SAIFI_SAIDI_Contour_Maps.pdf",
            "SG_UTown_Figure_10_Robust_Sensitivity_Mechanism_Source_Data.csv",
            "SG_UTown_Figure_11_Tail_Robustness_Resilience.svg",
            "SG_UTown_Figure_12_CCP_BESS_Capacity_Response.pdf",
        ]
        for name in examples:
            version, _ = classify_figure_file(Path(name))
            self.assertEqual(version, FIGURE_VERSION_2, name)

    def test_classifies_legacy_and_auxiliary_figures_as_version_1(self) -> None:
        examples = [
            "SG_UTown_Figure_4_Baseline_Cost_Comparison_Source_Data.csv",
            "SG_UTown_Figure_4_Reliability_Driven_Sizing_Trajectory.png",
            "SG_UTown_Figure_5_Cost_Resilience_Pareto_Frontier.svg",
            "SG_UTown_Figure_12_BESS_Capacity_Physical_Interpretation.pdf",
            "SG_UTown_Figure_X_Baseline_LOLP_BESS_Slices_Source_Data.csv",
            "SG_UTown_Uncertainty_Baseline_Distribution_Violins.png",
        ]
        for name in examples:
            version, _ = classify_figure_file(Path(name))
            self.assertEqual(version, FIGURE_VERSION_1, name)

    def test_detects_figure_like_files(self) -> None:
        self.assertTrue(is_figure_like_file(Path("SG_UTown_Figure_3A.png")))
        self.assertTrue(is_figure_like_file(Path("SG_UTown_Figure_3_Source_Data.csv")))
        self.assertTrue(is_figure_like_file(Path("SG_UTown_Figure_12_Event_Source_Data.csv")))
        self.assertFalse(is_figure_like_file(Path("SG_UTown_Baseline_Optimal_Design.csv")))
        self.assertFalse(is_figure_like_file(Path("notes.txt")))

    def test_archive_copies_files_and_writes_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_root = root / "outputs" / "SG"
            baseline = case_root / "SG_UTown_Baseline" / "SG_UTown_Baseline_Result_Figures"
            result = case_root / "SG_UTown_Result_Figures"
            uncertainty = case_root / "SG_UTown_Uncertainty_Analysis" / "SG_UTown_Uncertainty_Baseline"
            for directory in (baseline, result, uncertainty):
                directory.mkdir(parents=True)
            files = [
                baseline / "SG_UTown_Figure_3A_Baseline_PV_BESS_3D_Response_Surfaces.png",
                baseline / "SG_UTown_Figure_3_Baseline_PV_BESS_Design_Space_Source_Data.csv",
                baseline / "SG_UTown_Figure_4A_Baseline_Cost_Comparison.pdf",
                baseline / "SG_UTown_Figure_4B_Baseline_Cost_Comparison_Source_Data.csv",
                baseline / "SG_UTown_Figure_X_Baseline_LOLP_BESS_Slices.pdf",
                result / "SG_UTown_Figure_9_Robust_Sensitivity_Bands.svg",
                result / "SG_UTown_Figure_12_CCP_BESS_Capacity_Response.pdf",
                uncertainty / "SG_UTown_Uncertainty_Baseline_Capacity_Cost_Bubble.png",
            ]
            for path in files:
                path.write_text("figure", encoding="utf-8")
            (result / "SG_UTown_Result_Table.csv").write_text("not figure", encoding="utf-8")

            archive_root = case_root / "SG_UTown_Figure_Versions"
            paths = archive_case_figures(case_root, "SG_UTown", archive_root)

            manifest = pd.read_csv(paths["figure_version_manifest"])
            self.assertEqual(len(manifest), len(files))
            self.assertTrue((archive_root / FIGURE_VERSION_2 / files[0].name).exists())
            self.assertTrue((archive_root / FIGURE_VERSION_2 / files[1].name).exists())
            self.assertTrue((archive_root / FIGURE_VERSION_2 / files[2].name).exists())
            self.assertTrue((archive_root / FIGURE_VERSION_2 / files[3].name).exists())
            self.assertTrue((archive_root / FIGURE_VERSION_2 / files[5].name).exists())
            self.assertTrue((archive_root / FIGURE_VERSION_2 / files[6].name).exists())
            self.assertTrue((archive_root / FIGURE_VERSION_1 / files[4].name).exists())
            self.assertTrue((archive_root / FIGURE_VERSION_1 / files[7].name).exists())
            self.assertFalse((archive_root / FIGURE_VERSION_1 / "SG_UTown_Result_Table.csv").exists())
            self.assertEqual(set(manifest["figure_version"]), {FIGURE_VERSION_1, FIGURE_VERSION_2})
            self.assertIn("SG_UTown_Figure_3A_Baseline_PV_BESS_3D_Response_Surfaces", manifest["figure_id"].tolist())

            archive_case_figures(case_root, "SG_UTown", archive_root)
            self.assertFalse(any("__2" in path.name for path in archive_root.rglob("*")))

    def test_figure_id_removes_source_data_suffix(self) -> None:
        self.assertEqual(
            figure_id_from_name("SG_UTown_Figure_3_Baseline_PV_BESS_Design_Space_Source_Data.csv"),
            "SG_UTown_Figure_3_Baseline_PV_BESS_Design_Space",
        )


if __name__ == "__main__":
    unittest.main()
