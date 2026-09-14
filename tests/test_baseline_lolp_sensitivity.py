from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.figures.baseline_lolp import (
    DEFAULT_BASELINE_LOLP_THRESHOLDS,
    build_baseline_lolp_constraint_scan_source,
    make_baseline_lolp_bess_slice_figure,
    make_baseline_lolp_constraint_scan_figure,
    select_baseline_lolp_bess_slice_source,
    select_baseline_lolp_constraint_designs,
)


def _design_space_source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pv_capacity_kw": 0.0,
                "pv_capacity_mw": 0.0,
                "battery_energy_kwh": 0.0,
                "battery_energy_mwh": 0.0,
                "system_total_cost_sgd_per_year": 100.0,
                "system_total_cost_million_sgd_per_year": 0.0001,
                "UE_kwh_per_year": 300.0,
                "UE_mwh_per_year": 0.3,
                "outage_weighted_load_kwh": 1000.0,
                "energy_based_LOLP": 0.30,
            },
            {
                "pv_capacity_kw": 1000.0,
                "pv_capacity_mw": 1.0,
                "battery_energy_kwh": 500.0,
                "battery_energy_mwh": 0.5,
                "system_total_cost_sgd_per_year": 150.0,
                "system_total_cost_million_sgd_per_year": 0.00015,
                "UE_kwh_per_year": 100.0,
                "UE_mwh_per_year": 0.1,
                "outage_weighted_load_kwh": 1000.0,
                "energy_based_LOLP": 0.10,
            },
            {
                "pv_capacity_kw": 2000.0,
                "pv_capacity_mw": 2.0,
                "battery_energy_kwh": 1000.0,
                "battery_energy_mwh": 1.0,
                "system_total_cost_sgd_per_year": 140.0,
                "system_total_cost_million_sgd_per_year": 0.00014,
                "UE_kwh_per_year": 150.0,
                "UE_mwh_per_year": 0.15,
                "outage_weighted_load_kwh": 1000.0,
                "energy_based_LOLP": 0.15,
            },
            {
                "pv_capacity_kw": 3000.0,
                "pv_capacity_mw": 3.0,
                "battery_energy_kwh": 1500.0,
                "battery_energy_mwh": 1.5,
                "system_total_cost_sgd_per_year": 220.0,
                "system_total_cost_million_sgd_per_year": 0.00022,
                "UE_kwh_per_year": 40.0,
                "UE_mwh_per_year": 0.04,
                "outage_weighted_load_kwh": 1000.0,
                "energy_based_LOLP": 0.04,
            },
        ]
    )


def _slice_design_space_source() -> pd.DataFrame:
    rows = []
    for pv_mw in [8.0, 20.0, 32.0]:
        for bess_mwh in [0.0, 5.0, 10.0, 20.0]:
            lolp = max(0.0, min(1.0, 0.40 - 0.006 * pv_mw - 0.012 * bess_mwh))
            rows.append(
                {
                    "pv_capacity_kw": pv_mw * 1000.0,
                    "pv_capacity_mw": pv_mw,
                    "battery_energy_kwh": bess_mwh * 1000.0,
                    "battery_energy_mwh": bess_mwh,
                    "system_total_cost_sgd_per_year": 1_000_000.0 + 100_000.0 * pv_mw + 50_000.0 * bess_mwh,
                    "system_total_cost_million_sgd_per_year": 1.0 + 0.1 * pv_mw + 0.05 * bess_mwh,
                    "UE_kwh_per_year": lolp * 10_000.0,
                    "UE_mwh_per_year": lolp * 10.0,
                    "outage_weighted_load_kwh": 10_000.0,
                    "energy_based_LOLP": lolp,
                }
            )
    return pd.DataFrame(rows)


def _small_slice_design_space_source() -> pd.DataFrame:
    rows = []
    for pv_mw in [0.0, 0.05, 0.10]:
        for bess_mwh in [0.0, 0.02, 0.04, 0.06]:
            lolp = max(0.0, min(1.0, 0.40 - 0.8 * pv_mw - 2.0 * bess_mwh))
            rows.append(
                {
                    "pv_capacity_kw": pv_mw * 1000.0,
                    "pv_capacity_mw": pv_mw,
                    "battery_energy_kwh": bess_mwh * 1000.0,
                    "battery_energy_mwh": bess_mwh,
                    "system_total_cost_sgd_per_year": 1000.0 + 100.0 * pv_mw + 50.0 * bess_mwh,
                    "system_total_cost_million_sgd_per_year": 0.001 + 0.0001 * pv_mw + 0.00005 * bess_mwh,
                    "UE_kwh_per_year": lolp * 100.0,
                    "UE_mwh_per_year": lolp * 0.1,
                    "outage_weighted_load_kwh": 100.0,
                    "energy_based_LOLP": lolp,
                }
            )
    return pd.DataFrame(rows)


def _case_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour_index": [0, 1],
            "electric_load_kwh": [100.0, 100.0],
            "cooling_load_kwh_th": [0.0, 0.0],
            "heating_load_kwh_th": [0.0, 0.0],
        }
    )


def _outage_profile() -> pd.DataFrame:
    return pd.DataFrame({"hour_index": [0, 1], "outage_fraction": [1.0, 0.0]})


def _config() -> dict:
    return {"technical": {"cooling_cop": 5.0, "heating_cop": 3.5}}


class BaselineLOLPConstraintScanTests(unittest.TestCase):
    def test_scan_selects_cheapest_feasible_design_for_each_threshold(self) -> None:
        scan = select_baseline_lolp_constraint_designs(_design_space_source(), thresholds=[0.10, 0.15, 0.20])

        self.assertEqual(scan["design_lolp_constraint"].tolist(), [0.10, 0.15, 0.20])
        self.assertEqual(scan["available"].tolist(), [True, True, True])
        self.assertEqual(scan["actual_LOLP"].tolist(), [0.10, 0.15, 0.15])
        self.assertEqual(scan["system_total_cost_sgd_per_year"].tolist(), [150.0, 140.0, 140.0])
        self.assertNotIn("chance_constraint_probability", scan.columns)

    def test_scan_keeps_unavailable_rows_for_infeasible_thresholds(self) -> None:
        scan = select_baseline_lolp_constraint_designs(_design_space_source(), thresholds=[0.01, 0.05])

        unavailable = scan[scan["design_lolp_constraint"].eq(0.01)].iloc[0]
        available = scan[scan["design_lolp_constraint"].eq(0.05)].iloc[0]
        self.assertFalse(bool(unavailable["available"]))
        self.assertTrue(pd.isna(unavailable["actual_LOLP"]))
        self.assertTrue(bool(available["available"]))
        self.assertEqual(float(available["actual_LOLP"]), 0.04)

    def test_builder_uses_single_outage_energy_based_lolp(self) -> None:
        candidate_summary = pd.DataFrame(
            [
                {
                    "pv_capacity_kw": 0.0,
                    "battery_energy_kwh": 0.0,
                    "pv_bess_system_annual_cost_sgd": 50.0,
                    "unserved_energy_kwh_per_year": 20.0,
                },
                {
                    "pv_capacity_kw": 1000.0,
                    "battery_energy_kwh": 1000.0,
                    "pv_bess_system_annual_cost_sgd": 90.0,
                    "unserved_energy_kwh_per_year": 4.0,
                },
            ]
        )
        scan = build_baseline_lolp_constraint_scan_source(
            candidate_summary,
            _case_frame(),
            _outage_profile(),
            _config(),
            thresholds=[0.05, 0.20],
        )

        self.assertEqual(scan["actual_LOLP"].tolist(), [0.04, 0.20])
        self.assertEqual(scan["UE_kwh_per_year"].tolist(), [4.0, 20.0])
        self.assertNotIn("chance_constraint_probability", scan.columns)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_figure_exports_source_png_svg_and_pdf(self) -> None:
        scan = select_baseline_lolp_constraint_designs(_design_space_source(), DEFAULT_BASELINE_LOLP_THRESHOLDS)
        with TemporaryDirectory() as tmp:
            paths = make_baseline_lolp_constraint_scan_figure(Path(tmp), "SG_UTown", scan)

            self.assertTrue(paths["source_data"].exists())
            self.assertIn("SG_UTown_Figure_X_Baseline_LOLP_Constraint_Scan_png", paths)
            self.assertIn("SG_UTown_Figure_X_Baseline_LOLP_Constraint_Scan_svg", paths)
            self.assertIn("SG_UTown_Figure_X_Baseline_LOLP_Constraint_Scan_pdf", paths)
            written = pd.read_csv(paths["source_data"])
            self.assertEqual(written["design_lolp_constraint"].tolist(), [0.05, 0.10, 0.15, 0.20])

    def test_bess_slice_source_selects_fixed_pv_capacities(self) -> None:
        source = select_baseline_lolp_bess_slice_source(_slice_design_space_source())

        self.assertEqual(source["pv_capacity_mw"].drop_duplicates().tolist(), [8.0, 20.0, 32.0])
        self.assertEqual(set(source["pv_slice_label"]), {"Low PV", "Medium PV", "High PV"})
        self.assertTrue((source.groupby("pv_slice_label")["battery_energy_mwh"].nunique() >= 3).all())
        self.assertTrue(source["energy_based_LOLP"].between(0.0, 1.0).all())
        self.assertNotIn("chance_constraint_probability", source.columns)

    def test_bess_slice_source_falls_back_for_small_pv_grid(self) -> None:
        source = select_baseline_lolp_bess_slice_source(_small_slice_design_space_source())

        self.assertEqual(source["pv_capacity_mw"].drop_duplicates().tolist(), [0.0, 0.05, 0.10])
        self.assertEqual(set(source["pv_slice_label"]), {"Low PV", "Medium PV", "High PV"})
        self.assertEqual(set(source["pv_slice_source"]), {"adaptive_low_medium_high_mw"})
        self.assertTrue((source.groupby("pv_slice_label")["battery_energy_mwh"].nunique() >= 3).all())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_bess_slice_figure_exports_source_png_svg_and_pdf(self) -> None:
        source = select_baseline_lolp_bess_slice_source(_slice_design_space_source())
        with TemporaryDirectory() as tmp:
            paths = make_baseline_lolp_bess_slice_figure(Path(tmp), "SG_UTown", source)

            self.assertTrue(paths["source_data"].exists())
            self.assertIn("SG_UTown_Figure_X_Baseline_LOLP_BESS_Slices_png", paths)
            self.assertIn("SG_UTown_Figure_X_Baseline_LOLP_BESS_Slices_svg", paths)
            self.assertIn("SG_UTown_Figure_X_Baseline_LOLP_BESS_Slices_pdf", paths)
            written = pd.read_csv(paths["source_data"])
            self.assertEqual(set(written["pv_slice_label"]), {"Low PV", "Medium PV", "High PV"})
            self.assertEqual(written["pv_capacity_mw"].drop_duplicates().tolist(), [8.0, 20.0, 32.0])


if __name__ == "__main__":
    unittest.main()
