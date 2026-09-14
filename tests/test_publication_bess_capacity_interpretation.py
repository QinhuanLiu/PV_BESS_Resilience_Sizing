from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.io.data_loader import InputCase
from src.figures.bess_interpretation import (
    build_bess_capacity_interpretation_source,
    make_bess_capacity_interpretation_figure,
    make_demo_bess_capacity_interpretation_figure,
    validate_bess_event_source,
)


def _write_case_inputs(root: Path) -> InputCase:
    city_dir = root / "SG"
    city_dir.mkdir(parents=True)
    load = pd.DataFrame(
        {
            "Lighting": [1000.0] * 8760,
            "Equipment": [0.0] * 8760,
            "Cooling": [0.0] * 8760,
            "Heating": [0.0] * 8760,
        }
    )
    pv = pd.DataFrame({"PV_output_kWh_per_m2": [0.01] * 8760})
    tou = pd.DataFrame({"hour": list(range(24)), "tou_sgd_per_kwh": [0.25] * 24})
    load_csv = city_dir / "SG_Test_Load.csv"
    pv_csv = city_dir / "SG_Test_PV.csv"
    tou_csv = city_dir / "SG_Test_TOU.csv"
    parameter_csv = city_dir / "SG_Test_Parameter.csv"
    load.to_csv(load_csv, index=False)
    pv.to_csv(pv_csv, index=False)
    tou.to_csv(tou_csv, index=False)
    pd.DataFrame({"parameter": [], "base_value": []}).to_csv(parameter_csv, index=False)
    return InputCase("SG", "Test", "SG_Test", city_dir, load_csv, pv_csv, tou_csv, parameter_csv)


def _config() -> dict:
    return {
        "technical": {"cooling_cop": 5.0, "heating_cop": 3.5},
        "pv": {
            "capacity_kw_candidates": [90.0, 110.0],
            "rooftop_area_m2": 10_000.0,
            "module_power_density_kw_per_m2": 0.1,
        },
        "battery": {
            "soc_min_fraction": 0.10,
            "soc_max_fraction": 0.95,
            "roundtrip_efficiency": 0.92,
        },
        "economics": {},
        "scenarios": {
            "SG_Test": {
                "saidi_min_per_year": 180.0,
                "saifi_per_year": 3.0,
                "voll_sgd_per_kwh": 50.0,
                "battery_economics": {},
            }
        },
    }


def _profile(duration: int) -> pd.DataFrame:
    outage_fraction = [1.0] * duration + [0.0] * (8760 - duration)
    return pd.DataFrame({"hour_index": range(8760), "outage_fraction": outage_fraction})


def _outage_scenarios() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_key": "SG_Test",
                "city_key": "SG",
                "building_key": "Test",
                "frontier_scenario_id": idx,
                "outage_seed": 100 + idx,
                "max_event_duration_min": duration * 60.0,
                "_outage_profile": _profile(duration),
            }
            for idx, duration in enumerate([1, 2, 3], start=1)
        ]
    )


def _optimal_designs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "design_type": "reliability_constrained_design",
                "available": True,
                "pv_capacity_kw": 104.0,
                "battery_energy_kwh": 2000.0,
            }
        ]
    )


def _scenario_results() -> pd.DataFrame:
    rows = []
    for pv in [90.0, 110.0]:
        for bess in [0.0, 1000.0, 2000.0, 3000.0]:
            for scenario_id, base_ue in enumerate([3000.0, 2000.0, 1000.0], start=1):
                ue = max(0.0, base_ue - bess * 0.45 - pv * 0.2)
                rows.append(
                    {
                        "pv_capacity_kw": pv,
                        "battery_energy_kwh": bess,
                        "UE_kwh_per_year": ue,
                        "scenario_LOLP": ue / 5000.0,
                        "unserved_energy_cost_sgd_per_year": ue * 50.0,
                        "annualized_battery_cost_sgd": bess * 20.0,
                        "battery_om_cost_sgd": bess * 2.0,
                        "pv_total_annual_cost_sgd": pv * 100.0,
                        "pv_bess_grid_cost_sgd": 100_000.0 - bess * 3.0,
                        "pv_bess_private_annual_cost_sgd": 100_000.0 + bess * 19.0,
                        "pv_bess_system_annual_cost_sgd": 100_000.0 + bess * 19.0 + ue * 50.0,
                        "lolp_threshold": 0.10,
                    }
                )
    return pd.DataFrame(rows)


class PublicationBessCapacityInterpretationTests(unittest.TestCase):
    def test_missing_event_source_columns_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required column"):
            validate_bess_event_source(pd.DataFrame({"net_deficit_kwh": [1.0]}))

    def test_builder_computes_quantiles_coverage_and_nearest_pv_slice(self) -> None:
        with TemporaryDirectory() as tmp:
            case = _write_case_inputs(Path(tmp))
            events, capacity_slice, summary = build_bess_capacity_interpretation_source(
                case,
                _config(),
                _optimal_designs(),
                _outage_scenarios(),
                _scenario_results(),
            )
            row = summary.iloc[0]
            self.assertAlmostEqual(row["net_deficit_p50_kwh"], 1979.2)
            self.assertAlmostEqual(row["net_deficit_p90_kwh"], 2968.8)
            self.assertAlmostEqual(row["net_deficit_p95_kwh"], 2968.8)
            self.assertAlmostEqual(row["net_deficit_coverage_probability"], 1 / 3)
            self.assertAlmostEqual(row["slice_pv_capacity_kw"], 110.0)
            self.assertTrue(capacity_slice["is_optimal_bess"].any())
            self.assertEqual(len(events), 3)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_publication_figure_exports_all_formats_and_source_data(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_demo_bess_capacity_interpretation_figure(Path(tmp), "SG_UTown")
            png_key = (
                "SG_UTown_Figure_X_BESS_Capacity_Physical_Interpretation_Demo_"
                "Figure_12_BESS_Capacity_Physical_Interpretation_png"
            )
            svg_key = png_key.replace("_png", "_svg")
            self.assertIn(png_key, paths)
            self.assertIn(svg_key, paths)
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            svg_text = paths[svg_key].read_text()
            self.assertIn("Net-deficit coverage", svg_text)
            self.assertIn("Optimized usable BESS", svg_text)
            self.assertIn("Marginal UE reduction", svg_text)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_real_source_figure_exports_all_formats(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            case = _write_case_inputs(tmp_path / "input")
            events, capacity_slice, summary = build_bess_capacity_interpretation_source(
                case,
                _config(),
                _optimal_designs(),
                _outage_scenarios(),
                _scenario_results(),
            )
            paths = make_bess_capacity_interpretation_figure(tmp_path / "out", "SG_Test", events, capacity_slice, summary)
            expected = {
                "SG_Test_Figure_12_BESS_Capacity_Physical_Interpretation_png",
                "SG_Test_Figure_12_BESS_Capacity_Physical_Interpretation_svg",
                "SG_Test_Figure_12_BESS_Capacity_Physical_Interpretation_pdf",
                "SG_Test_Figure_12_BESS_Capacity_Physical_Interpretation_event_source_data",
                "SG_Test_Figure_12_BESS_Capacity_Physical_Interpretation_capacity_slice_source_data",
                "SG_Test_Figure_12_BESS_Capacity_Physical_Interpretation_summary_source_data",
            }
            self.assertTrue(expected.issubset(paths.keys()))


if __name__ == "__main__":
    unittest.main()
