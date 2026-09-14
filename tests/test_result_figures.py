from __future__ import annotations

import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.figures.registry import make_result_figure as make_result_figure_new
from src.figures.registry import make_result_figures as make_result_figures_new
from src.figures.registry import (
    build_result_sensitivity_source,
    make_result_figure,
    make_result_figures,
)
from src.figures.core import (
    _matplotlib,
    _save_all,
    _style,
)
from src.figures.sensitivity import METRICS, make_sensitivity_comparison_publication_figure
from src.figures.saifi_saidi import build_saifi_saidi_sensitivity_source
from src.figures.ccp_bess_response import (
    build_ccp_bess_capacity_response_source,
    make_ccp_bess_capacity_response_figure,
)
from src.app.workflow import _compute_saifi_saidi_grid_rows, run_selected


def _synthetic_design_space_source() -> pd.DataFrame:
    rows = []
    for pv in [0.0, 1.0, 2.0]:
        for bess in [0.0, 1.0, 2.0]:
            ue = max(0.0, 1.0 - 0.22 * pv - 0.28 * bess)
            lolp = max(0.0, 0.5 - 0.12 * pv - 0.14 * bess)
            rows.append(
                {
                    "pv_capacity_mw": pv,
                    "battery_energy_mwh": bess,
                    "system_total_cost_million_sgd_per_year": 10.0 + pv * 0.25 + bess * 0.2,
                    "mean_UE_mwh_per_year": ue,
                    "energy_based_LOLP": lolp,
                    "chance_constraint_probability": 1.0 - lolp,
                    "lolp_threshold": 0.2,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_reliability_candidates() -> pd.DataFrame:
    rows = []
    for pv_kw, bess_kwh, ccp, cost in [
        (1000.0, 1000.0, 0.50, 1_000_000.0),
        (2000.0, 2000.0, 0.80, 1_150_000.0),
        (3000.0, 3000.0, 0.95, 1_400_000.0),
    ]:
        rows.append(
            {
                "pv_capacity_kw": pv_kw,
                "battery_energy_kwh": bess_kwh,
                "system_total_cost_sgd_per_year": cost,
                "annualized_pv_cost_sgd_per_year": pv_kw * 80.0,
                "annualized_battery_cost_sgd_per_year": bess_kwh * 40.0,
                "battery_om_cost_sgd_per_year": bess_kwh * 2.0,
                "mean_grid_cost_sgd_per_year": cost * 0.55,
                "mean_unserved_energy_cost_sgd_per_year": (1.0 - ccp) * 100_000.0,
                "mean_UE_kwh_per_year": (1.0 - ccp) * 10_000.0,
                "mean_scenario_LOLP": 1.0 - ccp,
                "chance_constraint_probability": ccp,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_sensitivity_source() -> pd.DataFrame:
    rows = []
    for parameter, values in {
        "battery_capex_sgd_per_kwh": [350.0, 450.0, 550.0],
        "pv_capex_sgd_per_kw": [900.0, 1200.0, 1500.0],
        "saidi_min_per_year": [0.0, 5.0, 10.0],
        "saifi_per_year": [0.0, 1.0, 2.0],
        "voll_sgd_per_kwh": [10.0, 25.0, 40.0],
    }.items():
        for index, value in enumerate(values, start=1):
            rows.append(
                {
                    "sensitivity_parameter": parameter,
                    "sensitivity_value": value,
                    "pv_capacity_kw": 1000.0 + 10.0 * index,
                    "battery_energy_kwh": 2000.0 + 100.0 * index,
                    "pv_bess_system_annual_cost_sgd": 3_000_000.0 + 100_000.0 * index,
                    "annualized_pv_cost_sgd": 400_000.0,
                    "annualized_battery_cost_sgd": 500_000.0,
                    "pv_bess_grid_cost_sgd": 600_000.0,
                    "unserved_energy_cost_sgd_per_year": 700_000.0,
                    "unserved_energy_kwh_per_year": 8000.0 - 500.0 * index,
                    "outage_weighted_load_kwh": 20_000.0,
                }
            )
    return pd.DataFrame(rows)


def _apartment_sensitivity_source() -> pd.DataFrame:
    rows = []
    for parameter, values in {
        "battery_capex_sgd_per_kwh": [350.0, 450.0, 550.0],
        "pv_capex_sgd_per_kw": [900.0, 1200.0, 1500.0],
        "saidi_min_per_year": [0.0, 5.0, 10.0],
        "saifi_per_year": [0.0, 1.0, 2.0],
        "voll_sgd_per_kwh": [10.0, 25.0, 40.0],
    }.items():
        for index, value in enumerate(values, start=1):
            rows.append(
                {
                    "sensitivity_parameter": parameter,
                    "sensitivity_value": value,
                    "pv_capacity_kw": 100.0 + 12.5 * index,
                    "battery_energy_kwh": 25.0 * index,
                    "pv_bess_system_annual_cost_sgd": 58_000.0 + 800.0 * index,
                    "annualized_pv_cost_sgd": 12_000.0,
                    "annualized_battery_cost_sgd": 1_500.0,
                    "pv_bess_grid_cost_sgd": 44_000.0,
                    "unserved_energy_cost_sgd_per_year": 100.0,
                    "unserved_energy_kwh_per_year": 20.0 - 4.0 * index,
                    "outage_weighted_load_kwh": 112.0,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_saifi_saidi_source() -> pd.DataFrame:
    rows = []
    for saifi in [0.0, 1.0, 2.0]:
        for saidi in [0.0, 60.0, 120.0]:
            stress = saidi / 60.0 + saifi
            rows.append(
                {
                    "saifi_per_year": saifi,
                    "saidi_min_per_year": saidi,
                    "pv_capacity_kw": 1000.0 + 120.0 * stress,
                    "battery_energy_kwh": 2000.0 + 240.0 * stress,
                    "pv_bess_system_annual_cost_sgd": 3_000_000.0 + 150_000.0 * stress,
                    "unserved_energy_kwh_per_year": 250.0 * stress,
                    "outage_weighted_load_kwh": max(1.0, 10_000.0 * stress),
                }
            )
    return pd.DataFrame(rows)


def _synthetic_uncertainty_baseline_source() -> pd.DataFrame:
    rows = []
    for scenario_id, ue in enumerate([0.0, 100.0, 250.0, 500.0], start=1):
        rows.append(
            {
                "uncertainty_scenario_id": scenario_id,
                "outage_seed": 1000 + scenario_id,
                "pv_capacity_kw": 2000.0 + 100.0 * scenario_id,
                "battery_energy_kwh": 3000.0 + 150.0 * scenario_id,
                "pv_bess_system_annual_cost_sgd": 4_000_000.0 + 50_000.0 * scenario_id,
                "unserved_energy_kwh_per_year": ue,
                "unserved_energy_cost_sgd_per_year": ue * 50.0,
                "outage_weighted_load_kwh": 10_000.0,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_ccp_bess_candidates() -> pd.DataFrame:
    rows = []
    for pv_mw in [8.0, 20.0, 32.0]:
        for bess_mwh in [0.0, 20.0, 40.0, 60.0, 80.0]:
            stress = bess_mwh / 80.0 + pv_mw / 40.0
            ccp = min(1.0, max(0.0, 0.05 + stress * 0.62))
            rows.append(
                {
                    "pv_capacity_kw": pv_mw * 1000.0,
                    "pv_capacity_mw": pv_mw,
                    "battery_energy_kwh": bess_mwh * 1000.0,
                    "frontier_outage_scenario_count": 100,
                    "lolp_threshold": 0.10,
                    "mean_scenario_LOLP": max(0.0, 0.6 - stress * 0.45),
                    "chance_constraint_probability": ccp,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_uncertainty_sensitivity_source() -> pd.DataFrame:
    rows = []
    values_by_parameter = {
        "battery_capex_sgd_per_kwh": [350.0, 450.0, 550.0],
        "pv_capex_sgd_per_kw": [900.0, 1200.0, 1500.0],
        "saidi_min_per_year": [0.0, 60.0, 120.0],
        "saifi_per_year": [0.0, 1.0, 2.0],
        "voll_sgd_per_kwh": [10.0, 25.0, 40.0],
    }
    for p_index, (parameter, values) in enumerate(values_by_parameter.items(), start=1):
        for v_index, value in enumerate(values, start=1):
            for scenario_id in range(1, 4):
                stress = p_index + v_index + scenario_id
                rows.append(
                    {
                        "sensitivity_parameter": parameter,
                        "sensitivity_value": value,
                        "uncertainty_scenario_id": scenario_id,
                        "pv_capacity_kw": 1000.0 + 10.0 * stress,
                        "battery_energy_kwh": 2000.0 + 25.0 * stress,
                        "pv_bess_system_annual_cost_sgd": 3_000_000.0 + 100_000.0 * stress,
                        "annualized_pv_cost_sgd": 400_000.0,
                        "pv_om_cost_sgd": 20_000.0,
                        "annualized_battery_cost_sgd": 500_000.0,
                        "battery_om_cost_sgd": 25_000.0,
                        "pv_bess_grid_cost_sgd": 600_000.0,
                        "unserved_energy_cost_sgd_per_year": 7000.0 * stress,
                        "unserved_energy_kwh_per_year": 80.0 * stress,
                        "outage_weighted_load_kwh": 20_000.0,
                    }
                )
    return pd.DataFrame(rows)


class ResultFigureReconstructionTests(unittest.TestCase):
    def test_make_result_figure_3_is_no_longer_supported(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unsupported result figure number: 3"):
                make_result_figure(
                    Path(tmp),
                    "SG_UTown",
                    {"design_space": _synthetic_design_space_source()},
                    3,
                )

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_make_result_figure_4_exports_reliability_trajectory(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_result_figure(
                Path(tmp),
                "SG_UTown",
                {
                    "reliability_trajectory": _synthetic_reliability_candidates(),
                    "design_space": _synthetic_design_space_source(),
                },
                4,
            )
            self.assertIn("SG_UTown_Figure_4_Reliability_Driven_Sizing_Trajectory_png", paths)
            self.assertIn("source_data", paths)

    def test_removed_result_figure_3_is_not_generated_by_bulk_registry(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_result_figures_new(
                Path(tmp),
                "SG_UTown",
                {
                    "design_space": _synthetic_design_space_source(),
                    "reliability_trajectory": _synthetic_reliability_candidates(),
                    "pareto": _synthetic_reliability_candidates().assign(lolp_threshold=0.2),
                    "sensitivity": _synthetic_sensitivity_source(),
                },
                figures=[3],
            )
            self.assertEqual(paths, {})

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_make_result_figure_5_exports_pareto_frontier(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_result_figure(
                Path(tmp),
                "SG_UTown",
                {"pareto": _synthetic_reliability_candidates().assign(lolp_threshold=0.2)},
                5,
            )
            self.assertIn("SG_UTown_Figure_5_Cost_Resilience_Pareto_Frontier_png", paths)
            self.assertIn("source_data", paths)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_make_result_figure_7_exports_saifi_saidi_surfaces(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_result_figure(
                Path(tmp),
                "SG_UTown",
                {"saifi_saidi": _synthetic_saifi_saidi_source()},
                7,
            )
            self.assertIn("source_data", paths)
            self.assertIn("SG_UTown_Figure_7A_SAIFI_SAIDI_Response_Surfaces_png", paths)
            self.assertIn("SG_UTown_Figure_7B_SAIFI_SAIDI_Contour_Maps_png", paths)

    def test_figure_12_source_selects_fixed_figure_3c_pv_slices(self) -> None:
        source = build_ccp_bess_capacity_response_source(_synthetic_ccp_bess_candidates())

        self.assertEqual(source["pv_capacity_mw"].drop_duplicates().tolist(), [8.0, 20.0, 32.0])
        self.assertEqual(source.groupby("pv_capacity_mw")["battery_energy_mwh"].min().to_dict(), {8.0: 0.0, 20.0: 0.0, 32.0: 0.0})
        self.assertEqual(source.groupby("pv_capacity_mw")["battery_energy_mwh"].max().to_dict(), {8.0: 80.0, 20.0: 80.0, 32.0: 80.0})
        self.assertTrue(source["chance_constraint_probability"].between(0.0, 1.0).all())
        self.assertEqual(source["lolp_threshold"].drop_duplicates().tolist(), [0.1])
        self.assertEqual(source["frontier_outage_scenario_count"].drop_duplicates().tolist(), [100])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_make_result_figure_12_exports_ccp_bess_response(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_result_figure(
                Path(tmp),
                "SG_UTown",
                {"ccp_bess_capacity_response": build_ccp_bess_capacity_response_source(_synthetic_ccp_bess_candidates())},
                12,
            )

            self.assertIn("source_data", paths)
            self.assertIn("SG_UTown_Figure_12_CCP_BESS_Capacity_Response_png", paths)
            self.assertIn("SG_UTown_Figure_12_CCP_BESS_Capacity_Response_svg", paths)
            self.assertIn("SG_UTown_Figure_12_CCP_BESS_Capacity_Response_pdf", paths)
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            svg = paths["SG_UTown_Figure_12_CCP_BESS_Capacity_Response_svg"].read_text()
            self.assertIn("PV = 8 MW", svg)
            self.assertIn("PV = 20 MW", svg)
            self.assertIn("PV = 32 MW", svg)
            self.assertIn("100%", svg)

    def test_figure_7_source_requires_complete_saifi_saidi_grid(self) -> None:
        source = _synthetic_saifi_saidi_source()
        source = source[~((source["saifi_per_year"] == 2.0) & (source["saidi_min_per_year"] == 120.0))]

        with self.assertRaisesRegex(ValueError, "complete rectangular grid"):
            build_saifi_saidi_sensitivity_source(source)

    def test_figure_7_grid_requires_saidi_and_saifi_ranges(self) -> None:
        parameters = SimpleNamespace(sensitivity={"saidi_min_per_year": [0.0, 60.0]})

        with self.assertRaisesRegex(ValueError, "both saidi_min_per_year and saifi_per_year"):
            _compute_saifi_saidi_grid_rows(None, {}, parameters)

    def test_figure_7_grid_defaults_to_one_common_outage_seed(self) -> None:
        case = SimpleNamespace(case_key="SG_UTown")
        parameters = SimpleNamespace(
            base={"uncertainty_seed_start": 1000},
            sensitivity={"saidi_min_per_year": [0.0, 60.0], "saifi_per_year": [0.0, 1.0]},
        )

        def fake_load_input_case(_case, global_config, parameter_overrides=None):
            overrides = parameter_overrides or {}
            saidi = float(overrides.get("saidi_min_per_year", 0.0))
            saifi = float(overrides.get("saifi_per_year", 0.0))
            config = {
                "dispatch": {"backend": "greedy"},
                "outage": {},
                "scenarios": {
                    "SG_UTown": {
                        "saidi_min_per_year": saidi,
                        "saifi_per_year": saifi,
                    }
                },
            }
            return pd.DataFrame(), {}, config, parameters

        def fake_run_design_case(_case, config, case_key, sensitivity_parameter, sensitivity_value):
            scenario = config["scenarios"][case_key]
            seed_offset = int(config["outage"]["seed"]) - 1000
            saidi = float(scenario["saidi_min_per_year"])
            saifi = float(scenario["saifi_per_year"])
            row = pd.DataFrame(
                [
                    {
                        "saifi_per_year": saifi,
                        "saidi_min_per_year": saidi,
                        "pv_capacity_kw": 1000.0 + saidi + seed_offset,
                        "battery_energy_kwh": 2000.0 + 10.0 * saifi + seed_offset,
                        "pv_bess_system_annual_cost_sgd": 3_000_000.0 + saidi + saifi + seed_offset,
                        "unserved_energy_kwh_per_year": 100.0 + saidi + saifi + seed_offset,
                        "outage_weighted_load_kwh": 1000.0,
                        "dispatch_backend": "greedy",
                    }
                ]
            )
            return row, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

        with patch("src.app.workflow.load_input_case", side_effect=fake_load_input_case):
            with patch("src.app.workflow._run_design_case", side_effect=fake_run_design_case):
                rows = _compute_saifi_saidi_grid_rows(case, {}, parameters)

        self.assertEqual(len(rows), 4)
        source = build_saifi_saidi_sensitivity_source(rows)
        self.assertEqual(len(source), 4)
        self.assertEqual(source["scenario_count"].unique().tolist(), [1])
        self.assertEqual(source["outage_seed_start"].unique().tolist(), [1000])
        self.assertEqual(source["outage_seed_end"].unique().tolist(), [1000])
        self.assertEqual(source["dispatch_backend"].unique().tolist(), ["greedy"])

    def test_result_figure_7_does_not_trigger_frontier_generation(self) -> None:
        case = SimpleNamespace(city_key="SG", building_key="UTown", case_key="SG_UTown")
        parameters = SimpleNamespace(base={}, sensitivity={}, raw=pd.DataFrame())
        config = {"outputs": {"directory": "outputs"}}
        with TemporaryDirectory() as tmp:
            with patch("src.app.workflow.discover_input_cases", return_value=[case]):
                with patch("src.app.workflow.load_input_case", return_value=(pd.DataFrame(), {}, config, parameters)):
                    with patch("src.app.workflow.run_resilience_frontier") as frontier:
                        with patch("src.app.workflow.build_result_sources", return_value={"saifi_saidi": pd.DataFrame()}):
                            with patch("src.app.workflow.make_result_figures", return_value={}):
                                run_selected(
                                    config,
                                    Path(tmp) / "Input",
                                    Path(tmp) / "outputs",
                                    run_baseline=False,
                                    run_sensitivity=False,
                                    run_result_figures=True,
                                    case_filters=["SG_UTown"],
                                    result_figures=[7],
                                )
        frontier.assert_not_called()

    def test_make_result_figure_routes_robust_uncertainty_figures(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch("src.figures.robust_uncertainty.make_figure_9_robust_sensitivity_bands") as plot_bands:
                plot_bands.return_value = {"figure": out / "bands.svg"}
                result = make_result_figure_new(
                    out,
                    "SG_UTown",
                    {"robust_sensitivity_bands": _synthetic_uncertainty_sensitivity_source()},
                    9,
                )
            self.assertEqual(result, {"figure": out / "bands.svg"})
            plot_bands.assert_called_once()

            with patch("src.figures.robust_uncertainty.make_figure_10_robust_sensitivity_mechanism") as plot_mechanism:
                plot_mechanism.return_value = {"figure": out / "mechanism.svg"}
                result = make_result_figure_new(
                    out,
                    "SG_UTown",
                    {"robust_sensitivity_mechanism": _synthetic_uncertainty_sensitivity_source()},
                    10,
                )
            self.assertEqual(result, {"figure": out / "mechanism.svg"})
            plot_mechanism.assert_called_once()

    def test_cli_figure_choices_include_operation_figures(self) -> None:
        parser_source = Path("src/app/cli.py").read_text(encoding="utf-8")
        self.assertIn("--figure", parser_source)
        figure_section = parser_source[parser_source.index("--figure") :]
        choices_match = re.search(r"choices=\[(.*?)\]", figure_section, re.DOTALL)
        self.assertIsNotNone(choices_match)
        choices = {int(value) for value in re.findall(r"\d+", choices_match.group(1))}
        self.assertTrue({4, 5, 6, 7, 8, 9, 10, 11, 12}.issubset(choices))
        self.assertFalse({3}.intersection(choices))

    def test_make_result_figure_routes_tail_resilience_figure(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch("src.figures.robust_uncertainty.make_figure_11_tail_robustness") as plot:
                plot.return_value = {"figure": out / "figure.svg"}
                result = make_result_figure_new(
                    out,
                    "SG_UTown",
                    {"tail_robustness": pd.DataFrame({"source_type": ["baseline_exceedance"]})},
                    11,
                )
            self.assertEqual(result, {"figure": out / "figure.svg"})
            plot.assert_called_once()

    def test_legacy_result_figures_wrapper_matches_new_registry(self) -> None:
        self.assertIs(make_result_figure, make_result_figure_new)
        self.assertIs(make_result_figures, make_result_figures_new)

    def test_build_result_sensitivity_source_still_prepares_figure_6_data(self) -> None:
        source = build_result_sensitivity_source(
            pd.DataFrame(
                [
                    {
                        "sensitivity_parameter": "battery_capex_sgd_per_kwh",
                        "sensitivity_value": 450.0,
                        "pv_capacity_kw": 1000.0,
                        "battery_energy_kwh": 2000.0,
                        "pv_bess_system_annual_cost_sgd": 3_000_000.0,
                        "annualized_pv_cost_sgd": 400_000.0,
                        "annualized_battery_cost_sgd": 500_000.0,
                        "pv_bess_grid_cost_sgd": 600_000.0,
                        "unserved_energy_cost_sgd_per_year": 700_000.0,
                        "unserved_energy_kwh_per_year": 8000.0,
                        "outage_weighted_load_kwh": 20_000.0,
                    }
                ]
            )
        )
        self.assertEqual(source["parameter_label"].tolist(), ["Battery CAPEX"])
        self.assertEqual(source["pv_capacity_mw"].tolist(), [1.0])
        self.assertEqual(source["battery_energy_mwh"].tolist(), [2.0])
        self.assertEqual(source["system_total_cost_million_sgd_per_year"].tolist(), [3.0])
        self.assertEqual(source["UE_mwh_per_year"].tolist(), [8.0])
        self.assertEqual(source["energy_based_LOLP"].tolist(), [0.4])
        for metric in METRICS:
            self.assertIn(metric.key, source.columns)
        self.assertTrue(source["energy_based_LOLP"].between(0.0, 1.0).all())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_sensitivity_figure_exports_all_formats_and_source_data(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_sensitivity_comparison_publication_figure(
                Path(tmp),
                "SG_UTown",
                _synthetic_sensitivity_source(),
            )
            self.assertIn("source_data", paths)
            self.assertIn("SG_UTown_Figure_6_CAPEX_SAIDI_SAIFI_VoLL_Sensitivity_png", paths)
            self.assertIn("SG_UTown_Figure_6_CAPEX_SAIDI_SAIFI_VoLL_Sensitivity_svg", paths)
            self.assertIn("SG_UTown_Figure_6_CAPEX_SAIDI_SAIFI_VoLL_Sensitivity_pdf", paths)
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            source = pd.read_csv(paths["source_data"])
            self.assertEqual([metric.key for metric in METRICS], [
                "system_total_cost_million_sgd_per_year",
                "pv_capacity_mw",
                "battery_energy_mwh",
                "UE_mwh_per_year",
                "energy_based_LOLP",
            ])
            for metric in METRICS:
                self.assertIn(metric.key, source.columns)
            self.assertEqual(
                source["sensitivity_parameter"].drop_duplicates().tolist(),
                [
                    "battery_capex_sgd_per_kwh",
                    "pv_capex_sgd_per_kw",
                    "saidi_min_per_year",
                    "saifi_per_year",
                    "voll_sgd_per_kwh",
                ],
            )
            self.assertTrue(source["energy_based_LOLP"].between(0.0, 1.0).all())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_sensitivity_figure_uses_small_units_for_apartment_scale_source(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_sensitivity_comparison_publication_figure(
                Path(tmp),
                "SG_Apartment_Apartment",
                _apartment_sensitivity_source(),
            )
            svg = paths["SG_Apartment_Apartment_Figure_6_CAPEX_SAIDI_SAIFI_VoLL_Sensitivity_svg"].read_text()

        self.assertIn("PV capacity (kW)", svg)
        self.assertIn("Battery capacity (kWh)", svg)
        self.assertIn("UE (kWh", svg)
        self.assertIn("kSGD", svg)
        self.assertNotIn("PV capacity (MW)", svg)
        self.assertNotIn("Battery capacity (MWh)", svg)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_nature_style_sets_editable_vector_text(self) -> None:
        with TemporaryDirectory() as tmp:
            mpl, _ = _matplotlib(Path(tmp))
            _style(mpl)
            self.assertEqual(mpl.rcParams["svg.fonttype"], "none")
            self.assertEqual(mpl.rcParams["pdf.fonttype"], 42)
            self.assertFalse(mpl.rcParams["axes.spines.right"])
            self.assertFalse(mpl.rcParams["axes.spines.top"])
            self.assertFalse(mpl.rcParams["legend.frameon"])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_save_all_exports_png_svg_and_pdf(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            mpl, plt = _matplotlib(out)
            _style(mpl)
            fig, ax = plt.subplots(figsize=(3.0, 2.0), constrained_layout=True)
            ax.plot([0, 1], [0, 1], color="#0F4D92")
            ax.set_title("a  Test panel")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            paths = _save_all(fig, out / "test_figure")
            plt.close(fig)
            self.assertEqual(set(paths), {"test_figure_png", "test_figure_svg", "test_figure_pdf"})
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_save_all_rejects_text_outside_canvas(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            mpl, plt = _matplotlib(out)
            _style(mpl)
            fig, ax = plt.subplots(figsize=(2.0, 1.2))
            ax.plot([0, 1], [0, 1])
            ax.set_title("a  Far title", y=2.5)
            with self.assertRaisesRegex(ValueError, "extends beyond the figure canvas"):
                _save_all(fig, out / "bad_figure")
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
