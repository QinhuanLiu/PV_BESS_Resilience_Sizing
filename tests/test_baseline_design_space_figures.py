from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import pandas as pd

from pv_bess_sizing import run_selected
from src.io.config import DEFAULT_CONFIG
from src.figures.baseline_design_space import (
    DISPLAY_UE,
    DISPLAY_SYSTEM_COST,
    _contour_levels,
    _display_baseline_metrics,
    _figure_3b_reference_marker_coordinates,
    build_baseline_design_space_source,
    build_baseline_pv_capacity_slice_source,
    make_baseline_design_space_figures,
    plot_baseline_pv_capacity_bess_slices,
    plot_baseline_design_space_contours,
    select_baseline_pv_capacity_slices,
    validate_baseline_design_space_source,
)
from src.figures.display_units import add_display_unit_columns
from src.figures.baseline_cost_comparison import (
    build_baseline_cost_comparison_source,
    make_baseline_cost_comparison_figure,
)
from src.figures.baseline_outage_balance import (
    SELECTION_RULE_AVOIDED_UE,
    build_baseline_outage_balance_comparison_sources,
    make_baseline_outage_balance_comparison_figure,
    _select_baseline_design_rows,
)
from src.io.data_loader import discover_input_cases, load_input_case


def _write_fake_figure4_outputs(out: Path, case_key: str) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for stem in ("Figure_4A_Baseline_Cost_Comparison", "Figure_4B_Baseline_Cost_Comparison"):
        base = out / f"{case_key}_{stem}"
        for suffix in ("png", "svg", "pdf"):
            path = base.with_suffix(f".{suffix}")
            path.write_text("figure", encoding="utf-8")
            paths[f"{base.name}_{suffix}"] = path
        source = base.with_name(base.name + "_Source_Data.csv")
        source.write_text("design_case\nfake\n", encoding="utf-8")
        paths[f"{stem.lower()}_source_data"] = source
    return paths


def _write_fake_figure5_outputs(out: Path, case_key: str, *_args, **_kwargs) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{case_key}_Figure_5_Baseline_Outage_Energy_Balance_Comparison"
    paths: dict[str, Path] = {}
    for suffix in ("png", "svg", "pdf"):
        path = base.with_suffix(f".{suffix}")
        path.write_text("figure", encoding="utf-8")
        paths[f"{base.name}_{suffix}"] = path
    source = base.with_name(base.name + "_Source_Data.csv")
    source.write_text("design_case\nfake\n", encoding="utf-8")
    paths["source_data"] = source
    summary = base.with_name(base.name + "_Event_Summary.csv")
    summary.write_text("event_id\n1\n", encoding="utf-8")
    paths["event_summary_source_data"] = summary
    return paths


def _config() -> dict:
    return {
        "technical": {
            "cooling_cop": 5.0,
            "chiller_cop": 5.0,
            "heating_cop": 3.5,
        }
    }


def _case_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour_index": [0, 1, 2, 3],
            "electric_load_kwh": [10.0, 10.0, 10.0, 10.0],
            "cooling_load_kwh_th": [5.0, 5.0, 5.0, 5.0],
            "heating_load_kwh_th": [0.0, 0.0, 0.0, 0.0],
        }
    )


def _outage_profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour_index": [0, 1, 2, 3],
            "grid_availability": [1.0, 0.5, 0.0, 1.0],
            "outage_fraction": [0.0, 0.5, 1.0, 0.0],
        }
    )


def _candidate_summary() -> pd.DataFrame:
    rows = []
    for pv in [0.0, 8000.0, 22000.0, 32000.0]:
        for bess in [0.0, 1000.0, 2000.0]:
            ue = max(0.0, 12.0 - 0.003 * pv - 0.004 * bess)
            rows.append(
                {
                    "search_stage": "coarse",
                    "search_iteration": 0,
                    "pv_capacity_kw": pv,
                    "battery_energy_kwh": bess,
                    "pv_bess_system_annual_cost_sgd": 1_000_000.0 + 40.0 * pv + 60.0 * bess,
                    "unserved_energy_kwh_per_year": ue,
                }
            )
    return pd.DataFrame(rows)


def _cost_candidate_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pv_capacity_kw": 0.0,
                "battery_energy_kwh": 0.0,
                "pv_bess_system_annual_cost_sgd": 1_350_000.0,
                "pv_bess_grid_cost_sgd": 1_000_000.0,
                "annualized_pv_cost_sgd": 0.0,
                "pv_om_cost_sgd": 0.0,
                "annualized_battery_cost_sgd": 0.0,
                "battery_om_cost_sgd": 0.0,
                "unserved_energy_kwh_per_year": 10_000.0,
                "unserved_energy_cost_sgd_per_year": 350_000.0,
            },
            {
                "pv_capacity_kw": 20_000.0,
                "battery_energy_kwh": 0.0,
                "pv_bess_system_annual_cost_sgd": 1_220_000.0,
                "pv_bess_grid_cost_sgd": 760_000.0,
                "annualized_pv_cost_sgd": 480_000.0,
                "pv_om_cost_sgd": 18_000.0,
                "annualized_battery_cost_sgd": 0.0,
                "battery_om_cost_sgd": 0.0,
                "unserved_energy_kwh_per_year": 5_000.0,
                "unserved_energy_cost_sgd_per_year": 175_000.0,
            },
            {
                "pv_capacity_kw": 20_000.0,
                "battery_energy_kwh": 10_000.0,
                "pv_bess_system_annual_cost_sgd": 1_150_000.0,
                "pv_bess_grid_cost_sgd": 420_000.0,
                "annualized_pv_cost_sgd": 480_000.0,
                "pv_om_cost_sgd": 18_000.0,
                "annualized_battery_cost_sgd": 80_000.0,
                "battery_om_cost_sgd": 6_000.0,
                "unserved_energy_kwh_per_year": 1_000.0,
                "unserved_energy_cost_sgd_per_year": 40_000.0,
            },
            {
                "pv_capacity_kw": 32_000.0,
                "battery_energy_kwh": 40_000.0,
                "pv_bess_system_annual_cost_sgd": 1_450_000.0,
                "pv_bess_grid_cost_sgd": 300_000.0,
                "annualized_pv_cost_sgd": 720_000.0,
                "pv_om_cost_sgd": 24_000.0,
                "annualized_battery_cost_sgd": 320_000.0,
                "battery_om_cost_sgd": 16_000.0,
                "unserved_energy_kwh_per_year": 500.0,
                "unserved_energy_cost_sgd_per_year": 20_000.0,
            },
        ]
    )


def _small_cost_candidate_summary() -> pd.DataFrame:
    rows = []
    for pv_kw, bess_kwh, cost in [
        (0.0, 0.0, 13_500.0),
        (100.0, 0.0, 12_200.0),
        (200.0, 0.0, 12_000.0),
        (100.0, 100.0, 11_500.0),
        (200.0, 100.0, 11_000.0),
        (200.0, 200.0, 11_800.0),
    ]:
        rows.append(
            {
                "pv_capacity_kw": pv_kw,
                "battery_energy_kwh": bess_kwh,
                "pv_bess_system_annual_cost_sgd": cost,
                "pv_bess_grid_cost_sgd": cost * 0.45,
                "annualized_pv_cost_sgd": pv_kw * 10.0,
                "pv_om_cost_sgd": pv_kw,
                "annualized_battery_cost_sgd": bess_kwh * 5.0,
                "battery_om_cost_sgd": bess_kwh * 0.5,
                "unserved_energy_kwh_per_year": max(0.0, 100.0 - pv_kw * 0.2 - bess_kwh * 0.1),
                "unserved_energy_cost_sgd_per_year": max(0.0, 100.0 - pv_kw * 0.2 - bess_kwh * 0.1) * 35.0,
            }
        )
    return pd.DataFrame(rows)


def _manual_full_lolp_source() -> pd.DataFrame:
    rows = []
    for pv in [0.0, 1.0, 2.0]:
        for bess in [0.0, 1.0, 2.0]:
            lolp = max(0.0, min(1.0, 1.0 - 0.25 * pv - 0.25 * bess))
            rows.append(
                {
                    "pv_capacity_kw": pv * 1000.0,
                    "pv_capacity_mw": pv,
                    "battery_energy_kwh": bess * 1000.0,
                    "battery_energy_mwh": bess,
                    "system_total_cost_sgd_per_year": 1_000_000.0 + 500_000.0 * pv + 400_000.0 * bess,
                    "system_total_cost_million_sgd_per_year": 1.0 + 0.5 * pv + 0.4 * bess,
                    "UE_kwh_per_year": 10_000.0 * lolp,
                    "UE_mwh_per_year": 10.0 * lolp,
                    "outage_weighted_load_kwh": 10_000.0,
                    "energy_based_LOLP": lolp,
                }
            )
    return pd.DataFrame(rows)


def _sg_like_source_with_rooftop_reference() -> pd.DataFrame:
    rows = []
    for pv in [0.0, 8.0, 20.0, 32.0]:
        for bess in [0.0, 20.0, 40.0, 60.0, 80.0]:
            ue = max(0.0, 120.0 - 1.5 * pv - 0.7 * bess)
            rows.append(
                {
                    "pv_capacity_kw": pv * 1000.0,
                    "pv_capacity_mw": pv,
                    "battery_energy_kwh": bess * 1000.0,
                    "battery_energy_mwh": bess,
                    "system_total_cost_sgd_per_year": 5_000_000.0 + 120_000.0 * pv + 60_000.0 * bess,
                    "system_total_cost_million_sgd_per_year": 5.0 + 0.12 * pv + 0.06 * bess,
                    "UE_kwh_per_year": ue * 1000.0,
                    "UE_mwh_per_year": ue,
                    "outage_weighted_load_kwh": 500_000.0,
                    "energy_based_LOLP": ue * 1000.0 / 500_000.0,
                    "rooftop_pv_limit_kw": 40_500.0,
                    "rooftop_pv_limit_mw": 40.5,
                    "twice_rooftop_pv_limit_kw": 81_000.0,
                    "twice_rooftop_pv_limit_mw": 81.0,
                }
            )
    return pd.DataFrame(rows)


def _apartment_like_design_space_source() -> pd.DataFrame:
    rows = []
    for pv_kw in [0.0, 50.0, 100.0, 125.0, 150.0, 200.0]:
        for bess_kwh in [0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 600.0]:
            cost = 59_000.0 + 1.5 * (pv_kw - 125.0) ** 2 + 0.08 * (bess_kwh - 25.0) ** 2
            ue = max(0.0, 112.0 - 0.7 * pv_kw - 0.4 * bess_kwh)
            rows.append(
                {
                    "pv_capacity_kw": pv_kw,
                    "pv_capacity_mw": pv_kw / 1000.0,
                    "battery_energy_kwh": bess_kwh,
                    "battery_energy_mwh": bess_kwh / 1000.0,
                    "system_total_cost_sgd_per_year": cost,
                    "system_total_cost_million_sgd_per_year": cost / 1_000_000.0,
                    "UE_kwh_per_year": ue,
                    "UE_mwh_per_year": ue / 1000.0,
                    "outage_weighted_load_kwh": 112.0,
                    "energy_based_LOLP": ue / 112.0,
                }
            )
    return pd.DataFrame(rows)


PARAMETER_COLUMNS = [
    "parameter",
    "base_value",
    "unit",
    "sensitivity_start",
    "sensitivity_end",
    "sensitivity_step",
    "description",
    "中文说明",
]


def _write_valid_input_case(root: Path) -> Path:
    case_dir = root / "Input" / "SG"
    case_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Cooling": [25.0] * 8760,
            "Heating": [0.0] * 8760,
            "Lighting": [6.0] * 8760,
            "Equipment": [4.0] * 8760,
        }
    ).to_csv(case_dir / "SG_UTown_Load.csv", index=False)
    pd.DataFrame({"PV_output_kWh_per_m2": [0.000104] * 8760}).to_csv(case_dir / "SG_UTown_PV.csv", index=False)
    pd.DataFrame({"hour": range(24), "tou_sgd_per_kwh": [0.239] * 9 + [0.329] * 12 + [0.239] * 3}).to_csv(
        case_dir / "SG_UTown_TOU.csv",
        index=False,
    )
    rows = [
        ("pv_capex_sgd_per_kw", 1200, "SGD/kW", "", "", "", "PV unit CAPEX.", "光伏单位装机投资成本。"),
        ("battery_capex_sgd_per_kwh", 450, "SGD/kWh", "", "", "", "Battery energy CAPEX.", "电池储能单位能量容量投资成本。"),
        ("battery_power_capex_sgd_per_kw", 180, "SGD/kW", "", "", "", "Battery power CAPEX.", "电池储能单位功率容量投资成本。"),
        ("saidi_min_per_year", 0.26, "min/year", "", "", "", "Annual outage duration.", "年平均停电持续时间。"),
        ("saifi_per_year", 0.006, "events/year", "", "", "", "Annual outage count.", "年平均停电次数。"),
        ("voll_sgd_per_kwh", 35, "SGD/kWh", "", "", "", "Value of lost load.", "单位未供电电量对应的停电损失价值。"),
        ("city_name", "SG", "-", "", "", "", "City key.", "城市或地区名称。"),
        ("building_name", "UTown", "-", "", "", "", "Building key.", "建筑或建筑群名称。"),
        ("rooftop_area_m2", 1000, "m2", "", "", "", "Effective rooftop area available for PV.", "可安装光伏的有效屋顶面积。"),
        ("pv_module_power_density_kw_per_m2", 0.2, "kW/m2", "", "", "", "PV module power density.", "单位面积光伏组件装机功率。"),
        ("local_refine_step_kw", 500, "kW", "", "", "", "PV local refinement step.", "光伏容量局部细化步长。"),
        ("local_refine_step_kwh", 500, "kWh", "", "", "", "BESS local refinement step.", "储能容量局部细化步长。"),
        ("cooling_cop", 5.0, "-", "", "", "", "Cooling COP.", "制冷系统 COP。"),
        ("heating_cop", 3.5, "-", "", "", "", "Heating COP.", "供热系统 COP。"),
        ("battery_om_fraction_per_year", 0.02, "fraction/year", "", "", "", "Battery O&M.", "电池年度运维成本占 CAPEX 比例。"),
        ("uncertainty_baseline_scenario_count", 1, "scenarios", "", "", "", "Uncertainty baseline count.", "不确定性基准场景数量。"),
        ("uncertainty_sensitivity_scenario_count", 1, "scenarios", "", "", "", "Uncertainty sensitivity count.", "不确定性敏感性场景数量。"),
        ("uncertainty_seed_start", 2000, "-", "", "", "", "Uncertainty seed start.", "不确定性随机种子。"),
        ("parallel_workers", 1, "workers", "", "", "", "Parallel workers.", "并行进程数。"),
    ]
    pd.DataFrame([dict(zip(PARAMETER_COLUMNS, row, strict=True)) for row in rows], columns=PARAMETER_COLUMNS).to_csv(
        case_dir / "SG_UTown_Parameter.csv",
        index=False,
    )
    return root / "Input"


class BaselineDesignSpaceFigureTests(unittest.TestCase):
    def test_source_builder_uses_single_outage_profile_and_no_ccp_columns(self) -> None:
        source = build_baseline_design_space_source(_candidate_summary(), _case_frame(), _outage_profile(), _config())

        self.assertIn("system_total_cost_million_sgd_per_year", source.columns)
        self.assertIn("UE_mwh_per_year", source.columns)
        self.assertIn("energy_based_LOLP", source.columns)
        self.assertNotIn("chance_constraint_probability", source.columns)
        outage_weighted_load = (10.0 + 5.0 / 5.0) * 1.5
        first = source[(source["pv_capacity_kw"] == 0.0) & (source["battery_energy_kwh"] == 0.0)].iloc[0]
        self.assertAlmostEqual(first["outage_weighted_load_kwh"], outage_weighted_load)
        self.assertAlmostEqual(first["energy_based_LOLP"], first["UE_kwh_per_year"] / outage_weighted_load)

    def test_source_builder_returns_zero_lolp_when_no_outage_load_exists(self) -> None:
        outage = _outage_profile().copy()
        outage["outage_fraction"] = 0.0
        source = build_baseline_design_space_source(_candidate_summary(), _case_frame(), outage, _config())

        self.assertEqual(source["outage_weighted_load_kwh"].unique().tolist(), [0.0])
        self.assertEqual(source["energy_based_LOLP"].unique().tolist(), [0.0])

    def test_validate_rejects_ccp_only_frontier_style_source(self) -> None:
        broken = pd.DataFrame(
            {
                "pv_capacity_mw": [1.0],
                "battery_energy_mwh": [1.0],
                "system_total_cost_million_sgd_per_year": [1.0],
                "UE_mwh_per_year": [0.1],
                "chance_constraint_probability": [0.9],
            }
        )

        with self.assertRaisesRegex(ValueError, "missing required column"):
            validate_baseline_design_space_source(broken)

    def test_figure_3c_source_selects_informative_low_medium_high_pv_slices(self) -> None:
        source = _sg_like_source_with_rooftop_reference()

        slices = select_baseline_pv_capacity_slices(source)
        self.assertEqual([spec.label for spec in slices], ["Low PV", "Medium PV", "High PV"])
        self.assertEqual([spec.selected_mw for spec in slices], [8.0, 20.0, 32.0])
        self.assertEqual({spec.source for spec in slices}, {"fixed_8_20_32_mw"})

        slice_source = build_baseline_pv_capacity_slice_source(source)
        self.assertEqual(set(slice_source["pv_slice_label"]), {"Low PV", "Medium PV", "High PV"})
        self.assertEqual(slice_source["pv_capacity_mw"].nunique(), 3)
        self.assertIn("pv_slice_source", slice_source.columns)
        self.assertNotIn("chance_constraint_probability", slice_source.columns)

    def test_figure_3c_slice_selection_falls_back_when_fixed_targets_are_missing(self) -> None:
        source = _sg_like_source_with_rooftop_reference()
        source = source[~source["pv_capacity_mw"].eq(20.0)].copy()

        slices = select_baseline_pv_capacity_slices(source)

        self.assertEqual([spec.label for spec in slices], ["Low PV", "Medium PV", "High PV"])
        self.assertEqual([spec.selected_mw for spec in slices], [0.0, 8.0, 32.0])
        self.assertEqual({spec.source for spec in slices}, {"adaptive_low_medium_high_mw"})

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_figure_3c_uses_smooth_lines_without_markers(self) -> None:
        source = build_baseline_pv_capacity_slice_source(_sg_like_source_with_rooftop_reference())
        calls = []

        def fake_smooth_line(ax, x, y, **kwargs):
            calls.append(kwargs)
            (line,) = ax.plot(x, y, color=kwargs["color"], linewidth=kwargs["linewidth"], label=kwargs["label"])
            return line

        with TemporaryDirectory() as tmp, patch(
            "src.figures.baseline_design_space.plot_smooth_line",
            side_effect=fake_smooth_line,
        ):
            paths = plot_baseline_pv_capacity_bess_slices(Path(tmp), "SG_UTown", source)

            self.assertTrue((Path(tmp) / "SG_UTown_Figure_3C_Baseline_PV_Capacity_BESS_Slices.png").exists())
            self.assertIn("SG_UTown_Figure_3C_Baseline_PV_Capacity_BESS_Slices_svg", paths)
            self.assertEqual(len(calls), 9)
            self.assertTrue(all("marker" not in call or call["marker"] is None for call in calls))

            svg = paths["SG_UTown_Figure_3C_Baseline_PV_Capacity_BESS_Slices_svg"].read_text()
            self.assertIn("Low PV", svg)
            self.assertIn("Medium PV", svg)
            self.assertIn("High PV", svg)
            self.assertNotIn("CCP", svg)
            self.assertNotIn("Economically acceptable", svg)
            self.assertNotIn("Resilience feasible", svg)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is required for smooth Figure 3C curves")
    def test_baseline_figures_export_all_formats_without_feasibility_text(self) -> None:
        source = build_baseline_design_space_source(_candidate_summary(), _case_frame(), _outage_profile(), _config())
        with TemporaryDirectory() as tmp:
            paths = make_baseline_design_space_figures(Path(tmp), "SG_UTown", source)

            expected = {
                "source_data",
                "SG_UTown_Figure_3A_Baseline_PV_BESS_3D_Response_Surfaces_png",
                "SG_UTown_Figure_3A_Baseline_PV_BESS_3D_Response_Surfaces_svg",
                "SG_UTown_Figure_3A_Baseline_PV_BESS_3D_Response_Surfaces_pdf",
                "SG_UTown_Figure_3B_Baseline_PV_BESS_Contour_Maps_png",
                "SG_UTown_Figure_3B_Baseline_PV_BESS_Contour_Maps_svg",
                "SG_UTown_Figure_3B_Baseline_PV_BESS_Contour_Maps_pdf",
                "figure_3c_source_data",
                "SG_UTown_Figure_3C_Baseline_PV_Capacity_BESS_Slices_png",
                "SG_UTown_Figure_3C_Baseline_PV_Capacity_BESS_Slices_svg",
                "SG_UTown_Figure_3C_Baseline_PV_Capacity_BESS_Slices_pdf",
            }
            self.assertTrue(expected.issubset(paths))
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

            contour_svg = paths["SG_UTown_Figure_3B_Baseline_PV_BESS_Contour_Maps_svg"].read_text()
            self.assertNotIn("CCP", contour_svg)
            self.assertNotIn("Economically acceptable", contour_svg)
            self.assertNotIn("Resilience feasible", contour_svg)
            slice_source = pd.read_csv(paths["figure_3c_source_data"])
            self.assertEqual(set(slice_source["pv_slice_label"]), {"Low PV", "Medium PV", "High PV"})
            self.assertEqual(slice_source["pv_capacity_mw"].nunique(), 3)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_contour_colorbar_labels_use_five_equal_divisions(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = plot_baseline_design_space_contours(Path(tmp), "SG_UTown", _manual_full_lolp_source())
            contour_svg = paths["SG_UTown_Figure_3B_Baseline_PV_BESS_Contour_Maps_svg"].read_text()

            for label in ("20%", "40%", "60%", "80%"):
                self.assertIn(label, contour_svg)
        self.assertNotIn("25%", contour_svg)
        self.assertNotIn("75%", contour_svg)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_apartment_scale_figure_3b_uses_small_units_on_canvas(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = plot_baseline_design_space_contours(Path(tmp), "SG_Apartment_Apartment", _apartment_like_design_space_source())
            contour_svg = paths["SG_Apartment_Apartment_Figure_3B_Baseline_PV_BESS_Contour_Maps_svg"].read_text()

        self.assertIn("PV capacity (kW)", contour_svg)
        self.assertIn("BESS energy (kWh)", contour_svg)
        self.assertIn("Cost (kSGD", contour_svg)
        self.assertIn("UE (kWh", contour_svg)
        self.assertNotIn("PV capacity (MW)", contour_svg)
        self.assertNotIn("BESS energy (MWh)", contour_svg)
        self.assertNotIn("Cost (million SGD", contour_svg)

    def test_apartment_scale_figure_3b_uses_fine_display_levels_and_best_design_marker(self) -> None:
        display_source, units = add_display_unit_columns(_apartment_like_design_space_source())
        metrics = _display_baseline_metrics(units)

        cost_levels = _contour_levels(display_source[DISPLAY_SYSTEM_COST].to_numpy(dtype=float), metrics[0])
        ue_levels = _contour_levels(display_source[DISPLAY_UE].to_numpy(dtype=float), metrics[1])
        marker = _figure_3b_reference_marker_coordinates(display_source)

        self.assertGreaterEqual(len(cost_levels), 10)
        self.assertGreaterEqual(len(ue_levels), 10)
        self.assertAlmostEqual(marker[0], 125.0)
        self.assertAlmostEqual(marker[1], 25.0)

    def test_baseline_cost_comparison_source_selects_fixed_20_mw_reference_designs(self) -> None:
        source = build_baseline_cost_comparison_source(
            _cost_candidate_summary(),
            {"active_scenario": {"voll_sgd_per_kwh": 35.0}},
        )

        self.assertEqual(source["design_case"].tolist(), ["no_pv_bess", "pv20_no_bess", "pv20_bess10"])
        self.assertEqual(
            source["design_case_label"].tolist(),
            ["No PV/BESS", "20 MW PV / 0 MWh Battery", "20 MW PV / 10 MWh Battery"],
        )
        self.assertEqual(source["pv_capacity_mw"].tolist(), [0.0, 20.0, 20.0])
        self.assertEqual(source["battery_energy_mwh"].tolist(), [0.0, 0.0, 10.0])
        self.assertAlmostEqual(source.loc[1, "pv_annual_cost_sgd_per_year"], 498_000.0)
        self.assertAlmostEqual(source.loc[2, "pv_annual_cost_sgd_per_year"], 498_000.0)
        self.assertAlmostEqual(source.loc[2, "battery_annual_cost_sgd_per_year"], 86_000.0)
        self.assertAlmostEqual(source.loc[2, "lost_load_cost_sgd_per_year"], 40_000.0)

    def test_baseline_cost_comparison_source_falls_back_for_small_capacity_grid(self) -> None:
        source = build_baseline_cost_comparison_source(
            _small_cost_candidate_summary(),
            {"active_scenario": {"voll_sgd_per_kwh": 35.0}},
        )

        self.assertEqual(source["design_case"].tolist(), ["no_pv_bess", "pv20_no_bess", "pv20_bess10"])
        self.assertEqual(source["pv_capacity_kw"].tolist(), [0.0, 200.0, 200.0])
        self.assertEqual(source["battery_energy_kwh"].tolist(), [0.0, 0.0, 100.0])
        self.assertEqual(
            source["design_case_label"].tolist(),
            ["No PV/BESS", "0.2 MW PV / 0 MWh Battery", "0.2 MW PV / 0.1 MWh Battery"],
        )

    def test_baseline_outage_balance_design_selection_falls_back_for_small_capacity_grid(self) -> None:
        rows = _select_baseline_design_rows(_small_cost_candidate_summary())

        self.assertEqual(float(rows["no_pv_bess"]["pv_capacity_kw"]), 0.0)
        self.assertEqual(float(rows["pv20_no_bess"]["pv_capacity_kw"]), 200.0)
        self.assertEqual(float(rows["pv20_no_bess"]["battery_energy_kwh"]), 0.0)
        self.assertEqual(float(rows["pv20_bess10"]["pv_capacity_kw"]), 200.0)
        self.assertEqual(float(rows["pv20_bess10"]["battery_energy_kwh"]), 100.0)

    def test_baseline_cost_comparison_falls_back_to_ue_times_voll(self) -> None:
        summary = _cost_candidate_summary().drop(columns=["unserved_energy_cost_sgd_per_year"])

        source = build_baseline_cost_comparison_source(
            summary,
            {"active_scenario": {"voll_sgd_per_kwh": 35.0}},
        )

        self.assertAlmostEqual(source.loc[0, "lost_load_cost_sgd_per_year"], 350_000.0)
        self.assertAlmostEqual(source.loc[1, "lost_load_cost_sgd_per_year"], 175_000.0)
        self.assertAlmostEqual(source.loc[2, "lost_load_cost_sgd_per_year"], 35_000.0)

    def test_baseline_cost_comparison_requires_no_pv_no_bess_reference(self) -> None:
        summary = _cost_candidate_summary()
        summary = summary[~((summary["pv_capacity_kw"] == 0.0) & (summary["battery_energy_kwh"] == 0.0))]

        with self.assertRaisesRegex(ValueError, "no-PV/no-BESS reference"):
            build_baseline_cost_comparison_source(summary, {"active_scenario": {"voll_sgd_per_kwh": 35.0}})

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_baseline_cost_comparison_figure_exports_all_formats_and_source_data(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = make_baseline_cost_comparison_figure(
                Path(tmp),
                "SG_UTown",
                _cost_candidate_summary(),
                {"active_scenario": {"voll_sgd_per_kwh": 35.0}},
            )

            expected = {
                "figure_4a_source_data",
                "figure_4b_source_data",
                "SG_UTown_Figure_4A_Baseline_Cost_Comparison_png",
                "SG_UTown_Figure_4A_Baseline_Cost_Comparison_svg",
                "SG_UTown_Figure_4A_Baseline_Cost_Comparison_pdf",
                "SG_UTown_Figure_4B_Baseline_Cost_Comparison_png",
                "SG_UTown_Figure_4B_Baseline_Cost_Comparison_svg",
                "SG_UTown_Figure_4B_Baseline_Cost_Comparison_pdf",
            }
            self.assertTrue(expected.issubset(paths))
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            source_4a = pd.read_csv(paths["figure_4a_source_data"])
            source_4b = pd.read_csv(paths["figure_4b_source_data"])
            self.assertEqual(source_4a["design_case"].tolist(), ["no_pv_bess", "pv20_bess10"])
            self.assertEqual(source_4b["design_case"].tolist(), ["no_pv_bess", "pv20_no_bess", "pv20_bess10"])
            svg = paths["SG_UTown_Figure_4B_Baseline_Cost_Comparison_svg"].read_text()
            self.assertIn("Annual grid bill", svg)
            self.assertIn("PV depreciation + O&amp;M", svg)
            self.assertIn("Battery depreciation + O&amp;M", svg)
            self.assertIn("Lost load cost", svg)
            self.assertIn("20 MW PV", svg)

    def _baseline_outage_case(self, root: Path):
        input_dir = _write_valid_input_case(root)
        case = discover_input_cases(input_dir)[0]
        config = deepcopy(DEFAULT_CONFIG)
        config.setdefault("dispatch", {})["backend"] = "greedy"
        _, _, case_config, _ = load_input_case(case, config)
        outage_profile = pd.DataFrame(
            {
                "hour_index": range(8760),
                "grid_availability": [1.0] * 8760,
                "outage_fraction": [0.0] * 8760,
            }
        )
        for hour in [100, 200, 201]:
            outage_profile.loc[outage_profile["hour_index"].eq(hour), "grid_availability"] = 0.0
            outage_profile.loc[outage_profile["hour_index"].eq(hour), "outage_fraction"] = 1.0
        outage_events = pd.DataFrame(
            [
                {"event_id": 1, "start_hour": 100.0, "end_hour": 101.0, "duration_minutes": 60.0},
                {"event_id": 2, "start_hour": 200.0, "end_hour": 202.0, "duration_minutes": 120.0},
            ]
        )
        candidate_summary = pd.DataFrame(
            [
                {
                    "pv_capacity_kw": 0.0,
                    "battery_energy_kwh": 0.0,
                    "pv_bess_system_annual_cost_sgd": 2_000_000.0,
                },
                {
                    "pv_capacity_kw": 20_000.0,
                    "battery_energy_kwh": 0.0,
                    "pv_bess_system_annual_cost_sgd": 1_500_000.0,
                },
                {
                    "pv_capacity_kw": 20_000.0,
                    "battery_energy_kwh": 10_000.0,
                    "pv_bess_system_annual_cost_sgd": 1_000_000.0,
                },
            ]
        )
        return case, case_config, candidate_summary, outage_profile, outage_events

    def test_baseline_outage_balance_selects_max_avoided_ue_event(self) -> None:
        with TemporaryDirectory() as tmp:
            case, case_config, candidate_summary, outage_profile, outage_events = self._baseline_outage_case(Path(tmp))

            summary, source = build_baseline_outage_balance_comparison_sources(
                case,
                case_config,
                candidate_summary,
                outage_profile,
                outage_events,
            )

            self.assertEqual(summary["selection_rule"].tolist(), [SELECTION_RULE_AVOIDED_UE])
            self.assertEqual(summary["event_id"].tolist(), [2])
            expected_cases = {"no_pv_bess", "pv20_no_bess", "pv20_bess10"}
            self.assertEqual(set(source["design_case"]), expected_cases)
            self.assertEqual(source.groupby("design_case").size().to_dict(), {case: 72 for case in expected_cases})
            self.assertEqual(
                source.groupby("design_case")["event_id"].nunique().to_dict(),
                {case: 1 for case in expected_cases},
            )
            self.assertGreater(float(summary["avoided_UE_kwh"].iloc[0]), 0.0)
            self.assertEqual(summary["comparison_design_case"].tolist(), ["pv20_bess10"])
            self.assertEqual(summary["comparison_pv_capacity_kw"].tolist(), [20_000.0])
            self.assertEqual(summary["comparison_battery_energy_kwh"].tolist(), [10_000.0])
            for column in [
                "event_reference_UE_kwh",
                "event_optimal_UE_kwh",
                "event_comparison_UE_kwh",
                "avoided_UE_kwh",
                "event_pv_bess_support_kwh",
            ]:
                self.assertIn(column, source.columns)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    def test_baseline_outage_balance_figure_exports_all_formats_and_source_data(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            case, case_config, candidate_summary, outage_profile, outage_events = self._baseline_outage_case(root)

            paths = make_baseline_outage_balance_comparison_figure(
                root / "figures",
                "SG_UTown",
                case,
                case_config,
                candidate_summary,
                outage_profile,
                outage_events,
            )

            expected = {
                "source_data",
                "event_summary_source_data",
                "SG_UTown_Figure_5_Baseline_Outage_Energy_Balance_Comparison_png",
                "SG_UTown_Figure_5_Baseline_Outage_Energy_Balance_Comparison_svg",
                "SG_UTown_Figure_5_Baseline_Outage_Energy_Balance_Comparison_pdf",
            }
            self.assertTrue(expected.issubset(paths))
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            source = pd.read_csv(paths["source_data"])
            self.assertEqual(set(source["design_case"]), {"no_pv_bess", "pv20_no_bess", "pv20_bess10"})
            self.assertEqual(len(source), 216)
            svg = paths["SG_UTown_Figure_5_Baseline_Outage_Energy_Balance_Comparison_svg"].read_text()
            self.assertIn("No PV/BESS", svg)
            self.assertIn("20 MW PV / 0 MWh Battery", svg)
            self.assertIn("20 MW PV / 10 MWh Battery", svg)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is required for smooth Figure 3C curves")
    def test_run_selected_baseline_writes_baseline_figure_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _write_valid_input_case(root)
            output_dir = root / "outputs"
            config = DEFAULT_CONFIG.copy()
            config["capacity_search"] = dict(DEFAULT_CONFIG["capacity_search"])
            config["capacity_search"]["coarse_intervals"] = 8
            config["capacity_search"]["local_refine_divisions"] = 1
            config["capacity_search"]["max_upper_expansions"] = 0
            config["dispatch"] = dict(DEFAULT_CONFIG["dispatch"])
            config["dispatch"]["backend"] = "greedy"

            def fake_figure4(out, case_key, _summary, _config):
                return _write_fake_figure4_outputs(Path(out), case_key)

            with (
                patch("src.app.outputs.write_case_outputs", return_value={}),
                patch("src.app.workflow.make_baseline_cost_comparison_figure", side_effect=fake_figure4),
                patch("src.app.workflow.make_baseline_outage_balance_comparison_figure", side_effect=_write_fake_figure5_outputs),
            ):
                paths = run_selected(
                    config,
                    input_dir,
                    output_dir,
                    run_baseline=True,
                    run_sensitivity=False,
                    case_filters=["SG_UTown"],
                )

            figures = output_dir / "SG" / "SG_UTown_Baseline" / "SG_UTown_Baseline_Result_Figures"
            source = figures / "SG_UTown_Figure_3_Baseline_PV_BESS_Design_Space_Source_Data.csv"
            self.assertTrue(source.exists())
            source_data = pd.read_csv(source)
            self.assertIn("energy_based_LOLP", source_data.columns)
            self.assertNotIn("chance_constraint_probability", source_data.columns)
            self.assertTrue((figures / "SG_UTown_Figure_3A_Baseline_PV_BESS_3D_Response_Surfaces.png").exists())
            self.assertTrue((figures / "SG_UTown_Figure_3B_Baseline_PV_BESS_Contour_Maps.png").exists())
            self.assertTrue((figures / "SG_UTown_Figure_3C_Baseline_PV_Capacity_BESS_Slices.png").exists())
            self.assertTrue((figures / "SG_UTown_Figure_4A_Baseline_Cost_Comparison.png").exists())
            self.assertTrue((figures / "SG_UTown_Figure_4B_Baseline_Cost_Comparison.png").exists())
            self.assertFalse((figures / "SG_UTown_Figure_4_Baseline_Cost_Comparison.png").exists())
            self.assertTrue((figures / "SG_UTown_Figure_5_Baseline_Outage_Energy_Balance_Comparison.png").exists())
            self.assertTrue(
                (figures / "SG_UTown_Figure_3C_Baseline_PV_Capacity_BESS_Slices_Source_Data.csv").exists()
            )
            self.assertTrue(any(key.startswith("SG_UTown_baseline_figure3_") for key in paths))
            self.assertTrue(any(key.startswith("SG_UTown_baseline_figure4_") for key in paths))
            self.assertTrue(any(key.startswith("SG_UTown_baseline_figure5_") for key in paths))
            versions = output_dir / "SG" / "SG_UTown_Figure_Versions"
            version_2 = versions / "Figure_Version_2"
            version_1 = versions / "Figure_Version_1"
            manifest = versions / "Figure_Version_Manifest.csv"
            self.assertTrue(manifest.exists())
            self.assertTrue((version_2 / "SG_UTown_Figure_3A_Baseline_PV_BESS_3D_Response_Surfaces.png").exists())
            self.assertTrue((version_2 / "SG_UTown_Figure_4A_Baseline_Cost_Comparison.png").exists())
            self.assertTrue((version_2 / "SG_UTown_Figure_4B_Baseline_Cost_Comparison.png").exists())
            self.assertFalse((version_2 / "SG_UTown_Figure_4_Baseline_Cost_Comparison.png").exists())
            self.assertTrue((version_2 / "SG_UTown_Figure_5_Baseline_Outage_Energy_Balance_Comparison.png").exists())
            self.assertTrue((version_1 / "SG_UTown_Figure_X_Baseline_LOLP_BESS_Slices_Source_Data.csv").exists())
            manifest_rows = pd.read_csv(manifest)
            self.assertIn("Figure_Version_2", manifest_rows["figure_version"].tolist())
            self.assertIn("Figure_Version_1", manifest_rows["figure_version"].tolist())
            self.assertIn("SG_UTown_figure_version_manifest", paths)


if __name__ == "__main__":
    unittest.main()
