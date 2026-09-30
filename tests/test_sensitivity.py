from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.io.config import load_config
from src.io.data_loader import CaseParameters, load_case_parameters
from src.sensitivity import SensitivitySpec, build_sensitivity_plan, normalize_requested_parameters, run_sensitivity


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parameters(
    base: dict[str, float],
    ranges: dict[str, list[float]],
) -> CaseParameters:
    rows = [
        {
            "parameter": name,
            "base_value": value,
            "unit": "-",
            "sensitivity_start": values[0] if name in ranges else None,
            "sensitivity_end": values[-1] if name in ranges else None,
            "sensitivity_step": values[1] - values[0] if name in ranges and len(values) > 1 else 1.0,
            "description": "",
            "中文说明": "",
        }
        for name, value in base.items()
        for values in [ranges.get(name, [])]
    ]
    return CaseParameters(base=base, sensitivity=ranges, raw=pd.DataFrame(rows))


class SensitivityPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = _parameters(
            {
                "pv_capex_sgd_per_kw": 1200.0,
                "battery_capex_sgd_per_kwh": 400.0,
                "saidi_min_per_year": 120.0,
            },
            {
                "pv_capex_sgd_per_kw": [1000.0, 1200.0, 1400.0],
                "battery_capex_sgd_per_kwh": [300.0, 400.0, 500.0],
                "saidi_min_per_year": [60.0, 120.0, 180.0],
            },
        )

    def test_oat_keeps_every_nonvaried_selected_parameter_at_its_base_value(self) -> None:
        points = build_sensitivity_plan(
            self.parameters,
            SensitivitySpec(parameters=("pv_capex_sgd_per_kw", "battery_capex_sgd_per_kwh"), mode="oat"),
        )
        self.assertEqual(len(points), 5)  # baseline plus two non-base values on each axis
        self.assertTrue(points[0].is_baseline)
        self.assertTrue(all(point.chain_position == 0 for point in points))
        for point in points[1:]:
            changed = [
                name
                for name, base in points[0].overrides.items()
                if point.overrides[name] != base
            ]
            self.assertEqual(len(changed), 1)

    def test_factorial_respects_the_user_axis_order_and_deduplicates_baseline(self) -> None:
        points = build_sensitivity_plan(
            self.parameters,
            SensitivitySpec(
                parameters=("pv_capex_sgd_per_kw", "battery_capex_sgd_per_kwh", "saidi_min_per_year"),
                mode="factorial",
            ),
        )
        self.assertEqual(len(points), 27)  # 3 x 3 x 3, with the baseline combination reused once
        self.assertEqual(sum(point.is_baseline for point in points), 1)
        nonbaseline = [point for point in points if not point.is_baseline]
        by_prefix: dict[tuple[float, float], list] = {}
        for point in nonbaseline:
            key = (point.overrides["pv_capex_sgd_per_kw"], point.overrides["battery_capex_sgd_per_kwh"])
            by_prefix.setdefault(key, []).append(point)
        for chain in by_prefix.values():
            self.assertEqual(
                [point.overrides["saidi_min_per_year"] for point in chain],
                sorted(point.overrides["saidi_min_per_year"] for point in chain),
            )
            self.assertEqual([point.chain_position for point in chain], list(range(len(chain))))

    def test_large_factorial_requires_explicit_opt_in(self) -> None:
        values = [float(value) for value in range(8)]
        parameters = _parameters(
            {
                "pv_capex_sgd_per_kw": 1.0,
                "battery_capex_sgd_per_kwh": 1.0,
                "saidi_min_per_year": 1.0,
            },
            {
                "pv_capex_sgd_per_kw": values,
                "battery_capex_sgd_per_kwh": values,
                "saidi_min_per_year": values,
            },
        )
        spec = SensitivitySpec(parameters=tuple(parameters.sensitivity), mode="factorial")
        with self.assertRaisesRegex(ValueError, "exceeding the 500-run limit"):
            build_sensitivity_plan(parameters, spec)
        self.assertEqual(
            len(build_sensitivity_plan(parameters, replace(spec, allow_large=True))),
            512,
        )

    def test_all_selects_only_parameters_with_configured_ranges(self) -> None:
        self.assertEqual(
            normalize_requested_parameters(["all"], self.parameters),
            tuple(self.parameters.sensitivity),
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            normalize_requested_parameters(["all", "pv_capex_sgd_per_kw"], self.parameters)


class SensitivityParameterLoadingTests(unittest.TestCase):
    def _copy_parameter_file(self, directory: Path) -> Path:
        source = REPO_ROOT / "Input" / "Brisbane_SingleFamily" / "Brisbane_SingleFamily_Parameter.csv"
        target = directory / source.name
        shutil.copy2(source, target)
        return target

    def test_catalogued_numeric_parameter_can_define_a_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._copy_parameter_file(Path(temporary))
            frame = pd.read_csv(path)
            row = frame["parameter"].eq("battery_roundtrip_efficiency")
            frame.loc[row, ["sensitivity_start", "sensitivity_end", "sensitivity_step"]] = [0.85, 0.95, 0.05]
            frame.to_csv(path, index=False)
            parameters = load_case_parameters(path)
        self.assertEqual(parameters.sensitivity["battery_roundtrip_efficiency"], [0.85, 0.9, 0.95])

    def test_identity_control_boolean_and_unknown_ranges_are_rejected(self) -> None:
        for parameter, expected in (
            ("city_name", "not eligible"),
            ("analysis_years", "not eligible"),
            ("pv_export_enabled", "not eligible"),
        ):
            with self.subTest(parameter=parameter), tempfile.TemporaryDirectory() as temporary:
                path = self._copy_parameter_file(Path(temporary))
                frame = pd.read_csv(path)
                row = frame["parameter"].eq(parameter)
                frame.loc[row, ["sensitivity_start", "sensitivity_end", "sensitivity_step"]] = [1, 2, 1]
                frame.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, expected):
                    load_case_parameters(path)

        with tempfile.TemporaryDirectory() as temporary:
            path = self._copy_parameter_file(Path(temporary))
            frame = pd.read_csv(path)
            frame.loc[len(frame)] = {
                "parameter": "not_a_model_parameter",
                "base_value": 1.0,
                "unit": "-",
                "sensitivity_start": 1.0,
                "sensitivity_end": 2.0,
                "sensitivity_step": 1.0,
                "description": "",
                "中文说明": "",
            }
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "not eligible"):
                load_case_parameters(path)

    def test_incomplete_and_non_finite_ranges_are_rejected(self) -> None:
        for values, expected in (
            ((1000.0, None, 100.0), "define start, end, and step together"),
            ((float("inf"), 1400.0, 100.0), "finite numeric values"),
        ):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as temporary:
                path = self._copy_parameter_file(Path(temporary))
                frame = pd.read_csv(path)
                row = frame["parameter"].eq("pv_capex_sgd_per_kw")
                frame.loc[row, ["sensitivity_start", "sensitivity_end", "sensitivity_step"]] = values
                frame.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, expected):
                    load_case_parameters(path)


class SensitivityRunnerTests(unittest.TestCase):
    def _prepare_input(self, root: Path) -> Path:
        source = REPO_ROOT / "Input" / "Brisbane_SingleFamily"
        target = root / "Brisbane_SingleFamily"
        shutil.copytree(source, target)
        parameter_path = target / "Brisbane_SingleFamily_Parameter.csv"
        frame = pd.read_csv(parameter_path)
        ranges = {
            "pv_capex_sgd_per_kw": (1100.0, 1300.0, 200.0),
            "battery_capex_sgd_per_kwh": (350.0, 450.0, 100.0),
        }
        for name, values in ranges.items():
            row = frame["parameter"].eq(name)
            frame.loc[row, ["sensitivity_start", "sensitivity_end", "sensitivity_step"]] = values
        frame.to_csv(parameter_path, index=False)
        return root

    @staticmethod
    def _fake_evaluate(case, config, case_key, frame_transform=None, warm_start_best=None):
        pv = float(config["pv"]["capex_sgd_per_kw"])
        battery = float(config["economics"]["battery_capex_sgd_per_kwh"])
        design = pd.DataFrame(
            [
                {
                    "case_key": case_key,
                    "pv_capacity_kw": pv / 100.0,
                    "battery_energy_kwh": battery / 10.0,
                    "battery_power_kw": battery / 20.0,
                    "pv_bess_system_annual_cost_sgd": pv + battery,
                    "unserved_energy_kwh_per_year": 5.0,
                    "outage_weighted_load_kwh": 10.0,
                    "warm_start_used": warm_start_best is not None,
                    "warm_start_source_pv_capacity_kw": None if warm_start_best is None else warm_start_best["pv_capacity_kw"],
                    "warm_start_source_battery_energy_kwh": None if warm_start_best is None else warm_start_best["battery_energy_kwh"],
                    "warm_start_fallback_triggered": False,
                    "warm_start_fallback_reason": "",
                    "warm_start_candidate_count": 0,
                    "capacity_search_method": "fake",
                }
            ]
        )
        dispatch = pd.DataFrame(
            {
                "total_load_e_equiv_kwh": [100.0],
                "pv_available_kwh": [60.0],
                "pv_to_load_kwh": [50.0],
                "pv_to_battery_kwh": [10.0],
                "battery_discharge_from_pv_kwh": [10.0],
            }
        )
        return design, pd.DataFrame(), pd.DataFrame(), dispatch, pd.DataFrame(), {}, {}, pd.DataFrame()

    def _config(self) -> dict:
        config = load_config(REPO_ROOT / "config.yaml")
        config["capacity_search"].update({"coarse_intervals": 1, "local_refine_divisions": 1, "max_upper_expansions": 0})
        return config

    def test_factorial_runner_writes_standard_artifacts_and_uses_warm_start_only_in_chains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = self._prepare_input(root / "Input")
            output_root = root / "outputs"
            calls: list[dict | None] = []

            def fake(*args, **kwargs):
                calls.append(kwargs.get("warm_start_best"))
                return self._fake_evaluate(*args, **kwargs)

            with patch("src.sensitivity.runner.evaluate_design_case", side_effect=fake):
                paths = run_sensitivity(
                    config=self._config(),
                    input_dir=input_root,
                    output_dir=output_root,
                    requested_parameters=["pv_capex_sgd_per_kw", "battery_capex_sgd_per_kwh"],
                    mode="factorial",
                    workers=1,
                    case_filters=["Brisbane_SingleFamily"],
                )
                result = pd.read_csv(paths["Brisbane_SingleFamily_sensitivity_results"])
                self.assertEqual(len(result), 5)
                self.assertEqual(int(result["is_baseline"].sum()), 1)
                self.assertTrue({"parameter_pv_capex_sgd_per_kw", "parameter_battery_capex_sgd_per_kwh", "SSR", "SCR"} <= set(result))
                self.assertEqual(sum(anchor is not None for anchor in calls), 2)
                result_dir = paths["Brisbane_SingleFamily_sensitivity_results"].parent
                for name in ("manifest.json", "effective_config.json", "run_status.json", "validation.json", "sensitivity_parameter_catalog.csv"):
                    self.assertTrue((result_dir / name).exists())
                first_count = len(calls)
                run_sensitivity(
                    config=self._config(),
                    input_dir=input_root,
                    output_dir=output_root,
                    requested_parameters=["pv_capex_sgd_per_kw", "battery_capex_sgd_per_kwh"],
                    mode="factorial",
                    workers=1,
                    case_filters=["Brisbane_SingleFamily"],
                )
                self.assertEqual(len(calls), first_count)

                changed_config = self._config()
                changed_config["capacity_search"]["coarse_intervals"] = 2
                with self.assertRaisesRegex(ValueError, "resume refused"):
                    run_sensitivity(
                        config=changed_config,
                        input_dir=input_root,
                        output_dir=output_root,
                        requested_parameters=["pv_capex_sgd_per_kw", "battery_capex_sgd_per_kwh"],
                        mode="factorial",
                        workers=1,
                        case_filters=["Brisbane_SingleFamily"],
                    )

                calls.clear()
                run_sensitivity(
                    config=self._config(),
                    input_dir=input_root,
                    output_dir=output_root / "warm_disabled",
                    requested_parameters=["pv_capex_sgd_per_kw", "battery_capex_sgd_per_kwh"],
                    mode="factorial",
                    workers=1,
                    warm_start_enabled=False,
                    case_filters=["Brisbane_SingleFamily"],
                )
                self.assertTrue(calls and all(anchor is None for anchor in calls))

                calls.clear()
                run_sensitivity(
                    config=self._config(),
                    input_dir=input_root,
                    output_dir=output_root / "oat",
                    requested_parameters=["pv_capex_sgd_per_kw"],
                    mode="oat",
                    workers=1,
                    case_filters=["Brisbane_SingleFamily"],
                )
                self.assertTrue(calls and all(anchor is None for anchor in calls))

    def test_invalid_soc_point_is_rejected_before_any_sensitivity_output_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = self._prepare_input(root / "Input")
            parameter_path = input_root / "Brisbane_SingleFamily" / "Brisbane_SingleFamily_Parameter.csv"
            frame = pd.read_csv(parameter_path)
            row = frame["parameter"].eq("battery_soc_min_fraction")
            frame.loc[row, ["sensitivity_start", "sensitivity_end", "sensitivity_step"]] = [0.1, 0.95, 0.85]
            frame.to_csv(parameter_path, index=False)
            output_root = root / "outputs"
            with self.assertRaisesRegex(ValueError, "soc_min_fraction < soc_max_fraction"):
                run_sensitivity(
                    config=self._config(),
                    input_dir=input_root,
                    output_dir=output_root,
                    requested_parameters=["battery_soc_min_fraction"],
                    case_filters=["Brisbane_SingleFamily"],
                )
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
