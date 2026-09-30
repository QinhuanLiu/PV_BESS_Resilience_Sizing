from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import sys
import unittest
from unittest.mock import patch

from src.app import cli


def _config() -> dict[str, dict[str, str]]:
    return {
        "input": {"directory": "Input"},
        "outputs": {"directory": "outputs"},
    }


class CliTests(unittest.TestCase):
    def _run_main(self, *arguments: str) -> tuple[int, str]:
        stderr = StringIO()
        with patch.object(sys, "argv", ["pv_bess_sizing.py", *arguments]), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exit_error:
                cli.main()
        return int(exit_error.exception.code), stderr.getvalue()

    def test_removed_arguments_are_rejected_before_any_workflow(self) -> None:
        removed_arguments = (
            ("--outage-scenario-count", "10"),
            ("--outage-seed-start", "42"),
            ("--outage-seed", "42"),
            ("--cost-sensitivity-grid",),
            ("--outage-value-sensitivity-grid",),
            ("--sensitivity-grid-workers", "2"),
            ("--sensitivity-grid-analysis-years", "15"),
            ("--disable-warm-start",),
        )
        for arguments in removed_arguments:
            with self.subTest(arguments=arguments), patch.object(cli, "run_selected") as run_selected, patch.object(
                cli, "run_sensitivity"
            ) as run_sensitivity:
                code, stderr = self._run_main(*arguments)
                self.assertEqual(code, 2)
                self.assertIn("unrecognized arguments", stderr)
                run_selected.assert_not_called()
                run_sensitivity.assert_not_called()

    def test_disable_sensitivity_warm_start_requires_factorial_sensitivity(self) -> None:
        for arguments, expected in (
            (("--disable-sensitivity-warm-start",), "require at least one --sensitivity-parameter"),
            (
                ("--sensitivity-parameter", "pv_capex_sgd_per_kw", "--disable-sensitivity-warm-start"),
                "requires --sensitivity-mode factorial",
            ),
        ):
            with self.subTest(arguments=arguments), patch.object(cli, "run_sensitivity") as run_sensitivity:
                code, stderr = self._run_main(*arguments)
                self.assertEqual(code, 2)
                self.assertIn(expected, stderr)
                run_sensitivity.assert_not_called()

    def test_factorial_options_are_forwarded_to_unified_sensitivity_runner(self) -> None:
        with (
            patch.object(cli, "load_config", return_value=_config()),
            patch.object(cli, "run_sensitivity", return_value={}) as run_sensitivity,
            patch.object(cli, "run_selected") as run_selected,
            patch.object(
                sys,
                "argv",
                [
                    "pv_bess_sizing.py",
                    "--case",
                    "Brisbane_SingleFamily",
                    "--sensitivity-parameter",
                    "pv_capex_sgd_per_kw",
                    "--sensitivity-parameter",
                    "battery_capex_sgd_per_kwh",
                    "--sensitivity-mode",
                    "factorial",
                    "--sensitivity-analysis-years",
                    "2",
                    "--sensitivity-workers",
                    "3",
                    "--allow-large-sensitivity",
                    "--disable-sensitivity-warm-start",
                ],
            ),
            redirect_stdout(StringIO()),
        ):
            cli.main()

        run_selected.assert_not_called()
        self.assertEqual(
            run_sensitivity.call_args.kwargs,
            {
                "config": _config(),
                "input_dir": __import__("pathlib").Path("Input").resolve(),
                "output_dir": __import__("pathlib").Path("outputs").resolve(),
                "requested_parameters": ["pv_capex_sgd_per_kw", "battery_capex_sgd_per_kwh"],
                "mode": "factorial",
                "analysis_years": 2,
                "workers": 3,
                "allow_large": True,
                "warm_start_enabled": False,
                "city_filters": None,
                "case_filters": ["Brisbane_SingleFamily"],
            },
        )

    def test_baseline_and_tou_calls_do_not_receive_sensitivity_or_removed_parameters(self) -> None:
        for arguments in ((), ("--tou-analysis",)):
            with self.subTest(arguments=arguments):
                with (
                    patch.object(cli, "load_config", return_value=_config()),
                    patch.object(cli, "run_selected", return_value={}) as run_selected,
                    patch.object(cli, "run_sensitivity") as run_sensitivity,
                    patch.object(sys, "argv", ["pv_bess_sizing.py", *arguments]),
                    redirect_stdout(StringIO()),
                ):
                    cli.main()
                run_sensitivity.assert_not_called()
                forbidden = {
                    "outage_scenario_count",
                    "outage_seed_start",
                    "outage_seeds",
                    "warm_start_enabled",
                    "sensitivity_parameters",
                }
                self.assertFalse(forbidden & set(run_selected.call_args.kwargs))


if __name__ == "__main__":
    unittest.main()
