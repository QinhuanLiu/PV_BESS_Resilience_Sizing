from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import unittest
from typing import Any

import numpy as np
import pandas as pd

from src.figures import _capacity_contour_style as style
from src.figures.capacity_sweep_contours import (
    ECONOMIC_METRICS, METRICS, ROOT, aggregate_scenarios, build_figure,
    metrics_for_source, reconcile_summary,
    with_scr_display_values,
)
from src.figures.core import apply_nature_style, matplotlib_context


def legacy_namespace() -> dict[str, Any]:
    """Load only the required pure helpers from the frozen reference, without imports."""
    namespace = {"np": np, "pd": pd, "Any": Any}
    selected = {
        "single_outage_city_baseline.py": [
            "_apply_single_outage_readable_font_style", "_draw_contour_panel",
            "_contour_colormap", "_metric_levels", "_format_metric_colorbar",
            "_sparse_contour_lines", "_format_numeric_tick", "_format_percent_tick",
        ],
        "display_units.py": ["nice_step", "nice_levels", "sparse_ticks_from_levels"],
    }
    constants = {
        "DISPLAY_PV_CAPACITY", "DISPLAY_BATTERY_ENERGY", "SINGLE_OUTAGE_BASE_FONT_SIZE",
        "SINGLE_OUTAGE_TICK_FONT_SIZE", "SINGLE_OUTAGE_TITLE_FONT_SIZE",
        "SINGLE_OUTAGE_COLORBAR_LABEL_SIZE", "SINGLE_OUTAGE_COLORBAR_TICK_SIZE",
    }
    function_dumps = {}
    for filename, names in selected.items():
        code = subprocess.check_output(
            ["git", "show", f"{style.REFERENCE_REVISION}:src/figures/{filename}"], cwd=ROOT, text=True,
        )
        nodes = []
        for node in ast.parse(code).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                nodes.append(node)
                function_dumps[node.name] = ast.dump(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in constants for target in node.targets
            ):
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    namespace["function_dumps"] = function_dumps
    namespace["constant_names"] = constants
    return namespace


def compare_legacy_rendering(source: pd.DataFrame) -> dict[str, Any]:
    """Compare the same data and standalone labels using independent reference helpers."""
    legacy = legacy_namespace()
    mpl, plt = matplotlib_context()
    results = {}
    for metric in metrics_for_source(source):
        actual, metadata = build_figure(source, metric)
        apply_nature_style(mpl)
        legacy["_apply_single_outage_readable_font_style"](mpl)
        reference = plt.figure(figsize=(11.2, 11.4), constrained_layout=False)
        grid = reference.add_gridspec(
            5, 5, left=0.075, right=0.945, bottom=0.075, top=0.935,
            wspace=0.24, hspace=0.22, width_ratios=[1.0] * 4 + [0.055],
        )
        row = 1 if metric["key"] == "ue" else 2
        ax = reference.add_subplot(grid[row, 3])
        cax = reference.add_subplot(grid[row, 4])
        display = source.assign(**{
            legacy["DISPLAY_PV_CAPACITY"]: source.pv_capacity_kw,
            legacy["DISPLAY_BATTERY_ENERGY"]: source.battery_energy_kwh,
        })
        if metric["key"] == "cost":
            display[metric["column"]] = display[metric["column"]] / 1000.0
        if metric["key"] == "scr":
            display = with_scr_display_values(display)
        display_column = metric.get("display_column", metric["column"])
        plot_metric = {**metric, "column": display_column}
        plot_source = display
        level_values = (np.array([0., 1.]) if metric["key"] in {"ccp", "scr", "ssr"}
                        else display[display_column].to_numpy())
        levels = legacy["_metric_levels"](level_values, metric["unit_type"])
        artist = legacy["_draw_contour_panel"](mpl, ax, plot_source, plot_metric, levels)
        ax.set_title("Brisbane", loc="left", pad=5)
        ax.set_ylabel(f"{metric['title']}\nPV capacity (kW)")
        ax.set_xlabel("BESS energy (kWh)")
        ax.set_box_aspect(1.0)
        colorbar = reference.colorbar(artist, cax=cax)
        legacy["_format_metric_colorbar"](colorbar, levels, metric)
        reference.canvas.draw()
        try:
            np.testing.assert_array_equal(levels, metadata["levels"])
            np.testing.assert_array_equal(colorbar.get_ticks(), metadata["colorbar_ticks"])
            np.testing.assert_allclose(ax.bbox.bounds, actual.axes[0].bbox.bounds, rtol=0, atol=1e-10)
            np.testing.assert_allclose(cax.bbox.size, actual.axes[1].bbox.size, rtol=0, atol=1e-10)
            # Allow only the documented external whitespace adjustment, then
            # require the complete old/new rendering to be pixel-identical.
            cax.set_position(actual.axes[1].get_position())
            reference.canvas.draw()
            np.testing.assert_array_equal(np.asarray(reference.canvas.buffer_rgba()),
                                          np.asarray(actual.canvas.buffer_rgba()))
            renderer = actual.canvas.get_renderer()
            if actual.axes[1].get_tightbbox(renderer).x0 < actual.axes[0].bbox.x1:
                raise AssertionError("Colorbar text overlaps the data panel")
            results[metric["key"]] = {
                "levels_equal": True, "colorbar_ticks_equal": True,
                "panel_and_colorbar_dimensions_equal": True,
                "rendered_pixels_equal_after_documented_canvas_translation": True,
                "colorbar_text_clear_of_panel": True,
            }
        finally:
            plt.close(reference)
            plt.close(actual)
    return results


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.spec = {"case_key": "Brisbane_SingleFamily", "pv_capacities": [0, 1],
                     "battery_capacities": [0, 2.5], "scenario_count": 2,
                     "analysis_years": 15, "seed_start": 42}
        self.records = pd.DataFrame([
            dict(case_key=self.spec["case_key"], pv_capacity_kw=pv, battery_energy_kwh=battery,
                 scenario_id=scenario, outage_seed=41 + scenario,
                 outage_profile_sha256=f"profile-{scenario}", analysis_years=15,
                 EUE_kwh_total=1., EUE_kwh_per_year=1 / 15,
                 outage_load_kwh_total=demand, LOLP=1 / demand, no_outage_load=False)
            for pv in [0, 1] for battery in [0, 2.5] for scenario, demand in [(1, 1.), (2, 10.)]
        ])

    def enriched_records(self):
        rows = self.records.copy()
        zero_pv = rows.pv_capacity_kw.eq(0)
        first = rows.scenario_id.eq(1)
        rows["annual_cost_sgd_per_year"] = np.where(first, 2000., 4000.)
        rows["SSR"] = np.where(zero_pv, 0., np.where(first, .25, .75))
        rows["SCR"] = np.where(zero_pv, np.nan, np.where(first, .5, .9))
        rows["SCR_defined"] = ~zero_pv
        rows["SCR_status"] = np.where(zero_pv, "not_applicable_no_pv", "defined")
        rows["load_kwh_total"] = np.where(first, 1., 3.)
        rows["pv_available_kwh_total"] = np.where(zero_pv, 0., np.where(first, 1., 10.))
        return rows

    def test_economic_scenario_means_and_undefined_scr(self):
        rows = self.enriched_records()
        source = aggregate_scenarios(rows.sample(frac=1, random_state=3), self.spec)
        np.testing.assert_allclose(source.annual_cost_sgd_per_year_mean, 3000.)
        valid = source.pv_capacity_kw.gt(0)
        np.testing.assert_allclose(source.loc[valid, "SCR_mean"], .7)
        np.testing.assert_allclose(source.loc[valid, "SSR_mean"], .5)
        np.testing.assert_array_equal(source.SCR_defined_scenario_count, [0, 0, 2, 2])
        self.assertTrue(source.loc[~valid, "SCR_mean"].isna().all())
        display = with_scr_display_values(source)
        for battery in self.spec["battery_capacities"]:
            expected = display.loc[
                display.pv_capacity_kw.eq(1) & display.battery_energy_kwh.eq(battery), "SCR_mean"
            ].item()
            actual = display.loc[
                display.pv_capacity_kw.eq(0) & display.battery_energy_kwh.eq(battery), "SCR_display_mean"
            ].item()
            self.assertEqual(actual, expected)
        self.assertTrue(source.loc[~valid, "SSR_mean"].eq(0).all())
        self.assertNotAlmostEqual(source.loc[valid, "SCR_mean"].iloc[0], (.5 + .9 * 10) / 11)
        self.assertNotAlmostEqual(source.loc[valid, "SSR_mean"].iloc[0], (.25 + .75 * 3) / 4)
        self.assertEqual(len(metrics_for_source(source)), 6)
        base = aggregate_scenarios(self.records, self.spec)
        pd.testing.assert_frame_equal(source[base.columns], base)
        self.assertEqual(len(metrics_for_source(base)), 3)

    def test_unexpected_scr_nulls_and_bad_flags_rejected(self):
        for index, column, value in [
            (4, "SCR", np.nan), (0, "SCR", 0.), (0, "SCR_defined", True),
            (4, "SCR_status", "not_applicable_no_pv"), (0, "SSR", .1),
            (4, "SCR", 1.01), (4, "SSR", np.inf), (4, "annual_cost_sgd_per_year", np.nan),
        ]:
            rows = self.enriched_records()
            rows.loc[index, column] = value
            with self.subTest(index=index, column=column), self.assertRaises(ValueError):
                aggregate_scenarios(rows, self.spec)
        for rows in (self.enriched_records().iloc[:-1], self.enriched_records().drop(columns="SSR")):
            with self.assertRaises(ValueError):
                aggregate_scenarios(rows, self.spec)

    def test_economic_summary_null_pattern_and_counts_reconciled(self):
        source = aggregate_scenarios(self.enriched_records(), self.spec)
        differences = reconcile_summary(source, source.copy())
        self.assertEqual(differences["SCR_mean"], 0.)
        for index, column, value in [(0, "SCR_mean", 0.), (2, "SCR_mean", np.nan),
                                     (2, "SCR_defined_scenario_count", 1),
                                     (2, "annual_cost_sgd_per_year_mean", 200.), (2, "SSR_mean", .6)]:
            summary = source.copy()
            summary.loc[index, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                reconcile_summary(source, summary)

    def test_mean_of_ratios_and_annual_units(self):
        source = aggregate_scenarios(self.records.sample(frac=1, random_state=1), self.spec)
        self.assertEqual(len(source), 4)
        np.testing.assert_allclose(source.EUE_kwh_per_year_mean, 1 / 15)
        np.testing.assert_allclose(source.LOLP_mean, 0.55)
        self.assertNotAlmostEqual(source.LOLP_mean.iloc[0], 2 / 11)
        self.assertTrue(source.scenario_count.eq(2).all())
        np.testing.assert_allclose(source.CCP_LOLP_lt20, 0.5)

    def test_ccp_strict_threshold_and_all_or_none_pass(self):
        rows = self.records.copy()
        # Includes immediate floating-point neighbours of 0.20; no rounding or
        # numerical tolerance is allowed in this strict chance constraint.
        ratios = [0.20, np.nextafter(0.20, 1.), np.nextafter(0.20, 0.), 0.20,
                  0., 0.1, 0.19, 0.21]
        rows["LOLP"] = ratios
        rows["EUE_kwh_total"] = rows.LOLP * rows.outage_load_kwh_total
        rows["EUE_kwh_per_year"] = rows.EUE_kwh_total / 15
        source = aggregate_scenarios(rows.sample(frac=1, random_state=2), self.spec)
        np.testing.assert_array_equal(source.LOLP_lt20_scenario_count, [0, 1, 2, 1])
        np.testing.assert_array_equal(source.CCP_LOLP_lt20, [0., 0.5, 1., 0.5])
        np.testing.assert_array_equal(source.LOLP_threshold, [0.20] * 4)

    def test_incomplete_duplicate_or_unpaired_grid_rejected(self):
        mismatched = self.records.copy()
        mismatched.loc[0, "outage_profile_sha256"] = "different"
        for bad in (self.records.iloc[:-1], pd.concat([self.records, self.records.iloc[:1]]), mismatched):
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                aggregate_scenarios(bad, self.spec)

    def test_invalid_metric_or_horizon_rejected(self):
        for column, value in [("EUE_kwh_per_year", np.nan), ("LOLP", np.inf),
                              ("LOLP", 1.1), ("LOLP", 0.2), ("analysis_years", 1),
                              ("EUE_kwh_per_year", 1.), ("no_outage_load", True)]:
            bad = self.records.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                aggregate_scenarios(bad, self.spec)

    def test_zero_outage_demand_retains_equal_scenario_weight(self):
        rows = self.records.copy()
        zero = rows.scenario_id.eq(2)
        rows.loc[zero, ["EUE_kwh_total", "EUE_kwh_per_year", "outage_load_kwh_total", "LOLP"]] = 0.
        rows.loc[zero, "no_outage_load"] = True
        source = aggregate_scenarios(rows, self.spec)
        np.testing.assert_allclose(source.LOLP_mean, 0.5)
        np.testing.assert_allclose(source.EUE_kwh_per_year_mean, 1 / 30)
        np.testing.assert_allclose(source.CCP_LOLP_lt20, 0.5)

    def test_summary_mismatch_rejected(self):
        source = aggregate_scenarios(self.records, self.spec)
        summary = source.copy()
        reconcile_summary(source, summary)
        summary.loc[0, "LOLP_mean"] = 2 / 11
        with self.assertRaisesRegex(ValueError, "LOLP_mean"):
            reconcile_summary(source, summary)


class LegacyStyleTests(unittest.TestCase):
    def economic_source(self):
        return pd.DataFrame([
            {"pv_capacity_kw": pv, "battery_energy_kwh": battery,
             "EUE_kwh_per_year_mean": 1.86 / (1 + pv + battery),
             "LOLP_mean": 1 / (1 + pv + battery), "CCP_LOLP_lt20": (pv + battery) / 30,
             "annual_cost_sgd_per_year_mean": 2000 + 100 * battery + 200 * pv,
             "SSR_mean": .6 * pv / 10, "SCR_mean": np.nan if pv == 0 else .7 + .3 * battery / 20}
            for pv in [0., 1., 10.] for battery in [0., 2.5, 7.5, 20.]
        ])

    def test_economic_figures_match_original_drawing_helpers(self):
        self.assertEqual(set(compare_legacy_rendering(self.economic_source())),
                         {"ue", "lolp", "ccp", "cost", "scr", "ssr"})

    def test_cost_units_scr_display_fill_without_marker_and_percentage_scales(self):
        _, plt = matplotlib_context()
        source = self.economic_source()
        for metric in ECONOMIC_METRICS:
            fig, metadata = build_figure(source, metric)
            try:
                ax = fig.axes[0]
                self.assertEqual(metadata["colormap"], metric["cmap"])
                self.assertEqual(metadata["y_limits"], [0., 10.])
                self.assertEqual(metadata["x_limits"], [0., 20.])
                if metric["key"] == "cost":
                    self.assertEqual(metadata["display_divisor"], 1000.)
                    self.assertEqual(metadata["levels"][0], 2.)
                    self.assertEqual(metadata["levels"][-1], 6.)
                    self.assertIn("kSGD", fig.axes[1].get_ylabel())
                else:
                    self.assertEqual(metadata["colorbar_tick_labels"],
                                     ["0%", "20%", "40%", "60%", "80%", "100%"])
                self.assertIsNone(metadata["design_marker"])
                if metric["key"] == "scr":
                    self.assertEqual(metadata["data_pv_limits"], [0., 10.])
                    self.assertEqual(metadata["undefined_capacity_count"], 4)
                    self.assertEqual(metadata["display_filled_capacity_count"], 4)
                    self.assertIn("PV=1 kW", metadata["scr_display_policy"])
                    for path in ax.collections[0].get_paths():
                        self.assertTrue((path.vertices[:, 1] >= 0.).all())
                    self.assertFalse(ax.texts)
            finally:
                plt.close(fig)
        self.assertEqual(source.annual_cost_sgd_per_year_mean.max(), 6000.)

    def test_vendored_functions_and_constants_unchanged(self):
        legacy = legacy_namespace()
        current = {node.name: ast.dump(node) for node in ast.parse(
            Path(style.__file__).read_text()
        ).body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(current, legacy["function_dumps"])
        for name in legacy["constant_names"]:
            self.assertEqual(getattr(style, name), legacy[name])

    def test_asymmetric_grid_matches_reference_rendering(self):
        source = pd.DataFrame([
            {"pv_capacity_kw": pv, "battery_energy_kwh": battery,
             "EUE_kwh_per_year_mean": 1.86 / (1 + pv + battery),
             "LOLP_mean": 1 / (1 + pv + battery),
             "CCP_LOLP_lt20": 0.1 + 0.8 * (pv + battery) / 30}
            for pv in [0., 1., 10.] for battery in [0., 2.5, 7.5, 20.]
        ])
        results = compare_legacy_rendering(source)
        self.assertEqual(set(results), {"ue", "lolp", "ccp"})

    def test_ccp_uses_full_percentage_scale_for_constant_surfaces(self):
        _, plt = matplotlib_context()
        ccp = next(metric for metric in METRICS if metric["key"] == "ccp")
        for value in (0., 1.):
            source = pd.DataFrame([
                {"pv_capacity_kw": pv, "battery_energy_kwh": battery, "CCP_LOLP_lt20": value}
                for pv in (0., 10.) for battery in (0., 20.)
            ])
            figure, metadata = build_figure(source, ccp)
            try:
                self.assertEqual(metadata["colormap"], "Purples")
                self.assertEqual(metadata["levels"][0], 0.)
                self.assertEqual(metadata["levels"][-1], 1.)
                self.assertEqual(metadata["colorbar_tick_labels"],
                                 ["0%", "20%", "40%", "60%", "80%", "100%"])
                self.assertIn("LOLP < 20%", metadata["y_axis"])
            finally:
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
