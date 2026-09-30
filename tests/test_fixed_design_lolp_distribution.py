from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.figures import _capacity_contour_style as contour_style
from src.figures.core import NATURE_PALETTE, matplotlib_context
from src.figures.fixed_design_lolp_distribution import (
    DEFAULT_OUTPUT_DIR,
    EXPECTED_SCENARIO_COUNT,
    FIGURE_SIZE_INCHES,
    GRID_SPEC,
    build_empirical_cdf_figure,
    build_histogram_figure,
    cdf_upper_percent,
    design_metadata,
    empirical_cdf_source_data,
    histogram_bin_data,
    load_and_validate,
    ranked_source_data,
    run,
)


def valid_records() -> pd.DataFrame:
    seeds = np.arange(1, EXPECTED_SCENARIO_COUNT + 1)
    lolp = np.linspace(0.0, 0.782449599564832, EXPECTED_SCENARIO_COUNT)
    return pd.DataFrame(
        {
            "outage_seed": seeds,
            "analysis_years": 15,
            "pv_capacity_kw": 3.25,
            "battery_energy_kwh": 11.5,
            "LOLP": lolp,
        }
    )


class FixedDesignLOLPValidationTests(unittest.TestCase):
    def write_records(self, directory: Path, records: pd.DataFrame) -> Path:
        path = directory / "records.csv"
        records.to_csv(path, index=False)
        return path

    def test_validation_and_ranked_source_preserve_every_scenario(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = load_and_validate(self.write_records(Path(temporary), valid_records()))
        ranked = ranked_source_data(records.sample(frac=1.0, random_state=8))
        self.assertEqual(len(ranked), 1000)
        np.testing.assert_array_equal(ranked.rank_descending, np.arange(1000))
        self.assertTrue(ranked.LOLP.is_monotonic_decreasing)
        self.assertEqual(ranked.outage_seed.iloc[0], 1000)
        self.assertEqual(ranked.outage_seed.iloc[-1], 1)
        self.assertTrue(np.allclose(ranked.LOLP_percent, ranked.LOLP * 100.0))

    def test_histogram_uses_five_percentage_point_bins_and_all_records(self):
        histogram = histogram_bin_data(valid_records().LOLP)
        self.assertEqual(histogram.LOLP_lower_percent.iloc[0], 0.0)
        self.assertEqual(histogram.LOLP_upper_percent.iloc[-1], 80.0)
        np.testing.assert_allclose(
            np.diff(histogram.LOLP_lower_percent), 5.0,
        )
        self.assertEqual(int(histogram.frequency.sum()), 1000)

    def test_empirical_cdf_includes_equal_values_and_reaches_full_probability(self):
        values = np.array([0.0, 0.10, 0.20, 0.20, 0.30])
        cdf = empirical_cdf_source_data(values, upper_percent=80.0)
        at_20 = cdf.loc[cdf.LOLP_value_percent.eq(20.0)].iloc[0]
        self.assertEqual(at_20.cumulative_scenario_count, 4)
        self.assertEqual(at_20.CDF, 0.8)
        self.assertEqual(cdf.iloc[-1].LOLP_value_percent, 80.0)
        self.assertEqual(cdf.iloc[-1].cumulative_scenario_count, 5)
        self.assertEqual(cdf.iloc[-1].CDF, 1.0)
        self.assertTrue(cdf.CDF.is_monotonic_increasing)

    def test_cdf_upper_bound_expands_when_raw_lolp_exceeds_eighty_percent(self):
        self.assertEqual(cdf_upper_percent(np.array([0.0, 0.7824])), 80.0)
        self.assertEqual(cdf_upper_percent(np.array([0.0, 0.9411])), 95.0)
        self.assertEqual(cdf_upper_percent(np.array([0.95])), 100.0)

    def test_invalid_inputs_are_rejected(self):
        cases = []
        missing = valid_records().drop(columns="LOLP")
        cases.append(("missing", missing))
        duplicate_seed = valid_records()
        duplicate_seed.loc[0, "outage_seed"] = 2
        cases.append(("duplicate_seed", duplicate_seed))
        mixed_pv = valid_records()
        mixed_pv.loc[0, "pv_capacity_kw"] = 3.5
        cases.append(("mixed_pv", mixed_pv))
        mixed_battery = valid_records()
        mixed_battery.loc[0, "battery_energy_kwh"] = 12.5
        cases.append(("mixed_battery", mixed_battery))
        invalid_lolp = valid_records()
        invalid_lolp.loc[0, "LOLP"] = 1.01
        cases.append(("invalid_lolp", invalid_lolp))
        incomplete = valid_records().iloc[:-1]
        cases.append(("incomplete", incomplete))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name, records in cases:
                with self.subTest(name=name):
                    path = self.write_records(directory, records)
                    with self.assertRaises(ValueError):
                        load_and_validate(path)

    def test_design_metadata_is_derived_from_the_single_input_design(self):
        records = valid_records()
        records.loc[:, "pv_capacity_kw"] = 4.5
        records.loc[:, "battery_energy_kwh"] = 8.25
        with tempfile.TemporaryDirectory() as temporary:
            loaded = load_and_validate(self.write_records(Path(temporary), records))
        design = design_metadata(loaded)
        self.assertEqual(design["pv_capacity_kw"], 4.5)
        self.assertEqual(design["battery_energy_kwh"], 8.25)
        self.assertEqual(design["legend_label"], "PV 4.5 kW, BESS 8.25 kWh")


class FixedDesignLOLPFigureStyleTests(unittest.TestCase):
    def test_figures_follow_contour_canvas_typography_and_raw_data(self):
        ranked = ranked_source_data(valid_records())
        histogram = histogram_bin_data(ranked.LOLP)
        cdf = empirical_cdf_source_data(ranked.LOLP)
        design = design_metadata(ranked)
        _, plt = matplotlib_context()
        for builder in (
            lambda: build_histogram_figure(histogram, design=design),
            lambda: build_empirical_cdf_figure(cdf, design=design),
        ):
            figure, metadata = builder()
            try:
                axis = figure.axes[0]
                self.assertEqual(tuple(figure.get_size_inches()), FIGURE_SIZE_INCHES)
                self.assertEqual(metadata["grid_spec"], GRID_SPEC)
                self.assertEqual(axis.get_box_aspect(), 1.0)
                self.assertEqual(axis.title.get_fontsize(), contour_style.SINGLE_OUTAGE_TITLE_FONT_SIZE)
                self.assertEqual(axis.xaxis.label.get_fontsize(), contour_style.SINGLE_OUTAGE_BASE_FONT_SIZE)
                self.assertFalse(axis.xaxis.get_gridlines()[0].get_visible())
                self.assertFalse(axis.spines["top"].get_visible())
                self.assertFalse(axis.spines["right"].get_visible())
                self.assertFalse(metadata["smoothing"])
            finally:
                plt.close(figure)
        histogram_figure, histogram_metadata = build_histogram_figure(histogram, design=design)
        try:
            patches = histogram_figure.axes[0].patches
            self.assertEqual(histogram_metadata["color_mapping"], "core.py teal histogram series")
            self.assertEqual(patches[0].get_facecolor()[-1], 0.88)
            np.testing.assert_allclose(
                patches[0].get_facecolor()[:3], [66 / 255, 148 / 255, 158 / 255], atol=1e-12,
            )
            self.assertTrue(np.allclose(patches[0].get_facecolor(), patches[-1].get_facecolor()))
            self.assertEqual(histogram_figure.axes[0].get_title(loc="left"), "Brisbane")
            self.assertEqual([item.get_text() for item in histogram_figure.axes[0].get_legend().get_texts()],
                             [design["legend_label"]])
        finally:
            plt.close(histogram_figure)
        figure, metadata = build_empirical_cdf_figure(cdf, design=design)
        try:
            line = figure.axes[0].lines[0]
            self.assertEqual(line.get_drawstyle(), "steps-post")
            np.testing.assert_allclose(line.get_xdata(), cdf.LOLP_value_percent)
            np.testing.assert_allclose(line.get_ydata(), cdf.CDF * 100.0)
            self.assertEqual(metadata["x_axis"], "LOLP (%)")
            self.assertEqual(metadata["y_axis"], "CDF (%)")
            self.assertEqual(metadata["comparison"], "<=")
            self.assertEqual(metadata["color"], NATURE_PALETTE["blue"])
            self.assertEqual(len(figure.axes[0].lines), 1)
            self.assertFalse(figure.axes[0].collections)
            self.assertFalse(figure.axes[0].texts)
            self.assertEqual(figure.axes[0].get_title(loc="left"), "Brisbane")
            self.assertEqual([item.get_text() for item in figure.axes[0].get_legend().get_texts()],
                             [design["legend_label"]])
        finally:
            plt.close(figure)

    def test_run_exports_both_figures_and_source_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "records.csv"
            valid_records().to_csv(input_path, index=False)
            paths = run(input_path, directory / "figures")
            self.assertNotEqual(DEFAULT_OUTPUT_DIR, directory / "figures")
            for key in (
                "Brisbane_FixedDesign_LOLP_Histogram_png",
                "Brisbane_FixedDesign_LOLP_Histogram_svg",
                "Brisbane_FixedDesign_LOLP_Histogram_pdf",
                "Brisbane_FixedDesign_LOLP_CDF_png",
                "Brisbane_FixedDesign_LOLP_CDF_svg",
                "Brisbane_FixedDesign_LOLP_CDF_pdf",
                "source_data",
                "histogram_bins",
                "cdf_source",
                "validation",
            ):
                with self.subTest(key=key):
                    self.assertTrue(paths[key].is_file())
                    self.assertGreater(paths[key].stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
