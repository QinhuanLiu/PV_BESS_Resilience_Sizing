from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.figures import fixed_design_lolp_histogram_cdf as combined
from src.figures.core import matplotlib_context


def records(lolp: np.ndarray | None = None) -> pd.DataFrame:
    values = np.asarray(lolp if lolp is not None else np.linspace(0.0, 0.8, 1000), dtype=float)
    return pd.DataFrame(
        {
            "outage_seed": np.arange(1, 1001),
            "analysis_years": 15,
            "pv_capacity_kw": 3.25,
            "battery_energy_kwh": 11.5,
            "LOLP": values,
        }
    )


class FixedDesignHistogramCdfTests(unittest.TestCase):
    def test_fixed_bins_are_percentages_and_cover_every_scenario(self):
        values = np.repeat(np.arange(20) / 20, 50)
        histogram = combined.fixed_histogram_bin_data(values)
        self.assertEqual(len(histogram), 20)
        self.assertEqual(int(histogram.frequency.sum()), 1000)
        self.assertAlmostEqual(float(histogram.frequency_percent.sum()), 100.0)
        self.assertEqual(histogram.LOLP_lower_percent.iloc[0], 0.0)
        self.assertEqual(histogram.LOLP_upper_percent.iloc[-1], 100.0)
        self.assertTrue(histogram.frequency.eq(50).all())
        self.assertTrue(histogram.frequency_percent.eq(5.0).all())

    def test_cdf_is_right_continuous_with_repeats_and_endpoint_at_one(self):
        values = np.concatenate([np.zeros(500), np.full(499, 0.2), np.ones(1)])
        cdf = combined.fixed_empirical_cdf_source_data(values)
        self.assertEqual(cdf.LOLP_value_percent.tolist(), [0.0, 20.0, 100.0])
        np.testing.assert_allclose(cdf.CDF, [0.5, 0.999, 1.0])
        self.assertTrue(cdf.CDF.is_monotonic_increasing)
        self.assertEqual(cdf.value_source.iloc[-1], "observed_LOLP")

    def test_dual_axis_style_and_fixed_ranges(self):
        frame = records()
        histogram = combined.fixed_histogram_bin_data(frame.LOLP)
        cdf = combined.fixed_empirical_cdf_source_data(frame.LOLP)
        design = {"legend_label": "PV 3.25 kW, BESS 11.5 kWh"}
        _, plt = matplotlib_context()
        figure, metadata = combined.build_figure(histogram, cdf, design=design)
        try:
            self.assertEqual(len(figure.axes), 2)
            frequency_axis, cdf_axis = figure.axes
            self.assertEqual(metadata["x_limits_percent"], [0.0, 100.0])
            self.assertEqual(metadata["frequency_y_limits_percent"], [0.0, 40.0])
            self.assertEqual(metadata["cdf_y_limits_percent"], [0.0, 100.0])
            self.assertEqual(frequency_axis.get_ylabel(), "Frequency (%)")
            self.assertEqual(cdf_axis.get_ylabel(), "CDF (%)")
            np.testing.assert_allclose(
                frequency_axis.get_position().bounds, cdf_axis.get_position().bounds, rtol=0.0, atol=1e-12
            )
            self.assertEqual(metadata["histogram_color"], "#42949E")
            self.assertEqual(metadata["cdf_color"], "#0F4D92")
            self.assertEqual(metadata["legend_labels"], ["Histogram (frequency)", "CDF"])
            self.assertFalse(frequency_axis.xaxis.get_gridlines()[0].get_visible())
        finally:
            plt.close(figure)

    def test_shared_frequency_axis_can_expand_without_clipping(self):
        values = np.concatenate([np.zeros(650), np.full(350, 0.3)])
        histogram = combined.fixed_histogram_bin_data(values)
        cdf = combined.fixed_empirical_cdf_source_data(values)
        _, plt = matplotlib_context()
        figure, metadata = combined.build_figure(
            histogram, cdf, design={"legend_label": "PV 6.5 kW, BESS 14.5 kWh"},
            frequency_upper_percent=70.0,
        )
        try:
            self.assertEqual(metadata["frequency_y_limits_percent"], [0.0, 70.0])
            self.assertEqual(figure.axes[0].get_ylim(), (0.0, 70.0))
            self.assertLessEqual(float(histogram.frequency_percent.max()), figure.axes[0].get_ylim()[1])
        finally:
            plt.close(figure)

    def test_run_exports_three_figure_formats_and_auditable_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "records.csv"
            records().to_csv(source, index=False)
            paths = combined.run(source, root / "figures")
            for suffix in ("png", "svg", "pdf"):
                exported = [path for key, path in paths.items() if key.endswith(f"_{suffix}")]
                self.assertEqual(len(exported), 1)
                self.assertTrue(exported[0].is_file())
            bins = pd.read_csv(paths["bins_source"])
            cdf = pd.read_csv(paths["cdf_source"])
            self.assertEqual(len(bins), 20)
            self.assertAlmostEqual(float(bins.frequency_percent.sum()), 100.0)
            self.assertAlmostEqual(float(cdf.CDF.iloc[-1]), 1.0)
            self.assertTrue(paths["validation"].is_file())


if __name__ == "__main__":
    unittest.main()
