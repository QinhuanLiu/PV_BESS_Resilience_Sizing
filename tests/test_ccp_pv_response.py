from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.figures import _capacity_contour_style as style
from src.figures.ccp_pv_response import (
    BESS_CAPACITIES,
    COLOR_POSITIONS,
    FIGURE_SIZE_INCHES,
    GRID_SPEC,
    PV_CAPACITIES,
    SMOOTH_SAMPLES,
    build_figure,
    extract_response_source,
)
from src.figures.core import matplotlib_context, save_publication_figure


def fixture_source() -> pd.DataFrame:
    rows = []
    for bess in BESS_CAPACITIES:
        for pv in PV_CAPACITIES:
            ccp = min(1.0, 0.02 + (bess - 10.0) * 0.10 + pv * 0.06)
            rows.append(
                {
                    "pv_capacity_kw": pv,
                    "battery_energy_kwh": bess,
                    "scenario_count": 100,
                    "analysis_years": 15,
                    "LOLP_threshold_lt20": 0.20,
                    "LOLP_lt20_scenario_count": int(round(ccp * 100)),
                    "CCP_LOLP_lt20": ccp,
                }
            )
    return pd.DataFrame(rows)


class CCPPVResponseSourceTests(unittest.TestCase):
    def test_extracts_exact_twenty_seven_points_and_preserves_ccp_fraction(self):
        response = extract_response_source(fixture_source())
        self.assertEqual(len(response), 27)
        self.assertEqual(response.battery_energy_kwh.unique().tolist(), list(BESS_CAPACITIES))
        self.assertEqual(response.pv_capacity_kw.unique().tolist(), list(PV_CAPACITIES))
        self.assertTrue(response.scenario_count.eq(100).all())
        self.assertTrue(np.allclose(response.CCP_LOLP_lt20_percent, response.CCP_LOLP_lt20 * 100.0))

    def test_rejects_incomplete_grid_or_non_strict_threshold(self):
        with self.assertRaisesRegex(ValueError, "complete BESS"):
            extract_response_source(fixture_source().iloc[:-1])
        source = fixture_source()
        source.loc[0, "LOLP_threshold_lt20"] = 0.21
        with self.assertRaisesRegex(ValueError, "strict LOLP"):
            extract_response_source(source)


class CCPPVResponseFigureTests(unittest.TestCase):
    def test_delegates_to_the_battery_response_plotting_contract(self):
        response = extract_response_source(fixture_source())
        expected = (object(), {"style_provider": "ccp_battery_response"})
        with patch("src.figures.ccp_pv_response.build_ccp_response_figure", return_value=expected) as shared:
            result = build_figure(response, city_label="Test city")
        self.assertIs(result, expected)
        _, kwargs = shared.call_args
        self.assertEqual(kwargs["x_column"], "pv_capacity_kw")
        self.assertEqual(kwargs["series_column"], "battery_energy_kwh")
        self.assertEqual(kwargs["series_values"], BESS_CAPACITIES)
        self.assertEqual(kwargs["series_unit"], "kWh")
        self.assertEqual(kwargs["color_positions"], COLOR_POSITIONS)
        self.assertEqual(kwargs["x_limits"], (0.0, 8.0))
        self.assertEqual(kwargs["city_label"], "Test city")

    def test_pchip_curves_use_the_required_limits_colours_and_no_points(self):
        response = extract_response_source(fixture_source())
        mpl, plt = matplotlib_context()
        figure, metadata = build_figure(response)
        try:
            axis = figure.axes[0]
            self.assertEqual(tuple(figure.get_size_inches()), FIGURE_SIZE_INCHES)
            self.assertEqual(metadata["grid_spec"], GRID_SPEC)
            self.assertEqual(axis.get_box_aspect(), 1.0)
            self.assertEqual(axis.title.get_fontsize(), style.SINGLE_OUTAGE_TITLE_FONT_SIZE)
            self.assertEqual(axis.xaxis.label.get_fontsize(), style.SINGLE_OUTAGE_BASE_FONT_SIZE)
            self.assertEqual(metadata["x_limits"], [0.0, 8.0])
            self.assertEqual(metadata["y_limits"], [0.0, 1.0])
            self.assertEqual(metadata["y_tick_labels"], ["0%", "20%", "40%", "60%", "80%", "100%"])
            self.assertEqual(metadata["style_provider"], "ccp_battery_response")
            self.assertEqual(metadata["line_width"], 1.45)
            self.assertEqual(metadata["legend"], {
                "loc": "upper left", "bbox_to_anchor": [1.05, 1.0], "borderaxespad": 0.0,
            })
            self.assertIsNone(metadata["design_marker"])
            self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()], [
                "BESS 10 kWh", "BESS 12.5 kWh", "BESS 15 kWh",
            ])
            self.assertEqual(metadata["colors"], {
                f"BESS_{bess:g}": mpl.colors.to_hex(mpl.colormaps["Purples"](COLOR_POSITIONS[bess]))
                for bess in BESS_CAPACITIES
            })
            self.assertEqual(len(axis.lines), 3)
            self.assertFalse(metadata["show_data_points"])
            for line in axis.lines:
                self.assertEqual(len(line.get_xdata()), SMOOTH_SAMPLES)
                self.assertIn(line.get_marker(), {None, "None", ""})
                self.assertGreaterEqual(float(line.get_xdata().min()), 0.0)
                self.assertLessEqual(float(line.get_xdata().max()), 8.0)
                self.assertGreaterEqual(float(line.get_ydata().min()), 0.0)
                self.assertLessEqual(float(line.get_ydata().max()), 1.0)
        finally:
            plt.close(figure)

    def test_export_has_png_svg_and_pdf(self):
        response = extract_response_source(fixture_source())
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "response"
            _, plt = matplotlib_context(base)
            figure, _ = build_figure(response, out=base)
            try:
                paths = save_publication_figure(figure, base)
            finally:
                plt.close(figure)
            self.assertEqual(set(paths), {"response_png", "response_svg", "response_pdf"})
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
