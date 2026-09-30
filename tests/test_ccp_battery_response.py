from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.figures import _capacity_contour_style as style
from src.figures.ccp_battery_response import (
    BATTERY_CAPACITIES,
    FIGURE_SIZE_INCHES,
    GRID_SPEC,
    PV_CAPACITIES,
    RESPONSE_PROFILES,
    SMOOTH_SAMPLES,
    build_figure,
    extract_response_source,
)
from src.figures.core import matplotlib_context, save_publication_figure


def fixture_source() -> pd.DataFrame:
    rows = []
    for pv in PV_CAPACITIES:
        for battery in BATTERY_CAPACITIES:
            ccp = ((pv - 2.0) * 0.025) + battery * 0.01
            rows.append(
                {
                    "pv_capacity_kw": pv,
                    "battery_energy_kwh": battery,
                    "scenario_count": 100,
                    "analysis_years": 15,
                    "LOLP_threshold_lt20": 0.20,
                    "LOLP_lt20_scenario_count": int(round(ccp * 100)),
                    "CCP_LOLP_lt20": ccp,
                }
            )
    return pd.DataFrame(rows)


def pv3_pv6_pv9_fixture_source() -> pd.DataFrame:
    values = {
        3.0: (0.00, 0.00, 0.01, 0.05, 0.10, 0.39, 0.92, 1.00, 1.00),
        6.0: (0.02, 0.04, 0.07, 0.13, 0.20, 0.74, 0.98, 1.00, 1.00),
        9.0: (0.04, 0.06, 0.09, 0.17, 0.26, 0.85, 0.99, 1.00, 1.00),
    }
    return pd.DataFrame(
        {
            "pv_capacity_kw": pv,
            "battery_energy_kwh": battery,
            "scenario_count": 100,
            "analysis_years": 15,
            "LOLP_threshold_lt20": 0.20,
            "LOLP_lt20_scenario_count": int(round(ccp * 100)),
            "CCP_LOLP_lt20": ccp,
        }
        for pv, ccp_values in values.items()
        for battery, ccp in zip(BATTERY_CAPACITIES, ccp_values, strict=True)
    )


class CCPBatteryResponseSourceTests(unittest.TestCase):
    def test_extracts_exact_twenty_seven_points_and_preserves_ccp_fraction(self):
        response = extract_response_source(fixture_source())
        self.assertEqual(len(response), 27)
        self.assertEqual(response.pv_capacity_kw.unique().tolist(), list(PV_CAPACITIES))
        self.assertEqual(response.battery_energy_kwh.unique().tolist(), list(BATTERY_CAPACITIES))
        self.assertTrue(response.scenario_count.eq(100).all())
        self.assertTrue(np.allclose(response.CCP_LOLP_lt20_percent, response.CCP_LOLP_lt20 * 100.0))

    def test_rejects_incomplete_grid_or_non_strict_threshold(self):
        source = fixture_source().iloc[:-1]
        with self.assertRaisesRegex(ValueError, "complete PV"):
            extract_response_source(source)
        source = fixture_source()
        source.loc[0, "LOLP_threshold_lt20"] = 0.21
        with self.assertRaisesRegex(ValueError, "strict LOLP"):
            extract_response_source(source)

    def test_extracts_new_profile_points_and_star_value(self):
        profile = RESPONSE_PROFILES["pv3-pv6-pv9"]
        response = extract_response_source(
            pv3_pv6_pv9_fixture_source(), pv_capacities=tuple(profile["pv_capacities"]),
        )
        self.assertEqual(len(response), 27)
        self.assertEqual(response.pv_capacity_kw.unique().tolist(), [3.0, 6.0, 9.0])
        point = response.loc[
            response.pv_capacity_kw.eq(3.0) & response.battery_energy_kwh.eq(15.0)
        ].iloc[0]
        self.assertEqual(point.LOLP_lt20_scenario_count, 92)
        self.assertEqual(point.CCP_LOLP_lt20, 0.92)


class CCPBatteryResponseFigureTests(unittest.TestCase):
    def test_pchip_curves_have_no_points_or_extrapolation(self):
        response = extract_response_source(fixture_source())
        _, plt = matplotlib_context()
        figure, metadata = build_figure(response)
        try:
            axis = figure.axes[0]
            self.assertEqual(tuple(figure.get_size_inches()), FIGURE_SIZE_INCHES)
            self.assertEqual(metadata["grid_spec"], GRID_SPEC)
            self.assertEqual(axis.get_box_aspect(), 1.0)
            self.assertEqual(axis.title.get_fontsize(), style.SINGLE_OUTAGE_TITLE_FONT_SIZE)
            self.assertEqual(axis.xaxis.label.get_fontsize(), style.SINGLE_OUTAGE_BASE_FONT_SIZE)
            self.assertEqual(len(axis.lines), 3)
            self.assertEqual(metadata["rendered_samples_per_curve"], SMOOTH_SAMPLES)
            self.assertFalse(metadata["show_data_points"])
            self.assertEqual(metadata["x_limits"], [0.0, 20.0])
            self.assertEqual(metadata["y_limits"], [0.0, 0.4])
            self.assertEqual(metadata["y_tick_labels"], ["0%", "10%", "20%", "30%", "40%"])
            self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                             ["PV 2 kW", "PV 5 kW", "PV 8 kW"])
            for line in axis.lines:
                self.assertEqual(len(line.get_xdata()), SMOOTH_SAMPLES)
                self.assertIn(line.get_marker(), {None, "None", ""})
                self.assertGreaterEqual(float(line.get_xdata().min()), 0.0)
                self.assertLessEqual(float(line.get_xdata().max()), 20.0)
                self.assertGreaterEqual(float(line.get_ydata().min()), 0.0)
                self.assertLessEqual(float(line.get_ydata().max()), 0.4)
                self.assertGreaterEqual(float(np.diff(line.get_ydata()).min()), -1e-12)
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

    def test_new_profile_uses_core_colours_and_only_the_design_star(self):
        profile_name = "pv3-pv6-pv9"
        response = extract_response_source(
            pv3_pv6_pv9_fixture_source(),
            pv_capacities=tuple(RESPONSE_PROFILES[profile_name]["pv_capacities"]),
        )
        _, plt = matplotlib_context()
        figure, metadata = build_figure(response, profile_name=profile_name)
        try:
            axis = figure.axes[0]
            self.assertEqual(len(axis.lines), 3)
            self.assertEqual(metadata["pv_capacities_kw"], [3.0, 6.0, 9.0])
            self.assertEqual(metadata["colors"], {
                "PV_3": "#0f4d92", "PV_6": "#b64342", "PV_9": "#8c6bb1",
            })
            self.assertEqual(metadata["y_limits"], [0.0, 1.0])
            self.assertEqual(metadata["y_tick_labels"], ["0%", "20%", "40%", "60%", "80%", "100%"])
            self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                             ["PV 3 kW", "PV 6 kW", "PV 9 kW"])
            self.assertEqual(metadata["design_marker"]["pv_capacity_kw"], 3.0)
            self.assertEqual(metadata["design_marker"]["battery_energy_kwh"], 15.0)
            self.assertEqual(metadata["design_marker"]["CCP_LOLP_lt20"], 0.92)
            np.testing.assert_allclose(axis.collections[0].get_offsets(), [[15.0, 0.92]])
            np.testing.assert_allclose(axis.collections[0].get_edgecolors()[0], [1.0, 1.0, 1.0, 1.0])
            for line in axis.lines:
                self.assertEqual(len(line.get_xdata()), SMOOTH_SAMPLES)
                self.assertIn(line.get_marker(), {None, "None", ""})
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
