from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.figures.core import (
    FigureDependencyError,
    FigureLayoutError,
    apply_nature_style,
    format_colorbar,
    matplotlib_context,
    nice_display_ticks,
    plot_smooth_line,
    save_publication_figure,
)


class FigureCoreTests(unittest.TestCase):
    def test_apply_nature_style_sets_shared_rcparams(self) -> None:
        mpl, _ = matplotlib_context()
        apply_nature_style(mpl)

        self.assertEqual(mpl.rcParams["figure.facecolor"], "white")
        self.assertEqual(mpl.rcParams["font.size"], 7)
        self.assertEqual(mpl.rcParams["svg.fonttype"], "none")
        self.assertEqual(mpl.rcParams["pdf.fonttype"], 42)
        self.assertFalse(mpl.rcParams["legend.frameon"])

    def test_nice_display_ticks_prefers_integerish_ticks_for_costs(self) -> None:
        ticks = nice_display_ticks([1.2, 8.7, 13.4], unit_type="cost")

        self.assertGreaterEqual(min(ticks), 0)
        self.assertTrue(all(float(tick).is_integer() for tick in ticks))
        self.assertIn(10.0, ticks)

    def test_nice_display_ticks_prefers_canonical_probability_ticks(self) -> None:
        ticks = nice_display_ticks([0.04, 0.97], unit_type="ccp")

        self.assertEqual(ticks, [0.04, 0.25, 0.5, 0.75, 0.97])

    def test_format_colorbar_uses_readable_tick_labels(self) -> None:
        with TemporaryDirectory() as tmp:
            mpl, plt = matplotlib_context(Path(tmp) / "figure")
            apply_nature_style(mpl)
            fig, ax = plt.subplots(figsize=(3.0, 2.0))
            image = ax.imshow(np.array([[0.0, 0.5], [0.75, 1.0]]), vmin=0.0, vmax=1.0)
            cbar = fig.colorbar(image, ax=ax)

            format_colorbar(cbar, [0.0, 1.0], "Chance constraint", unit_type="ccp")

            labels = [tick.get_text() for tick in cbar.ax.get_yticklabels()]
            self.assertIn("0%", labels)
            self.assertIn("100%", labels)
            plt.close(fig)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is required for smooth-line rendering")
    def test_plot_smooth_line_draws_more_points_without_markers(self) -> None:
        with TemporaryDirectory() as tmp:
            mpl, plt = matplotlib_context(Path(tmp) / "figure")
            apply_nature_style(mpl)
            fig, ax = plt.subplots(figsize=(3.0, 2.0))

            line = plot_smooth_line(ax, [0, 1, 2, 3], [0, 1, 0.2, 1.4], color="#0F4D92", samples=80)

            self.assertGreater(len(line.get_xdata()), 4)
            self.assertIn(line.get_marker(), ("None", None, ""))
            plt.close(fig)

    def test_plot_smooth_line_reports_missing_scipy_clearly_when_unavailable(self) -> None:
        if importlib.util.find_spec("scipy"):
            self.skipTest("scipy is installed")
        with TemporaryDirectory() as tmp:
            mpl, plt = matplotlib_context(Path(tmp) / "figure")
            apply_nature_style(mpl)
            fig, ax = plt.subplots(figsize=(3.0, 2.0))

            with self.assertRaisesRegex(FigureDependencyError, "requires scipy"):
                plot_smooth_line(ax, [0, 1, 2], [0, 1, 0], color="#0F4D92")
            plt.close(fig)

    def test_plot_smooth_line_rejects_duplicate_x_values(self) -> None:
        with TemporaryDirectory() as tmp:
            mpl, plt = matplotlib_context(Path(tmp) / "figure")
            apply_nature_style(mpl)
            fig, ax = plt.subplots(figsize=(3.0, 2.0))

            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                plot_smooth_line(ax, [0, 1, 1], [0, 1, 0], color="#0F4D92")
            plt.close(fig)

    def test_save_publication_figure_exports_all_formats(self) -> None:
        with TemporaryDirectory() as tmp:
            mpl, plt = matplotlib_context(Path(tmp) / "figure")
            apply_nature_style(mpl)
            fig, ax = plt.subplots(figsize=(3.2, 2.2))
            ax.plot([0, 1, 2], [1, 2, 1], color="#0F4D92", label="Trend")
            ax.set_xlabel("Capacity")
            ax.set_ylabel("Cost")
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=1)
            fig.subplots_adjust(left=0.18, bottom=0.24, top=0.80)

            paths = save_publication_figure(fig, Path(tmp) / "publication")

            self.assertEqual(set(paths), {"publication_png", "publication_svg", "publication_pdf"})
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths.values()))
            plt.close(fig)

    def test_save_publication_figure_rejects_legend_inside_data_area(self) -> None:
        with TemporaryDirectory() as tmp:
            mpl, plt = matplotlib_context(Path(tmp) / "figure")
            apply_nature_style(mpl)
            fig, ax = plt.subplots(figsize=(3.2, 2.2))
            ax.plot([0, 1, 2], [1, 2, 1], color="#0F4D92", label="Trend")
            ax.legend(loc="upper left")

            with self.assertRaisesRegex(FigureLayoutError, "legend overlaps the data area"):
                save_publication_figure(fig, Path(tmp) / "bad_legend")
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
