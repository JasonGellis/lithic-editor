"""Tests for content-adaptive working resolution."""

import math

import numpy as np
import pytest

from lithic_editor.processing.resolution import (
    LineGeometry,
    choose_upscale_factor,
    measure_line_geometry,
    restore_to_grid,
)


def _drawing(line_width, gap, size=120):
    """A closed outline with horizontal hatching of given stroke width and clearance."""
    img = np.full((size, size), 255, dtype=np.uint8)
    w = line_width
    img[4:4 + w, 4:size - 4] = 0
    img[size - 4 - w:size - 4, 4:size - 4] = 0
    img[4:size - 4, 4:4 + w] = 0
    img[4:size - 4, size - 4 - w:size - 4] = 0
    row = 4 + w + gap
    while row + w < size - 4 - w - gap:
        img[row:row + w, 12:size - 30] = 0
        row += w + gap
    return img


class TestMeasureLineGeometry:
    def test_wide_strokes_and_clear_gaps_are_measured(self):
        geometry = measure_line_geometry(_drawing(line_width=5, gap=12))
        assert geometry.line_width == pytest.approx(5, abs=1)
        assert geometry.hatch_gap == pytest.approx(12, abs=2)

    def test_thin_close_strokes_are_measured(self):
        geometry = measure_line_geometry(_drawing(line_width=1, gap=2))
        assert geometry.line_width <= 2
        assert geometry.hatch_gap <= 3

    def test_blank_image_has_no_geometry(self):
        geometry = measure_line_geometry(np.full((40, 40), 255, dtype=np.uint8))
        assert math.isnan(geometry.line_width)
        assert math.isnan(geometry.hatch_gap)
        assert geometry.ink_fraction == 0

    def test_dots_do_not_shrink_the_gap(self):
        img = _drawing(line_width=5, gap=14)
        # Dense stipple in the open area at right, dots 2 px apart
        for y in range(20, 100, 4):
            for x in range(96, 112, 4):
                img[y:y + 2, x:x + 2] = 0
        geometry = measure_line_geometry(img)
        assert geometry.hatch_gap > 8


class TestChooseUpscaleFactor:
    def test_no_upscaling_when_geometry_is_fine(self):
        assert choose_upscale_factor(LineGeometry(10.0, 20.0, 0.1)) == 1

    def test_thin_lines_drive_the_factor(self):
        assert choose_upscale_factor(LineGeometry(1.8, 20.0, 0.1)) == 4
        assert choose_upscale_factor(LineGeometry(4.7, 20.0, 0.1)) == 2

    def test_close_hatching_drives_the_factor(self):
        assert choose_upscale_factor(LineGeometry(10.0, 5.0, 0.1)) == 3

    def test_factor_is_capped(self):
        assert choose_upscale_factor(LineGeometry(0.5, 0.5, 0.1)) == 4
        assert choose_upscale_factor(LineGeometry(0.5, 0.5, 0.1), max_factor=2) == 2

    def test_unmeasurable_geometry_means_no_upscaling(self):
        assert choose_upscale_factor(LineGeometry(math.nan, math.nan, 0.0)) == 1


class TestRestoreToGrid:
    def test_same_shape_is_returned_unchanged(self):
        img = np.full((30, 30), 255, dtype=np.uint8)
        assert restore_to_grid(img, (30, 30)) is img

    def test_thin_line_survives_a_four_times_reduction(self):
        big = np.full((120, 120), 255, dtype=np.uint8)
        big[58:64, 8:112] = 0  # a 6 px line at 4x becomes 1.5 px
        small = restore_to_grid(big, (30, 30))
        assert small.shape == (30, 30)
        assert small.dtype == np.uint8
        row_ink = (small < 128).any(axis=0)
        assert row_ink[2:28].all()
