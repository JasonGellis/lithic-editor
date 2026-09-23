"""Unit tests for the DPI evaluation metrics on tiny synthetic skeletons."""

import numpy as np
import pytest

from tools.dpi_eval import metrics, sources


def _line(shape, row, col_start, col_end):
    mask = np.zeros(shape, dtype=bool)
    mask[row, col_start:col_end] = True
    return mask


class TestToleranceScores:
    def test_identical_skeletons_score_one(self):
        line = _line((40, 40), 20, 5, 35)
        assert metrics.tolerance_scores(line, line, tolerance=4) == (1.0, 1.0, 1.0)

    def test_shift_within_tolerance_still_scores_one(self):
        reference = _line((40, 40), 20, 5, 35)
        shifted = _line((40, 40), 23, 5, 35)
        precision, recall, f1 = metrics.tolerance_scores(shifted, reference, tolerance=4)
        assert (precision, recall, f1) == (1.0, 1.0, 1.0)

    def test_shift_beyond_tolerance_scores_zero(self):
        reference = _line((40, 40), 20, 5, 35)
        shifted = _line((40, 40), 30, 5, 35)
        assert metrics.tolerance_scores(shifted, reference, tolerance=4) == (0.0, 0.0, 0.0)

    def test_partial_overlap_gives_partial_recall(self):
        reference = _line((40, 40), 20, 0, 40)
        half = _line((40, 40), 20, 0, 20)
        precision, recall, _ = metrics.tolerance_scores(half, reference, tolerance=1)
        assert precision == 1.0
        assert recall == pytest.approx(0.5, abs=0.1)

    def test_empty_cases(self):
        empty = np.zeros((10, 10), dtype=bool)
        line = _line((10, 10), 5, 0, 10)
        assert metrics.tolerance_scores(empty, empty) == (1.0, 1.0, 1.0)
        assert metrics.tolerance_scores(empty, line) == (0.0, 0.0, 0.0)
        assert metrics.tolerance_scores(line, empty) == (0.0, 0.0, 0.0)


class TestResidualRipples:
    def test_closed_outline_has_no_ripples(self):
        outline = np.zeros((50, 50), dtype=bool)
        outline[10, 10:40] = outline[40, 10:40] = True
        outline[10:41, 10] = outline[10:41, 40] = True
        assert metrics.residual_ripples(outline) == 0

    def test_short_free_lines_are_counted(self):
        mask = np.zeros((60, 60), dtype=bool)
        for row in (10, 20, 30):
            mask[row, 5:25] = True
        assert metrics.residual_ripples(mask, max_length=200) == 3

    def test_long_free_line_is_not_a_ripple(self):
        mask = _line((20, 400), 10, 0, 400)
        assert metrics.residual_ripples(mask, max_length=200) == 0

    def test_hatch_attached_to_closed_outline_is_counted(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10, 10:41] = mask[40, 10:41] = True
        mask[10:41, 10] = mask[10:41, 40] = True   # closed outline: no free endpoints
        mask[25, 11:30] = True                      # one hatch line hanging off it
        assert metrics.residual_ripples(mask) == 1

    def test_open_line_with_a_branch_counts_every_free_end(self):
        # A T-junction on an open line yields three free-ended segments. This is
        # the known limit of the proxy: it counts open structural lines too.
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:41, 10] = True
        mask[25, 11:30] = True
        assert metrics.residual_ripples(mask) == 3


class TestGridAndVariants:
    def test_to_common_grid_recovers_shape_and_ink(self):
        image = np.full((100, 80), 255, dtype=np.uint8)
        image[40:60, :] = 0
        small = sources.make_variant(image, 600, 300)
        assert small.shape == (50, 40)
        mask = metrics.to_common_grid(small, image.shape)
        assert mask.shape == image.shape
        assert mask[50, 40] and not mask[10, 10]

    def test_make_variant_rejects_upsampling(self):
        with pytest.raises(ValueError):
            sources.make_variant(np.zeros((10, 10), dtype=np.uint8), 300, 600)

    def test_structural_loss_zero_for_identical(self):
        ink = _line((30, 30), 15, 0, 30)
        assert metrics.structural_loss(ink, ink) == 0.0
        assert np.isnan(metrics.structural_loss(ink, np.zeros_like(ink)))
