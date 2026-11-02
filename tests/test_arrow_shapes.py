"""Tests for the arrow style shapes shared by the GUI and the CLI debug image."""

import pytest

from lithic_editor.annotations.arrow_shapes import (
    DEFAULT_STYLE, STYLE_NAMES, STYLES, arrow_parts, check_style, filled_outline, shaft_width,
)


def all_points(parts):
    points = list(parts.head)
    for shaft in parts.shafts:
        points.extend(shaft)
    if parts.bar is not None:
        points.extend(parts.bar)
    if parts.dot is not None:
        (cx, cy), r = parts.dot
        points.extend([(cx - r, cy - r), (cx + r, cy + r)])
    return points


class TestStyles:
    def test_five_styles_with_filled_first(self):
        assert STYLE_NAMES == ("filled", "open", "dot", "crossed", "dashed")
        assert DEFAULT_STYLE == "filled"
        assert [label for _, label in STYLES] == ["Filled", "Open", "Dot", "Crossed", "Dashed"]

    def test_unknown_style_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown arrow style 'solid'"):
            check_style("solid")
        with pytest.raises(ValueError):
            arrow_parts("solid", 50)


class TestFilledOutline:
    def test_is_the_seven_point_arrow_of_the_canvas(self):
        # shaft 1.3 x size, head 0.8 long and 0.6 wide, shaft width 5 percent (2.5 px at 50)
        assert filled_outline(50) == (
            (-32.5, -1.25), (-7.5, -1.25), (-7.5, -15.0), (32.5, 0.0), (-7.5, 15.0), (-7.5, 1.25), (-32.5, 1.25),
        )

    def test_shaft_width_has_a_floor_of_two_pixels(self):
        assert shaft_width(20) == 2.0
        assert shaft_width(100) == 5.0


class TestArrowParts:
    @pytest.mark.parametrize("style", STYLE_NAMES)
    def test_every_part_lies_inside_the_arrow_box(self, style):
        size = 60
        parts = arrow_parts(style, size)
        for x, y in all_points(parts):
            assert -0.65 * size - 1e-9 <= x <= 0.65 * size + 1e-9
            assert -0.3 * size - 1e-9 <= y <= 0.3 * size + 1e-9

    def test_the_head_is_the_same_in_every_style(self):
        heads = {arrow_parts(style, 50).head for style in STYLE_NAMES}
        assert heads == {((-7.5, -15.0), (32.5, 0.0), (-7.5, 15.0))}

    def test_filled_and_open_differ_only_in_the_head_fill(self):
        filled, open_ = arrow_parts("filled", 50), arrow_parts("open", 50)
        assert filled.filled_head and not open_.filled_head
        assert filled.shafts == open_.shafts and filled.dot is None and open_.dot is None

    def test_dot_sits_at_the_tail(self):
        (cx, cy), radius = arrow_parts("dot", 50).dot
        assert radius == 6.0 and cy == 0.0
        assert cx - radius == -32.5  # the dot touches the tail end

    def test_bar_crosses_the_shaft_near_the_tail(self):
        bar = arrow_parts("crossed", 50).bar
        xs = sorted({x for x, _ in bar})
        ys = sorted({y for _, y in bar})
        assert ys == [-10.0, 10.0]  # 0.4 x size long
        assert xs[1] - xs[0] == shaft_width(50)
        assert sum(xs) / 2 == -32.5 + 12.5  # 0.25 x size from the tail

    def test_dashed_shaft_is_four_dashes_with_equal_gaps(self):
        shafts = arrow_parts("dashed", 70).shafts
        assert len(shafts) == 4
        spans = [(min(x for x, _ in d), max(x for x, _ in d)) for d in shafts]
        lengths = [b - a for a, b in spans]
        gaps = [spans[i + 1][0] - spans[i][1] for i in range(3)]
        assert all(abs(length - lengths[0]) < 1e-9 for length in lengths)
        assert all(abs(gap - lengths[0]) < 1e-9 for gap in gaps)
        assert spans[0][0] == -0.65 * 70 and abs(spans[-1][1] - (0.65 * 70 - 0.8 * 70)) < 1e-9
