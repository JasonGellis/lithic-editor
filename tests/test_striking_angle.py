"""Tests for the striking angle measured from a scar's ripples."""

import math

import numpy as np
import pytest
from PIL import Image

from lithic_editor.config import AngleSettings, Config
from lithic_editor.processing import process_lithic_drawing
from lithic_editor.processing.striking_angle import (
    compass_from_vector,
    enclosed_regions,
    group_ripples_by_scar,
    mean_orientation,
    measure_scar,
    measure_striking_angles,
    ripple_arc_depth,
    ripple_orientation,
    vector_from_compass,
    widget_angle_from_compass,
)


def _arc(centre, radius, start_deg, end_deg, n=60):
    """Points of an arc around ``centre`` from ``start_deg`` to ``end_deg`` (image coordinates)."""
    angles = np.radians(np.linspace(start_deg, end_deg, n))
    return np.column_stack([centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)])


class TestCompassFrame:
    """The frame PyLithics uses for arrow_angle, checked against its own detector."""

    @pytest.mark.parametrize("dx, dy, compass", [
        (0, 1, 0.0), (-1, 0, 90.0), (0, -1, 180.0), (1, 0, 270.0), (-1, 1, 45.0), (1, 1, 315.0),
    ])
    def test_compass_from_vector(self, dx, dy, compass):
        assert compass_from_vector(dx, dy) == compass

    @pytest.mark.parametrize("compass", [0, 45, 90, 135, 180, 225, 270, 315])
    def test_vector_round_trip(self, compass):
        assert compass_from_vector(*vector_from_compass(compass)) == float(compass)

    def test_widget_angle_points_the_head_the_same_way(self):
        # Arrow.angle 0 draws a head that points right, which is compass 270
        assert widget_angle_from_compass(270) == 0
        assert widget_angle_from_compass(0) == 90
        assert widget_angle_from_compass(180) == 270


class TestRippleGeometry:
    def test_orientation_of_a_horizontal_line(self):
        pts = np.column_stack([np.arange(30.0), np.full(30, 5.0)])
        assert ripple_orientation(pts) == pytest.approx(0.0, abs=1e-6)

    def test_orientation_of_a_vertical_line(self):
        pts = np.column_stack([np.full(30, 5.0), np.arange(30.0)])
        assert ripple_orientation(pts) == pytest.approx(math.pi / 2, abs=1e-6)

    def test_straight_line_has_no_arc_depth(self):
        pts = np.column_stack([np.arange(30.0), np.arange(30.0)])
        assert np.allclose(ripple_arc_depth(pts), 0.0)

    def test_arc_depth_points_away_from_its_centre(self):
        # Arc below the centre (image y down): apex is at larger y than the chord
        pts = _arc((50, 50), 40, 45, 135)
        depth = ripple_arc_depth(pts)
        assert depth[1] > 5 and abs(depth[0]) < 1

    def test_mean_orientation_of_equal_angles(self):
        mean, resultant = mean_orientation([0.3, 0.3, 0.3], [1, 1, 1])
        assert mean == pytest.approx(0.3) and resultant == pytest.approx(1.0)

    def test_mean_orientation_treats_theta_and_theta_plus_pi_alike(self):
        mean, resultant = mean_orientation([0.2, 0.2 + math.pi], [1, 1])
        assert mean == pytest.approx(0.2) and resultant == pytest.approx(1.0)

    def test_perpendicular_orientations_have_no_resultant(self):
        _, resultant = mean_orientation([0.0, math.pi / 2], [1, 1])
        assert resultant == pytest.approx(0.0, abs=1e-9)


class TestMeasureScar:
    settings = AngleSettings()

    def test_fan_of_arcs_points_away_from_the_percussion_point(self):
        # Arcs around a percussion point at (100, 20), curving downward: the force went down, compass 0
        ripples = [_arc((100, 20), r, 60, 120) for r in (30, 45, 60, 75)]
        record = measure_scar(1, ripples, self.settings)
        assert abs((record.compass + 180.0) % 360.0 - 180.0) <= 3.0
        assert record.direction_certain
        assert record.ripples == 4 and record.spread < 0.05

    def test_fan_pointing_right(self):
        # Arcs around (20, 100) curving to the right: the force went right, compass 270
        ripples = [_arc((20, 100), r, -30, 30) for r in (30, 45, 60, 75)]
        record = measure_scar(2, ripples, self.settings)
        assert record.compass == pytest.approx(270.0, abs=3.0)
        assert record.direction_certain

    def test_straight_hatching_is_uncertain_but_has_a_line(self):
        ripples = [np.column_stack([np.arange(40.0), np.full(40, y)]) for y in (10.0, 20.0, 30.0, 40.0)]
        record = measure_scar(3, ripples, self.settings)
        assert record.direction_certain is False
        assert record.compass in (0.0, 180.0)

    def test_centroid_is_the_mean_of_all_ripple_pixels(self):
        ripples = [_arc((100, 20), r, 60, 120) for r in (30, 60)]
        record = measure_scar(1, ripples, self.settings)
        expected = np.concatenate(ripples).mean(axis=0)
        assert (record.x, record.y) == pytest.approx(tuple(expected))

    def test_on_grid_divides_the_centroid(self):
        record = measure_scar(1, [_arc((100, 20), r, 60, 120) for r in (30, 60)], self.settings)
        halved = record.on_grid(2)
        assert (halved.x, halved.y) == pytest.approx((record.x / 2, record.y / 2))
        assert halved.compass == record.compass
        assert halved.extent == pytest.approx(record.extent / 2)

    def test_extent_is_the_root_of_the_bounding_box_area(self):
        ripples = [np.column_stack([np.arange(40.0), np.full(40, y)]) for y in (10.0, 20.0, 30.0, 40.0)]
        record = measure_scar(1, ripples, self.settings)
        assert record.extent == pytest.approx(math.sqrt(40 * 31))

    def test_arrow_size_is_clamped(self):
        from lithic_editor.processing.striking_angle import arrow_size_for
        assert arrow_size_for(100, 50) == 50      # a fifth of 100 is below the minimum
        assert arrow_size_for(600, 50) == 120     # a fifth of the extent
        assert arrow_size_for(5000, 50) == 200    # four times the minimum


class TestGrouping:
    def test_enclosed_regions_ignore_the_outside(self):
        structure = np.zeros((40, 60), dtype=bool)
        structure[5, 5:55] = structure[34, 5:55] = True
        structure[5:35, 5] = structure[5:35, 54] = True
        structure[5:35, 30] = True  # a divider: two faces
        regions = enclosed_regions(structure)
        assert regions[0, 0] == 0 and regions[20, 2] == 0
        inside = {int(regions[20, 15]), int(regions[20, 45])}
        assert 0 not in inside and len(inside) == 2

    def test_ripples_are_grouped_by_the_face_they_lie_in(self):
        structure = np.zeros((40, 60), dtype=bool)
        structure[5, 5:55] = structure[34, 5:55] = True
        structure[5:35, 5] = structure[5:35, 54] = True
        structure[5:35, 30] = True
        regions = enclosed_regions(structure)
        labeled = np.zeros_like(structure, dtype=np.int32)
        labeled[10, 8:25] = 1
        labeled[15, 8:25] = 2
        labeled[10, 35:50] = 3
        labeled[25, 35:50] = 4
        labeled[20, 0:4] = 5   # outside: left out
        groups = group_ripples_by_scar(labeled, {1, 2, 3, 4, 5}, regions)
        sizes = sorted(len(g) for g in groups.values())
        assert sizes == [2, 2]

    def test_small_groups_are_skipped(self):
        structure = np.zeros((40, 60), dtype=bool)
        structure[5, 5:55] = structure[34, 5:55] = True
        structure[5:35, 5] = structure[5:35, 54] = True
        labeled = np.zeros_like(structure, dtype=np.int32)
        for i, y in enumerate((10, 15, 20), start=1):
            labeled[y, 8:40] = i
        records, skipped = measure_striking_angles(labeled, {1, 2, 3}, structure, AngleSettings(min_ripples=4))
        assert records == [] and skipped == [(1, 3, "ripples")]
        records, skipped = measure_striking_angles(labeled, {1, 2, 3}, structure, AngleSettings(min_ripples=3))
        assert len(records) == 1 and skipped == []


    def test_cortex_faces_are_skipped(self):
        structure = np.zeros((40, 60), dtype=bool)
        structure[5, 5:55] = structure[34, 5:55] = True
        structure[5:35, 5] = structure[5:35, 54] = True
        labeled = np.zeros_like(structure, dtype=np.int32)
        for i, y in enumerate((10, 15, 20, 25), start=1):
            labeled[y, 8:40] = i
        cortex = np.zeros_like(structure)
        for x in (42, 46, 50):
            cortex[29:31, x:x + 2] = True   # three dots
        settings = AngleSettings(min_ripples=4, max_cortex_dots=2)
        records, skipped = measure_striking_angles(labeled, {1, 2, 3, 4}, structure, settings, cortex=cortex)
        assert records == [] and skipped == [(1, 4, "cortex")]
        # two dots or fewer, or no cortex mask at all, keep the scar
        records, skipped = measure_striking_angles(labeled, {1, 2, 3, 4}, structure,
                                                   AngleSettings(min_ripples=4, max_cortex_dots=3), cortex=cortex)
        assert len(records) == 1 and skipped == []
        records, skipped = measure_striking_angles(labeled, {1, 2, 3, 4}, structure, settings)
        assert len(records) == 1


def _two_scar_drawing(path, width=6):
    """A drawing with two closed outlines, each with four curved hatch lines.

    The left scar's arcs are centred above it and curve downward, so the force went down (compass 0).
    The right scar's arcs are centred to its right and curve leftward, so the force went left (compass 90).
    A third outline holds two hatch lines only, so it is skipped. A fourth outline
    holds four hatch lines and cortex dots, so it is skipped as cortex.
    """
    import cv2
    size = 520
    img = np.full((size, size), 255, dtype=np.uint8)

    def box(x0, y0, x1, y1):
        cv2.rectangle(img, (x0, y0), (x1, y1), 0, width)

    def hatch(centre, radii, start_deg, end_deg):
        for r in radii:
            pts = _arc(centre, r, start_deg, end_deg, n=80).astype(np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, 0, width)

    box(20, 20, 240, 300)
    hatch((130, -60), (110, 140, 170, 200, 230), 65, 115)      # arcs curve downward: force down
    box(280, 20, 500, 300)
    hatch((600, 160), (130, 160, 190, 220, 250), 160, 200)     # arcs curve leftward: force left
    box(20, 340, 240, 500)
    hatch((130, 250), (110, 140), 70, 110)
    box(280, 340, 500, 500)
    hatch((390, 250), (110, 125, 140, 155), 75, 105)
    for y in range(440, 490, 14):
        for x in range(300, 480, 14):
            cv2.circle(img, (x, y), 3, 0, -1)      # cortex dots, 28 px area at 300 DPI
    Image.fromarray(img).save(path, dpi=(300, 300))
    return path


class TestPipeline:
    def test_two_scars_get_angles_and_a_small_one_is_skipped(self, tmp_path, capsys):
        path = _two_scar_drawing(str(tmp_path / "scars.png"))
        result = process_lithic_drawing(path, output_folder=str(tmp_path / "debug"), save_debug=True,
                                        measure_angles=True)
        assert isinstance(result, dict)
        records = result['striking_angles']
        by_x = sorted(records, key=lambda r: r['x'])
        assert len(by_x) == 2, records
        left, right = by_x
        assert abs((left['compass'] + 180) % 360 - 180) <= 10, left
        assert abs(right['compass'] - 90) <= 10, right
        assert left['ripples'] >= 4 and right['ripples'] >= 4
        assert 20 < left['x'] < 240 and 20 < left['y'] < 300
        assert 280 < right['x'] < 500 and 20 < right['y'] < 300
        out = capsys.readouterr().out
        assert "fewer than 4" in out and "cortex stipple" in out
        assert (tmp_path / "debug" / "6a_scars_striking_angles.png").exists()
        assert (tmp_path / "debug" / "10_scars_striking_arrows.png").exists()

    def test_no_measurement_without_the_flag(self, tmp_path):
        path = _two_scar_drawing(str(tmp_path / "scars.png"))
        result = process_lithic_drawing(path, output_folder=str(tmp_path / "debug"))
        assert isinstance(result, np.ndarray)

    def test_centroid_is_on_the_input_grid_after_upscaling(self, tmp_path):
        # A thin-lined drawing is upscaled for processing; the centroid must come back
        path = _two_scar_drawing(str(tmp_path / "thin.png"), width=2)
        result = process_lithic_drawing(path, output_folder=str(tmp_path / "debug"),
                                        measure_angles=True, upscale_low_dpi=True)
        assert result['working_scale_factor'] > 1 and result['scale_factor'] == 1
        for r in result['striking_angles']:
            assert 0 <= r['x'] < 520 and 0 <= r['y'] < 520

    def test_min_ripples_from_config(self, tmp_path):
        path = _two_scar_drawing(str(tmp_path / "scars.png"))
        config = Config(angles=AngleSettings(min_ripples=2))
        result = process_lithic_drawing(path, output_folder=str(tmp_path / "debug"),
                                        measure_angles=True, config=config)
        assert len(result['striking_angles']) == 3   # the cortex face stays out

    def test_cortex_face_gets_no_arrow(self, tmp_path):
        path = _two_scar_drawing(str(tmp_path / "scars.png"))
        kept = process_lithic_drawing(path, output_folder=str(tmp_path / "debug"), measure_angles=True)
        assert [r['scar'] for r in kept['striking_angles']] and all(r['y'] < 320 for r in kept['striking_angles'])
