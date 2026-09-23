"""Tests for individual pipeline stages: endpoint detection, bridging, and line weight."""

import numpy as np
import pytest
from PIL import Image

from lithic_editor.processing import process_lithic_drawing
from lithic_editor.processing.resolution import measure_line_geometry
from lithic_editor.processing.ripple_removal import (
    _bridge_close_endpoints,
    _find_endpoints_and_junctions,
)


def _outline_with_hatching(size=240, width=6, gap=18, cortex=False):
    """A closed outline with hatch lines inside that end free, black on white."""
    img = np.full((size, size), 255, dtype=np.uint8)
    m = 20
    img[m:m + width, m:size - m] = 0
    img[size - m - width:size - m, m:size - m] = 0
    img[m:size - m, m:m + width] = 0
    img[m:size - m, size - m - width:size - m] = 0
    row = m + width + gap
    while row + width < size - m - width - gap:
        img[row:row + width, m + width:size // 2] = 0   # attached to the left edge, free at the right
        row += width + gap
    if cortex:
        # Dots of 8x8 px: within the cortex area limits at 600 DPI (48 to 960 px)
        for y in range(size // 2 + 10, size - m - 12, 16):
            for x in range(size // 2 + 20, size - m - 12, 16):
                img[y:y + 8, x:x + 8] = 0
    return img


class TestEndpointsAndJunctions:
    def test_open_line_has_two_endpoints_and_no_junction(self):
        skel = np.zeros((20, 40), dtype=bool)
        skel[10, 5:35] = True
        endpoints, junctions = _find_endpoints_and_junctions(skel)
        assert sorted(endpoints) == [(5, 10), (34, 10)]
        assert junctions == []

    def test_t_shape_has_three_endpoints_and_one_junction(self):
        skel = np.zeros((40, 40), dtype=bool)
        skel[20, 5:35] = True
        skel[5:20, 20] = True
        endpoints, junctions = _find_endpoints_and_junctions(skel)
        assert len(endpoints) == 3
        # Every pixel with three or more neighbours is a junction, so a T gives a small cluster
        assert (20, 20) in junctions
        assert all(abs(x - 20) <= 1 and abs(y - 20) <= 1 for x, y in junctions)

    def test_border_pixels_are_ignored(self):
        skel = np.zeros((10, 10), dtype=bool)
        skel[0, :] = True
        endpoints, junctions = _find_endpoints_and_junctions(skel)
        assert endpoints == [] and junctions == []


class TestBridging:
    def test_close_free_ends_are_joined(self):
        skel = np.zeros((20, 40), dtype=bool)
        skel[10, 5:18] = True
        skel[10, 21:35] = True          # 3 px gap
        endpoints, _ = _find_endpoints_and_junctions(skel)
        assert len(endpoints) == 4
        assert _bridge_close_endpoints(skel, endpoints, max_distance=4.0) == 1
        endpoints, _ = _find_endpoints_and_junctions(skel)
        assert len(endpoints) == 2

    def test_distant_free_ends_are_left_alone(self):
        skel = np.zeros((20, 60), dtype=bool)
        skel[10, 5:20] = True
        skel[10, 40:55] = True          # 20 px gap
        endpoints, _ = _find_endpoints_and_junctions(skel)
        assert _bridge_close_endpoints(skel, endpoints, max_distance=4.0) == 0
        assert skel.sum() == 30

    def test_each_end_is_bridged_at_most_once(self):
        skel = np.zeros((30, 30), dtype=bool)
        skel[10, 5:12] = True     # ends at (11,10)
        skel[10, 14:20] = True    # ends at (14,10) and (19,10)
        skel[10, 22:28] = True    # ends at (22,10)
        endpoints, _ = _find_endpoints_and_junctions(skel)
        bridges = _bridge_close_endpoints(skel, endpoints, max_distance=3.0)
        assert bridges == 2
        endpoints, _ = _find_endpoints_and_junctions(skel)
        assert len(endpoints) == 2

    def test_broken_outline_survives_the_pipeline(self, tmp_path):
        """A small break in a closed outline must not turn the outline into a ripple."""
        img = _outline_with_hatching(width=6, gap=18)
        img[20:26, 118:120] = 255           # a 2 px break in the top edge
        result = process_lithic_drawing(img, output_folder=str(tmp_path), dpi_info=600)
        top_band = result[16:30, 30:110]
        assert (top_band < 128).sum() >= 4 * 80, "top edge lost after the break"
        assert (result[16:30, 130:210] < 128).sum() >= 4 * 80, "top edge lost beyond the break"


class TestRippleRemovalOnSyntheticDrawing:
    def test_hatching_is_removed_and_outline_is_kept(self, tmp_path):
        img = _outline_with_hatching()
        result = process_lithic_drawing(img, output_folder=str(tmp_path), dpi_info=600)
        interior = result[40:200, 40:200]
        assert (interior < 128).sum() < 0.02 * interior.size, "hatching survived"
        assert (result[16:30, 30:210] < 128).sum() >= 4 * 180, "top edge lost"
        assert (result[30:210, 16:30] < 128).sum() >= 4 * 180, "left edge lost"

    def test_line_weight_is_preserved(self, tmp_path):
        img = _outline_with_hatching(width=8, gap=24)
        result = process_lithic_drawing(img, output_folder=str(tmp_path), dpi_info=600)
        width_in = measure_line_geometry(img).line_width
        width_out = measure_line_geometry(result).line_width
        assert abs(width_out - width_in) <= 2

    def test_cortex_is_kept(self, tmp_path):
        img = _outline_with_hatching(cortex=True)
        result = process_lithic_drawing(img, output_folder=str(tmp_path), dpi_info=600)
        cortex_region = result[130:210, 140:210]
        assert (cortex_region < 128).sum() >= 20

    def test_never_downscales(self, tmp_path):
        img = _outline_with_hatching(width=14, gap=40)
        result = process_lithic_drawing(img, output_folder=str(tmp_path), dpi_info=1200,
                                        upscale_low_dpi=True, return_scale_factor=True)
        assert result["working_scale_factor"] == 1
        assert result["processed_image"].shape == img.shape

    def test_debug_files_are_written_in_order(self, tmp_path):
        img = _outline_with_hatching(width=2, gap=6, size=120)
        process_lithic_drawing(img, output_folder=str(tmp_path), dpi_info=150,
                               upscale_low_dpi=True, save_debug=True, debug_filename="t")
        names = sorted(p.name for p in tmp_path.glob("*.png"))
        assert names[0] == "0_t_input.png"
        assert "0a_t_upscaled.png" in names
        assert "1b_t_smoothed.png" in names
        assert "8_t_final_cleaned.png" in names
        assert names[-1] == "9_t_restored.png"

    def test_debug_image_dpi_tags_are_truthful(self, tmp_path):
        img = _outline_with_hatching(width=2, gap=6, size=120)
        process_lithic_drawing(img, output_folder=str(tmp_path), dpi_info=150,
                               upscale_low_dpi=True, save_debug=True, debug_filename="t")
        with Image.open(tmp_path / "0a_t_upscaled.png") as up, \
                Image.open(tmp_path / "9_t_restored.png") as restored:
            factor = up.size[0] // 120
            assert round(up.info["dpi"][0]) == 150 * factor
            assert round(restored.info["dpi"][0]) == 150


class TestScaleImageHandling:
    def test_scale_is_unchanged_when_result_is_restored(self, tmp_path):
        img = _outline_with_hatching(width=2, gap=6, size=120)
        scale = np.full((10, 60), 255, dtype=np.uint8)
        scale[4:6, 5:55] = 0
        scale_path = tmp_path / "bar.png"
        Image.fromarray(scale).save(scale_path)
        result = process_lithic_drawing(img, output_folder=str(tmp_path), dpi_info=150,
                                        upscale_low_dpi=True, scale_image_path=str(scale_path))
        assert result["scale_factor"] == 1
        assert result["processed_scale"].shape == scale.shape

    def test_scale_is_upscaled_with_a_kept_result(self, tmp_path):
        img = _outline_with_hatching(width=2, gap=6, size=120)
        scale = np.full((10, 60), 255, dtype=np.uint8)
        scale[4:6, 5:55] = 0
        scale_path = tmp_path / "bar.png"
        Image.fromarray(scale).save(scale_path)
        result = process_lithic_drawing(img, output_folder=str(tmp_path), dpi_info=150,
                                        upscale_low_dpi=True, scale_image_path=str(scale_path),
                                        restore_original_size=False)
        factor = result["scale_factor"]
        assert factor > 1
        assert result["processed_scale"].shape == (10 * factor, 60 * factor)
        assert result["final_dpi"] == 150 * factor


class TestRealExampleImages:
    """The measured geometry of the shipped example images gives the expected factors."""

    @pytest.mark.parametrize("name,expected_factor", [
        ("lithic_75dpi", 4), ("lithic_150", 2), ("lithic_300dpi", 1),
        ("369", 2), ("371", 2), ("365", 2),
    ])
    def test_expected_upscale_factor(self, name, expected_factor):
        from pathlib import Path
        from lithic_editor.processing.resolution import choose_upscale_factor
        path = Path(__file__).resolve().parents[1] / "example_images" / f"{name}.png"
        if not path.exists():
            pytest.skip(f"{path.name} not present")
        gray = np.array(Image.open(path).convert("L"))
        assert choose_upscale_factor(measure_line_geometry(gray)) == expected_factor
