"""Tests for the configuration file and loader."""

import numpy as np
import pytest
import yaml

from lithic_editor.config import (
    ENV_VAR,
    Config,
    ConfigError,
    config_from_mapping,
    default_config_path,
    load_config,
    resolve_config,
    to_mapping,
)


class TestShippedFile:
    def test_shipped_file_matches_dataclass_defaults(self):
        assert to_mapping(load_config()) == to_mapping(Config())

    def test_default_path_exists(self):
        assert default_config_path().is_file()

    def test_source_records_the_file(self):
        assert load_config().source.endswith("config.yaml")


class TestOverrides:
    def test_partial_file_keeps_other_defaults(self, tmp_path):
        path = tmp_path / "mine.yaml"
        path.write_text("resolution:\n  min_line_width_px: 3.0\n")
        config = load_config(path)
        assert config.resolution.min_line_width_px == 3.0
        assert config.resolution.min_hatch_gap_px == Config().resolution.min_hatch_gap_px
        assert config.smoothing == Config().smoothing

    def test_environment_variable_selects_the_file(self, tmp_path, monkeypatch):
        path = tmp_path / "env.yaml"
        path.write_text("skeleton:\n  spur_length_px: 9\n")
        monkeypatch.setenv(ENV_VAR, str(path))
        assert load_config().skeleton.spur_length_px == 9
        assert resolve_config(None).skeleton.spur_length_px == 9

    def test_resolve_accepts_config_and_path(self, tmp_path):
        config = Config()
        assert resolve_config(config) is config
        path = tmp_path / "c.yaml"
        path.write_text("lines:\n  radius_in_line_widths: 0.6\n")
        assert resolve_config(path).lines.radius_in_line_widths == 0.6


class TestValidation:
    def test_unknown_section_is_an_error(self):
        with pytest.raises(ConfigError, match="unknown section"):
            config_from_mapping({"typo": {}})

    def test_unknown_key_names_the_section(self):
        with pytest.raises(ConfigError, match="resolution"):
            config_from_mapping({"resolution": {"min_line_width": 3}})

    def test_wrong_type_names_the_key(self):
        with pytest.raises(ConfigError, match="max_upscale_factor"):
            config_from_mapping({"resolution": {"max_upscale_factor": 2.5}})

    def test_out_of_range_value_is_an_error(self):
        with pytest.raises(ConfigError, match="upscale_model"):
            config_from_mapping({"resolution": {"upscale_model": "bicubic"}})

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "absent.yaml")

    def test_malformed_yaml_is_an_error(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("resolution: [unclosed\n")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_config(path)


class TestDerivedValues:
    """The helper methods reproduce the pipeline's original DPI rules."""

    def test_sauvola_window_by_dpi(self):
        t = Config().threshold
        assert (t.window_for_dpi(1200), t.window_for_dpi(600), t.window_for_dpi(300),
                t.window_for_dpi(150), t.window_for_dpi(None)) == (51, 51, 25, 15, 15)

    def test_cortex_limits_scale_quadratically(self):
        c = Config().cortex
        assert c.area_limits_for_dpi(150) == (3, 60)
        assert c.area_limits_for_dpi(300) == (12, 240)
        assert c.area_limits_for_dpi(600) == (48, 960)
        assert c.area_limits_for_dpi(75) == (2, 30)
        assert c.area_limits_for_dpi(None) == (3, 60)

    def test_y_tip_distance_by_dpi(self):
        r = Config().ripples
        assert (r.y_tip_distance_for_dpi(300), r.y_tip_distance_for_dpi(600),
                r.y_tip_distance_for_dpi(75), r.y_tip_distance_for_dpi(None)) == (5, 3, 2, 5)

    def test_bridge_distance(self):
        s = Config().skeleton
        assert s.bridge_distance(None, None) == 3.0
        assert s.bridge_distance(6.0, 20.0) == 9.0
        assert s.bridge_distance(6.0, 10.0) == 5.0
        assert s.bridge_distance(1.0, None) == 3.0

    def test_line_radius_range(self):
        assert Config().lines.radius_range(18.0) == (9, 14)
        assert Config().lines.radius_range(3.0) == (2, 2)


class TestConfigDrivesTheFunctions:
    def test_floors_change_the_upscale_factor(self):
        from lithic_editor.processing.resolution import LineGeometry, choose_upscale_factor
        geometry = LineGeometry(4.7, 20.0, 0.1)
        assert choose_upscale_factor(geometry) == 2
        low_floor = config_from_mapping({"resolution": {"min_line_width_px": 3.0}}).resolution
        assert choose_upscale_factor(geometry, settings=low_floor) == 1

    def test_smoothing_can_be_switched_off(self):
        from lithic_editor.processing.resolution import smooth_for_threshold
        gray = np.full((40, 40), 255, dtype=np.uint8)
        gray[20, :] = 0
        off = config_from_mapping({"smoothing": {"enabled": False}}).smoothing
        assert smooth_for_threshold(gray, 8.0, off) is gray
        assert smooth_for_threshold(gray, 8.0) is not gray

    def test_pipeline_accepts_a_config_path(self, tmp_path):
        from lithic_editor.processing import process_lithic_drawing
        img = np.full((80, 80), 255, dtype=np.uint8)
        img[10:14, 10:70] = img[66:70, 10:70] = 0
        img[10:70, 10:14] = img[10:70, 66:70] = 0
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"resolution": {"min_line_width_px": 1.0, "min_hatch_gap_px": 1.0}}))
        result = process_lithic_drawing(img, output_folder=str(tmp_path), dpi_info=300,
                                        upscale_low_dpi=True, return_scale_factor=True, config=path)
        assert result["working_scale_factor"] == 1
