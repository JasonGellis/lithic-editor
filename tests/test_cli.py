"""
Tests for the command-line interface.
"""

import sys

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from lithic_editor.cli.main import (
    create_parser, 
    validate_input_file,
    process_image_cli,
    open_docs
)


def mock_args(**values):
    """Parsed-argument stand-in with the angle options unset, as argparse leaves them."""
    defaults = {'min_ripples': None, 'min_arc_depth': None, 'max_cortex_dots': None}
    return MagicMock(**{**defaults, **values})


class TestCLIParser:
    """Test CLI argument parser."""
    
    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == 'lithic-editor'
    
    def test_version_flag(self):
        """Test version flag."""
        parser = create_parser()
        
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['--version'])
        
        # Version flag should exit with 0
        assert exc_info.value.code == 0
    
    def test_gui_flag(self):
        parser = create_parser()
        args = parser.parse_args(['--gui'])
        assert args.gui is True and args.input_image is None

    def test_drawing_is_the_positional_argument(self):
        parser = create_parser()
        args = parser.parse_args(['image.png'])
        assert args.input_image == 'image.png'
        assert args.output == 'image_debug'  # default
        assert args.debug is False and args.quiet is False
        assert args.gui is False and args.docs is False

    def test_processing_options(self):
        parser = create_parser()
        args = parser.parse_args(['image.png', '--output', 'results/', '--debug', '--quiet'])
        assert args.input_image == 'image.png'
        assert args.output == 'results/'
        assert args.debug is True and args.quiet is True

    def test_cortex_flag(self):
        parser = create_parser()
        assert parser.parse_args(['image.png', '--no-preserve-cortex']).no_preserve_cortex is True
        assert parser.parse_args(['image.png']).no_preserve_cortex is False

    def test_no_subcommands_remain(self):
        parser = create_parser()
        args = parser.parse_args(['process'])
        assert args.input_image == 'process'  # 'process' is a file name now, not a command
        with pytest.raises(SystemExit):
            parser.parse_args(['help', 'api'])

    def test_help_flags_print_the_full_text_and_exit(self, capsys):
        parser = create_parser()
        for flag, marker in (('--help', 'USAGE'), ('-h', 'USAGE'), ('--help-api', 'LITHIC EDITOR API')):
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([flag])
            assert exc_info.value.code == 0
            assert marker in capsys.readouterr().out

    def test_docs_flag(self):
        parser = create_parser()
        args = parser.parse_args(['--docs'])
        assert args.docs is True and args.offline is False and args.input_image is None

    def test_docs_offline(self):
        parser = create_parser()
        args = parser.parse_args(['--docs', '--offline'])
        assert args.docs is True and args.offline is True


class TestInputValidation:
    """Test input file validation."""
    
    def test_valid_file(self, sample_image):
        """Test validation of valid image file."""
        result = validate_input_file(str(sample_image))
        assert result == sample_image
    
    def test_nonexistent_file(self):
        """Test validation of non-existent file."""
        with pytest.raises(FileNotFoundError):
            validate_input_file("nonexistent.png")
    
    def test_directory_instead_of_file(self, temp_dir):
        """Test validation fails for directory."""
        with pytest.raises(ValueError):
            validate_input_file(str(temp_dir))
    
    def test_unsupported_format(self, temp_dir):
        """Test validation of unsupported format."""
        bad_file = temp_dir / "test.txt"
        bad_file.write_text("not an image")
        
        with pytest.raises(ValueError) as exc_info:
            validate_input_file(str(bad_file))
        
        assert "Unsupported file format" in str(exc_info.value)
    
    @pytest.mark.parametrize("ext", ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'])
    def test_supported_formats(self, temp_dir, ext):
        """Test all supported formats pass validation."""
        test_file = temp_dir / f"test{ext}"
        test_file.touch()  # Create empty file
        
        # Should not raise exception for file existence
        # (would fail on actual processing, but validation passes)
        result = validate_input_file(str(test_file))
        assert result == test_file


class TestProcessImageCLI:
    """Test process_image_cli function."""
    
    def test_process_success(self, sample_image, tmp_path):
        """Test successful image processing."""
        args = mock_args()
        args.input_image = str(sample_image)
        args.output = str(tmp_path / "out")
        args.debug = False
        args.quiet = False
        
        with patch('lithic_editor.cli.main.process_lithic_drawing') as mock_process:
            mock_process.return_value = np.full((20, 20), 255, dtype=np.uint8)
            
            result = process_image_cli(args)
            
            assert result == 0
            mock_process.assert_called_once()
    
    def test_process_quiet_mode(self, sample_image, capsys, tmp_path):
        """Test quiet mode suppresses output."""
        args = mock_args()
        args.input_image = str(sample_image)
        args.output = str(tmp_path / "out")
        args.debug = False
        args.quiet = True
        
        with patch('lithic_editor.cli.main.process_lithic_drawing') as mock_process:
            mock_process.return_value = np.full((20, 20), 255, dtype=np.uint8)
            
            result = process_image_cli(args)
            
            # Check no output was printed
            captured = capsys.readouterr()
            assert captured.out == ""
    
    def test_process_file_not_found(self, tmp_path):
        """Test handling of missing file."""
        args = mock_args()
        args.input_image = "nonexistent.png"
        args.output = str(tmp_path / "out")
        args.debug = False
        args.quiet = False
        
        result = process_image_cli(args)
        assert result == 1
    
    def test_process_with_cortex_parameter(self, sample_image, tmp_path):
        """Test processing with cortex preservation parameter."""
        args = mock_args()
        args.input_image = str(sample_image)
        args.output = str(tmp_path / "out")
        args.debug = False
        args.quiet = False
        args.no_preserve_cortex = True
        
        with patch('lithic_editor.cli.main.process_lithic_drawing') as mock_process:
            mock_process.return_value = np.full((20, 20), 255, dtype=np.uint8)
            
            result = process_image_cli(args)
            
            assert result == 0
            # Should pass preserve_cortex=False when no_preserve_cortex=True
            mock_process.assert_called_once()
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs['preserve_cortex'] == False


class TestMainDispatch:
    """main() routes each flag to its function."""

    def _run(self, argv, monkeypatch):
        from lithic_editor.cli import main as cli
        monkeypatch.setattr(sys, 'argv', ['lithic-editor'] + argv)
        return cli.main()

    def test_no_arguments_shows_the_full_help(self, monkeypatch):
        with patch('lithic_editor.cli.main.show_help') as mock_help:
            assert self._run([], monkeypatch) == 0
            mock_help.assert_called_once()

    def test_gui_flag_starts_the_interface(self, monkeypatch):
        with patch('lithic_editor.cli.main.launch_gui_cli', return_value=0) as mock_gui:
            assert self._run(['--gui'], monkeypatch) == 0
            mock_gui.assert_called_once()

    def test_docs_flag_opens_the_documentation(self, monkeypatch):
        with patch('lithic_editor.cli.main.open_docs', return_value=0) as mock_docs:
            assert self._run(['--docs'], monkeypatch) == 0
            mock_docs.assert_called_once()

    def test_a_drawing_is_processed(self, monkeypatch):
        with patch('lithic_editor.cli.main.process_image_cli', return_value=0) as mock_process:
            assert self._run(['drawing.png', '--quiet'], monkeypatch) == 0
            assert mock_process.call_args[0][0].input_image == 'drawing.png'


class TestDocsCLI:
    """Test docs command functionality."""
    
    def test_open_docs_online(self):
        """Test opening online documentation."""
        args = mock_args()
        args.offline = False
        
        with patch('webbrowser.open') as mock_open:
            result = open_docs(args)
            
            assert result == 0
            mock_open.assert_called_once_with("https://jasongellis.github.io/lithic-editor/")
    
    def test_open_docs_offline(self):
        """Test opening offline documentation."""
        args = mock_args()
        args.offline = True
        
        with patch('lithic_editor.cli.docs_server.serve_docs') as mock_serve:
            mock_serve.return_value = 0
            
            result = open_docs(args)
            
            assert result == 0
            mock_serve.assert_called_once()

class TestKeepUpscaledOutput:
    """The CLI keeps the working size on request and scales the scale image with it."""

    @staticmethod
    def _thin_drawing(size=60):
        img = np.full((size, size), 255, dtype=np.uint8)
        img[5, 5:55] = img[54, 5:55] = 0
        img[5:55, 5] = img[5:55, 54] = 0
        for row in range(12, 48, 3):
            img[row, 10:40] = 0
        return img

    def _args(self, image_path, out_dir, **overrides):
        args = mock_args()
        args.input_image = str(image_path)
        args.output = str(out_dir)
        args.debug = False
        args.quiet = True
        args.no_preserve_cortex = False
        args.auto_upscale = True
        args.default_dpi = None
        args.upscale_model = 'espcn'
        args.keep_upscaled = False
        args.scale_image = None
        args.config = None
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_default_returns_input_size_and_dpi(self, tmp_path):
        from PIL import Image
        from lithic_editor.cli.main import process_image_cli
        image_path = tmp_path / "thin.png"
        Image.fromarray(self._thin_drawing()).save(image_path, dpi=(150, 150))

        assert process_image_cli(self._args(image_path, tmp_path / "out")) == 0
        with Image.open(tmp_path / "out" / "thin_cleaned.png") as out:
            assert out.size == (60, 60)
            assert tuple(round(d) for d in out.info["dpi"]) == (150, 150)
        assert not (tmp_path / "out" / "thin_scale.png").exists()

    def test_keep_upscaled_scales_scale_image_and_dpi(self, tmp_path):
        from PIL import Image
        from lithic_editor.cli.main import process_image_cli
        image_path = tmp_path / "thin.png"
        Image.fromarray(self._thin_drawing()).save(image_path, dpi=(150, 150))
        scale = np.full((10, 40), 255, dtype=np.uint8)
        scale[4:6, 2:38] = 0
        scale_path = tmp_path / "bar.png"
        Image.fromarray(scale).save(scale_path, dpi=(150, 150))

        args = self._args(image_path, tmp_path / "out", keep_upscaled=True, scale_image=str(scale_path))
        assert process_image_cli(args) == 0
        with Image.open(tmp_path / "out" / "thin_cleaned.png") as out, \
                Image.open(tmp_path / "out" / "thin_scale.png") as out_scale:
            factor = out.size[0] // 60
            assert factor > 1
            assert out.size == (60 * factor, 60 * factor)
            assert out_scale.size == (40 * factor, 10 * factor)
            assert tuple(round(d) for d in out.info["dpi"]) == (150 * factor, 150 * factor)
            assert tuple(round(d) for d in out_scale.info["dpi"]) == (150 * factor, 150 * factor)


class TestParserOptions:
    def test_process_options_parse(self):
        parser = create_parser()
        args = parser.parse_args(["in.png", "--auto-upscale", "--keep-upscaled",
                                  "--scale-image", "bar.png", "--config", "c.yaml",
                                  "--upscale-model", "fsrcnn", "--default-dpi", "150"])
        assert args.auto_upscale and args.keep_upscaled
        assert args.scale_image == "bar.png" and args.config == "c.yaml"
        assert args.upscale_model == "fsrcnn" and args.default_dpi == 150

    def test_upscale_model_defaults_to_configuration(self):
        args = create_parser().parse_args(["in.png"])
        assert args.upscale_model is None and args.config is None

    def test_removed_option_is_rejected(self):
        import pytest
        with pytest.raises(SystemExit):
            create_parser().parse_args(["in.png", "--upscale-threshold", "300"])

    def test_config_path_reaches_the_pipeline(self, sample_image, tmp_path):
        from unittest.mock import patch
        args = mock_args()
        args.input_image = str(sample_image)
        args.output = str(tmp_path / "out")
        args.debug = False
        args.quiet = True
        args.no_preserve_cortex = False
        args.auto_upscale = False
        args.default_dpi = None
        args.upscale_model = None
        args.keep_upscaled = False
        args.scale_image = None
        args.config = "my.yaml"
        with patch('lithic_editor.cli.main.process_lithic_drawing') as mock_process:
            mock_process.return_value = np.full((20, 20), 255, dtype=np.uint8)
            assert process_image_cli(args) == 0
            assert mock_process.call_args[1]["config"] == "my.yaml"
            assert mock_process.call_args[1]["restore_original_size"] is True


class TestStrikingAnglesCLI:
    """The --angles option and its CSV file."""

    def test_angles_flag_parses(self):
        parser = create_parser()
        args = parser.parse_args(['in.png', '--angles'])
        assert args.angles is True
        assert parser.parse_args(['in.png']).angles is False

    def test_angles_are_written_as_csv(self, sample_image, tmp_path, capsys):
        import csv
        args = mock_args()
        args.input_image = str(sample_image)
        args.output = str(tmp_path / "out")
        args.debug = False
        args.quiet = False
        args.config = None
        args.angles = True
        record = {'scar': 1, 'x': 12.5, 'y': 30.0, 'compass': 45.0, 'ripples': 5,
                  'spread': 0.02, 'direction_certain': True, 'extent': 80.0}
        with patch('lithic_editor.cli.main.process_lithic_drawing') as mock_process:
            mock_process.return_value = {
                'processed_image': np.full((20, 20), 255, dtype=np.uint8),
                'scale_factor': 1, 'working_scale_factor': 1,
                'original_dpi': None, 'final_dpi': None,
                'striking_angles': [record],
            }
            assert process_image_cli(args) == 0
            assert mock_process.call_args[1]['measure_angles'] is True
        angles_file = tmp_path / "out" / f"{sample_image.stem}_angles.csv"
        with open(angles_file, newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert rows == [{'scar': '1', 'x': '12.5', 'y': '30.0', 'compass': '45.0', 'ripples': '5',
                         'spread': '0.02', 'direction_certain': 'True', 'extent': '80.0'}]
        out = capsys.readouterr().out
        assert "Scar 1: 45.0 degrees from 5 ripples" in out
        assert "Striking angles written to" in out

    def test_angle_options_parse(self):
        parser = create_parser()
        args = parser.parse_args(['in.png', '--angles', '--min-ripples', '3',
                                  '--min-arc-depth', '1.5', '--max-cortex-dots', '0'])
        assert (args.min_ripples, args.min_arc_depth, args.max_cortex_dots) == (3, 1.5, 0)
        args = parser.parse_args(['in.png'])
        assert args.min_ripples is None and args.min_arc_depth is None and args.max_cortex_dots is None

    def test_angle_options_replace_the_configuration_values(self):
        from lithic_editor.cli.main import config_with_angle_options
        from lithic_editor.config import Config
        args = MagicMock(config=None, min_ripples=3, min_arc_depth=None, max_cortex_dots=0)
        config = config_with_angle_options(args)
        assert isinstance(config, Config)
        assert config.angles.min_ripples == 3
        assert config.angles.max_cortex_dots == 0
        assert config.angles.min_arc_depth_px == 2.0
        assert "command line" in config.source

    def test_without_angle_options_the_config_path_passes_through(self):
        from lithic_editor.cli.main import config_with_angle_options
        args = MagicMock(config='some.yaml', min_ripples=None, min_arc_depth=None, max_cortex_dots=None)
        assert config_with_angle_options(args) == 'some.yaml'
        args.config = None
        assert config_with_angle_options(args) is None

    def test_invalid_angle_option_is_reported(self, sample_image, tmp_path, capsys):
        args = MagicMock(input_image=str(sample_image), output=str(tmp_path / "out"), debug=False,
                         quiet=False, config=None, angles=True,
                         min_ripples=0, min_arc_depth=None, max_cortex_dots=None)
        assert process_image_cli(args) == 1
        assert "angles.min_ripples must be at least 1" in capsys.readouterr().out

    def test_no_csv_without_the_flag(self, sample_image, tmp_path):
        args = mock_args()
        args.input_image = str(sample_image)
        args.output = str(tmp_path / "out")
        args.debug = False
        args.quiet = True
        args.config = None
        args.angles = False
        with patch('lithic_editor.cli.main.process_lithic_drawing') as mock_process:
            mock_process.return_value = np.full((20, 20), 255, dtype=np.uint8)
            assert process_image_cli(args) == 0
            assert mock_process.call_args[1]['measure_angles'] is False
        assert not list((tmp_path / "out").glob("*_angles.csv"))
