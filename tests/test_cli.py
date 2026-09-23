"""
Tests for the command-line interface.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from lithic_editor.cli.main import (
    create_parser, 
    validate_input_file,
    process_image_cli,
    show_help_cli,
    open_docs
)


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
    
    def test_gui_command(self):
        """Test GUI command parsing."""
        parser = create_parser()
        args = parser.parse_args(['gui'])
        
        assert args.command == 'gui'
    
    def test_gui_flag(self):
        """Test --gui flag."""
        parser = create_parser()
        args = parser.parse_args(['--gui'])
        
        assert args.gui == True
    
    def test_process_command(self):
        """Test process command parsing."""
        parser = create_parser()
        args = parser.parse_args(['process', 'image.png'])
        
        assert args.command == 'process'
        assert args.input_image == 'image.png'
        assert args.output == 'image_debug'  # default
        assert args.debug == False  # default
        assert args.quiet == False  # default
    
    def test_process_with_options(self):
        """Test process command with all options."""
        parser = create_parser()
        args = parser.parse_args([
            'process', 'image.png',
            '--output', 'results/',
            '--debug',
            '--quiet'
        ])
        
        assert args.input_image == 'image.png'
        assert args.output == 'results/'
        assert args.debug == True
        assert args.quiet == True
    
    def test_process_with_cortex_preservation(self):
        """Test process command with cortex preservation flag."""
        parser = create_parser()
        args = parser.parse_args([
            'process', 'image.png',
            '--no-preserve-cortex'
        ])
        
        assert args.input_image == 'image.png'
        assert args.no_preserve_cortex == True
    
    def test_process_cortex_default(self):
        """Test that cortex preservation is enabled by default."""
        parser = create_parser()
        args = parser.parse_args(['process', 'image.png'])
        
        # no_preserve_cortex should be False by default (cortex preserved)
        assert args.no_preserve_cortex == False
    
    def test_help_command(self):
        """Test help command parsing."""
        parser = create_parser()
        args = parser.parse_args(['help'])
        
        assert args.command == 'help'
        assert args.topic is None
    
    def test_help_api_topic(self):
        """Test help command with API topic."""
        parser = create_parser()
        args = parser.parse_args(['help', 'api'])
        
        assert args.command == 'help'
        assert args.topic == 'api'
    
    def test_docs_command(self):
        """Test docs command parsing."""
        parser = create_parser()
        args = parser.parse_args(['docs'])
        
        assert args.command == 'docs'
        assert args.offline == False  # default
    
    def test_docs_offline(self):
        """Test docs command with offline flag."""
        parser = create_parser()
        args = parser.parse_args(['docs', '--offline'])
        
        assert args.command == 'docs'
        assert args.offline == True


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
        args = MagicMock()
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
        args = MagicMock()
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
        args = MagicMock()
        args.input_image = "nonexistent.png"
        args.output = str(tmp_path / "out")
        args.debug = False
        args.quiet = False
        
        result = process_image_cli(args)
        assert result == 1
    
    def test_process_with_cortex_parameter(self, sample_image, tmp_path):
        """Test processing with cortex preservation parameter."""
        args = MagicMock()
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


class TestHelpCLI:
    """Test help command functionality."""
    
    def test_show_help_default(self):
        """Test default help display."""
        args = MagicMock()
        args.topic = None
        
        with patch('lithic_editor.cli.main.show_help') as mock_help:
            show_help_cli(args)
            mock_help.assert_called_once()
    
    def test_show_api_help(self):
        """Test API help display."""
        args = MagicMock()
        args.topic = 'api'
        
        with patch('lithic_editor.cli.main.show_api_help') as mock_api_help:
            show_help_cli(args)
            mock_api_help.assert_called_once()


class TestDocsCLI:
    """Test docs command functionality."""
    
    def test_open_docs_online(self):
        """Test opening online documentation."""
        args = MagicMock()
        args.offline = False
        
        with patch('webbrowser.open') as mock_open:
            result = open_docs(args)
            
            assert result == 0
            mock_open.assert_called_once_with("https://jasongellis.github.io/lithic-editor/")
    
    def test_open_docs_offline(self):
        """Test opening offline documentation."""
        args = MagicMock()
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
        from unittest.mock import MagicMock
        args = MagicMock()
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
        args = parser.parse_args(["process", "in.png", "--auto-upscale", "--keep-upscaled",
                                  "--scale-image", "bar.png", "--config", "c.yaml",
                                  "--upscale-model", "fsrcnn", "--default-dpi", "150"])
        assert args.auto_upscale and args.keep_upscaled
        assert args.scale_image == "bar.png" and args.config == "c.yaml"
        assert args.upscale_model == "fsrcnn" and args.default_dpi == 150

    def test_upscale_model_defaults_to_configuration(self):
        args = create_parser().parse_args(["process", "in.png"])
        assert args.upscale_model is None and args.config is None

    def test_removed_option_is_rejected(self):
        import pytest
        with pytest.raises(SystemExit):
            create_parser().parse_args(["process", "in.png", "--upscale-threshold", "300"])

    def test_config_path_reaches_the_pipeline(self, sample_image, tmp_path):
        from unittest.mock import MagicMock, patch
        args = MagicMock()
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
