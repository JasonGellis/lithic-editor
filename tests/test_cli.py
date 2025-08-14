"""
Tests for the command-line interface.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from lithic_editor.cli.main import (
    create_parser, 
    validate_input_file,
    process_image_cli,
    show_help_cli,
    open_docs
)
from lithic_editor import __version__


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
    
    def test_process_success(self, sample_image):
        """Test successful image processing."""
        args = MagicMock()
        args.input_image = str(sample_image)
        args.output = "test_output"
        args.debug = False
        args.quiet = False
        
        with patch('lithic_editor.cli.main.process_lithic_drawing_improved') as mock_process:
            mock_process.return_value = MagicMock()
            
            result = process_image_cli(args)
            
            assert result == 0
            mock_process.assert_called_once()
    
    def test_process_quiet_mode(self, sample_image, capsys):
        """Test quiet mode suppresses output."""
        args = MagicMock()
        args.input_image = str(sample_image)
        args.output = "test_output"
        args.debug = False
        args.quiet = True
        
        with patch('lithic_editor.cli.main.process_lithic_drawing_improved') as mock_process:
            mock_process.return_value = MagicMock()
            
            result = process_image_cli(args)
            
            # Check no output was printed
            captured = capsys.readouterr()
            assert captured.out == ""
    
    def test_process_file_not_found(self):
        """Test handling of missing file."""
        args = MagicMock()
        args.input_image = "nonexistent.png"
        args.output = "test_output"
        args.debug = False
        args.quiet = False
        
        result = process_image_cli(args)
        assert result == 1


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