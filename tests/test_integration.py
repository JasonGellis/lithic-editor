"""
Integration tests for combined features.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image


class TestUpscalingIntegration:
    """Test integration of upscaling with main processing."""
    
    def test_low_dpi_image_processing_with_upscaling(self, temp_dir):
        """Test processing low DPI image with automatic upscaling."""
        from lithic_editor.processing import process_lithic_drawing
        
        # Create low DPI test image with DPI metadata
        img = Image.new('L', (50, 50), 255)
        # Add some content
        img_array = np.array(img)
        img_array[20:30, 20:30] = 0  # Black square
        img = Image.fromarray(img_array)
        
        image_path = temp_dir / "low_dpi_test.png"
        img.save(image_path, dpi=(150, 150))  # Low DPI
        
        result = process_lithic_drawing(
            image_path=str(image_path),
            output_folder=str(temp_dir),
            target_dpi=300,
            upscale_model='espcn',
            upscale_low_dpi=True,
            save_debug=True
        )
        
        assert result is not None
        # Should be upscaled
        assert result.shape[0] >= 50
        assert result.shape[1] >= 50
        
        # Check upscaling debug images were created
        debug_files = list(temp_dir.glob("*.png"))
        debug_names = [f.stem for f in debug_files]
        assert any("upscaled" in name.lower() for name in debug_names)
    
    def test_high_dpi_image_no_upscaling(self, temp_dir):
        """Test that high DPI images skip upscaling."""
        from lithic_editor.processing import process_lithic_drawing
        
        # Create high DPI test image
        img = Image.new('L', (100, 100), 255)
        image_path = temp_dir / "high_dpi_test.png"
        img.save(image_path, dpi=(600, 600))  # High DPI
        
        result = process_lithic_drawing(
            image_path=str(image_path),
            output_folder=str(temp_dir),
            target_dpi=300,
            save_debug=True
        )
        
        assert result is not None
        # Should maintain original size (no upscaling)
        assert result.shape == (100, 100)


class TestCortexUpscalingIntegration:
    """Test cortex preservation with upscaling."""
    
    def test_cortex_preservation_with_upscaling(self, temp_dir):
        """Test that cortex is preserved even after upscaling."""
        from lithic_editor.processing import process_lithic_drawing
        
        # Create low DPI image with cortex stippling
        cortex_img = np.zeros((50, 50), dtype=np.uint8)
        # Add small cortex dots
        for i in range(10, 40, 8):
            for j in range(10, 40, 8):
                cortex_img[i:i+2, j:j+2] = 255
        
        # Add structural line
        cortex_img[25:28, 5:45] = 255
        
        # Save as low DPI image
        img = Image.fromarray(cortex_img)
        image_path = temp_dir / "cortex_low_dpi.png"
        img.save(image_path, dpi=(150, 150))
        
        result = process_lithic_drawing(
            image_path=str(image_path),
            output_folder=str(temp_dir),
            target_dpi=300,
            preserve_cortex=True,
            upscale_model='espcn',
            upscale_low_dpi=True,
            save_debug=True
        )
        
        assert result is not None
        # Should have both upscaling and cortex preservation
        assert result.shape[0] >= 50  # Upscaled
        assert np.sum(result > 0) > 0  # Cortex preserved
        
        # Check both upscaling and cortex debug images
        debug_files = list(temp_dir.glob("*.png"))
        debug_names = [f.stem for f in debug_files]
        assert any("upscaled" in name.lower() for name in debug_names)
        assert any("cortex" in name.lower() for name in debug_names)


class TestFullWorkflowIntegration:
    """Test complete workflow integration."""
    
    def test_cli_to_processing_integration(self, sample_image, temp_dir):
        """Test CLI command integration with processing."""
        from lithic_editor.cli.main import process_image_cli
        
        args = MagicMock()
        args.input_image = str(sample_image)
        args.output = str(temp_dir)
        args.debug = True
        args.quiet = False
        args.no_preserve_cortex = False
        
        with patch('lithic_editor.cli.main.process_lithic_drawing') as mock_process:
            mock_process.return_value = np.zeros((100, 100), dtype=np.uint8)
            
            result = process_image_cli(args)
            
            assert result == 0
            # Should call with correct parameters
            mock_process.assert_called_once()
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs['preserve_cortex'] == True
            assert call_kwargs['save_debug'] == True
    
    @patch('lithic_editor.gui.main_window.process_lithic_drawing')
    @patch('os.path.exists')
    def test_gui_to_processing_integration(self, mock_exists, mock_process, qapp, sample_image):
        """Test GUI processing integration."""
        from lithic_editor.gui.main_window import ProcessingThread
        
        mock_process.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_exists.return_value = True  # Mock file existence
        
        thread = ProcessingThread(
            input_path=str(sample_image),  # Use real test image path
            output_folder="output",
            save_debug=True,
            preserve_cortex=True
        )
        
        thread.run()
        
        # Should call processing with correct parameters
        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args[1]
        assert call_kwargs['preserve_cortex'] == True
        assert call_kwargs['save_debug'] == True


class TestDebugImageIntegration:
    """Test debug image generation across features."""
    
    def test_all_debug_images_generated(self, temp_dir):
        """Test that all expected debug images are generated."""
        from lithic_editor.processing import process_lithic_drawing
        
        # Create complex test image
        test_img = np.zeros((100, 100), dtype=np.uint8)
        # Add cortex stippling
        for i in range(10, 30, 5):
            for j in range(10, 30, 5):
                test_img[i:i+2, j:j+2] = 255
        
        # Add structural lines
        test_img[50:55, 10:90] = 255
        test_img[60:90, 20:25] = 255
        
        # Save as low DPI (trigger upscaling)
        img = Image.fromarray(test_img)
        image_path = temp_dir / "complex_test.png"
        img.save(image_path, dpi=(150, 150))
        
        result = process_lithic_drawing(
            image_path=str(image_path),
            output_folder=str(temp_dir),
            target_dpi=300,
            preserve_cortex=True,
            upscale_model='espcn',
            upscale_low_dpi=True,
            save_debug=True
        )
        
        debug_files = list(temp_dir.glob("*.png"))
        debug_names = [f.stem for f in debug_files]
        
        # Check for all expected debug stages
        expected_stages = [
            "original",
            "upscaled",
            "cortex",
            "structural",  # Updated to match actual debug image name 
            "skeleton",
            "endpoints"
        ]
        
        for stage in expected_stages:
            assert any(stage in name.lower() for name in debug_names), f"Missing {stage} debug image"
    
    def test_debug_image_naming_consistency(self, temp_dir):
        """Test that debug images follow consistent naming pattern."""
        from lithic_editor.processing import process_lithic_drawing
        
        # Create simple test image
        test_img = np.zeros((50, 50), dtype=np.uint8)
        test_img[20:30, 20:30] = 255
        
        result = process_lithic_drawing(
            image_path=test_img,
            output_folder=str(temp_dir),
            preserve_cortex=True,
            save_debug=True
        )
        
        debug_files = list(temp_dir.glob("*.png"))
        
        # All debug files should follow naming convention
        for debug_file in debug_files:
            name = debug_file.stem
            # Should start with number and underscore
            assert name[0].isdigit(), f"Debug file {name} doesn't start with number"
            assert '_' in name, f"Debug file {name} missing underscore separator"


class TestParameterCombinations:
    """Test various parameter combinations."""
    
    @pytest.mark.parametrize("preserve_cortex,save_debug", [
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    ])
    def test_cortex_debug_combinations(self, preserve_cortex, save_debug, temp_dir):
        """Test all combinations of cortex and debug parameters."""
        from lithic_editor.processing import process_lithic_drawing
        
        # Create test image with cortex
        cortex_img = np.zeros((50, 50), dtype=np.uint8)
        cortex_img[10:15, 10:15] = 255
        cortex_img[30:35, 30:35] = 255
        
        result = process_lithic_drawing(
            image_path=cortex_img,
            output_folder=str(temp_dir),
            preserve_cortex=preserve_cortex,
            save_debug=save_debug
        )
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        
        debug_files = list(temp_dir.glob("*.png"))
        if save_debug:
            assert len(debug_files) > 0
        else:
            # Only original files, no debug images
            assert len([f for f in debug_files if "debug" in f.name.lower()]) == 0
    
    def test_upscaling_model_fallback(self, temp_dir):
        """Test upscaling with model fallback to interpolation."""
        from lithic_editor.processing import process_lithic_drawing
        
        # Create low DPI image
        img = Image.new('L', (25, 25), 255)
        image_path = temp_dir / "fallback_test.png"
        img.save(image_path, dpi=(75, 75))  # Very low DPI
        
        # Use non-existent model to trigger fallback
        with patch('lithic_editor.processing.upscaling.load_upscaling_model') as mock_load:
            mock_load.return_value = None  # Model loading fails
            
            result = process_lithic_drawing(
                image_path=str(image_path),
                output_folder=str(temp_dir),
                target_dpi=300,
                upscale_model='nonexistent',
                upscale_low_dpi=True,
                save_debug=False
            )
            
            assert result is not None
            # Should still be upscaled via interpolation
            assert result.shape[0] >= 25
            assert result.shape[1] >= 25


class TestErrorHandlingIntegration:
    """Test error handling across integrated features."""
    
    def test_invalid_upscaling_model_with_cortex(self, temp_dir):
        """Test error handling with invalid model and cortex preservation."""
        from lithic_editor.processing import process_lithic_drawing
        
        # Create test image
        test_img = np.zeros((50, 50), dtype=np.uint8)
        test_img[20:25, 20:25] = 255
        
        # Should handle gracefully even with invalid model
        result = process_lithic_drawing(
            image_path=test_img,
            output_folder=str(temp_dir),
            preserve_cortex=True,
            upscale_model='invalid_model',
            upscale_low_dpi=True,
            save_debug=False
        )
        
        assert result is not None
        # Should fall back to interpolation and still work
    
    def test_missing_output_directory_creation(self, temp_dir):
        """Test that missing output directories are created."""
        from lithic_editor.processing import process_lithic_drawing
        
        # Create test image
        test_img = np.zeros((30, 30), dtype=np.uint8)
        test_img[10:20, 10:20] = 255
        
        # Use non-existent output directory
        output_dir = temp_dir / "nested" / "output" / "dir"
        
        result = process_lithic_drawing(
            image_path=test_img,
            output_folder=str(output_dir),
            save_debug=True
        )
        
        assert result is not None
        assert output_dir.exists()
        
        # Debug files should be created
        debug_files = list(output_dir.glob("*.png"))
        assert len(debug_files) > 0