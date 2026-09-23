"""
Integration tests for combined features.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image


class TestUpscalingIntegration:
    """Test integration of adaptive upscaling with main processing."""

    @staticmethod
    def _thin_hatched_drawing(size=60):
        """One-pixel strokes with two-pixel gaps: too thin and too close to skeletonize well."""
        img = np.full((size, size), 255, dtype=np.uint8)
        img[5, 5:55] = img[54, 5:55] = 0
        img[5:55, 5] = img[5:55, 54] = 0
        for row in range(12, 48, 3):
            img[row, 10:40] = 0
        return img

    def test_thin_drawing_is_upscaled_and_restored(self, temp_dir):
        """A thin, tightly hatched drawing is upscaled for processing and returned at input size."""
        from lithic_editor.processing import process_lithic_drawing

        img = Image.fromarray(self._thin_hatched_drawing())
        image_path = temp_dir / "thin_test.png"
        img.save(image_path, dpi=(150, 150))

        result = process_lithic_drawing(
            image_path=str(image_path),
            output_folder=str(temp_dir),
            upscale_model='espcn',
            upscale_low_dpi=True,
            save_debug=True
        )

        assert result is not None
        assert result.shape == (60, 60)
        debug_names = [f.stem for f in temp_dir.glob("*.png")]
        assert any("upscaled" in name.lower() for name in debug_names)
        assert any("restored" in name.lower() for name in debug_names)

    def test_thin_drawing_can_keep_upscaled_size(self, temp_dir):
        """With restore_original_size=False the result is larger and the factor is reported."""
        from lithic_editor.processing import process_lithic_drawing

        img = Image.fromarray(self._thin_hatched_drawing())
        image_path = temp_dir / "thin_keep.png"
        img.save(image_path, dpi=(150, 150))

        result = process_lithic_drawing(
            image_path=str(image_path),
            output_folder=str(temp_dir),
            upscale_low_dpi=True,
            restore_original_size=False,
            return_scale_factor=True,
        )

        factor = result['scale_factor']
        assert factor > 1
        assert result['working_scale_factor'] == factor
        assert result['processed_image'].shape == (60 * factor, 60 * factor)
        assert result['final_dpi'] == 150 * factor

    def test_thick_drawing_is_not_upscaled(self, temp_dir):
        """Wide, well-separated strokes need no upscaling even when upscaling is allowed."""
        from lithic_editor.processing import process_lithic_drawing

        img = np.full((120, 120), 255, dtype=np.uint8)
        img[10:22, 10:110] = img[98:110, 10:110] = 0
        img[10:110, 10:22] = img[10:110, 98:110] = 0
        image_path = temp_dir / "thick_test.png"
        Image.fromarray(img).save(image_path, dpi=(600, 600))

        result = process_lithic_drawing(
            image_path=str(image_path),
            output_folder=str(temp_dir),
            upscale_low_dpi=True,
            return_scale_factor=True,
            save_debug=True
        )

        assert result['working_scale_factor'] == 1
        assert result['processed_image'].shape == (120, 120)
        debug_names = [f.stem for f in temp_dir.glob("*.png")]
        assert not any("upscaled" in name.lower() for name in debug_names)


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
            preserve_cortex=True,
            upscale_model='espcn',
            upscale_low_dpi=True,
            save_debug=True
        )
        
        assert result is not None
        # Should have both upscaling and cortex preservation
        assert result.shape == (50, 50)  # Returned on the input grid
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