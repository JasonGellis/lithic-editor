"""
Tests for the image processing module.
"""

import pytest
import numpy as np
from pathlib import Path
from lithic_editor.processing import process_lithic_drawing_improved


class TestProcessingModule:
    """Test suite for ripple removal processing."""
    
    def test_process_image_from_file(self, sample_image, temp_dir):
        """Test processing an image from file path."""
        result = process_lithic_drawing_improved(
            image_path=str(sample_image),
            output_folder=str(temp_dir),
            save_debug=False
        )
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100)  # Should maintain dimensions
    
    def test_process_image_from_numpy_array(self, sample_numpy_array, temp_dir):
        """Test processing a numpy array directly."""
        result = process_lithic_drawing_improved(
            image_path=sample_numpy_array,
            output_folder=str(temp_dir),
            save_debug=False
        )
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_numpy_array.shape
    
    def test_process_with_debug_output(self, sample_image, temp_dir):
        """Test that debug mode saves intermediate images."""
        result = process_lithic_drawing_improved(
            image_path=str(sample_image),
            output_folder=str(temp_dir),
            save_debug=True
        )
        
        # Check that debug images were created
        debug_files = list(temp_dir.glob("*.png"))
        assert len(debug_files) > 0
        
        # Check for expected debug stages
        expected_stages = [
            "01_original",
            "02_inverted",
            "03_skeleton"
        ]
        
        debug_names = [f.stem for f in debug_files]
        for stage in expected_stages:
            assert any(stage in name for name in debug_names)
    
    def test_process_with_dpi_preservation(self, sample_image_with_dpi, temp_dir):
        """Test that DPI information is preserved."""
        result = process_lithic_drawing_improved(
            image_path=str(sample_image_with_dpi),
            output_folder=str(temp_dir),
            dpi_info=(300, 300),
            save_debug=False
        )
        
        assert result is not None
        # Result should be numpy array
        assert isinstance(result, np.ndarray)
    
    def test_invalid_image_path(self, temp_dir):
        """Test handling of invalid image path."""
        with pytest.raises(Exception):
            process_lithic_drawing_improved(
                image_path="nonexistent.png",
                output_folder=str(temp_dir),
                save_debug=False
            )
    
    def test_output_folder_creation(self, sample_image, temp_dir):
        """Test that output folder is created if it doesn't exist."""
        output_dir = temp_dir / "new_output_dir"
        
        result = process_lithic_drawing_improved(
            image_path=str(sample_image),
            output_folder=str(output_dir),
            save_debug=True
        )
        
        assert output_dir.exists()
        assert result is not None
    
    def test_process_without_debug(self, sample_image, temp_dir):
        """Test processing without saving debug images."""
        result = process_lithic_drawing_improved(
            image_path=str(sample_image),
            output_folder=str(temp_dir),
            save_debug=False
        )
        
        # Should not create debug images
        debug_files = list(temp_dir.glob("*.png"))
        # Only the original test image should be there
        assert len([f for f in debug_files if "test_image" not in f.name]) == 0
        assert result is not None
    
    def test_empty_image_handling(self, temp_dir):
        """Test processing of an empty/white image."""
        # Create completely white image
        white_img = np.ones((100, 100), dtype=np.uint8) * 255
        
        result = process_lithic_drawing_improved(
            image_path=white_img,
            output_folder=str(temp_dir),
            save_debug=False
        )
        
        assert result is not None
        # Should return an image even if no lines detected
        assert isinstance(result, np.ndarray)
    
    def test_binary_image_processing(self, temp_dir):
        """Test processing of already binary image."""
        # Create a simple binary pattern
        binary_img = np.zeros((50, 50), dtype=np.uint8)
        binary_img[10:40, 20:30] = 255
        
        result = process_lithic_drawing_improved(
            image_path=binary_img,
            output_folder=str(temp_dir),
            save_debug=False
        )
        
        assert result is not None
        assert result.shape == binary_img.shape


class TestImageFormats:
    """Test different image format support."""
    
    @pytest.mark.parametrize("format_ext", [".png", ".jpg", ".tiff", ".bmp"])
    def test_format_support(self, temp_dir, format_ext):
        """Test processing different image formats."""
        from PIL import Image
        
        # Create test image in specified format
        img = Image.new('L', (50, 50), 255)
        image_path = temp_dir / f"test{format_ext}"
        img.save(image_path)
        
        result = process_lithic_drawing_improved(
            image_path=str(image_path),
            output_folder=str(temp_dir),
            save_debug=False
        )
        
        assert result is not None
        assert isinstance(result, np.ndarray)