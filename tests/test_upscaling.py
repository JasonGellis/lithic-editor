"""
Tests for the upscaling module.
"""

import pytest
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from unittest.mock import patch, MagicMock

from lithic_editor.processing.upscaling import (
    detect_image_dpi,
    calculate_upscale_factor,
    get_model_path,
    load_upscaling_model,
    upscale_with_model,
    upscale_with_interpolation,
    needs_upscaling,
    upscale_image_to_target_dpi,
    validate_upscaling_inputs,
    get_cached_model
)


class TestDPIDetection:
    """Test DPI detection functionality."""
    
    def test_detect_dpi_with_metadata(self, temp_dir):
        """Test DPI detection from image metadata."""
        # Create image with DPI metadata
        img = Image.new('L', (100, 100), 255)
        img.info['dpi'] = (300, 300)
        image_path = temp_dir / "test_dpi.png"
        img.save(image_path, dpi=(300, 300))
        
        dpi = detect_image_dpi(str(image_path))
        assert abs(dpi - 300) < 1  # Allow for floating point precision
    
    def test_detect_dpi_tuple_metadata(self, temp_dir):
        """Test DPI detection with tuple metadata."""
        img = Image.new('L', (100, 100), 255)
        img.info['dpi'] = (300, 150)  # Different x,y DPI
        image_path = temp_dir / "test_dpi_tuple.png"
        img.save(image_path, dpi=(300, 150))
        
        dpi = detect_image_dpi(str(image_path))
        assert dpi is not None and abs(dpi - 300) < 1  # Should return max
    
    def test_detect_dpi_no_metadata(self, temp_dir):
        """Test DPI detection without metadata."""
        img = Image.new('L', (100, 100), 255)
        image_path = temp_dir / "test_no_dpi.png"
        img.save(image_path)
        
        dpi = detect_image_dpi(str(image_path))
        assert dpi is None
    
    def test_detect_dpi_invalid_file(self):
        """Test DPI detection with invalid file."""
        dpi = detect_image_dpi("nonexistent.png")
        assert dpi is None


class TestUpscaleCalculations:
    """Test upscaling calculation functions."""
    
    def test_calculate_upscale_factor(self):
        """Test upscale factor calculation."""
        assert calculate_upscale_factor(150, 300) == 2.0
        assert calculate_upscale_factor(100, 300) == 3.0
        assert calculate_upscale_factor(75, 300) == 4.0
        assert calculate_upscale_factor(300, 300) == 1.0
    
    def test_needs_upscaling(self):
        """Test upscaling threshold check."""
        assert needs_upscaling(150, 300) == True
        assert needs_upscaling(300, 300) == False
        assert needs_upscaling(400, 300) == False
        assert needs_upscaling(200, 250) == True


class TestModelManagement:
    """Test model file management."""
    
    def test_get_model_path(self):
        """Test model path generation."""
        path = get_model_path('espcn', 2)
        assert 'ESPCN_x2.pb' in path
        
        path = get_model_path('fsrcnn', 4)
        assert 'FSRCNN_x4.pb' in path
    
    @patch('os.path.exists')
    @patch('cv2.dnn_superres.DnnSuperResImpl_create')
    def test_load_upscaling_model_success(self, mock_create, mock_exists):
        """Test successful model loading."""
        mock_exists.return_value = True
        mock_sr = MagicMock()
        mock_create.return_value = mock_sr
        
        result = load_upscaling_model('espcn', 2)
        
        assert result == mock_sr
        mock_sr.readModel.assert_called_once()
        mock_sr.setModel.assert_called_once_with('espcn', 2)
    
    @patch('os.path.exists')
    def test_load_upscaling_model_file_not_found(self, mock_exists):
        """Test model loading with missing file."""
        mock_exists.return_value = False
        
        result = load_upscaling_model('espcn', 2)
        
        assert result is None
    
    def test_get_cached_model(self):
        """Test model caching functionality."""
        with patch('lithic_editor.processing.upscaling.load_upscaling_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            # First call should load
            result1 = get_cached_model('espcn', 2)
            assert result1 == mock_model
            mock_load.assert_called_once()
            
            # Second call should use cache
            mock_load.reset_mock()
            result2 = get_cached_model('espcn', 2)
            assert result2 == mock_model
            mock_load.assert_not_called()


class TestUpscalingFunctions:
    """Test upscaling functions."""
    
    def test_upscale_with_interpolation(self):
        """Test interpolation-based upscaling."""
        image = np.zeros((50, 50), dtype=np.uint8)
        image[20:30, 20:30] = 255
        
        result = upscale_with_interpolation(image, 2.0)
        
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
    
    @patch('lithic_editor.processing.upscaling.load_upscaling_model')
    def test_upscale_with_model_success(self, mock_load):
        """Test neural network upscaling."""
        # Mock model
        mock_model = MagicMock()
        mock_upscaled = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_model.upsample.return_value = mock_upscaled
        mock_load.return_value = mock_model
        
        image = np.zeros((50, 50), dtype=np.uint8)
        result = upscale_with_model(image, 2.0, 'espcn')
        
        assert result is not None
        mock_model.upsample.assert_called_once()
    
    @patch('lithic_editor.processing.upscaling.load_upscaling_model')
    def test_upscale_with_model_fallback(self, mock_load):
        """Test fallback to interpolation when model fails."""
        mock_load.return_value = None
        
        image = np.zeros((50, 50), dtype=np.uint8)
        result = upscale_with_model(image, 2.0, 'espcn')
        
        assert result.shape == (100, 100)  # Should fallback to interpolation
    
    def test_upscale_image_to_target_dpi_no_upscaling_needed(self):
        """Test when image already meets target DPI."""
        image = np.zeros((100, 100), dtype=np.uint8)
        
        result_img, scale_factor = upscale_image_to_target_dpi(image, 300, 300)
        
        assert np.array_equal(result_img, image)
        assert scale_factor == 1.0
    
    @patch('lithic_editor.processing.upscaling.upscale_with_model')
    def test_upscale_image_to_target_dpi_upscaling_needed(self, mock_upscale):
        """Test upscaling when DPI is below target."""
        image = np.zeros((50, 50), dtype=np.uint8)
        mock_upscaled = np.zeros((100, 100), dtype=np.uint8)
        mock_upscale.return_value = mock_upscaled
        
        result_img, scale_factor = upscale_image_to_target_dpi(image, 150, 300, 'espcn')
        
        assert scale_factor == 2.0
        mock_upscale.assert_called_once_with(image, 2.0, 'espcn')


class TestUpscalingValidation:
    """Test upscaling parameter validation."""
    
    def test_validate_upscaling_inputs_valid(self):
        """Test validation with valid inputs."""
        is_valid, error = validate_upscaling_inputs(150, 300, 'espcn')
        assert is_valid == True
        assert error == ""
    
    def test_validate_upscaling_inputs_dpi_too_low(self):
        """Test validation with DPI too low."""
        is_valid, error = validate_upscaling_inputs(50, 300, 'espcn')
        assert is_valid == False
        assert "too low" in error
    
    def test_validate_upscaling_inputs_target_lower_than_current(self):
        """Test validation with target DPI lower than current."""
        is_valid, error = validate_upscaling_inputs(300, 150, 'espcn')
        assert is_valid == False
        assert "lower than current" in error
    
    def test_validate_upscaling_inputs_invalid_model(self):
        """Test validation with invalid model."""
        is_valid, error = validate_upscaling_inputs(150, 300, 'invalid')
        assert is_valid == False
        assert "Invalid model" in error
    
    def test_validate_upscaling_inputs_scale_factor_too_high(self):
        """Test validation with scale factor exceeding 4x."""
        is_valid, error = validate_upscaling_inputs(50, 300, 'espcn')
        assert is_valid == False
        assert "too low" in error or "exceeds 4x" in error  # Either error is valid


class TestModelScaling:
    """Test model scale factor selection."""
    
    @pytest.mark.parametrize("scale_factor,expected_model_scale", [
        (1.5, 2),
        (2.0, 2),
        (2.5, 2),
        (2.8, 3),
        (3.0, 3),
        (3.5, 3),
        (3.8, 4),
        (4.0, 4),
        (5.0, 4)  # Capped at 4
    ])
    def test_model_scale_selection(self, scale_factor, expected_model_scale):
        """Test that correct model scale is selected for different scale factors."""
        with patch('lithic_editor.processing.upscaling.load_upscaling_model') as mock_load:
            mock_load.return_value = None  # Will fallback to interpolation
            
            image = np.zeros((50, 50), dtype=np.uint8)
            upscale_with_model(image, scale_factor, 'espcn')
            
            mock_load.assert_called_once_with('espcn', expected_model_scale)