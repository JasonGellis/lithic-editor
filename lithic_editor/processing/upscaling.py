"""
ESPCN and FSRCNN upscaling module for lithic images.

This module provides neural network-based upscaling to improve line quality
in low-resolution lithic drawings before processing.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union


def detect_image_dpi(image_path: str) -> Optional[int]:
    """
    Detect DPI from image metadata.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        DPI value if found in metadata, None otherwise
    """
    try:
        with Image.open(image_path) as img:
            if hasattr(img, 'info') and 'dpi' in img.info:
                dpi_info = img.info['dpi']
                if isinstance(dpi_info, tuple):
                    # Return the higher of x,y DPI
                    return max(dpi_info[0], dpi_info[1])
                elif isinstance(dpi_info, (int, float)):
                    return int(dpi_info)
    except Exception as e:
        print(f"Warning: Could not read DPI from {image_path}: {e}")
    
    return None


def calculate_upscale_factor(current_dpi: int, target_dpi: int = 300) -> float:
    """
    Calculate the upscaling factor needed to reach target DPI.
    
    Args:
        current_dpi: Current image DPI
        target_dpi: Target DPI (default: 300)
        
    Returns:
        Upscaling factor (e.g., 2.0 for 2x upscaling)
    """
    return target_dpi / current_dpi


def get_model_path(model_name: str, scale_factor: int) -> str:
    """
    Get the path to the specified model file.
    
    Args:
        model_name: 'espcn' or 'fsrcnn'
        scale_factor: 2, 3, or 4
        
    Returns:
        Path to the model file
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_filename = f"{model_name.upper()}_x{scale_factor}.pb"
    return os.path.join(models_dir, model_filename)


def load_upscaling_model(model_name: str, scale_factor: int):
    """
    Load and initialize the upscaling model.
    
    Args:
        model_name: 'espcn' or 'fsrcnn'
        scale_factor: 2, 3, or 4
        
    Returns:
        Initialized DnnSuperRes model
    """
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = get_model_path(model_name, scale_factor)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        sr.readModel(model_path)
        sr.setModel(model_name.lower(), scale_factor)
        
        print(f"Loaded {model_name.upper()} {scale_factor}x model")
        return sr
        
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")
        return None


def upscale_with_model(image: np.ndarray, scale_factor: float, model: str = 'espcn') -> np.ndarray:
    """
    Upscale an image using the specified neural network model.
    
    Args:
        image: Input image as numpy array
        scale_factor: Upscaling factor (2.0, 3.0, or 4.0)
        model: Model to use ('espcn' or 'fsrcnn')
        
    Returns:
        Upscaled image as numpy array
    """
    # Round to nearest supported scale factor
    if scale_factor <= 2.5:
        model_scale = 2
    elif scale_factor <= 3.5:
        model_scale = 3
    else:
        model_scale = 4
    
    print(f"Upscaling with {model.upper()} {model_scale}x (target factor: {scale_factor:.1f})")
    
    # Load the model
    sr_model = load_upscaling_model(model, model_scale)
    
    if sr_model is None:
        print(f"Falling back to INTER_LANCZOS4 interpolation")
        return upscale_with_interpolation(image, scale_factor)
    
    try:
        # Convert to 3-channel if grayscale (required by models)
        if len(image.shape) == 2:
            input_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            input_image = image
        
        # Apply super-resolution
        upscaled = sr_model.upsample(input_image)
        
        # Convert back to grayscale if original was grayscale
        if len(image.shape) == 2:
            upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        
        # If model scale doesn't exactly match target, resize to exact target
        if model_scale != scale_factor:
            target_height = int(image.shape[0] * scale_factor)
            target_width = int(image.shape[1] * scale_factor)
            upscaled = cv2.resize(upscaled, (target_width, target_height), 
                                interpolation=cv2.INTER_LANCZOS4)
        
        print(f"Upscaling successful: {image.shape} → {upscaled.shape}")
        return upscaled
        
    except Exception as e:
        print(f"Error during upscaling with {model}: {e}")
        print("Falling back to INTER_LANCZOS4 interpolation")
        return upscale_with_interpolation(image, scale_factor)


def upscale_with_interpolation(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Fallback upscaling using traditional interpolation.
    
    Args:
        image: Input image as numpy array
        scale_factor: Upscaling factor
        
    Returns:
        Upscaled image using INTER_LANCZOS4
    """
    target_height = int(image.shape[0] * scale_factor)
    target_width = int(image.shape[1] * scale_factor)
    
    upscaled = cv2.resize(image, (target_width, target_height), 
                         interpolation=cv2.INTER_LANCZOS4)
    
    print(f"Interpolation upscaling: {image.shape} → {upscaled.shape}")
    return upscaled


def needs_upscaling(dpi: int, threshold: int = 300) -> bool:
    """
    Check if an image needs upscaling based on DPI threshold.
    
    Args:
        dpi: Current image DPI
        threshold: DPI threshold (default: 300)
        
    Returns:
        True if upscaling is needed
    """
    return dpi < threshold


def upscale_image_to_target_dpi(image: np.ndarray, current_dpi: int, 
                               target_dpi: int = 300, model: str = 'espcn') -> Tuple[np.ndarray, float]:
    """
    Upscale an image from current DPI to target DPI.
    
    Args:
        image: Input image as numpy array
        current_dpi: Current image DPI
        target_dpi: Target DPI (default: 300)
        model: Model to use ('espcn' or 'fsrcnn')
        
    Returns:
        Tuple of (upscaled_image, scale_factor)
    """
    if current_dpi >= target_dpi:
        print(f"Image DPI ({current_dpi}) already meets target ({target_dpi})")
        return image, 1.0
    
    scale_factor = calculate_upscale_factor(current_dpi, target_dpi)
    
    # Check maximum upscaling limit
    if scale_factor > 4.0:
        print(f"Warning: Scale factor {scale_factor:.1f}x exceeds 4x limit. Using 4x instead.")
        scale_factor = 4.0
        actual_target_dpi = int(current_dpi * scale_factor)
        print(f"Actual target DPI will be {actual_target_dpi} instead of {target_dpi}")
    
    upscaled_image = upscale_with_model(image, scale_factor, model)
    return upscaled_image, scale_factor


def validate_upscaling_inputs(current_dpi: int, target_dpi: int = 300, 
                            model: str = 'espcn') -> Tuple[bool, str]:
    """
    Validate upscaling parameters.
    
    Args:
        current_dpi: Current image DPI
        target_dpi: Target DPI
        model: Model name
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if current_dpi < 75:
        return False, f"Input DPI ({current_dpi}) too low. Minimum: 75 DPI"
    
    if target_dpi < current_dpi:
        return False, f"Target DPI ({target_dpi}) lower than current DPI ({current_dpi})"
    
    if model not in ['espcn', 'fsrcnn']:
        return False, f"Invalid model: {model}. Must be 'espcn' or 'fsrcnn'"
    
    scale_factor = target_dpi / current_dpi
    if scale_factor > 4.0:
        return False, f"Scale factor {scale_factor:.1f}x exceeds 4x maximum"
    
    return True, ""


# Model cache for batch processing
_model_cache = {}

def get_cached_model(model_name: str, scale_factor: int):
    """Get a cached model or load and cache it."""
    cache_key = f"{model_name}_{scale_factor}"
    
    if cache_key not in _model_cache:
        _model_cache[cache_key] = load_upscaling_model(model_name, scale_factor)
    
    return _model_cache[cache_key]