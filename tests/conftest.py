"""
Pytest configuration and shared fixtures for all tests.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

# Ensure QApplication exists for GUI tests
@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # Don't quit the app as it might be reused


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image."""
    # Create a simple black and white image with lines
    img = Image.new('L', (100, 100), 255)  # White background
    pixels = img.load()
    
    # Draw some horizontal lines (ripples)
    for y in range(20, 80, 5):
        for x in range(20, 80):
            pixels[x, y] = 0  # Black
    
    # Draw a vertical line (structure)
    for y in range(10, 90):
        pixels[50, y] = 0  # Black
    
    # Save to temp directory
    image_path = temp_dir / "test_image.png"
    img.save(image_path)
    
    return image_path


@pytest.fixture
def sample_image_with_dpi(temp_dir):
    """Create a sample test image with DPI information."""
    img = Image.new('L', (300, 300), 255)
    pixels = img.load()
    
    # Draw pattern
    for y in range(50, 250, 10):
        for x in range(50, 250):
            pixels[x, y] = 0
    
    # Save with DPI
    image_path = temp_dir / "test_image_dpi.png"
    img.save(image_path, dpi=(300, 300))
    
    return image_path


@pytest.fixture
def sample_numpy_array():
    """Create a sample numpy array representing an image."""
    # Create 100x100 binary image
    img = np.ones((100, 100), dtype=np.uint8) * 255
    
    # Add horizontal lines
    for y in range(20, 80, 5):
        img[y, 20:80] = 0
    
    # Add vertical line
    img[10:90, 50] = 0
    
    return img


@pytest.fixture
def mock_processed_image():
    """Create a mock processed image result."""
    return {
        'processed_image': np.ones((100, 100), dtype=np.uint8) * 255,
        'dpi_info': (300, 300),
        'format_info': 'PNG',
        'debug_stages': [
            ('original', np.ones((100, 100), dtype=np.uint8) * 255),
            ('skeleton', np.zeros((100, 100), dtype=np.uint8)),
            ('final', np.ones((100, 100), dtype=np.uint8) * 255)
        ]
    }


@pytest.fixture
def sample_pixmap():
    """Create a sample QPixmap for GUI testing."""
    pixmap = QPixmap(100, 100)
    pixmap.fill(Qt.white)
    return pixmap