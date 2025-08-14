"""
Tests for the annotations module (arrows).
"""

import pytest
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QPixmap
from lithic_editor.annotations.arrows import Arrow, ArrowCanvasWidget


class TestArrow:
    """Test suite for Arrow class."""
    
    def test_arrow_creation(self):
        """Test creating an arrow with default parameters."""
        arrow = Arrow()
        
        assert arrow.position == (0, 0)
        assert arrow.angle == 0
        assert arrow.size == 30
        assert arrow.color == Qt.black
        assert arrow.selected == False
    
    def test_arrow_with_custom_parameters(self):
        """Test creating an arrow with custom parameters."""
        arrow = Arrow(
            position=(100, 200),
            angle=45,
            size=50,
            color=Qt.red
        )
        
        assert arrow.position == (100, 200)
        assert arrow.angle == 45
        assert arrow.size == 50
        assert arrow.color == Qt.red
    
    def test_arrow_detection_status(self):
        """Test arrow detection status calculation."""
        # Small arrow - should be error
        small_arrow = Arrow(size=10)
        assert small_arrow.get_detection_status() == "error"
        
        # Medium arrow - should be warning or good
        medium_arrow = Arrow(size=30)
        status = medium_arrow.get_detection_status()
        assert status in ["warning", "good"]
        
        # Large arrow - should be good
        large_arrow = Arrow(size=100)
        assert large_arrow.get_detection_status() == "good"
    
    def test_arrow_make_detectable(self):
        """Test making arrow detectable."""
        # Create small arrow
        arrow = Arrow(size=10)
        original_size = arrow.size
        
        # Make it detectable
        adjusted = arrow.make_detectable()
        
        # Should have been adjusted
        assert arrow.size > original_size
        assert arrow.get_detection_status() != "error"
    
    def test_arrow_dpi_awareness(self):
        """Test arrow DPI-aware sizing."""
        arrow = Arrow(size=50)
        
        # Test with low DPI
        status_low_dpi = arrow.get_detection_status(image_dpi=150)
        
        # Test with high DPI
        status_high_dpi = arrow.get_detection_status(image_dpi=600)
        
        # Detection status might differ based on DPI
        assert status_low_dpi in ["error", "warning", "good"]
        assert status_high_dpi in ["error", "warning", "good"]


class TestArrowCanvasWidget:
    """Test suite for ArrowCanvasWidget."""
    
    def test_canvas_creation(self, qapp):
        """Test creating arrow canvas widget."""
        canvas = ArrowCanvasWidget()
        
        assert canvas is not None
        assert canvas.arrows == []
        assert canvas.selected_arrow is None
        assert canvas.dragging == False
    
    def test_set_base_image(self, qapp, sample_pixmap):
        """Test setting base image on canvas."""
        canvas = ArrowCanvasWidget()
        canvas.set_base_image(sample_pixmap)
        
        assert canvas.base_pixmap is not None
        assert canvas.base_pixmap.width() == sample_pixmap.width()
        assert canvas.base_pixmap.height() == sample_pixmap.height()
    
    def test_add_arrow(self, qapp, sample_pixmap):
        """Test adding arrow to canvas."""
        canvas = ArrowCanvasWidget()
        canvas.set_base_image(sample_pixmap)
        
        # Add arrow at center
        position = QPoint(50, 50)
        arrow = canvas.add_arrow(position, size=40, color=Qt.black)
        
        assert arrow is not None
        assert len(canvas.arrows) == 1
        assert canvas.selected_arrow == arrow
    
    def test_select_arrow(self, qapp, sample_pixmap):
        """Test arrow selection."""
        canvas = ArrowCanvasWidget()
        canvas.set_base_image(sample_pixmap)
        
        # Add two arrows
        arrow1 = canvas.add_arrow(QPoint(25, 25))
        arrow2 = canvas.add_arrow(QPoint(75, 75))
        
        # Select first arrow
        canvas.select_arrow(arrow1)
        assert canvas.selected_arrow == arrow1
        assert arrow1.selected == True
        assert arrow2.selected == False
        
        # Select second arrow
        canvas.select_arrow(arrow2)
        assert canvas.selected_arrow == arrow2
        assert arrow1.selected == False
        assert arrow2.selected == True
    
    def test_delete_selected_arrow(self, qapp, sample_pixmap):
        """Test deleting selected arrow."""
        canvas = ArrowCanvasWidget()
        canvas.set_base_image(sample_pixmap)
        
        # Add arrows
        arrow1 = canvas.add_arrow(QPoint(25, 25))
        arrow2 = canvas.add_arrow(QPoint(75, 75))
        
        # Select and delete first arrow
        canvas.select_arrow(arrow1)
        canvas.delete_selected_arrow()
        
        assert len(canvas.arrows) == 1
        assert arrow1 not in canvas.arrows
        assert arrow2 in canvas.arrows
        assert canvas.selected_arrow is None
    
    def test_clear_arrows(self, qapp, sample_pixmap):
        """Test clearing all arrows."""
        canvas = ArrowCanvasWidget()
        canvas.set_base_image(sample_pixmap)
        
        # Add multiple arrows
        canvas.add_arrow(QPoint(25, 25))
        canvas.add_arrow(QPoint(50, 50))
        canvas.add_arrow(QPoint(75, 75))
        
        assert len(canvas.arrows) == 3
        
        # Clear all
        canvas.clear_arrows()
        
        assert len(canvas.arrows) == 0
        assert canvas.selected_arrow is None
    
    def test_coordinate_mapping(self, qapp, sample_pixmap):
        """Test coordinate mapping between view and pixmap."""
        canvas = ArrowCanvasWidget()
        canvas.set_base_image(sample_pixmap)
        canvas.resize(200, 200)  # Different from pixmap size
        
        # Test mapping to pixmap coords
        view_point = QPoint(100, 100)
        pixmap_coords = canvas.map_to_pixmap_coords(view_point)
        
        assert pixmap_coords is not None
        assert isinstance(pixmap_coords, tuple)
        assert len(pixmap_coords) == 2
        
        # Test reverse mapping
        back_to_view = canvas.map_to_view_coords(pixmap_coords[0], pixmap_coords[1])
        assert isinstance(back_to_view, QPoint)
    
    def test_set_image_dpi(self, qapp, sample_pixmap):
        """Test setting image DPI."""
        canvas = ArrowCanvasWidget()
        canvas.set_base_image(sample_pixmap)
        
        # Set DPI
        canvas.set_image_dpi(300)
        assert canvas.image_dpi == 300
        
        # Set DPI as tuple
        canvas.set_image_dpi((300, 300))
        assert canvas.image_dpi == 300
        
        # Add arrow and check it's adjusted for DPI
        arrow = canvas.add_arrow(QPoint(50, 50))
        assert arrow.get_detection_status(canvas.image_dpi) != "error"
    
    def test_get_final_image(self, qapp, sample_pixmap):
        """Test getting final image with arrows."""
        canvas = ArrowCanvasWidget()
        canvas.set_base_image(sample_pixmap)
        
        # Add some arrows
        canvas.add_arrow(QPoint(25, 25))
        canvas.add_arrow(QPoint(75, 75))
        
        # Get final image
        final_image = canvas.get_final_image()
        
        assert final_image is not None
    
    def test_minimum_arrow_size(self, qapp):
        """Test minimum arrow size calculation."""
        canvas = ArrowCanvasWidget()
        
        # Default (no DPI)
        min_size = canvas.get_minimum_arrow_size()
        assert min_size == 50
        
        # With DPI
        canvas.image_dpi = 150
        min_size_low_dpi = canvas.get_minimum_arrow_size()
        assert min_size_low_dpi > 50  # Should be larger for low DPI
        
        canvas.image_dpi = 600
        min_size_high_dpi = canvas.get_minimum_arrow_size()
        assert min_size_high_dpi == 50  # Should stay at minimum for high DPI