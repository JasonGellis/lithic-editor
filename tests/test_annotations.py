"""
Tests for the annotations module (arrows).
"""

from PyQt5.QtCore import QPoint, Qt

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
        assert not arrow.selected
    
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
        arrow.make_detectable()
        
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

    def test_small_arrow_is_detectable_at_low_dpi(self):
        """A 20 px arrow is good at 75 DPI, the same arrow on the page as 80 px at 300 DPI."""
        arrow = Arrow(size=20)
        assert arrow.get_detection_status(image_dpi=75) == "good"
        assert arrow.get_detection_status(image_dpi=300) != "good"
        assert arrow.make_detectable(image_dpi=75) is False
        assert arrow.size == 20


class TestArrowCanvasWidget:
    """Test suite for ArrowCanvasWidget."""
    
    def test_canvas_creation(self, qapp):
        """Test creating arrow canvas widget."""
        canvas = ArrowCanvasWidget()
        
        assert canvas is not None
        assert canvas.arrows == []
        assert canvas.selected_arrow is None
        assert not canvas.dragging
    
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
        assert arrow1.selected
        assert not arrow2.selected
        
        # Select second arrow
        canvas.select_arrow(arrow2)
        assert canvas.selected_arrow == arrow2
        assert not arrow1.selected
        assert arrow2.selected
    
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
        
        # Below 300 DPI the minimum shrinks in proportion: the same size on the page
        canvas.image_dpi = 150
        assert canvas.get_minimum_arrow_size() == 25

        # Never below the floor
        canvas.image_dpi = (75, 75)
        assert canvas.get_minimum_arrow_size() == 20
        assert canvas.add_arrow(QPoint(50, 50)).size == 20

        canvas.image_dpi = 600
        assert canvas.get_minimum_arrow_size() == 50  # Should stay at minimum for high DPI

class TestArrowStyles:
    """The arrow styles on the model and the canvas."""

    def test_default_style_is_filled(self):
        assert Arrow().style == "filled"

    def test_unknown_style_is_rejected(self):
        import pytest
        with pytest.raises(ValueError):
            Arrow(style="solid")

    def test_every_style_draws_offscreen(self, qapp):
        from PyQt5.QtGui import QPainter, QPixmap

        from lithic_editor.annotations.arrow_shapes import STYLE_NAMES
        pixmap = QPixmap(200, 200)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        for i, style in enumerate(STYLE_NAMES):
            Arrow(position=(100, 100), angle=30 * i, size=60, style=style).draw(painter)
        painter.end()
        image = pixmap.toImage()
        dark = sum(1 for x in range(200) for y in range(200) if image.pixelColor(x, y).lightness() < 128)
        assert dark > 500

    def test_styles_differ_in_the_drawing(self, qapp):
        from PyQt5.QtGui import QPainter, QPixmap

        from lithic_editor.annotations.arrow_shapes import STYLE_NAMES
        renders = {}
        for style in STYLE_NAMES:
            pixmap = QPixmap(120, 60)
            pixmap.fill(Qt.white)
            painter = QPainter(pixmap)
            Arrow(position=(60, 30), size=60, style=style).draw(painter)
            painter.end()
            image = pixmap.toImage()
            renders[style] = tuple(image.pixelColor(x, y).lightness() < 128
                                   for x in range(120) for y in range(60))
        assert len(set(renders.values())) == len(STYLE_NAMES)

    def test_add_arrow_takes_a_style(self, qapp, sample_pixmap):
        canvas = ArrowCanvasWidget()
        canvas.set_base_image(sample_pixmap)
        arrow = canvas.add_arrow(QPoint(50, 50), style="dot")
        assert arrow.style == "dot"
        assert canvas.add_arrow(QPoint(50, 50)).style == "filled"

    def test_set_selected_style_changes_only_the_selected_arrow(self, qapp, sample_pixmap):
        canvas = ArrowCanvasWidget()
        canvas.set_base_image(sample_pixmap)
        first = canvas.add_arrow(QPoint(30, 30))
        second = canvas.add_arrow(QPoint(70, 70))  # selected
        canvas.set_selected_style("crossed")
        assert (first.style, second.style) == ("filled", "crossed")
        canvas.select_arrow(None)
        canvas.set_selected_style("dashed")  # nothing selected: no change
        assert (first.style, second.style) == ("filled", "crossed")

    def test_selection_changed_reports_the_selected_arrow(self, qapp, sample_pixmap):
        canvas = ArrowCanvasWidget()
        canvas.set_base_image(sample_pixmap)
        seen = []
        canvas.selection_changed.connect(seen.append)
        arrow = canvas.add_arrow(QPoint(50, 50))
        canvas.select_arrow(None)
        assert seen == [arrow, None]
