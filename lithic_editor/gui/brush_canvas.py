"""
The brush canvas of the Input Image panel: shows the input and lets the user paint on it
in black or white before processing.
"""

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel


class CanvasWidget(QLabel):
    """Custom canvas widget for drawing annotations"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.last_point = QPoint()
        self.drawing = False
        self.brush_active = False
        self.current_mouse_pos = QPoint(0, 0)
        self.input_image_path = None
        self.processed_image_data = None  # Store processed image in memory
        self.processed_dpi_info = None
        self.processed_format_info = None

        # Drawing properties
        self.brush_size = 5
        self.brush_color = Qt.black

        # Initialize empty pixmap
        self.setAlignment(Qt.AlignCenter)
        self.clear_canvas()

    def paintEvent(self, event):
        """Override paintEvent to draw brush cursor preview"""
        super().paintEvent(event)

        # Only draw cursor if brush is active and we have a base image
        if self.brush_active and hasattr(self, 'base_pixmap'):
            painter = QPainter(self)
            pen = QPen(self.brush_color)
            pen.setWidth(1)
            painter.setPen(pen)

            # Draw circle at current mouse position to show brush size
            painter.setBrush(Qt.transparent)
            painter.drawEllipse(self.current_mouse_pos,
                              self.brush_size // 2,
                              self.brush_size // 2)

    def clear_canvas(self):
        """Clear the canvas"""
        # Create a transparent pixmap
        self.annotation_pixmap = QPixmap(self.width(), self.height())
        self.annotation_pixmap.fill(Qt.transparent)
        self.setPixmap(self.annotation_pixmap)

    def set_base_image(self, pixmap):
        """Set the base image for annotation"""
        # Create a copy of the input pixmap
        self.base_pixmap = pixmap.copy()

        # Initialize annotation layer to match base image size
        self.annotation_pixmap = QPixmap(self.base_pixmap.size())
        self.annotation_pixmap.fill(Qt.transparent)

        # Combine base with empty annotations
        self.update_display()

    def update_display(self):
        """Update the display with base image + annotations"""
        if hasattr(self, 'base_pixmap'):
            # Create a copy of the base image
            result = self.base_pixmap.copy()

            # Draw annotations on top
            painter = QPainter(result)
            painter.drawPixmap(0, 0, self.annotation_pixmap)
            painter.end()

            # Scale for display within the widget bounds, maintaining aspect ratio
            if self.width() > 0 and self.height() > 0:  # Prevent division by zero
                display_width = self.width() - 10
                display_height = self.height() - 10

                w, h = result.width(), result.height()
                if w > 0 and h > 0:  # Prevent division by zero
                    scale = min(display_width / w, display_height / h)

                    if scale < 1:  # Only scale down, not up
                        result = result.scaled(int(w * scale), int(h * scale),
                                             Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Update the display
            self.setPixmap(result)

    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        if hasattr(self, 'base_pixmap'):
            self.update_display()

    def set_brush_properties(self, size, color):
        """Set brush size and color"""
        self.brush_size = size
        self.brush_color = color

    def set_brush_active(self, active):
        """Set whether the brush is active"""
        self.brush_active = active
        # Update cursor
        if active:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def adjustSize(self):
        """Adjust the widget size to fit the parent widget"""
        if self.parent():
            self.setMinimumWidth(self.parent().width() - 20)
            self.setMinimumHeight(self.parent().height() - 20)

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton and hasattr(self, 'base_pixmap') and self.brush_active:
            self.drawing = True
            self.last_point = self.map_to_pixmap(event.pos())

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        # Track current mouse position for cursor display
        self.current_mouse_pos = event.pos()

        # Trigger repaint to update cursor
        self.update()

        # Handle drawing if active
        if self.drawing and hasattr(self, 'base_pixmap') and self.brush_active:
            current_point = self.map_to_pixmap(event.pos())

            # Draw on the annotation layer
            painter = QPainter(self.annotation_pixmap)
            pen = QPen()
            pen.setWidth(self.brush_size)
            pen.setColor(self.brush_color)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, current_point)
            painter.end()

            self.last_point = current_point

            # Update the display
            self.update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def map_to_pixmap(self, pos):
        """Map screen coordinates to pixmap coordinates"""
        if not hasattr(self, 'base_pixmap'):
            return pos

        # Get the current displayed pixmap from the widget
        displayed_pixmap = self.pixmap()
        if not displayed_pixmap or displayed_pixmap.isNull():
            return pos

        # Get widget and pixmap sizes
        widget_size = self.size()
        base_pixmap_size = self.base_pixmap.size()
        displayed_size = displayed_pixmap.size()

        # Calculate the scale factor used for display
        scale_x = displayed_size.width() / base_pixmap_size.width()
        scale_y = displayed_size.height() / base_pixmap_size.height()

        # Calculate the offset to center the image in the widget
        x_offset = (widget_size.width() - displayed_size.width()) / 2
        y_offset = (widget_size.height() - displayed_size.height()) / 2

        # Map the position from widget coordinates to displayed pixmap coordinates
        display_x = pos.x() - x_offset
        display_y = pos.y() - y_offset

        # Scale from displayed coordinates to original pixmap coordinates
        pixmap_x = display_x / scale_x
        pixmap_y = display_y / scale_y

        return QPoint(int(pixmap_x), int(pixmap_y))

    def get_annotated_image(self):
        """Get the base image with annotations as QImage"""
        if hasattr(self, 'base_pixmap'):
            # Create a copy of the base image
            result = self.base_pixmap.copy()

            # Draw annotations on top
            painter = QPainter(result)
            painter.drawPixmap(0, 0, self.annotation_pixmap)
            painter.end()

            return result.toImage()
        return None
