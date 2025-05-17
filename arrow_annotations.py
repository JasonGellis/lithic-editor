"""
Arrow annotation module for the Lithic Editor and Annotator

This module provides classes for adding, manipulating, and rendering
arrow annotations on images.
"""

from PyQt5.QtWidgets import QLabel, QColorDialog
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath, QPixmap, QPolygon
from PyQt5.QtCore import Qt, QPoint, QLineF


class Arrow:
    """Class representing an arrow annotation that can be placed and rotated"""
    def __init__(self, position=(0, 0), angle=0, size=30, color=Qt.black):
        self.position = position  # (x, y) center position
        self.angle = angle  # angle in degrees
        self.size = size  # size in pixels
        self.color = color  # color of the arrow
        self.selected = False  # whether arrow is selected for manipulation

    def draw(self, painter):
        """Draw the arrow on the given painter with vector-like quality"""
        # Save the current painter state
        painter.save()

        # CRITICAL: Disable ALL anti-aliasing for vector-like rendering
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.setRenderHint(QPainter.HighQualityAntialiasing, False)

        # Set up the pen with sharp joins
        pen = QPen(self.color)
        pen.setWidth(2)  # Slightly thinner outline
        pen.setJoinStyle(Qt.MiterJoin)  # Sharp corners
        painter.setPen(pen)
        painter.setBrush(self.color)

        # Move to the arrow position and rotate
        painter.translate(self.position[0], self.position[1])
        painter.rotate(self.angle)

        # Calculate arrow dimensions based on size
        shaft_length = self.size * 0.8
        head_length = self.size * 0.6
        head_width = self.size * 0.6
        shaft_width = max(3, int(self.size * 0.1))  # Min width 3px

        # Create points for the entire arrow as a single polygon
        points = []

        # Shaft left side
        points.append(QPoint(int(-shaft_length/2), int(-shaft_width/2)))

        # Shaft right side + arrow base
        points.append(QPoint(int(shaft_length/2 - head_length), int(-shaft_width/2)))

        # Arrow head top
        points.append(QPoint(int(shaft_length/2 - head_length), int(-head_width/2)))

        # Arrow tip
        points.append(QPoint(int(shaft_length/2), 0))

        # Arrow head bottom
        points.append(QPoint(int(shaft_length/2 - head_length), int(head_width/2)))

        # Shaft right side bottom
        points.append(QPoint(int(shaft_length/2 - head_length), int(shaft_width/2)))

        # Shaft left side bottom
        points.append(QPoint(int(-shaft_length/2), int(shaft_width/2)))

        # Draw the polygon
        painter.drawPolygon(QPolygon(points))

        # If selected, draw a selection indicator
        if self.selected:
            painter.setPen(QPen(Qt.blue, 1, Qt.DashLine))
            painter.setBrush(Qt.transparent)
            painter.drawEllipse(QPoint(0, 0), int(self.size / 2 + 5), int(self.size / 2 + 5))

        # Restore the painter state
        painter.restore()

class ArrowCanvasWidget(QLabel):
    """A canvas widget that can display an image and arrow annotations"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        # Image properties
        self.base_pixmap = None
        self.temp_pixmap = None  # For rendering without saving changes

        # Arrow properties
        self.arrows = []  # List of arrows
        self.selected_arrow = None  # Currently selected arrow
        self.dragging = False  # Whether we're dragging an arrow
        self.drag_start_pos = None  # Starting position for drag
        self.drag_start_angle = 0  # Starting angle for rotation
        self.resizing = False  # Whether we're resizing an arrow
        self.resize_start_pos = None  # Starting position for resize
        self.resize_start_size = 0  # Starting size for resize

        # Alignment
        self.setAlignment(Qt.AlignCenter)

    def set_base_image(self, pixmap):
        """Set the base image to display"""
        self.base_pixmap = pixmap.copy()
        self.update_display()

    def update_display(self):
        """Update the display with base image + high-res vector arrows"""
        if self.base_pixmap:
            # Get original dimensions
            width = self.base_pixmap.width()
            height = self.base_pixmap.height()

            # STEP 1: Create high-resolution pixmap (2x)
            scale_factor = 2.0
            high_res = QPixmap(int(width * scale_factor), int(height * scale_factor))
            high_res.fill(Qt.transparent)

            # STEP 2: Draw base image scaled up
            high_painter = QPainter(high_res)
            high_painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            high_painter.drawPixmap(0, 0, self.base_pixmap.scaled(
                int(width * scale_factor), int(height * scale_factor),
                Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            ))

            # STEP 3: Draw all arrows at high resolution
            # Scale coordinates, but keep arrow drawing sharp
            high_painter.scale(scale_factor, scale_factor)
            for arrow in self.arrows:
                arrow.draw(high_painter)
            high_painter.end()

            # STEP 4: Downsample with nearest-neighbor to keep sharp edges
            final = high_res.scaled(
                width, height,
                Qt.IgnoreAspectRatio, Qt.FastTransformation
            )

            # Store this for final image retrieval
            self.temp_pixmap = final

            # STEP 5: Scale for display within the widget bounds
            if self.width() > 0 and self.height() > 0:
                display_width = self.width() - 10
                display_height = self.height() - 10

                w, h = final.width(), final.height()
                if w > 0 and h > 0:
                    # Calculate scaling factor to fit within display
                    scale = min(display_width / w, display_height / h)

                    # Scale to fit the display area
                    scaled_pixmap = final.scaled(
                        int(w * scale), int(h * scale),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )

                    # Display the scaled result
                    super().setPixmap(scaled_pixmap)
                    return

            # If no scaling applied
            super().setPixmap(final)

    def add_arrow(self, position, size=50, color=Qt.black):
        """Add a new arrow at the specified position"""
        # Convert from view coordinates to pixmap coordinates if needed
        pixmap_pos = self.map_to_pixmap_coords(position)

        # Create and add the arrow
        arrow = Arrow(position=pixmap_pos, size=size, color=color)
        self.arrows.append(arrow)

        # Select the new arrow
        self.select_arrow(arrow)

        # Update the display
        self.update_display()

        return arrow

    def select_arrow(self, arrow):
        """Select an arrow for manipulation"""
        # Deselect all arrows
        for a in self.arrows:
            a.selected = False

        # Select the specified arrow
        if arrow:
            arrow.selected = True
            self.selected_arrow = arrow
        else:
            self.selected_arrow = None

        # Update the display
        self.update_display()

    def delete_selected_arrow(self):
        """Delete the currently selected arrow"""
        if self.selected_arrow:
            self.arrows.remove(self.selected_arrow)
            self.selected_arrow = None
            self.update_display()

    def clear_arrows(self):
        """Remove all arrows"""
        self.arrows.clear()
        self.selected_arrow = None
        self.update_display()

    def clear_canvas(self):
        """Clear all arrows and reset the canvas"""
        self.arrows.clear()
        self.selected_arrow = None

        # If we have a base image, update the display to show just the base image
        if hasattr(self, 'base_pixmap'):
            self.update_display()
        else:
            # Initialize empty pixmap if we don't have a base image
            empty_pixmap = QPixmap(self.width(), self.height())
            empty_pixmap.fill(Qt.transparent)
            self.setPixmap(empty_pixmap)

    def get_final_image(self):
        """Get the final image with vector-like arrows"""
        if hasattr(self, 'temp_pixmap'):
            return self.temp_pixmap.toImage()
        return None

    def map_to_pixmap_coords(self, pos):
        """Map view coordinates to pixmap coordinates"""
        if not self.base_pixmap:
            return pos.x(), pos.y()

        # Get widget and pixmap sizes
        label_size = self.size()
        pixmap_size = self.base_pixmap.size()

        # Calculate scale factor for display
        scale_w = pixmap_size.width() / float(label_size.width())
        scale_h = pixmap_size.height() / float(label_size.height())

        # Use max scale to maintain aspect ratio (same as display logic)
        if scale_w > scale_h:
            # Width is the limiting factor
            scale = scale_w
            y_offset = (label_size.height() - pixmap_size.height() / scale) / 2
            x_offset = 0
        else:
            # Height is the limiting factor
            scale = scale_h
            x_offset = (label_size.width() - pixmap_size.width() / scale) / 2
            y_offset = 0

        # Map the position
        pixmap_x = (pos.x() - x_offset) * scale
        pixmap_y = (pos.y() - y_offset) * scale

        return pixmap_x, pixmap_y

    def map_to_view_coords(self, pixmap_x, pixmap_y):
        """Map pixmap coordinates to view coordinates"""
        if not self.base_pixmap:
            return QPoint(pixmap_x, pixmap_y)

        # Get widget and pixmap sizes
        label_size = self.size()
        pixmap_size = self.base_pixmap.size()

        # Calculate scaling factors
        scale_w = pixmap_size.width() / label_size.width()
        scale_h = pixmap_size.height() / label_size.height()

        # Use the same scaling logic as in your display code
        if scale_w > scale_h:
            # Width is the limiting factor
            scale = scale_w
            y_offset = (label_size.height() - pixmap_size.height() / scale) / 2
            x_offset = 0
        else:
            # Height is the limiting factor
            scale = scale_h
            x_offset = (label_size.width() - pixmap_size.width() / scale) / 2
            y_offset = 0

        # Map the position
        view_x = pixmap_x / scale + x_offset
        view_y = pixmap_y / scale + y_offset

        return QPoint(int(view_x), int(view_y))

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if not self.base_pixmap or not event.button() == Qt.LeftButton:
            return

        # Check if we clicked on an existing arrow
        for arrow in reversed(self.arrows):  # Check in reverse order to select top-most arrow
            pixmap_point = self.map_to_pixmap_coords(event.pos())

            # Calculate distance in pixmap coordinates
            dx = pixmap_point[0] - arrow.position[0]
            dy = pixmap_point[1] - arrow.position[1]
            distance = (dx * dx + dy * dy) ** 0.5

            # If within selection radius, select the arrow
            if distance <= arrow.size / 1.5:
                self.select_arrow(arrow)

                # Use Qt.AltModifier which works cross-platform
                # (corresponds to Alt on Windows/Linux, Option on Mac)
                if event.modifiers() & Qt.AltModifier:
                    self.resizing = True
                    self.resize_start_pos = event.pos()
                    self.resize_start_size = arrow.size
                else:
                    self.dragging = True
                    self.drag_start_pos = event.pos()
                    self.drag_start_angle = arrow.angle
                return

        # If we didn't click on an arrow, deselect all
        self.select_arrow(None)

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if not self.base_pixmap:
            return

        if self.resizing and self.selected_arrow and self.resize_start_pos:
            # Calculate the distance moved from start position
            dx = event.pos().x() - self.resize_start_pos.x()

            # Use horizontal movement for resizing (simpler than radial calculation)
            # Positive dx = increase size, negative dx = decrease size
            size_change = dx * 0.5  # Scale the change for better control

            # Update the arrow size with limits
            new_size = max(10, min(200, self.resize_start_size + size_change))
            self.selected_arrow.size = new_size

            # Update the display
            self.update_display()

        elif self.dragging and self.selected_arrow and self.drag_start_pos:
            if event.modifiers() & Qt.ShiftModifier:
                # Rotation mode
                # Calculate angle from center of arrow to current mouse position
                center = self.map_to_view_coords(self.selected_arrow.position[0], self.selected_arrow.position[1])

                # Get the vectors from center to start and current positions
                start_vector = QLineF(center, self.drag_start_pos)
                current_vector = QLineF(center, event.pos())

                # Calculate the angle between them
                angle_change = start_vector.angle() - current_vector.angle()

                # Update the arrow's angle
                self.selected_arrow.angle = (self.drag_start_angle + angle_change) % 360
                if event.modifiers() & Qt.ControlModifier:
                    # Snap to 15-degree increments if Ctrl is also held
                    self.selected_arrow.angle = round(self.selected_arrow.angle / 15) * 15
            else:
                # Movement mode
                # Calculate the offset
                dx = event.pos().x() - self.drag_start_pos.x()
                dy = event.pos().y() - self.drag_start_pos.y()

                # Update the drag start position
                self.drag_start_pos = event.pos()

                # Convert to pixmap coordinates and update the arrow's position
                pixmap_pos = self.map_to_pixmap_coords(event.pos())
                self.selected_arrow.position = pixmap_pos

            # Update the display
            self.update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.resizing = False  # Add this line
            self.drag_start_pos = None
            self.resize_start_pos = None  # Add this line

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Delete and self.selected_arrow:
            self.delete_selected_arrow()
        elif event.key() == Qt.Key_Left and self.selected_arrow:
            self.selected_arrow.angle = (self.selected_arrow.angle - 5) % 360
            self.update_display()
        elif event.key() == Qt.Key_Right and self.selected_arrow:
            self.selected_arrow.angle = (self.selected_arrow.angle + 5) % 360
            self.update_display()
        else:
            super().keyPressEvent(event)

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Delete and self.selected_arrow:
            self.delete_selected_arrow()
        elif event.key() == Qt.Key_Left and self.selected_arrow:
            self.selected_arrow.angle = (self.selected_arrow.angle - 5) % 360
            self.update_display()
        elif event.key() == Qt.Key_Right and self.selected_arrow:
            self.selected_arrow.angle = (self.selected_arrow.angle + 5) % 360
            self.update_display()
        else:
            super().keyPressEvent(event)

    # Add this method at the end of the class
    def setText(self, text):
        """Set text on the canvas (for showing messages)"""
        empty_pixmap = QPixmap(self.width(), self.height())
        empty_pixmap.fill(Qt.white)
        painter = QPainter(empty_pixmap)
        painter.setPen(Qt.black)
        painter.drawText(empty_pixmap.rect(), Qt.AlignCenter, text)
        painter.end()
        self.setPixmap(empty_pixmap)

        # Clear any existing image or arrows
        self.base_pixmap = None
        self.arrows.clear()
        self.selected_arrow = None