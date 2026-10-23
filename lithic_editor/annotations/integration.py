"""
Helper functions for the arrow tools of the main window.

The main window builds the Arrows pane and calls these functions from its buttons.
"""
from PyQt5.QtWidgets import (
    QColorDialog
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap

from lithic_editor.annotations.arrows import ArrowCanvasWidget

def create_arrow_canvas():
    """
    Create and return an arrow canvas widget.

    Returns:
        ArrowCanvasWidget: The canvas widget
    """
    canvas = ArrowCanvasWidget()
    canvas.setStyleSheet("border: 1px solid #cccccc; background-color: #f5f5f5;")
    canvas.setMinimumSize(400, 300)
    canvas.setAlignment(Qt.AlignCenter)
    return canvas

# Arrow manipulation functions
def add_arrow(parent):
    """Add an arrow to the center of the processed image"""
    if not hasattr(parent, 'canvas') or not hasattr(parent.canvas, 'base_pixmap'):
        return

    # Calculate the center of the image
    center_x = parent.canvas.width() // 2
    center_y = parent.canvas.height() // 2

    # Add the arrow with a larger default size (40px instead of 30px)
    arrow = parent.canvas.add_arrow(QPoint(center_x, center_y), size=50, color=parent.arrow_color)

    # Enable the delete buttons
    parent.delete_arrow_button.setEnabled(True)
    parent.clear_arrows_button.setEnabled(True)

def delete_selected_arrow(parent):
    """Delete the currently selected arrow"""
    if hasattr(parent, 'canvas'):
        parent.canvas.delete_selected_arrow()

        # Disable the delete button if no arrows left
        if not parent.canvas.arrows:
            parent.delete_arrow_button.setEnabled(False)
            parent.clear_arrows_button.setEnabled(False)

def clear_arrows(parent):
    """Clear all arrows from the canvas"""
    if hasattr(parent, 'canvas'):
        parent.canvas.clear_arrows()
        parent.delete_arrow_button.setEnabled(False)
        parent.clear_arrows_button.setEnabled(False)

def show_color_swatch(button, color):
    """Show a colour as a square swatch on the button, so the text stays readable."""
    swatch = QPixmap(16, 16)
    swatch.fill(color)
    painter = QPainter(swatch)  # a thin grey border, so a white swatch is visible on a white button
    painter.setPen(QColor("#9ca3af"))
    painter.drawRect(0, 0, 15, 15)
    painter.end()
    button.setIcon(QIcon(swatch))
    button.setToolTip(f"{button.text()}: {color.name()}")


def select_arrow_color(parent):
    """Open a color dialog to select the arrow color"""
    color = QColorDialog.getColor(parent.arrow_color, parent)
    if color.isValid():
        parent.arrow_color = color
        show_color_swatch(parent.arrow_color_button, color)

def enable_arrow_controls(parent):
    """Enable arrow controls after an image is processed"""
    parent.add_arrow_button.setEnabled(True)

def clear_arrows_on_new_image(parent):
    """Clear arrows when a new image is loaded"""
    if hasattr(parent, 'canvas') and hasattr(parent.canvas, 'clear_arrows'):
        parent.canvas.clear_arrows()
        parent.delete_arrow_button.setEnabled(False)
        parent.clear_arrows_button.setEnabled(False)

def get_image_with_arrows(canvas):
    """Get the final image with arrows"""
    if hasattr(canvas, 'get_final_image'):
        return canvas.get_final_image()
    return None