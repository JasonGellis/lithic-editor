"""
Integration helpers for arrow annotations in the Lithic Editor and Annotator.

This file provides helper functions to integrate arrow annotations
into the main application without cluttering the main code.
"""
import sys
from PyQt5.QtWidgets import (
    QPushButton, QLabel, QHBoxLayout, QGroupBox,
    QSlider, QColorDialog
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor

from arrow_annotations import ArrowCanvasWidget

def setup_arrow_controls(parent):
    # Create arrow annotation controls
    arrow_tools = QGroupBox("Arrow Annotation")
    arrow_layout = QHBoxLayout(arrow_tools)

    # Add arrow button
    parent.add_arrow_button = QPushButton("Add Arrow")
    parent.add_arrow_button.clicked.connect(lambda: add_arrow(parent))
    parent.add_arrow_button.setEnabled(False)

    # Add tooltip for the button
    import sys
    if sys.platform == 'darwin':  # macOS
        parent.add_arrow_button.setToolTip("Add an arrow to the center of the image\n"
                                     "Drag to move, Option+drag to resize, Shift+drag to rotate")
    else:  # Windows, Linux, etc.
        parent.add_arrow_button.setToolTip("Add an arrow to the center of the image\n"
                                     "Drag to move, Alt+drag to resize, Shift+drag to rotate")

    # Arrow color
    parent.arrow_color_button = QPushButton("Arrow Color")
    parent.arrow_color_button.setStyleSheet("background-color: black;")
    parent.arrow_color_button.clicked.connect(lambda: select_arrow_color(parent))
    parent.arrow_color = QColor(Qt.black)

    # Delete arrow button
    parent.delete_arrow_button = QPushButton("Delete Arrow")
    parent.delete_arrow_button.clicked.connect(lambda: delete_selected_arrow(parent))
    parent.delete_arrow_button.setEnabled(False)

    # Clear all arrows button
    parent.clear_arrows_button = QPushButton("Clear Arrows")
    parent.clear_arrows_button.clicked.connect(lambda: clear_arrows(parent))
    parent.clear_arrows_button.setEnabled(False)

    # Create a platform-aware hint message
    import sys
    if sys.platform == 'darwin':  # macOS
        interaction_hint = QLabel("Shift+drag to rotate, Option+drag to resize")
    else:  # Windows, Linux, etc.
        interaction_hint = QLabel("Shift+drag to rotate, Alt+drag to resize")
    interaction_hint.setStyleSheet("font-style: italic; color: gray;")

    # Add widgets to arrow layout
    arrow_layout.addWidget(parent.add_arrow_button)
    arrow_layout.addWidget(parent.arrow_color_button)
    arrow_layout.addWidget(parent.delete_arrow_button)
    arrow_layout.addWidget(parent.clear_arrows_button)
    arrow_layout.addWidget(interaction_hint)  # Add the interaction hint

    return arrow_tools

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

def select_arrow_color(parent):
    """Open a color dialog to select the arrow color"""
    color = QColorDialog.getColor(parent.arrow_color, parent)
    if color.isValid():
        parent.arrow_color = color
        parent.arrow_color_button.setStyleSheet(f"background-color: {color.name()};")

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