"""
Comprehensive help system for the Lithic Editor and Annotator CLI.

This module provides detailed help documentation covering installation,
usage, API documentation, and GUI workflows as specified in PythonPackaging.md.
"""

from lithic_editor import __version__
import sys


def show_help():
    """
    Display comprehensive help information for the Lithic Editor and Annotator.
    """
    help_text = f"""
LITHIC EDITOR AND ANNOTATOR v{__version__}
========================================

A specialized image processing tool for archaeological lithic drawings that automatically
removes ripple lines while preserving structural elements, and provides annotation
capabilities for technical analysis.

INSTALLATION
-----------
Install from git repository:
  pip install git+https://github.com/user/lithic-editor.git

COMMAND LINE USAGE
-----------------
Launch GUI application:
  lithic-editor --gui
  python -m lithic_editor

Process image via CLI:
  lithic-editor process <input_image> [options]

Show version:
  lithic-editor --version

Show this help:
  lithic-editor --help

CLI PROCESSING OPTIONS
---------------------
  --output DIR          Output directory (default: image_debug)
  --debug               Save debug images showing processing steps
  --quiet               Suppress processing output

Examples:
  lithic-editor process lithic.png --output results/ --debug
  lithic-editor process image.jpg --quiet

PROGRAMMATIC API USAGE
---------------------

Basic Processing:
  from lithic_editor.processing import process_lithic_drawing_improved
  
  result = process_lithic_drawing_improved(
      image_path="lithic.png",
      output_folder="results/",
      save_debug=True
  )

GUI Integration:
  from lithic_editor.gui import LithicEditorWidget, launch_gui
  
  # Standalone application
  launch_gui()
  
  # Embed in PyQt application
  from PyQt5.QtWidgets import QApplication
  app = QApplication([])
  editor = LithicEditorWidget()
  editor.show()

Arrow Annotations:
  from lithic_editor.annotations import Arrow, ArrowCanvasWidget
  
  # Create arrow programmatically
  arrow = Arrow(position=(100, 200), angle=45, size=30)
  
  # Use canvas widget
  canvas = ArrowCanvasWidget()
  canvas.load_image("processed_image.png")

DEVELOPMENT USAGE
----------------
During development, you can still use:
  python lithic_GUI.py

This preserves the original development workflow while the package provides
the production-ready API and CLI interface.

SUPPORT
-------
For issues and documentation, see the project repository.

VERSION INFORMATION
------------------
Version: {__version__}
Python: {sys.version}
Platform: {sys.platform}

"""
    print(help_text)


def show_version():
    """Display version information."""
    print(f"Lithic Editor and Annotator v{__version__}")


def show_api_help():
    """Display API-specific help information."""
    api_help = f"""
LITHIC EDITOR API REFERENCE v{__version__}
=========================================

PROCESSING MODULE
----------------
from lithic_editor.processing import process_lithic_drawing_improved

process_lithic_drawing_improved(
    image_path,                 # Path to input image or numpy array
    output_folder="image_debug", # Directory for output files  
    dpi_info=None,              # DPI tuple (x_dpi, y_dpi)
    format_info=None,           # Original image format
    output_dpi=None,            # Desired output DPI
    save_debug=False            # Save intermediate steps
) -> numpy.ndarray              # Returns processed image

GUI MODULE
----------
from lithic_editor.gui import LithicEditorWidget, launch_gui

# Standalone application
launch_gui() -> int             # Returns exit code

# Embeddable widget
widget = LithicEditorWidget()
widget.show()

ANNOTATIONS MODULE
-----------------
from lithic_editor.annotations import Arrow, ArrowCanvasWidget

# Arrow class usage
arrow = Arrow(
    position=(x, y),            # Center position
    angle=0,                    # Rotation in degrees
    size=30,                    # Size in pixels
    color=Qt.black             # Arrow color
)

# Canvas widget usage
canvas = ArrowCanvasWidget()
canvas.load_image(image_path)

EXAMPLES
--------

Basic Processing:
    from lithic_editor.processing import process_lithic_drawing_improved
    
    result = process_lithic_drawing_improved(
        "input.png",
        output_folder="results",
        save_debug=True
    )

Custom GUI Integration:
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from lithic_editor.gui import LithicEditorWidget
    
    class MyApp(QMainWindow):
        def __init__(self):
            super().__init__()
            self.lithic_editor = LithicEditorWidget()
            self.setCentralWidget(self.lithic_editor)
    
    app = QApplication([])
    window = MyApp()
    window.show()
    app.exec_()

"""
    print(api_help)


if __name__ == "__main__":
    show_help()