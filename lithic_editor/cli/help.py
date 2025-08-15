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

📚 COMPREHENSIVE DOCUMENTATION:
  lithic-editor docs              # Open full documentation online
  lithic-editor docs --offline    # Serve documentation locally
  
  Online: https://jasongellis.github.io/lithic-editor/
  
  The documentation includes:
  • Installation Guide - Detailed setup instructions
  • Quick Start Tutorial - Process your first image in minutes  
  • User Guide - Complete feature documentation with examples
  • Developer Guide - Contributing, testing, and extending
  • API Reference - Python API documentation

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
  from lithic_editor.processing import process_lithic_drawing
  
  result = process_lithic_drawing(
      image_path="lithic.png",
      output_folder="results/",
      save_debug=True
  )

GUI Integration:
  from lithic_editor.gui.main_window import LithicProcessorGUI
  
  # Embed in PyQt application
  from PyQt5.QtWidgets import QApplication
  app = QApplication([])
  editor = LithicProcessorGUI()
  editor.show()

Arrow Annotations:
  from lithic_editor.annotations.arrows import Arrow, ArrowCanvasWidget
  
  # Create arrow programmatically
  arrow = Arrow(position=(100, 200), angle=45, size=30)
  
  # Use canvas widget
  canvas = ArrowCanvasWidget()
  canvas.set_base_image(pixmap)

DEVELOPMENT USAGE
----------------
For developers and contributors:

Setup development environment:
  git clone https://github.com/JasonGellis/lithic-editor.git
  cd lithic-editor
  pip install -e ".[test]"    # Install with test dependencies
  pip install -e ".[dev]"     # Install with all dev tools
  pip install -e ".[docs]"    # Install with documentation tools

Run tests:
  pytest                      # Run test suite
  pytest --cov=lithic_editor # Run with coverage
  pytest -v tests/test_processing.py  # Run specific tests

Build documentation:
  mkdocs serve               # Serve docs locally
  mkdocs build               # Build static docs

Code quality:
  black lithic_editor tests  # Format code
  flake8 lithic_editor tests # Check style

The package provides a complete, self-contained application with
CLI interface and programmatic API.

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
from lithic_editor.processing import process_lithic_drawing

process_lithic_drawing(
    image_path,                 # Path to input image or numpy array
    output_folder="image_debug", # Directory for output files  
    dpi_info=None,              # DPI tuple (x_dpi, y_dpi)
    format_info=None,           # Original image format
    output_dpi=None,            # Desired output DPI
    save_debug=False            # Save intermediate steps
) -> numpy.ndarray              # Returns processed image

GUI MODULE
----------
from lithic_editor.gui.main_window import LithicProcessorGUI

# Embeddable widget
widget = LithicProcessorGUI()
widget.show()

ANNOTATIONS MODULE
-----------------
from lithic_editor.annotations.arrows import Arrow, ArrowCanvasWidget

# Arrow class usage
arrow = Arrow(
    position=(x, y),            # Center position
    angle=0,                    # Rotation in degrees
    size=30,                    # Size in pixels
    color=Qt.black             # Arrow color
)

# Canvas widget usage
canvas = ArrowCanvasWidget()
canvas.set_base_image(pixmap)

EXAMPLES
--------

Basic Processing:
    from lithic_editor.processing import process_lithic_drawing
    
    result = process_lithic_drawing(
        "input.png",
        output_folder="results",
        save_debug=True
    )

Custom GUI Integration:
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from lithic_editor.gui.main_window import LithicProcessorGUI
    
    class MyApp(QMainWindow):
        def __init__(self):
            super().__init__()
            self.lithic_editor = LithicProcessorGUI()
            self.setCentralWidget(self.lithic_editor)
    
    app = QApplication([])
    window = MyApp()
    window.show()
    app.exec_()

"""
    print(api_help)


if __name__ == "__main__":
    show_help()