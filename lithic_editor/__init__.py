"""
Lithic Editor and Annotator

A specialized image processing tool for archaeological lithic drawings that automatically 
removes ripple lines while preserving structural elements, and provides annotation 
capabilities for technical analysis.

This package preserves exact functionality from the working implementation
while providing a production-ready API and CLI interface.
"""

__version__ = "1.0.0"
__author__ = "Jason Jacob Gellis"
__email__ = "jg760@cam.ac.uk"
__institution__ = "University of Cambridge, Department of Archaeology and Anthropology"

# Main package exports as specified in PythonPackaging.md
from lithic_editor.processing import process_lithic_drawing
from lithic_editor.gui import launch_gui

__all__ = [
    "process_lithic_drawing",
    "launch_gui",
    "__version__"
]