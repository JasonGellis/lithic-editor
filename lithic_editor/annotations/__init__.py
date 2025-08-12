"""
Annotations module that wraps the working arrow annotation system.

This module provides access to the proven working arrow annotation
implementation, preserving exact functionality from the original files.
"""

from lithic_editor.annotations.arrows import *
from lithic_editor.annotations.integration import *

# Re-export key classes as specified in PythonPackaging.md
try:
    # Import the main Arrow class and ArrowCanvasWidget from the working code
    from lithic_editor.annotations.arrows import Arrow, ArrowCanvasWidget
    __all__ = ["Arrow", "ArrowCanvasWidget"]
except ImportError as e:
    print(f"Warning: Could not import arrow annotation classes: {e}")
    __all__ = []