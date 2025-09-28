"""
Processing module for lithic image analysis.

This module contains the core image processing algorithms for ripple removal
and quality enhancement, refactored from the working ripple_remover.py.
"""

from lithic_editor.processing.ripple_removal import (
    process_lithic_drawing,
    improve_line_quality_antialias
)

__all__ = [
    "process_lithic_drawing",
    "improve_line_quality_antialias"
]