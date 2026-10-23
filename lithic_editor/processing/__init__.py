"""
Processing module for lithic image analysis.

This module contains the image processing algorithms: ripple removal, resolution
handling, upscaling and the striking angle measurement.
"""

from lithic_editor.processing.ripple_removal import (
    process_lithic_drawing,
    improve_line_quality_antialias
)
from lithic_editor.processing.striking_angle import StrikingAngle, measure_striking_angles

__all__ = [
    "process_lithic_drawing",
    "improve_line_quality_antialias",
    "StrikingAngle",
    "measure_striking_angles",
]