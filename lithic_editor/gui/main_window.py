"""
Main GUI window for the Lithic Editor and Annotator.

This module contains the primary PyQt5 interface, exactly preserving the 
functionality from lithic_GUI.py without any changes to layout or behavior.
"""

# Import the exact working GUI implementation
import sys
import os

# Add parent directory to access the working lithic_GUI.py
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

# Import everything from the working lithic_GUI.py
from lithic_GUI import *

# Export the main class for package use
__all__ = ["LithicProcessorGUI"]