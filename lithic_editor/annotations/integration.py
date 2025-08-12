"""
Integration helpers for arrow annotation in the GUI.

This module preserves the exact working arrow integration system
from arrow_integration.py without any changes to functionality.
"""

# Import the exact working integration implementation
import sys
import os

# Add parent directory to access the working arrow_integration.py
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

# Import everything from the working arrow_integration.py
from arrow_integration import *