"""
Arrow class and canvas widget for lithic image annotation.

This module preserves the exact working arrow annotation system
from arrow_annotations.py without any changes to functionality.
"""

# Import the exact working arrow implementation
import sys
import os

# Add parent directory to access the working arrow_annotations.py
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

# Import everything from the working arrow_annotations.py
from arrow_annotations import *