"""
CLI entry point for the lithic-editor package.

This module provides the main command-line interface entry point
when the package is run with `python -m lithic_editor`.
"""

import sys
from lithic_editor.cli.main import main

if __name__ == '__main__':
    sys.exit(main())