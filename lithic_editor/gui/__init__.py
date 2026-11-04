"""
The PyQt5 graphical interface: the main window, its processing thread and the theme.
"""

import sys

from PyQt5.QtWidgets import QApplication

from lithic_editor.gui.main_window import LithicProcessorGUI


def launch_gui():
    """Launch the working GUI implementation with exact same functionality."""
    app = QApplication(sys.argv)
    window = LithicProcessorGUI()
    window.show()
    return app.exec_()

# Export for the API as specified in PythonPackaging.md
LithicEditorWidget = LithicProcessorGUI  # Alias for the spec

__all__ = ["launch_gui", "LithicEditorWidget", "LithicProcessorGUI"]