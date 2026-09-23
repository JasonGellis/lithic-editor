"""
Annotations module: the arrow annotation system.

Exposes the Arrow model and the ArrowCanvasWidget that renders and edits arrows.
GUI helper functions live in ``lithic_editor.annotations.integration``.
"""

from lithic_editor.annotations.arrows import Arrow, ArrowCanvasWidget

__all__ = ["Arrow", "ArrowCanvasWidget"]
