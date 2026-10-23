"""The window's theme: Fusion with the light stylesheet."""

from PyQt5.QtWidgets import QApplication

from lithic_editor.gui.theme import ACCENT, CHECK_ICON, stylesheet


def test_the_stylesheet_names_the_accent_and_the_check_icon():
    sheet = stylesheet()
    assert ACCENT in sheet
    assert CHECK_ICON.exists()
    assert CHECK_ICON.as_posix() in sheet


def test_the_window_applies_the_theme(qapp):
    from lithic_editor.gui.main_window import LithicProcessorGUI
    window = LithicProcessorGUI()
    app = QApplication.instance()
    assert ACCENT in app.styleSheet()
    assert window.process_button.objectName() == "primary"
    assert window.input_image_display.objectName() == "input_image_display"


def test_the_arrow_colour_is_a_swatch_not_the_button_background(qapp):
    from PyQt5.QtGui import QColor
    from lithic_editor.annotations.integration import show_color_swatch
    from lithic_editor.gui.main_window import LithicProcessorGUI
    window = LithicProcessorGUI()
    assert window.arrow_color_button.styleSheet() == ""
    assert not window.arrow_color_button.icon().isNull()
    show_color_swatch(window.arrow_color_button, QColor("#ff0000"))
    assert "#ff0000" in window.arrow_color_button.toolTip()
    assert window.arrow_color_button.styleSheet() == ""


def test_the_image_panels_get_most_of_the_window(qapp):
    from lithic_editor.gui.main_window import LithicProcessorGUI
    window = LithicProcessorGUI()
    window.show()
    qapp.processEvents()
    assert window.input_image_display.height() >= 400
    assert window.canvas.height() >= 400
    assert window.canvas.width() >= 400
    window.close()


def test_the_brush_colour_button_sets_the_brush(qapp, monkeypatch):
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import QColorDialog
    from lithic_editor.gui.main_window import LithicProcessorGUI
    window = LithicProcessorGUI()
    assert window.brush_color == QColor("#ffffff")  # white erases, the default
    assert not hasattr(window, "exit_button")
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#000000")))
    window.select_brush_color()
    assert window.brush_color == QColor("#000000")
    assert window.input_image_display.brush_color == QColor("#000000")
    assert "#000000" in window.brush_color_button.toolTip()
