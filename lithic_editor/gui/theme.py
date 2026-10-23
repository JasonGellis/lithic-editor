"""
The look of the window: Qt's Fusion style with a light stylesheet.

Fusion draws the same widgets on every platform, so the window looks the same on
macOS, Windows and Linux, and the offscreen tests render what the user sees. The
stylesheet puts white panels on a light grey background, rounds the fields and
buttons, and uses one accent colour for the primary button and the checked boxes.
Every colour of the window is defined here.
"""

from pathlib import Path

from PyQt5.QtWidgets import QApplication

ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
ACCENT_PALE = "#bfdbfe"
WINDOW = "#f7f8fa"
PANEL = "#ffffff"
PANEL_ALT = "#f3f4f6"
PANEL_PRESSED = "#e5e7eb"
PANEL_DISABLED = "#f9fafb"
LINE = "#e5e7eb"
LINE_STRONG = "#d1d5db"
LINE_INDICATOR = "#9ca3af"
TEXT = "#1f2937"
TEXT_SOFT = "#374151"
TEXT_MUTED = "#6b7280"
TEXT_DISABLED = "#9ca3af"
BRUSH_ACTIVE = "#bbf7d0"

ICONS = Path(__file__).with_name("icons")
CHECK_ICON = ICONS / "check.svg"
UP_ICON = ICONS / "chevron-up.svg"
DOWN_ICON = ICONS / "chevron-down.svg"


def stylesheet() -> str:
    """The stylesheet, with the colours and the icon path filled in."""
    check = CHECK_ICON.as_posix()
    up = UP_ICON.as_posix()
    down = DOWN_ICON.as_posix()
    return f"""
QMainWindow, QWidget {{ background: {WINDOW}; color: {TEXT}; font-size: 13px; }}

QGroupBox {{
    background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px;
    margin-top: 22px; padding: 12px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left; left: 4px; top: 2px;
    padding: 0 2px; color: {TEXT_SOFT}; font-weight: 600;
}}
QGroupBox QWidget {{ background: transparent; }}

QPushButton {{
    background: {PANEL}; border: 1px solid {LINE_STRONG}; border-radius: 6px;
    padding: 4px 8px; min-height: 20px;
}}
QPushButton:hover {{ background: {PANEL_ALT}; }}
QPushButton:pressed {{ background: {PANEL_PRESSED}; }}
QPushButton:checked {{ background: {BRUSH_ACTIVE}; border-color: {BRUSH_ACTIVE}; }}
QPushButton#log_toggle:checked {{ background: {PANEL_PRESSED}; border-color: {LINE_STRONG}; }}
QPushButton:disabled {{ color: {TEXT_DISABLED}; border-color: {LINE}; background: {PANEL_DISABLED}; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border: none; font-weight: 600; }}
QPushButton#primary:hover {{ background: {ACCENT_DARK}; }}
QPushButton#primary:disabled {{ background: {ACCENT_PALE}; color: white; }}

QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {{
    background: {PANEL}; border: 1px solid {LINE_STRONG}; border-radius: 6px; padding: 3px 6px;
}}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled, QComboBox:disabled {{
    color: {TEXT_DISABLED}; background: {PANEL_DISABLED}; border-color: {LINE};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right; width: 18px;
    border-left: 1px solid {LINE}; border-top-right-radius: 6px; background: {PANEL_ALT};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right; width: 18px;
    border-left: 1px solid {LINE}; border-bottom-right-radius: 6px; background: {PANEL_ALT};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {PANEL_PRESSED}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({up}); width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({down}); width: 10px; height: 10px; }}

QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QCheckBox, QRadioButton {{ spacing: 8px; }}
QCheckBox:disabled, QRadioButton:disabled, QLabel:disabled {{ color: {TEXT_DISABLED}; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px; border: 1px solid {LINE_INDICATOR}; background: {PANEL};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url({check}); }}
QRadioButton::indicator {{ border-radius: 8px; }}
QRadioButton::indicator:checked {{ border: 5px solid {ACCENT}; background: {PANEL}; }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ background: {PANEL_DISABLED}; border-color: {LINE}; }}
QCheckBox::indicator:checked:disabled {{ background: {ACCENT_PALE}; border-color: {ACCENT_PALE}; }}
QRadioButton::indicator:checked:disabled {{ border-color: {ACCENT_PALE}; }}

QTextEdit {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; color: {TEXT_SOFT}; }}
QScrollArea {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; }}
QScrollArea > QWidget > QWidget {{ background: {PANEL}; }}
QLabel#hint {{ color: {TEXT_MUTED}; }}
QLabel#input_image_display {{ background: {PANEL_ALT}; border: 1px solid {LINE}; border-radius: 8px; }}
QLabel#debug_image {{ background: {PANEL}; border: 1px solid {LINE}; }}

QProgressBar {{ background: {LINE}; border: none; border-radius: 4px; max-height: 8px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {LINE_STRONG}; border-radius: 5px; min-height: 30px; min-width: 30px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QToolTip {{ background: {TEXT}; color: white; border: none; padding: 4px 6px; }}
"""


def apply_theme(app: QApplication) -> None:
    """Set the Fusion style and the stylesheet on the application."""
    app.setStyle("Fusion")
    app.setStyleSheet(stylesheet())
