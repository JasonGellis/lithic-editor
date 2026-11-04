"""
The panes of the main window.

Each function builds one pane, puts its controls on ``window`` under the names the
rest of the window and the tests use, and returns the pane widget.
"""

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lithic_editor.annotations import integration as arrow_integration
from lithic_editor.annotations.arrow_shapes import STYLES as ARROW_STYLES
from lithic_editor.config import AngleSettings
from lithic_editor.gui.brush_canvas import CanvasWidget


def build_control_column(window):
    """The left column: the File, Brush, Arrows and Options panes, one under the other."""
    left_column = QWidget()
    left_layout = QVBoxLayout(left_column)
    left_layout.setSpacing(10)
    left_layout.setContentsMargins(0, 0, 0, 0)

    panes = (build_file_pane(window), build_brush_pane(window), build_arrows_pane(window))
    for group in panes:
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        left_layout.addWidget(group)

    # The Options pane scrolls when the column is shorter than its content
    left_layout.addWidget(build_options_pane(window), 1)
    left_column.setMinimumWidth(320)
    left_column.setMaximumWidth(420)
    window.left_column = left_column
    return left_column


def build_file_pane(window):
    """The File pane: Load Image, Process Image and Save Result in one row."""
    # File pane: one row of buttons
    file_controls = QGroupBox("File")
    file_layout = QHBoxLayout(file_controls)
    file_layout.setSpacing(6)
    file_layout.setContentsMargins(4, 2, 4, 4)

    window.load_button = QPushButton('Load Image')
    window.process_button = QPushButton('Process Image')
    window.process_button.setObjectName('primary')  # the accent button of the theme
    window.process_button.setEnabled(False)
    window.save_button = QPushButton('Save Result')
    window.save_button.setEnabled(False)

    # Connect signals
    window.load_button.clicked.connect(window.load_image)
    window.process_button.clicked.connect(window.process_image)
    window.save_button.clicked.connect(window.save_result)

    for button in (window.load_button, window.process_button, window.save_button):
        file_layout.addWidget(button, 1)  # three buttons of equal width
    return file_controls


def build_brush_pane(window):
    """The Brush pane: activate, colour, size and clear."""
    # Brush pane: two rows
    drawing_tools = QGroupBox("Brush")
    drawing_layout = QGridLayout(drawing_tools)
    drawing_layout.setSpacing(6)
    drawing_layout.setContentsMargins(4, 2, 4, 4)

    window.brush_button = QPushButton("Activate Brush")
    window.brush_button.setCheckable(True)
    window.brush_button.setChecked(False)
    window.brush_button.clicked.connect(window.toggle_brush)

    # Brush colour: a button with a swatch, like the arrow colour button.
    # White erases, black draws. White is the default.
    window.brush_color = QColor(Qt.white)
    window.brush_color_button = QPushButton("Brush Color")
    window.brush_color_button.setToolTip('The colour of the brush. White erases, black draws.')
    arrow_integration.show_color_swatch(window.brush_color_button, window.brush_color)
    window.brush_color_button.clicked.connect(window.select_brush_color)

    # Brush size
    size_label = QLabel("Brush Size:")
    window.brush_size_spin = QSpinBox()
    window.brush_size_spin.setMinimum(1)
    window.brush_size_spin.setMaximum(20)
    window.brush_size_spin.setValue(5)
    window.brush_size_spin.valueChanged.connect(window.update_brush)

    window.clear_annotations_button = QPushButton("Clear Brush")
    window.clear_annotations_button.clicked.connect(window.clear_annotations)
    window.clear_annotations_button.setEnabled(False)

    size_row = QHBoxLayout()
    size_row.setSpacing(4)
    size_row.addWidget(size_label)
    size_row.addWidget(window.brush_size_spin, 1)

    drawing_layout.addWidget(window.brush_button, 0, 0)
    drawing_layout.addWidget(window.brush_color_button, 0, 1)
    drawing_layout.addLayout(size_row, 1, 0)
    drawing_layout.addWidget(window.clear_annotations_button, 1, 1)
    drawing_layout.setColumnStretch(0, 1)  # two columns of equal width
    drawing_layout.setColumnStretch(1, 1)
    return drawing_tools


def build_arrows_pane(window):
    """The Arrows pane: add, colour, clear, delete, the style box and the mouse hint."""
    # Arrows pane: two rows and the mouse hint
    arrow_tools = QGroupBox("Arrows")
    arrow_layout = QGridLayout(arrow_tools)
    arrow_layout.setSpacing(6)
    arrow_layout.setContentsMargins(4, 2, 4, 4)

    window.add_arrow_button = QPushButton("Add Arrow")
    window.add_arrow_button.clicked.connect(lambda: arrow_integration.add_arrow(window))
    window.add_arrow_button.setEnabled(False)

    window.arrow_color_button = QPushButton("Arrow Color")
    arrow_integration.show_color_swatch(window.arrow_color_button, QColor(Qt.black))
    window.arrow_color_button.clicked.connect(lambda: arrow_integration.select_arrow_color(window))
    window.arrow_color = QColor(Qt.black)

    window.delete_arrow_button = QPushButton("Delete Arrow")
    window.delete_arrow_button.clicked.connect(lambda: arrow_integration.delete_selected_arrow(window))
    window.delete_arrow_button.setEnabled(False)

    window.clear_arrows_button = QPushButton("Clear Arrows")
    window.clear_arrows_button.clicked.connect(lambda: arrow_integration.clear_arrows(window))
    window.clear_arrows_button.setEnabled(False)

    # Add interaction hint
    if sys.platform == 'darwin':  # macOS
        interaction_hint = QLabel("Shift+drag to rotate, Option+drag to resize")
    else:  # Windows, Linux, etc.
        interaction_hint = QLabel("Shift+drag to rotate, Alt+drag to resize")
    interaction_hint.setObjectName("hint")  # the colour comes from the theme
    interaction_hint.setStyleSheet("font-style: italic; font-size: 10px;")

    # Arrow style: for new arrows and for the selected arrow
    style_row = QHBoxLayout()
    style_row.setSpacing(6)
    style_label = QLabel("Arrow Style:")
    window.arrow_style_combo = QComboBox()
    for name, label in ARROW_STYLES:
        window.arrow_style_combo.addItem(label, name)
    window.arrow_style_combo.setToolTip("The style of new arrows. Select an arrow to change its style.")
    window.arrow_style_combo.setEnabled(False)
    window.arrow_style_combo.currentIndexChanged.connect(
        lambda _index: arrow_integration.set_arrow_style(window))
    style_row.addWidget(style_label)
    style_row.addWidget(window.arrow_style_combo, 1)

    arrow_layout.addWidget(window.add_arrow_button, 0, 0)
    arrow_layout.addWidget(window.arrow_color_button, 0, 1)
    arrow_layout.addWidget(window.clear_arrows_button, 1, 0)
    arrow_layout.addWidget(window.delete_arrow_button, 1, 1)
    arrow_layout.addLayout(style_row, 2, 0, 1, 2)
    interaction_hint.setWordWrap(True)
    arrow_layout.addWidget(interaction_hint, 3, 0, 1, 2)
    arrow_layout.setColumnStretch(0, 1)  # two columns of equal width
    arrow_layout.setColumnStretch(1, 1)
    return arrow_tools


def build_options_pane(window):
    """The Options pane in a scroll area: processing, output, debug, configuration and DPI."""
    # RIGHT COLUMN: Options & DPI settings (single tall panel)
    options_group = QGroupBox("Options")
    options_layout = QVBoxLayout(options_group)
    options_layout.setSpacing(6)
    options_layout.setContentsMargins(4, 4, 4, 4)
    options_layout.setSpacing(6)
    options_layout.setContentsMargins(10, 10, 10, 10)

    def heading(text):
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold; font-size: 11px; margin-top: 10px;")
        return label

    # --- Processing ---
    window.preserve_cortex_checkbox = QCheckBox('Keep the cortex stipple')
    window.preserve_cortex_checkbox.setChecked(True)  # Default ON
    window.preserve_cortex_checkbox.setToolTip('Keep the small cortex dots when the lines are processed.')

    # The striking angle of a scar is measured from its ripples before they are removed
    window.measure_angles_checkbox = QCheckBox('Add striking angle arrows')
    window.measure_angles_checkbox.setChecked(True)
    window.measure_angles_checkbox.setToolTip(
        'Measure the direction of the blow from the ripples of each scar.\n'
        'An arrow is added at the centre of the ripples. Move, turn or delete it as necessary.\n'
        'A scar with fewer ripples than the minimum gets no arrow.')
    window.measure_angles_checkbox.toggled.connect(window.update_angle_options_state)

    # The three thresholds of the measurement. They start at the configuration
    # values and replace them for the next run.
    window.min_ripples_spin = QSpinBox()
    window.min_ripples_spin.setRange(1, 99)
    window.min_ripples_spin.setToolTip('A scar with fewer ripples than this gets no arrow.')
    window.min_arc_depth_spin = QDoubleSpinBox()
    window.min_arc_depth_spin.setRange(0.0, 100.0)
    window.min_arc_depth_spin.setDecimals(1)
    window.min_arc_depth_spin.setSingleStep(0.5)
    window.min_arc_depth_spin.setSuffix(' px')
    window.min_arc_depth_spin.setToolTip(
        'The arc depth of a ripple is the distance from the straight line between its two ends\n'
        'to its farthest point. The ripples curve in the direction of the force. When the mean\n'
        'arc depth is less than this,\n'
        'the direction is not certain and the log says so.')
    window.max_cortex_dots_spin = QSpinBox()
    window.max_cortex_dots_spin.setRange(0, 999)
    window.max_cortex_dots_spin.setToolTip(
        'A face with more cortex dots than this is cortex, not a scar, and gets no arrow.')
    for spin in (window.min_ripples_spin, window.min_arc_depth_spin, window.max_cortex_dots_spin):
        spin.setMaximumWidth(90)
    window.set_angle_options(AngleSettings())

    window.angle_options_widget = QWidget()
    angle_options_layout = QVBoxLayout(window.angle_options_widget)
    angle_options_layout.setContentsMargins(20, 0, 0, 0)
    angle_options_layout.setSpacing(2)
    for text, spin in (('Minimum ripples:', window.min_ripples_spin),
                       ('Minimum arc depth:', window.min_arc_depth_spin),
                       ('Maximum cortex dots in a scar:', window.max_cortex_dots_spin)):
        row = QHBoxLayout()
        row.addWidget(QLabel(text))
        row.addWidget(spin)
        row.addStretch()
        angle_options_layout.addLayout(row)

    # --- Output ---
    # The default returns the result at the input's pixel size so a separately
    # scanned scale bar keeps its meaning. Keeping the working size gives smoother
    # lines but the scale image must then be scaled by the same factor.
    window.keep_upscaled_checkbox = QCheckBox('Keep the upscaled size')
    window.keep_upscaled_checkbox.setChecked(False)
    window.keep_upscaled_checkbox.toggled.connect(window._warn_keep_upscaled)
    window.keep_upscaled_checkbox.setToolTip(
        'Off: the result has the pixel size and DPI of the input.\n'
        'On: the result keeps the upscaled size. The DPI value increases by the same factor.\n'
        'Load the scale image below. It is scaled by the same factor and saved with the result.')

    window.scale_image_path = None
    scale_layout = QHBoxLayout()
    window.load_scale_button = QPushButton('Load Scale Image...')
    window.load_scale_button.setToolTip('The scale bar image scanned with this drawing. It is saved with the result, scaled by the same factor.')
    window.load_scale_button.clicked.connect(window.load_scale_image)
    window.scale_image_label = QLabel('No scale image')
    window.scale_image_label.setStyleSheet("font-size: 10px; font-style: italic;")
    scale_layout.addWidget(window.load_scale_button)
    scale_layout.addWidget(window.scale_image_label)
    scale_layout.addStretch()

    # --- Debug ---
    window.debug_images_checkbox = QCheckBox('Show and save the debug images')
    window.debug_images_checkbox.setChecked(False)
    window.debug_images_checkbox.stateChanged.connect(window.toggle_debug_images)
    # Keep the old attributes for backward compatibility
    window.show_debug_images = window.debug_images_checkbox
    window.save_debug_images = window.debug_images_checkbox

    # --- Configuration ---
    # Every size threshold of the pipeline comes from a configuration file.
    config_layout = QHBoxLayout()
    window.load_config_button = QPushButton('Load Configuration...')
    window.load_config_button.setToolTip('A copy of the shipped config.yaml with your changes.')
    window.load_config_button.clicked.connect(window.load_configuration)
    window.reset_config_button = QPushButton('Reset to default')
    window.reset_config_button.setToolTip('Use the configuration file shipped with the package.')
    window.reset_config_button.clicked.connect(window.reset_configuration)
    config_layout.addWidget(window.load_config_button)
    config_layout.addWidget(window.reset_config_button)
    config_layout.addStretch()
    window.config_label = QLabel('Configuration: default')
    window.config_label.setStyleSheet("font-size: 10px; font-style: italic;")
    window.config_label.setWordWrap(True)

    # --- DPI ---
    window.current_dpi_label = QLabel("DPI of the image: none")
    window.current_dpi_label.setStyleSheet("font-size: 11px;")

    window.dpi_explanation_label = QLabel("The DPI of the input is kept when the file has one.")
    window.dpi_explanation_label.setStyleSheet("font-style: italic; font-size: 10px;")
    window.dpi_explanation_label.setWordWrap(True)

    # The choice below applies only to a file with no DPI tag, and is shown only then
    window.dpi_group = QButtonGroup(window)
    window.keep_unset_dpi = QRadioButton("No DPI value")
    window.use_custom_dpi = QRadioButton("Set the DPI:")
    window.dpi_group.addButton(window.keep_unset_dpi)
    window.dpi_group.addButton(window.use_custom_dpi)

    window.custom_dpi_spinner = QSpinBox()
    window.custom_dpi_spinner.setRange(72, 1200)
    window.custom_dpi_spinner.setValue(300)
    window.custom_dpi_spinner.setSuffix(" DPI")
    window.custom_dpi_spinner.setMaximumWidth(100)
    window.custom_dpi_spinner.setToolTip("Usual values: 96 (screen), 300 (print), 600 (high quality)")

    window.dpi_choice_widget = QWidget()
    dpi_choice_layout = QVBoxLayout(window.dpi_choice_widget)
    dpi_choice_layout.setContentsMargins(0, 0, 0, 0)
    dpi_choice_layout.setSpacing(4)
    custom_dpi_layout = QHBoxLayout()
    custom_dpi_layout.addWidget(window.use_custom_dpi)
    custom_dpi_layout.addWidget(window.custom_dpi_spinner)
    custom_dpi_layout.addStretch()
    dpi_choice_layout.addWidget(window.keep_unset_dpi)
    dpi_choice_layout.addLayout(custom_dpi_layout)
    window.dpi_choice_widget.setVisible(False)

    # Assemble the panel
    options_layout.addWidget(heading("Processing"))
    options_layout.addWidget(window.preserve_cortex_checkbox)
    options_layout.addWidget(window.measure_angles_checkbox)
    options_layout.addWidget(window.angle_options_widget)
    options_layout.addWidget(heading("Output"))
    options_layout.addWidget(window.keep_upscaled_checkbox)
    options_layout.addLayout(scale_layout)
    options_layout.addWidget(heading("Debug"))
    options_layout.addWidget(window.debug_images_checkbox)
    options_layout.addWidget(heading("Configuration"))
    options_layout.addLayout(config_layout)
    options_layout.addWidget(window.config_label)
    options_layout.addWidget(heading("DPI"))
    options_layout.addWidget(window.current_dpi_label)
    options_layout.addWidget(window.dpi_explanation_label)
    options_layout.addWidget(window.dpi_choice_widget)
    options_layout.addStretch()  # Push everything to top

    # The Options pane scrolls when the column is shorter than its content
    options_scroll = QScrollArea()
    options_scroll.setWidgetResizable(True)
    options_scroll.setFrameShape(QScrollArea.NoFrame)
    options_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    options_scroll.setWidget(options_group)
    return options_scroll


def build_image_panels(window):
    """The Input Image and Result panels side by side."""
    # Left side: Image panels (side by side)
    images_splitter = QSplitter(Qt.Horizontal)

    # Input image panel
    input_panel = QWidget()
    input_layout = QVBoxLayout(input_panel)
    input_layout.setContentsMargins(2, 2, 2, 2)
    input_layout.setSpacing(2)

    input_title = QLabel('Input Image')
    input_title.setAlignment(Qt.AlignCenter)
    input_title.setStyleSheet("font-weight: bold; font-size: 12px; padding: 2px;")

    window.input_image_display = CanvasWidget()
    window.input_image_display.setObjectName("input_image_display")
    window.input_image_display.setText("No image loaded")
    window.input_image_display.setMinimumSize(200, 200)

    input_layout.addWidget(input_title, 0)
    input_scroll = QScrollArea()
    input_scroll.setWidgetResizable(True)
    input_scroll.setWidget(window.input_image_display)
    input_layout.addWidget(input_scroll, 1)

    # Output image panel
    output_panel = QWidget()
    output_layout = QVBoxLayout(output_panel)
    output_layout.setContentsMargins(2, 2, 2, 2)
    output_layout.setSpacing(2)

    output_title = QLabel('Result')
    output_title.setAlignment(Qt.AlignCenter)
    output_title.setStyleSheet("font-weight: bold; font-size: 12px; padding: 2px;")

    window.canvas = arrow_integration.create_arrow_canvas()
    window.canvas.selection_changed.connect(
        lambda arrow: arrow_integration.show_arrow_style(window, arrow))
    window.canvas.setObjectName("output_canvas")
    window.canvas.setText("No processed image")
    window.canvas.setMinimumSize(200, 200)

    output_layout.addWidget(output_title, 0)
    output_scroll = QScrollArea()
    output_scroll.setWidgetResizable(True)
    output_scroll.setWidget(window.canvas)
    output_layout.addWidget(output_scroll, 1)

    # Add panels to images splitter
    images_splitter.addWidget(input_panel)
    images_splitter.addWidget(output_panel)
    images_splitter.setSizes([500, 500])
    images_splitter.setStretchFactor(0, 1)
    images_splitter.setStretchFactor(1, 1)
    return images_splitter


def build_status_row(window):
    """The log panel (hidden until asked for) and the status row under it."""
    # Bottom section: the log, hidden until the user asks for it, and one status row
    window.log_display = QTextEdit()
    window.log_display.setReadOnly(True)
    window.log_display.setMaximumHeight(140)
    window.log_display.setStyleSheet("font-family: 'Monaco', 'Courier New', monospace; font-size: 10px;")
    window.log_display.setVisible(False)

    window.status_label = QLabel('Ready')
    window.status_label.setStyleSheet("font-weight: bold; font-size: 11px;")

    window.progress_bar = QProgressBar()
    window.progress_bar.setTextVisible(False)
    window.progress_bar.setRange(0, 0)
    window.progress_bar.hide()
    window.progress_bar.setMaximumHeight(8)
    window.progress_bar.setMaximumWidth(240)

    window.log_toggle = QPushButton('Show log')
    window.log_toggle.setCheckable(True)
    window.log_toggle.setObjectName('log_toggle')  # a neutral checked state in the theme
    window.log_toggle.setToolTip('Show or hide the messages from the application and the pipeline.')
    window.log_toggle.toggled.connect(window.toggle_log)

    status_row = QHBoxLayout()
    status_row.setContentsMargins(4, 0, 4, 0)
    status_row.setSpacing(10)
    status_row.addWidget(window.status_label, 1)
    status_row.addWidget(window.progress_bar, 0)
    status_row.addWidget(window.log_toggle, 0)

    return status_row

