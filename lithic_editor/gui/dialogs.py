"""
The dialogs of the main window: the DPI question for a file with no DPI tag, and the
upscaling offer with its warning about the kept size.
"""

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


def upscaled_size_warning(factor=None):
    """The warning shown when the user selects to keep the upscaled size."""
    if factor:
        size_line = (f"The saved result is {factor}x the pixel size of the input image. "
                     f"The DPI value is {factor}x the input DPI.")
    else:
        size_line = ("The saved result has more pixels than the input image. "
                     "The DPI value increases by the same factor.")
    return (f"{size_line}\n\n"
            "The result and the input image do not have the same pixel size. "
            "A scale image scanned with the input does not match the result.\n\n"
            "Load the scale image before you process. Lithic Editor then saves a copy "
            "of the scale image, scaled by the same factor, with the result.\n\n"
            "Clear this option to save the result at the pixel size and DPI of the input.")


def warn_upscaled_size(parent, factor=None):
    QMessageBox.warning(parent, "Keep the upscaled size", upscaled_size_warning(factor))


class DPISelectionDialog(QDialog):
    """Dialog for selecting DPI when metadata is missing"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Specify Image DPI")
        self.setModal(True)
        self.selected_dpi = None

        layout = QVBoxLayout()

        # Info label
        info_label = QLabel("The image has no DPI value.\nSelect the DPI of the image:")
        layout.addWidget(info_label)

        # DPI buttons
        button_layout = QHBoxLayout()
        for dpi in [72, 96, 150, 200]:
            btn = QPushButton(str(dpi))
            btn.clicked.connect(lambda checked, d=dpi: self.select_dpi(d))
            button_layout.addWidget(btn)
        layout.addLayout(button_layout)

        # Custom DPI input
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("Custom:"))
        self.custom_input = QLineEdit()
        self.custom_input.setPlaceholderText("Enter DPI")
        custom_layout.addWidget(self.custom_input)
        custom_btn = QPushButton("Use Custom")
        custom_btn.clicked.connect(self.use_custom_dpi)
        custom_layout.addWidget(custom_btn)
        layout.addLayout(custom_layout)

        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

        self.setLayout(layout)

    def select_dpi(self, dpi):
        self.selected_dpi = dpi
        self.accept()

    def use_custom_dpi(self):
        try:
            custom_dpi = int(self.custom_input.text())
            if 50 <= custom_dpi <= 2400:  # Reasonable range
                self.selected_dpi = custom_dpi
                self.accept()
            else:
                QMessageBox.warning(self, "Invalid DPI", "The DPI must be between 50 and 2400.")
        except ValueError:
            QMessageBox.warning(self, "Invalid DPI", "Enter a whole number.")


class UpscalingDialog(QDialog):
    """Dialog that offers upscaling, with the model and the output size choices"""

    def __init__(self, geometry, factor, parent=None, keep_upscaled=False, default_model='espcn',
                 has_scale_image=False):
        super().__init__(parent)
        self.setWindowTitle("Upscale Image")
        self.setModal(True)
        self.geometry_info = geometry
        self.factor = factor
        self.upscale_confirmed = False
        self.selected_model = default_model
        self.keep_upscaled = keep_upscaled
        self.has_scale_image = has_scale_image

        layout = QVBoxLayout()

        warning_text = (f"Measured {geometry.describe()}.\n"
                        f"For lines this thin or this near to each other, processing at {factor}x "
                        f"gives a better result.\n\n"
                        f"Upscale {factor}x for processing?")
        warning_label = QLabel(warning_text)
        layout.addWidget(warning_label)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("ESPCN (Recommended)", "espcn")
        self.model_combo.addItem("FSRCNN (Alternative)", "fsrcnn")
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(default_model)))
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # Output size: the choice is made here, where its effect is explained, and is
        # kept in step with the checkbox in the options panel
        self.keep_checkbox = QCheckBox("Keep the upscaled size")
        self.keep_checkbox.setChecked(keep_upscaled)
        self.keep_checkbox.toggled.connect(self._update_size_text)
        self.keep_checkbox.toggled.connect(self._warn_if_kept)
        layout.addWidget(self.keep_checkbox)
        self.size_label = QLabel()
        self.size_label.setWordWrap(True)
        self.size_label.setStyleSheet("font-style: italic;")
        layout.addWidget(self.size_label)
        self._update_size_text(keep_upscaled)

        # Buttons
        button_layout = QHBoxLayout()
        yes_btn = QPushButton("Upscale")
        yes_btn.clicked.connect(self.confirm_upscale)
        no_btn = QPushButton("Do not upscale")
        no_btn.clicked.connect(self.reject)
        button_layout.addWidget(yes_btn)
        button_layout.addWidget(no_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _warn_if_kept(self, keep):
        if keep:
            warn_upscaled_size(self, self.factor)

    def _update_size_text(self, keep):
        if keep:
            text = (f"The result is {self.factor}x the input size. "
                    f"The DPI value is {self.factor}x the input DPI.")
            if self.has_scale_image:
                text += " The scale image is scaled by the same factor and saved with the result."
            else:
                text += (" No scale image is loaded: scale a separate scale bar image "
                         f"{self.factor}x before you measure.")
        else:
            text = "The result has the pixel size and DPI of the input."
        self.size_label.setText(text)

    def confirm_upscale(self):
        self.upscale_confirmed = True
        self.selected_model = self.model_combo.currentData()
        self.keep_upscaled = self.keep_checkbox.isChecked()
        self.accept()
