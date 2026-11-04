"""
The Processing Steps panel: one picture for each debug image of the last run.
"""

import os

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

# The debug images in pipeline order, as ``<step>_<base>_<name>.png``
DEBUG_STEPS = (
    ('0', 'input'), ('0a', 'upscaled'), ('0b', 'upscaled_scale'),
    ('1', 'original_image'), ('1b', 'smoothed'), ('1c', 'binary_thresholded'),
    ('2a', 'structural_only'), ('2b', 'cortex_mask'),
    ('3', 'skeleton'), ('4', 'endpoints_junctions'), ('5', 'labeled_segments'),
    ('6', 'ripple_identification'), ('6a', 'striking_angles'),
    ('7', 'skeleton_cleaned'), ('7a', 'endpoint_filtering'),
    ('8', 'final_cleaned'), ('9', 'restored'), ('10', 'striking_arrows'),
)

# Short titles for the panel
TITLES = {
    'input': 'Input', 'upscaled': 'Upscaled', 'upscaled_scale': 'Scale Bar',
    'original_image': 'Original', 'smoothed': 'Smoothed', 'binary_thresholded': 'Binary Threshold',
    'structural_only': 'Structural Only', 'cortex_mask': 'Cortex Mask', 'skeleton': 'Skeleton',
    'endpoints_junctions': 'Endpoints', 'labeled_segments': 'Segments',
    'ripple_identification': 'Ripple ID', 'striking_angles': 'Striking Angles',
    'skeleton_cleaned': 'Cleaned', 'endpoint_filtering': 'Filtered', 'final_cleaned': 'Final',
    'restored': 'Restored', 'striking_arrows': 'Striking Arrows',
}

MAX_WIDTH = 350


class DebugPanel(QWidget):
    """A titled, scrolling column of the debug images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_paths = []
        self._containers = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        title = QLabel('Processing Steps')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 12px; padding: 2px;")
        layout.addWidget(title, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        scroll.setWidget(self.content)
        layout.addWidget(scroll, 1)

        self.setMaximumWidth(420)

    def clear(self):
        """Remove every picture."""
        self.image_paths = []
        for i in reversed(range(self.content_layout.count())):
            widget = self.content_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self._containers = []

    def show_images(self, folder, base_name):
        """Show the debug images of ``base_name`` found in ``folder``. Return their paths."""
        for step, name in DEBUG_STEPS:
            path = os.path.join(folder, f'{step}_{base_name}_{name}.png')
            if not os.path.exists(path):
                continue
            self.image_paths.append(path)
            self._add_picture(path, f"{step}. {TITLES[name]}")
        self.content_layout.addStretch()
        return list(self.image_paths)

    def _add_picture(self, path, title):
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(5, 10, 5, 10)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; padding: 2px;")
        title_label.setWordWrap(True)
        title_label.setMinimumHeight(30)

        picture = QLabel()
        picture.setAlignment(Qt.AlignCenter)
        picture.setObjectName("debug_image")  # the border comes from the theme
        pixmap = load_pixmap(path)
        if pixmap is None:
            picture.setText("Failed to load image")
        else:
            picture.setPixmap(pixmap)

        container_layout.addWidget(title_label)
        container_layout.addWidget(picture)
        self.content_layout.addWidget(container)
        self._containers.append(container)


def load_pixmap(path, max_width=MAX_WIDTH):
    """Read an image file as a pixmap no wider than ``max_width``, or None when it cannot be read."""
    img = cv2.imread(path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    if w > max_width:
        scale = max_width / w
        w, h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)
    q_img = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(q_img.copy())
