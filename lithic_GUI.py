import sys
import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QProgressBar, QMessageBox, QCheckBox, QSplitter,
    QTextEdit, QComboBox, QGroupBox, QScrollArea,
    QRadioButton, QButtonGroup, QSpinBox, QSlider,
    QColorDialog
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor,
    QPainterPath
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QPoint, QPointF, QLineF
)

# Import arrow annotation functionality
from arrow_annotations import ArrowCanvasWidget
import arrow_integration

# Available style options
# GUI_STYLES = ['Fusion', 'Windows', 'WindowsVista', 'Macintosh']

# Import your processing function
try:
    from ripple_remover import process_lithic_drawing_improved
    print("Successfully imported process_lithic_drawing_improved from ripple_remover")
except ImportError as e:
    print(f"Error importing from ripple_remover: {e}")
    print(f"Python path: {sys.path}")
    # Try to list the directory content
    import os
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    sys.exit(1)

class ProcessingThread(QThread):
    """Thread for processing images without freezing the UI"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, input_path, output_folder):
        super().__init__()
        self.input_path = input_path
        self.output_folder = output_folder

    def run(self):
        try:
            # Override print to capture progress updates
            def progress_print(msg):
                self.progress_signal.emit(msg)
                original_print(msg)

            # Store the original print function
            original_print = print
            import builtins
            builtins.print = progress_print

            # Run the processing function
            process_lithic_drawing_improved(self.input_path, self.output_folder)

            # Restore the original print function
            builtins.print = original_print

            # Signal completion
            self.finished_signal.emit(os.path.join(self.output_folder, '9_high_quality.png'))
        except Exception as e:
            self.progress_signal.emit(f"Error: {str(e)}")
            self.finished_signal.emit("")

class CanvasWidget(QLabel):
    """Custom canvas widget for drawing annotations"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.last_point = QPoint()
        self.drawing = False
        self.brush_active = False
        self.current_mouse_pos = QPoint(0, 0)
        self.input_image_path = None
        self.processed_image_path = None

        # Drawing properties
        self.brush_size = 5
        self.brush_color = Qt.black

        # Initialize empty pixmap
        self.setAlignment(Qt.AlignCenter)
        self.clear_canvas()

    def paintEvent(self, event):
        """Override paintEvent to draw brush cursor preview"""
        super().paintEvent(event)

        # Only draw cursor if brush is active and we have a base image
        if self.brush_active and hasattr(self, 'base_pixmap'):
            painter = QPainter(self)
            pen = QPen(self.brush_color)
            pen.setWidth(1)
            painter.setPen(pen)

            # Draw circle at current mouse position to show brush size
            painter.setBrush(Qt.transparent)
            painter.drawEllipse(self.current_mouse_pos,
                              self.brush_size // 2,
                              self.brush_size // 2)

    def clear_canvas(self):
        """Clear the canvas"""
        # Create a transparent pixmap
        self.annotation_pixmap = QPixmap(self.width(), self.height())
        self.annotation_pixmap.fill(Qt.transparent)
        self.setPixmap(self.annotation_pixmap)

    def set_base_image(self, pixmap):
        """Set the base image for annotation"""
        # Create a copy of the input pixmap
        self.base_pixmap = pixmap.copy()

        # Initialize annotation layer to match base image size
        self.annotation_pixmap = QPixmap(self.base_pixmap.size())
        self.annotation_pixmap.fill(Qt.transparent)

        # Combine base with empty annotations
        self.update_display()

    def update_display(self):
        """Update the display with base image + annotations"""
        if hasattr(self, 'base_pixmap'):
            # Create a copy of the base image
            result = self.base_pixmap.copy()

            # Draw annotations on top
            painter = QPainter(result)
            painter.drawPixmap(0, 0, self.annotation_pixmap)
            painter.end()

            # Scale for display within the widget bounds, maintaining aspect ratio
            if self.width() > 0 and self.height() > 0:  # Prevent division by zero
                display_width = self.width() - 10
                display_height = self.height() - 10

                w, h = result.width(), result.height()
                if w > 0 and h > 0:  # Prevent division by zero
                    scale = min(display_width / w, display_height / h)

                    if scale < 1:  # Only scale down, not up
                        result = result.scaled(int(w * scale), int(h * scale),
                                             Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Update the display
            self.setPixmap(result)

    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        if hasattr(self, 'base_pixmap'):
            self.update_display()

    def set_brush_properties(self, size, color):
        """Set brush size and color"""
        self.brush_size = size
        self.brush_color = color

    def set_brush_active(self, active):
        """Set whether the brush is active"""
        self.brush_active = active
        # Update cursor
        if active:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def adjustSize(self):
        """Adjust the widget size to fit the parent widget"""
        if self.parent():
            self.setMinimumWidth(self.parent().width() - 20)
            self.setMinimumHeight(self.parent().height() - 20)

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton and hasattr(self, 'base_pixmap') and self.brush_active:
            self.drawing = True
            self.last_point = self.map_to_pixmap(event.pos())

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        # Track current mouse position for cursor display
        self.current_mouse_pos = event.pos()

        # Trigger repaint to update cursor
        self.update()

        # Handle drawing if active
        if self.drawing and hasattr(self, 'base_pixmap') and self.brush_active:
            current_point = self.map_to_pixmap(event.pos())

            # Draw on the annotation layer
            painter = QPainter(self.annotation_pixmap)
            pen = QPen()
            pen.setWidth(self.brush_size)
            pen.setColor(self.brush_color)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, current_point)
            painter.end()

            self.last_point = current_point

            # Update the display
            self.update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def map_to_pixmap(self, pos):
        """Map screen coordinates to pixmap coordinates"""
        if not hasattr(self, 'base_pixmap'):
            return pos

        # Get widget and pixmap sizes
        label_size = self.size()
        pixmap_size = self.base_pixmap.size()

        # Calculate scaling and offset
        scale_w = pixmap_size.width() / label_size.width()
        scale_h = pixmap_size.height() / label_size.height()

        # Use minimum scale to maintain aspect ratio
        scale = max(scale_w, scale_h)

        # Calculate the centered pixmap position
        x_offset = (label_size.width() - pixmap_size.width() / scale) / 2
        y_offset = (label_size.height() - pixmap_size.height() / scale) / 2

        # Map the position
        pixmap_x = (pos.x() - x_offset) * scale
        pixmap_y = (pos.y() - y_offset) * scale

        return QPoint(int(pixmap_x), int(pixmap_y))

    def get_annotated_image(self):
        """Get the base image with annotations as QImage"""
        if hasattr(self, 'base_pixmap'):
            # Create a copy of the base image
            result = self.base_pixmap.copy()

            # Draw annotations on top
            painter = QPainter(result)
            painter.drawPixmap(0, 0, self.annotation_pixmap)
            painter.end()

            return result.toImage()
        return None

class LithicProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        QApplication.setStyle("Fusion")
        self.input_image_path = None
        self.output_folder = None
        self.processed_image_path = None
        self.debug_images = []  # To store paths to debug images
        self.debug_image_widgets = []  # To store debug image labels
        self.initUI()

    def initUI(self):
        # Main window setup
        self.setWindowTitle('Lithic Drawing Processor')
        self.setGeometry(100, 100, 1400, 900)
        self.resize(1200, 800)  # Smaller starting size
        self.setMinimumSize(800, 600)  # Set minimum size

        # Main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Top section: Controls
        top_controls = QHBoxLayout()

        # File controls group
        file_controls = QGroupBox("File Controls")
        file_layout = QHBoxLayout(file_controls)

        self.load_button = QPushButton('Load Image')
        self.load_button.clicked.connect(self.load_image)
        self.process_button = QPushButton('Process Image')
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        self.save_button = QPushButton('Save Result')
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        self.exit_button = QPushButton('Exit')
        self.exit_button.clicked.connect(self.close)

        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.process_button)
        file_layout.addWidget(self.save_button)
        file_layout.addWidget(self.exit_button)

        # Options group
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout(options_group)

        self.show_debug_images = QCheckBox('Show Debug Images')
        self.show_debug_images.setChecked(True)
        self.show_debug_images.stateChanged.connect(self.toggle_debug_images)

        options_layout.addWidget(self.show_debug_images)

        # Add groups to top controls
        top_controls.addWidget(file_controls, 3)
        top_controls.addWidget(options_group, 2)

        # Add top controls to main layout
        main_layout.addLayout(top_controls)

        # Middle section: Image display and debug
        middle_section = QSplitter(Qt.Horizontal)

        # Left side with images and tools
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # Drawing tools section
        drawing_tools = QGroupBox("Drawing Tools")
        drawing_layout = QHBoxLayout(drawing_tools)

        # Brush color selection
        color_group = QButtonGroup(self)
        self.white_brush = QRadioButton("White")
        self.white_brush.setChecked(True)
        self.black_brush = QRadioButton("Black")
        color_group.addButton(self.white_brush)
        color_group.addButton(self.black_brush)
        color_group.buttonClicked.connect(self.update_brush)

        # Brush size spinbox with up/down buttons
        size_label = QLabel("Brush Size:")
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setMinimum(1)
        self.brush_size_spin.setMaximum(20)
        self.brush_size_spin.setValue(5)
        self.brush_size_spin.valueChanged.connect(self.update_brush)
        self.brush_button = QPushButton("Activate Brush")
        self.brush_button.setCheckable(True)
        self.brush_button.setChecked(False)  # Start with brush inactive
        self.brush_button.clicked.connect(self.toggle_brush)

        # Clear annotations button
        self.clear_annotations_button = QPushButton("Clear Brush")
        self.clear_annotations_button.clicked.connect(self.clear_annotations)
        self.clear_annotations_button.setEnabled(False)

        # Add widgets to drawing layout
        drawing_layout.addWidget(self.brush_button)
        drawing_layout.addWidget(QLabel("Brush Color:"))
        drawing_layout.addWidget(self.white_brush)
        drawing_layout.addWidget(self.black_brush)
        drawing_layout.addStretch(1)  # Add flexible space
        drawing_layout.addWidget(size_label)
        drawing_layout.addWidget(self.brush_size_spin)
        drawing_layout.addStretch(1)  # Add flexible space
        drawing_layout.addWidget(self.clear_annotations_button)

        # Add drawing tools to left layout
        left_layout.addWidget(drawing_tools)

        # Add arrow annotation controls - THIS IS THE NEW SECTION
        arrow_tools = arrow_integration.setup_arrow_controls(self)
        left_layout.addWidget(arrow_tools)

        # Images splitter (vertical) - for input and output images
        images_splitter = QSplitter(Qt.Vertical)

        # Input image panel
        input_panel = QWidget()
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(5, 5, 5, 5)

        input_title = QLabel('Input Image')
        input_title.setAlignment(Qt.AlignCenter)
        input_title.setStyleSheet("font-weight: bold; font-size: 14px;")

        # Use custom canvas widget for input image
        self.input_image_display = CanvasWidget()
        self.input_image_display.setObjectName("input_image_display")
        self.input_image_display.setText("No image loaded")
        self.input_image_display.setStyleSheet("border: 1px solid #cccccc; background-color: #f5f5f5;")
        self.input_image_display.setMinimumSize(400, 300)

        input_layout.addWidget(input_title)
        input_scroll = QScrollArea()
        input_scroll.setWidgetResizable(True)
        input_scroll.setWidget(self.input_image_display)
        input_layout.addWidget(input_scroll)

        # Output/Annotation image panel with canvas
        output_panel = QWidget()
        output_layout = QVBoxLayout(output_panel)
        output_layout.setContentsMargins(5, 5, 5, 5)

        output_title = QLabel('Processed Image / Arrow Annotations')
        output_title.setAlignment(Qt.AlignCenter)
        output_title.setStyleSheet("font-weight: bold; font-size: 14px;")

        # Use ArrowCanvasWidget for output display - THIS IS THE UPDATED PART
        self.canvas = arrow_integration.create_arrow_canvas()
        self.canvas.setObjectName("output_canvas")
        self.canvas.setText("No processed image")

        output_layout.addWidget(output_title)
        # Create scroll area for output/annotation canvas
        output_scroll = QScrollArea()
        output_scroll.setWidgetResizable(True)
        output_scroll.setWidget(self.canvas)
        output_layout.addWidget(output_scroll)

        # Add input and output panels to images splitter
        images_splitter.addWidget(input_panel)
        images_splitter.addWidget(output_panel)

        # Set initial sizes for the image panels (equal sizes)
        images_splitter.setSizes([1, 1])

        # Add images splitter to left layout
        left_layout.addWidget(images_splitter)

        # Right side: Debug images
        self.debug_panel = QWidget()
        debug_outer_layout = QVBoxLayout(self.debug_panel)
        debug_outer_layout.setContentsMargins(5, 5, 5, 5)

        debug_title = QLabel('Processing Steps')
        debug_title.setAlignment(Qt.AlignCenter)
        debug_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        debug_outer_layout.addWidget(debug_title)

        # Scroll area for debug images
        debug_scroll = QScrollArea()
        debug_scroll.setWidgetResizable(True)
        debug_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        debug_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.debug_content = QWidget()
        self.debug_layout = QVBoxLayout(self.debug_content)

        debug_scroll.setWidget(self.debug_content)
        debug_outer_layout.addWidget(debug_scroll)

        # Add panels to middle section
        middle_section.addWidget(left_panel)
        middle_section.addWidget(self.debug_panel)
        middle_section.setSizes([700, 700])  # Equal initial sizes
        middle_section.setStretchFactor(0, 2)  # Give more stretch weight to the left panel
        middle_section.setStretchFactor(1, 1)  # Less stretch weight to the debug panel

        # Add middle section to main layout
        main_layout.addWidget(middle_section, 5)  # Give it more space

        # Bottom section: Progress bar and log
        bottom_section = QSplitter(Qt.Vertical)

        # Processing log
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(100)
        self.log_display.setStyleSheet("font-family: monospace;")
        log_layout.addWidget(self.log_display)

        # Progress area
        progress_group = QGroupBox("Processing Status")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.hide()

        self.status_label = QLabel('Ready')
        self.status_label.setStyleSheet("font-weight: bold;")

        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)

        # Add log and progress to bottom section
        bottom_section.addWidget(log_group)
        bottom_section.addWidget(progress_group)
        bottom_section.setSizes([150, 50])  # More space for log

        # Add bottom section to main layout
        main_layout.addWidget(bottom_section, 1)  # Less space than middle

        # Set central widget
        self.setCentralWidget(central_widget)

        # Hide debug panel initially if not checked
        self.debug_panel.setVisible(self.show_debug_images.isChecked())

        # Initialize brush
        self.update_brush()

        # Initial log message
        self.log("Lithic Drawing Processor started. Ready to load images.")

    def update_brush(self):
        """Update brush properties based on UI settings"""
        # Get brush size
        size = self.brush_size_spin.value()

        # Get brush color
        color = Qt.black if self.black_brush.isChecked() else Qt.white

        # Update input canvas brush
        self.input_image_display.set_brush_properties(size, color)

        # Log the change
        color_name = "Black" if color == Qt.black else "White"
        self.log(f"Brush settings updated: Color = {color_name}, Size = {size}")

    def clear_annotations(self):
        """Clear all annotations from the input canvas"""
        if hasattr(self, 'input_image_display'):
            # Reset the canvas but keep the base image
            if hasattr(self.input_image_display, 'base_pixmap'):
                # Create a new empty annotation layer
                self.input_image_display.annotation_pixmap = QPixmap(
                    self.input_image_display.base_pixmap.size())
                self.input_image_display.annotation_pixmap.fill(Qt.transparent)

                # Update the display
                self.input_image_display.update_display()
                self.log("Annotations cleared")

    def toggle_brush(self):
        """Toggle brush tool activation"""
        is_active = self.brush_button.isChecked()

        if is_active:
            self.brush_button.setText("Brush Active")
            self.brush_button.setStyleSheet("background-color: #aaffaa;")
            self.input_image_display.set_brush_active(True)
            self.log("Brush tool activated")
        else:
            self.brush_button.setText("Activate Brush")
            self.brush_button.setStyleSheet("")
            self.input_image_display.set_brush_active(False)
            self.log("Brush tool deactivated")

    # Dropdown to change GUI style
    # def change_style(self, index):
    #     """Change the application style"""
    #     style_name = GUI_STYLES[index]
    #     QApplication.setStyle(style_name)
    #     self.log(f"Changed GUI style to {style_name}")

    def toggle_debug_images(self, state):
        """Show or hide debug images panel"""
        self.debug_panel.setVisible(state == Qt.Checked)

    def log(self, message):
        """Add a message to the log display"""
        self.log_display.append(message)
        # Auto-scroll to bottom
        scroll_bar = self.log_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def load_image(self):
        """Load an image using native file dialog"""
        # Create a synchronous file dialog with explicit options
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open Image",
            directory="",
            filter="Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
            options=QFileDialog.DontUseNativeDialog  # Use Qt's dialog instead of the OS native one
        )

        if file_path:
            # Process the selected file
            self.input_image_path = file_path

            # Load and crop the image to content
            img = cv2.imread(file_path)
            if img is not None:
                # Crop image to content with padding
                from ripple_remover import crop_to_content
                cropped_img = crop_to_content(img, padding=10)

                # Save the cropped version
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                self.output_folder = os.path.join('output', base_name)
                os.makedirs(self.output_folder, exist_ok=True)

                # Save cropped version for processing
                cropped_path = os.path.join(self.output_folder, "cropped_input.png")
                cv2.imwrite(cropped_path, cropped_img)

                # Use cropped version for display and processing
                self.display_image(cropped_path, self.input_image_display)
                self.input_image_path = cropped_path  # Update path to cropped version

                self.process_button.setEnabled(True)
                self.status_label.setText(f'Loaded and cropped: {os.path.basename(file_path)}')
                self.log(f"Loaded image: {file_path}")
                self.log(f"Cropped to content size: {cropped_img.shape[1]}x{cropped_img.shape[0]}")
                self.clear_annotations_button.setEnabled(True)

                # Clear previous debug images
                self.clear_debug_images()

                # Reset output image/canvas
                self.canvas.clear_canvas()
                self.canvas.setText("No processed image")
                self.save_button.setEnabled(False)
                self.clear_annotations_button.setEnabled(False)

    def clear_debug_images(self):
        """Clear all debug images from the panel"""
        # Clear the list of debug image paths
        self.debug_images = []

        # Remove all widgets from the debug layout
        for i in reversed(range(self.debug_layout.count())):
            widget = self.debug_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Clear the list of debug image widgets
        self.debug_image_widgets = []

    def process_image(self):
        if not self.input_image_path:
            return

        # Get the annotated input image
        if hasattr(self.input_image_display, 'get_annotated_image'):
            annotated_image = self.input_image_display.get_annotated_image()
            if annotated_image:
                # Convert QImage to numpy array
                width = annotated_image.width()
                height = annotated_image.height()
                ptr = annotated_image.bits()
                ptr.setsize(height * width * 4)  # RGBA
                arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

                # Convert RGBA to BGR (OpenCV format)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

                # Convert to grayscale
                grayscale = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

                # Save the annotated image to use as input
                annotated_path = os.path.join(self.output_folder, "annotated_input.png")
                cv2.imwrite(annotated_path, grayscale)

                # Use this as the new input path
                self.input_image_path = annotated_path

                self.log(f"Using annotated image for processing: {annotated_path}")

        # Disable controls during processing
        self.process_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.clear_annotations_button.setEnabled(False)

        # Show progress indicators
        self.progress_bar.show()
        self.status_label.setText('Processing...')

        # Clear previous debug images
        self.clear_debug_images()

        # Clear the log
        self.log_display.clear()
        self.log(f"Processing image: {self.input_image_path}")

        # Start processing thread
        self.processing_thread = ProcessingThread(self.input_image_path, self.output_folder)
        self.processing_thread.progress_signal.connect(self.update_progress)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.start()

    def update_progress(self, message):
            self.status_label.setText(message)
            self.log(message)

    def processing_finished(self, output_path):
        # Hide progress indicators
        self.progress_bar.hide()

        # Re-enable controls
        self.load_button.setEnabled(True)
        self.process_button.setEnabled(True)

        if output_path:
            self.processed_image_path = output_path
            self.log(f"Processing complete. Result path: {output_path}")

            # Debug dimensions for processed image
            if os.path.exists(output_path):
                proc_img = cv2.imread(output_path)
                if proc_img is not None:
                    proc_h, proc_w = proc_img.shape[:2]
                    self.log(f"DIMENSIONS: Processed image: {proc_w}x{proc_h}")
                else:
                    self.log(f"ERROR: Could not read processed image: {output_path}")
            else:
                self.log(f"ERROR: Processed image file does not exist: {output_path}")

            # Display the processed image
            self.display_image(output_path, self.canvas)
            # Enable arrow tools
            arrow_integration.enable_arrow_controls(self)
            self.log("You can now add arrows to the processed image (Alt+drag to resize, Shift+drag to rotate)")

            self.save_button.setEnabled(True)
            self.clear_annotations_button.setEnabled(True)
            self.status_label.setText('Processing complete!')
            self.log("You can draw on the input image with the brush tools")

            # Load debug images
            self.load_debug_images()
        else:
            self.status_label.setText('Processing failed!')
            self.log("ERROR: Processing failed!")
            QMessageBox.critical(self, 'Error', 'Image processing failed!')

    def load_debug_images(self):
        """Load all debug images from the output folder into the debug panel"""
        debug_files = [
            '1_original_image.png',
            '2_skeleton.png',
            '3_endpoints_junctions.png',
            '4_labeled_segments.png',
            '5_ripple_identification.png',
            '6_skeleton_cleaned.png',
            '7_final_cleaned.png',
            '8_improved_quality.png',
            '9_high_quality.png',
            # '10_comparison_all.png'
        ]

        for debug_file in debug_files:
            file_path = os.path.join(self.output_folder, debug_file)
            if os.path.exists(file_path):
                self.debug_images.append(file_path)

                # Create a container for each debug image
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(5, 10, 5, 10)

                # Image title based on filename
                title = debug_file.replace('.png', '').replace('_', ' ').title()
                image_title = QLabel(title)
                image_title.setAlignment(Qt.AlignCenter)
                image_title.setStyleSheet("font-weight: bold;")

                # Image display
                image_label = QLabel()
                image_label.setAlignment(Qt.AlignCenter)
                image_label.setStyleSheet("border: 1px solid #dddddd; background-color: white;")

                # Load and display the image
                self.display_debug_image(file_path, image_label)

                container_layout.addWidget(image_title)
                container_layout.addWidget(image_label)

                # Add to debug panel
                self.debug_layout.addWidget(container)
                self.debug_image_widgets.append(container)

        # Add a spacer at the end
        self.debug_layout.addStretch()

    def resizeEvent(self, event):
        """Handle window resize event to adjust image displays"""
        super().resizeEvent(event)

        # Refresh displayed images if they exist
        if hasattr(self, 'input_image_path') and self.input_image_path and hasattr(self, 'input_image_display'):
            if self.input_image_display.pixmap() and not self.input_image_display.pixmap().isNull():
                self.display_image(self.input_image_path, self.input_image_display)

        # Refresh processed image if it exists
        if hasattr(self, 'processed_image_path') and self.processed_image_path and hasattr(self, 'canvas'):
            if hasattr(self.canvas, 'base_pixmap') and not self.canvas.base_pixmap.isNull():
                # Update the canvas display when window is resized
                self.canvas.update_display()

    def display_debug_image(self, image_path, label_widget):
        """Display a debug image in the given label widget"""
        img = cv2.imread(image_path)
        if img is None:
            label_widget.setText("Failed to load image")
            return

        # Convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize for display in the debug panel
        h, w = img_rgb.shape[:2]
        max_width = 350
        if w > max_width:
            scale = max_width / w
            new_width, new_height = int(w * scale), int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            h, w = new_height, new_width

        # Create QImage and QPixmap for display
        q_img = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Set pixmap to label
        label_widget.setPixmap(pixmap)

    def save_result(self):
        """Save the processed image with annotations"""
        if not self.processed_image_path:
            return

        # Use native file dialog
        options = QFileDialog.Options()
        file_path, filter_used = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg)",
            options=QFileDialog.DontUseNativeDialog
        )

        if file_path:
            # Check and enforce file extension based on filter selected
            if filter_used == "PNG Files (*.png)" and not file_path.lower().endswith('.png'):
                file_path += '.png'
            elif filter_used == "JPEG Files (*.jpg)" and not file_path.lower().endswith(('.jpg', '.jpeg')):
                file_path += '.jpg'
            # Default to PNG if extension is missing and filter doesn't indicate format
            elif not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                file_path += '.png'

            # Get the annotated image from the canvas
            final_image = arrow_integration.get_image_with_arrows(self.canvas)

            if final_image:
                try:
                    # Convert QImage to numpy array
                    width = final_image.width()
                    height = final_image.height()
                    ptr = final_image.bits()
                    ptr.setsize(height * width * 4)  # RGBA
                    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

                    # Convert RGBA to BGR (OpenCV format)
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

                    # Save the image
                    success = cv2.imwrite(file_path, bgr)

                    if success:
                        self.status_label.setText(f'Saved to: {os.path.basename(file_path)}')
                        self.log(f"Processed image saved to: {file_path}")
                    else:
                        self.log(f"ERROR: Failed to save image to {file_path}")
                        QMessageBox.critical(self, 'Error', f'Failed to save image to {file_path}')

                except Exception as e:
                    self.log(f"ERROR saving file: {str(e)}")
                    QMessageBox.critical(self, 'Error', f'Error saving file: {str(e)}')

    def display_image(self, image_path, display_widget):
        """Load and display an image in the given widget, ensuring it fits within bounds"""
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return

        # Convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create QImage and QPixmap
        h, w = img_rgb.shape[:2]
        q_img = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # If the widget is a CanvasWidget or ArrowCanvasWidget, use set_base_image
        if isinstance(display_widget, CanvasWidget) or isinstance(display_widget, ArrowCanvasWidget):
            display_widget.set_base_image(pixmap)
        else:
            # For regular QLabel, do scaling and use setPixmap
            display_width = display_widget.width() - 10
            display_height = display_widget.height() - 10

            # Calculate scaling factor to fit within display
            scale = min(display_width / w, display_height / h)

            # Calculate new dimensions
            new_width = int(w * scale)
            new_height = int(h * scale)

            # Scale pixmap
            scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Update display
            display_widget.setPixmap(scaled_pixmap)

        # Log the scaling
        # self.log(f"Displayed image {os.path.basename(image_path)} - Original: {w}x{h}")

        # Clear arrows when a new image is loaded in the output canvas
        if display_widget == self.canvas:
            arrow_integration.clear_arrows_on_new_image(self)

    def resizeEvent(self, event):
        """Handle window resize event to adjust image displays"""
        super().resizeEvent(event)

        # Refresh displayed images if they exist
        if hasattr(self, 'input_image_path') and self.input_image_path and hasattr(self, 'input_image_display'):
            if self.input_image_display.pixmap() and not self.input_image_display.pixmap().isNull():
                self.display_image(self.input_image_path, self.input_image_display)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LithicProcessorGUI()
    window.show()
    sys.exit(app.exec_())