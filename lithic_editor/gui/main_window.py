import os
import sys

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QColorDialog,
    QDialog,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from lithic_editor.annotations import integration as arrow_integration

# Import arrow annotation functionality
from lithic_editor.annotations.arrows import ArrowCanvasWidget
from lithic_editor.config import (
    AngleSettings,
    ConfigError,
    default_config_path,
    load_config,
    replace_values,
)
from lithic_editor.gui.brush_canvas import CanvasWidget
from lithic_editor.gui.debug_panel import DebugPanel
from lithic_editor.gui.dialogs import DPISelectionDialog, UpscalingDialog, warn_upscaled_size
from lithic_editor.gui.panes import build_control_column, build_image_panels, build_status_row
from lithic_editor.gui.processing_thread import ProcessingThread
from lithic_editor.gui.theme import apply_theme
from lithic_editor.processing.resolution import choose_upscale_factor, measure_line_geometry
from lithic_editor.processing.striking_angle import arrow_size_for, widget_angle_from_compass


class LithicProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        apply_theme(QApplication.instance())
        self.input_image_path = None
        self.output_folder = None
        self.processed_image_path = None
        self.debug_images = []  # To store paths to debug images
        self.initUI()

    def get_output_dpi(self):
        """Get the DPI settings based on user selection"""
        if hasattr(self, 'image_dpi') and self.image_dpi:
            # If image has DPI, use it
            return self.image_dpi

        # No DPI in image, use selected option
        if self.keep_unset_dpi.isChecked():
            return None  # Leave unset
        elif self.use_custom_dpi.isChecked():
            dpi_value = self.custom_dpi_spinner.value()
            return (dpi_value, dpi_value)  # Custom value

        # No option selected, leave unset as safest default
        return None

    def initUI(self):
        # Main window setup
        self.setWindowTitle('Lithic Editor and Annotator')
        self.setGeometry(100, 100, 1600, 900)
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)

        # Main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        left_column = build_control_column(self)
        images_splitter = build_image_panels(self)
        self.debug_panel = DebugPanel()

        # Main content area: the control column, the two images, and the Processing Steps panel
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.addWidget(left_column)
        content_splitter.addWidget(images_splitter)
        content_splitter.addWidget(self.debug_panel)
        content_splitter.setCollapsible(0, False)
        content_splitter.setCollapsible(1, False)
        content_splitter.setSizes([340, 1060, 360])
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setStretchFactor(2, 0)
        main_layout.addWidget(content_splitter, 1)

        status_row = build_status_row(self)
        main_layout.addWidget(self.log_display, 0)
        main_layout.addLayout(status_row, 0)

        self.setCentralWidget(central_widget)

        # Hide debug panel initially if not checked
        self.debug_panel.setVisible(self.show_debug_images.isChecked())

        # Initialize brush
        self.update_brush()

        # Initial log message
        self.log("Lithic Editor and Annotator started. Load an image.")
        self.apply_configuration(None)

    def update_brush(self):
        """Update brush properties based on UI settings"""
        # Get brush size
        size = self.brush_size_spin.value()

        color = QColor(self.brush_color)
        self.input_image_display.set_brush_properties(size, color)
        self.log(f"Brush settings: Color = {color.name()}, Size = {size}")

    def select_brush_color(self):
        """Open a colour dialog for the brush"""
        color = QColorDialog.getColor(self.brush_color, self, 'Brush Color')
        if color.isValid():
            self.brush_color = color
            arrow_integration.show_color_swatch(self.brush_color_button, color)
            self.update_brush()

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
            self.input_image_display.set_brush_active(True)
            self.log("Brush tool activated")
        else:
            self.brush_button.setText("Activate Brush")
            self.input_image_display.set_brush_active(False)
            self.log("Brush tool deactivated")

    # Dropdown to change GUI style
    # def change_style(self, index):
    #     """Change the application style"""
    #     style_name = GUI_STYLES[index]
    #     QApplication.setStyle(style_name)
    #     self.log(f"Changed GUI style to {style_name}")

    def toggle_log(self, checked):
        """Show or hide the processing log"""
        self.log_display.setVisible(checked)
        self.log_toggle.setText('Hide log' if checked else 'Show log')

    def toggle_debug_images(self, state):
        """Show or hide debug images panel"""
        self.debug_panel.setVisible(state == Qt.Checked)

    def log(self, message):
        """Add a message to the log display"""
        self.log_display.append(message)
        # Auto-scroll to bottom
        scroll_bar = self.log_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def _warn_keep_upscaled(self, checked):
        """Warn about the size difference when the user selects to keep the upscaled size"""
        if checked:
            warn_upscaled_size(self)

    def update_angle_options_state(self, checked):
        """The angle thresholds apply only when the angles are measured"""
        self.angle_options_widget.setEnabled(bool(checked))

    def set_angle_options(self, angles):
        """Show the values of an ``AngleSettings`` in the three spin boxes"""
        self.min_ripples_spin.setValue(angles.min_ripples)
        self.min_arc_depth_spin.setValue(angles.min_arc_depth_px)
        self.max_cortex_dots_spin.setValue(angles.max_cortex_dots)

    def angle_options(self):
        """The three angle thresholds from the panel, as configuration keys"""
        return {
            'min_ripples': self.min_ripples_spin.value(),
            'min_arc_depth_px': self.min_arc_depth_spin.value(),
            'max_cortex_dots': self.max_cortex_dots_spin.value(),
        }

    def processing_config(self):
        """The configuration for the next run: the loaded file with the panel's angle values"""
        base = getattr(self, 'config', None) or load_config(default_config_path())
        return replace_values(base, 'angles', source=f"{base.source} with the panel values",
                              **self.angle_options())

    def apply_configuration(self, path):
        """Read a configuration file (None: environment variable or shipped default) and show it"""
        try:
            self.config = load_config(path)
        except ConfigError as e:
            self.config = None
            self.config_label.setText('Configuration: defaults (the file was not read)')
            self.set_angle_options(AngleSettings())
            self.log(f"Warning: the configuration file was not read: {e}. The defaults are used.")
            QMessageBox.warning(self, 'Configuration', f'The configuration file was not read:\n{e}')
            return
        self.set_angle_options(self.config.angles)
        source = self.config.source
        shown = 'default' if source == str(default_config_path()) else os.path.basename(source)
        self.config_label.setText(f'Configuration: {shown}')
        self.config_label.setToolTip(source)
        self.log(f"Configuration: {source}")

    def load_configuration(self):
        """Choose a configuration file"""
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open Configuration",
            directory="",
            filter="YAML Files (*.yaml *.yml)",
            options=QFileDialog.DontUseNativeDialog
        )
        if file_path:
            self.apply_configuration(file_path)

    def reset_configuration(self):
        """Use the configuration file shipped with the package"""
        self.apply_configuration(default_config_path())

    def load_scale_image(self):
        """Choose the scale bar image scanned with the drawing, so it can be scaled with it"""
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open Scale Image",
            directory="",
            filter="Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
            options=QFileDialog.DontUseNativeDialog
        )
        if file_path:
            self.scale_image_path = file_path
            self.scale_image_label.setText(os.path.basename(file_path))
            self.log(f"Scale image: {file_path}")

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
            # Store original filename for debug naming (before any processing changes it)
            self.original_filename = os.path.splitext(os.path.basename(file_path))[0]

            # Process the selected file
            self.input_image_path = file_path

            # Read metadata with PIL first
            try:
                pil_img = Image.open(file_path)
                if hasattr(pil_img, 'info') and 'dpi' in pil_img.info:
                    dpi_info = pil_img.info['dpi']
                    self.log(f"Image DPI: {dpi_info}")
                    # Store DPI info with the class for later use (properly rounded)
                    self.image_dpi = (round(dpi_info[0]), round(dpi_info[1]))
                    # Update the UI with rounded values
                    self.current_dpi_label.setText(f"DPI of the image: {self.image_dpi[0]}x{self.image_dpi[1]}")
                else:
                    self.log("The image has no DPI value")
                    self.image_dpi = None
                    self.current_dpi_label.setText("DPI of the image: none")

                # Also log the image format
                self.log(f"Image format: {pil_img.format}")
                self.image_format = pil_img.format

                # Get image dimensions
                self.log(f"Original dimensions: {pil_img.width}x{pil_img.height} pixels")
            except Exception as e:
                self.log(f"Warning: the image metadata was not read: {str(e)}")
                self.image_dpi = None
                self.image_format = None

            # Set appropriate UI state based on loaded image
            if self.image_dpi:
                self.dpi_group.setExclusive(False)  # Allow deselecting all buttons
                self.keep_unset_dpi.setChecked(False)
                self.use_custom_dpi.setChecked(False)
                self.dpi_group.setExclusive(True)  # Restore exclusive behavior
                dpi_options_explanation = f"The input DPI ({round(self.image_dpi[0])}x{round(self.image_dpi[1])}) is kept."
                self.dpi_explanation_label.setText(dpi_options_explanation)
                self.dpi_choice_widget.setVisible(False)
            else:
                self.dpi_explanation_label.setText("The image has no DPI value. Select the DPI for the output:")
                # Default to leave unset when no DPI in image
                self.keep_unset_dpi.setChecked(True)
                self.dpi_choice_widget.setVisible(True)

            # Load and crop the image to content
            img = cv2.imread(file_path)
            if img is not None:
                # Crop image to content with padding
                from lithic_editor.processing.image_io import crop_to_content
                cropped_img = crop_to_content(img, padding=10)

                # Define output folder path but don't create it yet
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                self.output_folder = os.path.join('image_debug', base_name)

                # Use a temp location for the cropped input to avoid creating folders prematurely
                import tempfile
                temp_dir = tempfile.mkdtemp()
                cropped_path = os.path.join(temp_dir, "cropped_input.png")

                # Save with PIL to preserve metadata
                try:
                    # Convert OpenCV BGR to RGB for PIL
                    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                    cropped_pil = Image.fromarray(cropped_rgb)

                    # Set DPI if we have it
                    if self.image_dpi:
                        cropped_pil.save(cropped_path, dpi=self.image_dpi)
                    else:
                        cropped_pil.save(cropped_path)
                except Exception as e:
                    self.log(f"Warning: the metadata was not written: {str(e)}")
                    # Fallback to OpenCV
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
                # self.clear_annotations_button.setEnabled(False)

    def clear_debug_images(self):
        """Clear the Processing Steps panel."""
        self.debug_images = []
        self.debug_panel.clear()

    def process_image(self):
        if not self.input_image_path:
            return

        # Use the original filename stored during load_image (before cropping/annotation changes it)
        original_filename = getattr(self, 'original_filename', 'image')

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

                # Save the annotated image to a temp location to avoid creating folders prematurely
                import tempfile
                temp_dir = tempfile.mkdtemp()
                annotated_path = os.path.join(temp_dir, "annotated_input.png")
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

        # Pass metadata to the processing function
        dpi_info = self.image_dpi if hasattr(self, 'image_dpi') else None
        format_info = self.image_format if hasattr(self, 'image_format') else None
        output_dpi = self.get_output_dpi()  # Get user's DPI preference
        # Debug images are both viewed and saved when checkbox is checked
        debug_enabled = self.debug_images_checkbox.isChecked()

        # Handle upscaling logic
        upscale_params = self.check_and_prompt_upscaling(dpi_info)

        # Pass output folder for debug images if enabled
        output_folder = self.output_folder if debug_enabled else None

        # Use the filename extracted earlier

        self.processing_thread = ProcessingThread(self.input_image_path, output_folder,
                dpi_info, format_info, output_dpi, debug_enabled, debug_filename=original_filename,
                preserve_cortex=self.preserve_cortex_checkbox.isChecked(),
                keep_upscaled=self.keep_upscaled_checkbox.isChecked(),
                scale_image_path=self.scale_image_path,
                config=self.processing_config(),
                measure_angles=self.measure_angles_checkbox.isChecked(),
                **upscale_params)
        self.processing_thread.progress_signal.connect(self.update_progress)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.start()

    def check_and_prompt_upscaling(self, dpi_info):
        """Ask for a DPI when the file has none, then offer upscaling if the drawing needs it"""
        settings = getattr(self, 'config', None)
        default_model = settings.resolution.upscale_model if settings else 'espcn'
        upscale_params = {
            'upscale_low_dpi': False,
            'default_dpi': None,
            'upscale_model': default_model,
        }

        # Determine current DPI
        if not dpi_info:
            # No DPI metadata - prompt user to specify
            dialog = DPISelectionDialog(self)
            if dialog.exec_() == QDialog.Accepted and dialog.selected_dpi:
                upscale_params['default_dpi'] = dialog.selected_dpi
            else:
                return upscale_params

        # The decision comes from the drawing, not the DPI: too-thin strokes or too-close
        # hatching make the skeleton unreliable, and only upscaling repairs that.
        try:
            gray = np.array(Image.open(self.input_image_path).convert('L'))
            if settings is not None:
                geometry = measure_line_geometry(gray, settings.resolution)
                factor = choose_upscale_factor(geometry, settings=settings.resolution)
            else:
                geometry = measure_line_geometry(gray)
                factor = choose_upscale_factor(geometry)
        except Exception as e:
            self.log(f"The line width was not measured: {e}")
            return upscale_params

        self.log(f"Measured {geometry.describe()}")
        if factor > 1:
            upscale_dialog = UpscalingDialog(
                geometry, factor, self,
                keep_upscaled=self.keep_upscaled_checkbox.isChecked(),
                default_model=default_model,
                has_scale_image=self.scale_image_path is not None,
            )
            if upscale_dialog.exec_() == QDialog.Accepted and upscale_dialog.upscale_confirmed:
                upscale_params['upscale_low_dpi'] = True
                upscale_params['upscale_model'] = upscale_dialog.selected_model
                # The dialog's choice becomes the panel's, so the processing thread reads it.
                # The dialog already warned, so the panel does not warn again.
                self.keep_upscaled_checkbox.blockSignals(True)
                self.keep_upscaled_checkbox.setChecked(upscale_dialog.keep_upscaled)
                self.keep_upscaled_checkbox.blockSignals(False)
                size_note = "kept at the upscaled size" if upscale_dialog.keep_upscaled else "returned to the input size"
                self.log(f"Upscale {factor}x for processing with {upscale_dialog.selected_model.upper()}; result {size_note}")
            else:
                self.log("No upscaling. Processing at the input size")
        else:
            self.log("The line width and the hatch gap are sufficient. No upscaling is necessary.")

        return upscale_params

    def update_progress(self, message):
            self.status_label.setText(message)
            self.log(message)

    def processing_finished(self, image_data, dpi_info, format_info):
        # Hide progress indicators
        self.progress_bar.hide()

        # Re-enable controls
        self.load_button.setEnabled(True)
        self.process_button.setEnabled(True)

        if image_data is not None:
            # Store the processed image data
            self.processed_image_data = image_data
            self.processed_dpi_info = dpi_info
            self.processed_format_info = format_info
            thread = getattr(self, 'processing_thread', None)
            self.processed_scale_factor = getattr(thread, 'result_scale_factor', 1)
            self.processed_scale_image = getattr(thread, 'processed_scale_image', None)

            self.log("Processing complete. Add arrows to the result if necessary.")
            if self.processed_scale_factor > 1:
                self.log(f"The result is {self.processed_scale_factor}x the input size. The DPI value is now {dpi_info}.")
                if self.processed_scale_image is None:
                    self.log("Warning: no scale image is loaded. Scale the scale bar image "
                             f"{self.processed_scale_factor}x before you measure.")

            # Debug dimensions for processed image
            if len(image_data.shape) == 2:
                proc_h, proc_w = image_data.shape
            else:
                proc_h, proc_w = image_data.shape[:2]
            self.log(f"Result size: {proc_w}x{proc_h} pixels")

            # Display the processed image from memory
            self.display_image_from_array(image_data, self.canvas)

            # Enable arrow tools
            arrow_integration.enable_arrow_controls(self)
            self.log("Add arrows to the result: drag to move, Shift+drag to rotate, Alt+drag to resize")
            self.place_striking_arrows(getattr(thread, 'striking_angles', []))

            # Update DPI display if we have DPI info
            if dpi_info:
                self.image_dpi = dpi_info
                self.current_dpi_label.setText(f"DPI of the result: {round(dpi_info[0])}x{round(dpi_info[1])}")
                self.log(f"DPI of the result: {round(dpi_info[0])}x{round(dpi_info[1])}")

            self.save_button.setEnabled(True)
            self.clear_annotations_button.setEnabled(True)
            self.status_label.setText('Processing complete')
            self.log("Use the brush tools to edit the input image")

            # Load and handle debug images if checkbox is enabled
            if self.debug_images_checkbox.isChecked():
                self.load_debug_images()
        else:
            # Processing failed
            self.status_label.setText('Processing failed')
            self.log("Error: processing failed")
            QMessageBox.critical(self, 'Error', 'Processing failed. See the log.')

    def place_striking_arrows(self, records):
        """Add one arrow for each measured scar, at the centroid of its ripples."""
        if not records:
            return
        for record in records:
            self.canvas.add_arrow_at_pixel(
                record['x'], record['y'], widget_angle_from_compass(record['compass']),
                size=arrow_size_for(record['extent'], self.canvas.get_minimum_arrow_size()),
                color=self.arrow_color)
            note = "" if record['direction_certain'] else " The direction is not certain."
            self.log(f"Scar {record['scar']}: striking angle {record['compass']} degrees "
                     f"from {record['ripples']} ripples.{note}")
        self.delete_arrow_button.setEnabled(True)
        self.clear_arrows_button.setEnabled(True)
        self.log(f"{len(records)} arrows added for the striking angles. "
                 "Move, turn or delete them as necessary.")

    def load_debug_images(self):
        """Show the debug images of the last run in the Processing Steps panel."""
        self.debug_images = self.debug_panel.show_images(
            self.output_folder, getattr(self, 'original_filename', '*'))

    def resizeEvent(self, event):
        """Handle window resize event to adjust image displays"""
        super().resizeEvent(event)

        # Refresh displayed images if they exist
        if hasattr(self, 'input_image_path') and self.input_image_path and hasattr(self, 'input_image_display'):
            if self.input_image_display.pixmap() and not self.input_image_display.pixmap().isNull():
                self.display_image(self.input_image_path, self.input_image_display)

        # Refresh processed image if it exists
        if hasattr(self, 'processed_image_data') and self.processed_image_data is not None and hasattr(self, 'canvas'):
            if hasattr(self.canvas, 'base_pixmap') and not self.canvas.base_pixmap.isNull():
                # Update the canvas display when window is resized
                self.canvas.update_display()

    def save_result(self):
        """Save the processed image with arrows"""
        if not hasattr(self, 'processed_image_data') or self.processed_image_data is None:
            QMessageBox.warning(self, 'No result', 'There is no result to save. Process an image first.')
            return

        # Ensure all arrows are detectable before saving
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'make_all_arrows_detectable'):
            adjusted_count = self.canvas.make_all_arrows_detectable()
            if adjusted_count > 0:
                self.log(f"{adjusted_count} arrows made larger so they stay visible in the saved image")
                # Update the display to show adjusted arrows
                self.canvas.update_display()

        # Use native file dialog
        QFileDialog.Options()
        file_path, filter_used = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;TIFF Files (*.tif)",
            options=QFileDialog.DontUseNativeDialog
        )

        if file_path:
            # Check and enforce file extension based on filter selected
            if filter_used == "PNG Files (*.png)" and not file_path.lower().endswith('.png'):
                file_path += '.png'
                save_format = 'PNG'
            elif filter_used == "JPEG Files (*.jpg)" and not file_path.lower().endswith(('.jpg', '.jpeg')):
                file_path += '.jpg'
                save_format = 'JPEG'
            elif filter_used == "TIFF Files (*.tif)" and not file_path.lower().endswith(('.tif', '.tiff')):
                file_path += '.tif'
                save_format = 'TIFF'
            # Default to PNG if extension is missing and filter doesn't indicate format
            elif not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']):
                file_path += '.png'
                save_format = 'PNG'
            else:
                # Determine format from extension
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.jpg', '.jpeg']:
                    save_format = 'JPEG'
                elif ext in ['.tif', '.tiff']:
                    save_format = 'TIFF'
                else:
                    save_format = 'PNG'

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

                    # Convert RGBA to RGB (PIL format)
                    rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

                    # Create PIL Image
                    pil_img = Image.fromarray(rgb)

                    # DPI of the processed result: the input's, or the user's choice, raised
                    # by the upscale factor when the result was kept at the working size
                    dpi_info = self.get_output_dpi()
                    factor = getattr(self, 'processed_scale_factor', 1)
                    if dpi_info and factor > 1:
                        dpi_info = (dpi_info[0] * factor, dpi_info[1] * factor)

                    # Log the DPI setting
                    if dpi_info:
                        self.log(f"DPI of the saved image: {dpi_info}")
                    else:
                        self.log("The saved image has no DPI value")

                    # Save the image with metadata. An image with no known DPI is saved
                    # without a DPI tag; a guessed tag would mislead measurement tools.
                    save_kwargs = {'dpi': dpi_info} if dpi_info else {}
                    pil_img.save(file_path, format=save_format, **save_kwargs)

                    self.status_label.setText(f'Saved to: {os.path.basename(file_path)}')
                    self.log(f"Result saved to: {file_path} (DPI: {dpi_info})")

                    # The scale image goes next to the result, scaled by the same factor,
                    # so the pair keeps one pixel scale
                    scale_image = getattr(self, 'processed_scale_image', None)
                    if scale_image is not None and factor > 1:
                        stem, ext = os.path.splitext(file_path)
                        scale_path = f"{stem}_scale{ext}"
                        Image.fromarray(np.asarray(scale_image)).save(scale_path, format=save_format, **save_kwargs)
                        self.log(f"Scale image saved to: {scale_path} (scaled {factor}x)")
                except Exception as e:
                    self.log(f"Error: the file was not saved: {str(e)}")
                    QMessageBox.critical(self, 'Error', f'The file was not saved: {str(e)}')

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

        # If displaying in the arrow canvas, update the DPI information
        if display_widget == self.canvas and hasattr(self, 'image_dpi'):
            # Set the DPI information in the canvas
            self.canvas.set_image_dpi(self.image_dpi)

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

        # Log DPI information if available
        if hasattr(self, 'image_dpi') and self.image_dpi:
            self.log(f"Image displayed with DPI: {self.image_dpi}")

    def display_image_from_array(self, image_array, display_widget):
        """Display an image from a numpy array in the given widget"""
        if image_array is None:
            return

        # Ensure image is in the right format for display
        if len(image_array.shape) == 2:
            # Grayscale image
            img_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Already RGB
            img_rgb = image_array
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # RGBA
            img_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        else:
            # BGR
            img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Create QImage and QPixmap
        h, w = img_rgb.shape[:2]
        bytes_per_line = 3 * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # If displaying in the arrow canvas, update the DPI information
        if display_widget == self.canvas and hasattr(self, 'image_dpi'):
            self.canvas.set_image_dpi(self.image_dpi)

        # If the widget is a CanvasWidget or ArrowCanvasWidget, use set_base_image
        if isinstance(display_widget, CanvasWidget) or isinstance(display_widget, ArrowCanvasWidget):
            display_widget.set_base_image(pixmap)
        else:
            # For regular QLabel, do scaling and use setPixmap
            display_width = display_widget.width() - 10
            display_height = display_widget.height() - 10

            # Calculate scaling factor to fit within display
            scale = min(display_width / w, display_height / h, 1.0)
            new_width = int(w * scale)
            new_height = int(h * scale)

            # Scale pixmap
            scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Update display
            display_widget.setPixmap(scaled_pixmap)

        # Clear arrows when a new image is loaded in the output canvas
        if display_widget == self.canvas:
            arrow_integration.clear_arrows_on_new_image(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LithicProcessorGUI()
    window.show()
    sys.exit(app.exec_())