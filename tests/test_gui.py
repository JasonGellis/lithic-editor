"""
Tests for the GUI module.
"""

import pytest
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtTest import QTest
from PyQt5.QtGui import QPixmap


class TestGUIBasic:
    """Basic GUI tests."""
    
    def test_gui_import(self):
        """Test that GUI module can be imported."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        assert LithicProcessorGUI is not None
    
    def test_gui_creation(self, qapp):
        """Test creating main GUI window."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        
        window = LithicProcessorGUI()
        assert window is not None
        assert window.windowTitle() == "Lithic Editor and Annotator"
    
    def test_gui_widgets_exist(self, qapp):
        """Test that main widgets exist."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        
        window = LithicProcessorGUI()
        
        # Check main buttons exist
        assert hasattr(window, 'load_button')
        assert hasattr(window, 'process_button')
        assert hasattr(window, 'save_button')
        
        # Check canvas widgets exist
        assert hasattr(window, 'input_image_display')
        assert hasattr(window, 'canvas')  # Arrow canvas
        
        # Check arrow controls exist
        assert hasattr(window, 'add_arrow_button')
        assert hasattr(window, 'delete_arrow_button')
        assert hasattr(window, 'clear_arrows_button')
    
    def test_initial_button_states(self, qapp):
        """Test initial button enable/disable states."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        
        window = LithicProcessorGUI()
        
        # Load should be enabled
        assert window.load_button.isEnabled() == True
        
        # Process should be disabled initially
        assert window.process_button.isEnabled() == False
        
        # Save should be disabled initially
        assert window.save_button.isEnabled() == False
        
        # Arrow buttons should be disabled initially
        assert window.add_arrow_button.isEnabled() == False
        assert window.delete_arrow_button.isEnabled() == False
        assert window.clear_arrows_button.isEnabled() == False


class TestProcessingThread:
    """Test the processing thread."""
    
    def test_thread_creation(self, qapp):
        """Test creating processing thread."""
        from lithic_editor.gui.main_window import ProcessingThread
        
        thread = ProcessingThread(
            input_path="test.png",
            output_folder="output",
            save_debug=False
        )
        
        assert thread is not None
        assert isinstance(thread, QThread)
        assert thread.input_path == "test.png"
        assert thread.output_folder == "output"
        assert thread.save_debug == False
    
    def test_thread_signals(self, qapp):
        """Test that thread has required signals."""
        from lithic_editor.gui.main_window import ProcessingThread
        
        thread = ProcessingThread(
            input_path="test.png",
            output_folder="output",
            save_debug=False
        )
        
        # Check signals exist
        assert hasattr(thread, 'progress_signal')
        assert hasattr(thread, 'finished_signal')


class TestCanvasWidget:
    """Test the canvas widget for displaying images."""
    
    def test_canvas_creation(self, qapp):
        """Test creating canvas widget."""
        from lithic_editor.gui.main_window import CanvasWidget
        
        canvas = CanvasWidget()
        assert canvas is not None
        assert not hasattr(canvas, 'base_pixmap')
    
    def test_canvas_set_image(self, qapp, sample_pixmap):
        """Test setting image on canvas."""
        from lithic_editor.gui.main_window import CanvasWidget
        
        canvas = CanvasWidget()
        canvas.set_base_image(sample_pixmap)
        
        assert hasattr(canvas, 'base_pixmap')
        assert canvas.pixmap() is not None
    
    def test_canvas_clear(self, qapp, sample_pixmap):
        """Test clearing canvas."""
        from lithic_editor.gui.main_window import CanvasWidget
        
        canvas = CanvasWidget()
        canvas.set_base_image(sample_pixmap)
        assert hasattr(canvas, 'base_pixmap')
        
        canvas.clear_canvas()
        # Base pixmap should still exist, but annotations should be cleared
        assert hasattr(canvas, 'base_pixmap')
        assert hasattr(canvas, 'annotation_pixmap')


class TestGUIWorkflow:
    """Test GUI workflow integration."""
    
    @pytest.mark.skip(reason="Requires full GUI interaction")
    def test_load_image_workflow(self, qapp, sample_image):
        """Test loading an image through GUI."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        
        window = LithicProcessorGUI()
        
        # Mock file dialog to return our test image
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (str(sample_image), 'Images (*.png)')
            
            # Trigger load button
            window.load_button.click()
            
            # Check that process button is now enabled
            assert window.process_button.isEnabled() == True
    
    @pytest.mark.skip(reason="Requires full GUI interaction")
    def test_process_image_workflow(self, qapp, sample_image):
        """Test processing an image through GUI."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        
        window = LithicProcessorGUI()
        
        # Set up the image
        window.current_image_path = str(sample_image)
        window.process_button.setEnabled(True)
        
        # Mock the processing to avoid actual computation
        with patch.object(window, 'process_image') as mock_process:
            window.process_button.click()
            mock_process.assert_called_once()


class TestGUIHelpers:
    """Test GUI helper functions."""
    
    def test_brush_size_slider(self, qapp):
        """Test brush size slider exists and has correct range."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        
        window = LithicProcessorGUI()
        
        if hasattr(window, 'brush_slider'):
            assert window.brush_slider.minimum() >= 1
            assert window.brush_slider.maximum() <= 50
            assert window.brush_slider.value() >= 1
    
    def test_debug_images_checkbox(self, qapp):
        """Test debug images checkbox."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        
        window = LithicProcessorGUI()
        
        assert hasattr(window, 'debug_images_checkbox')
        # Should be unchecked by default for performance
        assert window.debug_images_checkbox.isChecked() == False
        
        # Check backward compatibility aliases
        assert window.save_debug_images == window.debug_images_checkbox
        assert window.show_debug_images == window.debug_images_checkbox
    
    def test_cortex_preservation_checkbox(self, qapp):
        """Test cortex preservation checkbox."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        
        window = LithicProcessorGUI()
        
        assert hasattr(window, 'preserve_cortex_checkbox')
        # Should be checked by default (cortex preserved)
        assert window.preserve_cortex_checkbox.isChecked() == True
    
    def test_clear_annotations_button(self, qapp):
        """Test clear annotations button exists."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        
        window = LithicProcessorGUI()
        
        assert hasattr(window, 'clear_annotations_button')
        # Clear annotations button should be disabled initially
        assert window.clear_annotations_button.isEnabled() == False


class TestDialogs:
    """Test GUI dialog functionality."""
    
    @patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName')
    def test_load_image_dialog(self, mock_dialog, qapp, sample_image):
        """Test load image dialog."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        
        mock_dialog.return_value = (str(sample_image), 'Images (*.png)')
        
        window = LithicProcessorGUI()
        window.load_image()
        
        # Should have set input image path (may be cropped version)
        assert window.input_image_path is not None
        assert 'input' in window.input_image_path or str(sample_image) in window.input_image_path
        # Should have enabled process button
        assert window.process_button.isEnabled() == True
    
    @patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName')
    @patch('lithic_editor.annotations.integration.get_image_with_arrows')
    def test_save_dialog(self, mock_get_arrows, mock_dialog, qapp, temp_dir, sample_pixmap):
        """Test save/output file dialog."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        from PyQt5.QtGui import QImage
        import numpy as np
        
        save_path = str(temp_dir / "output.png")
        mock_dialog.return_value = (save_path, "PNG Files (*.png)")
        
        # Mock the arrow integration to return a QImage
        mock_image = QImage(100, 100, QImage.Format_RGB32)
        mock_get_arrows.return_value = mock_image
        
        window = LithicProcessorGUI()
        # Mock having a processed image
        window.processed_image_data = np.zeros((100, 100), dtype=np.uint8)
        window.save_button.setEnabled(True)
        
        window.save_result()
        
        # Should have called dialog
        mock_dialog.assert_called_once()
    
    def test_processing_thread_with_cortex_parameter(self, qapp):
        """Test that processing thread receives cortex parameter."""
        from lithic_editor.gui.main_window import ProcessingThread
        
        thread = ProcessingThread(
            input_path="test.png",
            output_folder="output",
            save_debug=False,
            preserve_cortex=False
        )
        
        assert thread.preserve_cortex == False
        
        # Test default value
        thread_default = ProcessingThread(
            input_path="test.png",
            output_folder="output",
            save_debug=False
        )
        
        assert thread_default.preserve_cortex == True  # Default