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