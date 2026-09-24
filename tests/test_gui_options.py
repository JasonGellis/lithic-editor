"""Tests for the GUI's resolution options, dialogs, and its debug-image list."""

import os

import numpy as np
import pytest
from PIL import Image

from lithic_editor.config import Config
from lithic_editor.processing import process_lithic_drawing
from lithic_editor.processing.resolution import LineGeometry


@pytest.fixture(autouse=True)
def no_modal_warnings(monkeypatch):
    """Ticking 'Keep the upscaled size' opens a modal warning; record it instead of blocking."""
    from PyQt5.QtWidgets import QMessageBox
    calls = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: calls.append((title, text)))
    return calls


class TestOptionsPanel:
    def test_new_controls_exist_with_safe_defaults(self, qapp):
        from lithic_editor.gui.main_window import LithicProcessorGUI
        window = LithicProcessorGUI()
        assert window.keep_upscaled_checkbox.isChecked() is False
        assert window.preserve_cortex_checkbox.isChecked() is True
        assert window.scale_image_path is None
        assert window.scale_image_label.text() == "No scale image"
        assert window.load_scale_button.isEnabled()

    def test_configuration_is_loaded_at_startup(self, qapp):
        from lithic_editor.gui.main_window import LithicProcessorGUI
        window = LithicProcessorGUI()
        assert isinstance(window.config, Config)
        assert "Configuration:" in window.log_display.toPlainText()

    def test_no_dpi_saves_without_a_tag(self, qapp):
        from lithic_editor.gui.main_window import LithicProcessorGUI
        window = LithicProcessorGUI()
        window.image_dpi = None
        window.keep_unset_dpi.setChecked(True)
        assert window.get_output_dpi() is None
        window.use_custom_dpi.setChecked(True)
        window.custom_dpi_spinner.setValue(200)
        assert window.get_output_dpi() == (200, 200)


class TestUpscalingDialog:
    def test_text_follows_the_output_size_choice(self, qapp):
        from lithic_editor.gui.main_window import UpscalingDialog
        dialog = UpscalingDialog(LineGeometry(3.0, 9.0, 0.05), 2, keep_upscaled=False)
        assert "pixel size and DPI of the input" in dialog.size_label.text()
        dialog.keep_checkbox.setChecked(True)
        assert "2x the input size" in dialog.size_label.text()
        assert "No scale image is loaded" in dialog.size_label.text()

    def test_scale_image_is_mentioned_when_loaded(self, qapp):
        from lithic_editor.gui.main_window import UpscalingDialog
        dialog = UpscalingDialog(LineGeometry(3.0, 9.0, 0.05), 4, keep_upscaled=True, has_scale_image=True)
        assert "scaled by the same factor" in dialog.size_label.text()

    def test_confirm_returns_model_and_size_choice(self, qapp):
        from lithic_editor.gui.main_window import UpscalingDialog
        dialog = UpscalingDialog(LineGeometry(3.0, 9.0, 0.05), 2, default_model="fsrcnn")
        assert dialog.model_combo.currentData() == "fsrcnn"
        dialog.keep_checkbox.setChecked(True)
        dialog.confirm_upscale()
        assert dialog.upscale_confirmed
        assert dialog.selected_model == "fsrcnn"
        assert dialog.keep_upscaled is True

    def test_dialog_choice_updates_the_panel(self, qapp, tmp_path, monkeypatch):
        """Confirming with the box ticked in the dialog ticks the panel's checkbox."""
        from PyQt5.QtWidgets import QDialog
        from lithic_editor.gui.main_window import LithicProcessorGUI, UpscalingDialog
        img = np.full((60, 60), 255, dtype=np.uint8)
        img[5, 5:55] = img[54, 5:55] = 0
        img[5:55, 5] = img[5:55, 54] = 0
        for row in range(12, 48, 3):
            img[row, 10:40] = 0
        path = tmp_path / "thin.png"
        Image.fromarray(img).save(path, dpi=(150, 150))

        def fake_exec(self):
            self.keep_checkbox.setChecked(True)
            self.confirm_upscale()
            return QDialog.Accepted
        monkeypatch.setattr(UpscalingDialog, "exec_", fake_exec)

        window = LithicProcessorGUI()
        window.input_image_path = str(path)
        assert not window.keep_upscaled_checkbox.isChecked()
        params = window.check_and_prompt_upscaling((150, 150))
        assert params["upscale_low_dpi"] is True
        assert window.keep_upscaled_checkbox.isChecked()


class TestProcessingThread:
    def test_thread_carries_the_new_options(self, qapp):
        from lithic_editor.gui.main_window import ProcessingThread
        config = Config()
        thread = ProcessingThread("in.png", None, keep_upscaled=True, scale_image_path="bar.png", config=config)
        assert thread.keep_upscaled is True
        assert thread.scale_image_path == "bar.png"
        assert thread.config is config
        assert thread.result_scale_factor == 1
        assert thread.processed_scale_image is None

    def test_run_reports_factor_and_raised_dpi(self, qapp, tmp_path):
        from lithic_editor.gui.main_window import ProcessingThread
        img = np.full((60, 60), 255, dtype=np.uint8)
        img[5, 5:55] = img[54, 5:55] = 0
        img[5:55, 5] = img[5:55, 54] = 0
        for row in range(12, 48, 3):
            img[row, 10:40] = 0
        path = tmp_path / "thin.png"
        Image.fromarray(img).save(path, dpi=(150, 150))
        received = []
        thread = ProcessingThread(str(path), None, dpi_info=(150, 150), format_info="PNG",
                                  upscale_low_dpi=True, keep_upscaled=True)
        thread.finished_signal.connect(lambda image, dpi, fmt: received.append((image.shape, dpi)))
        thread.run()
        assert thread.result_scale_factor > 1
        shape, dpi = received[0]
        assert shape == (60 * thread.result_scale_factor, 60 * thread.result_scale_factor)
        assert dpi == (150 * thread.result_scale_factor, 150 * thread.result_scale_factor)


class TestDebugPanelMatchesPipeline:
    def test_every_debug_file_the_pipeline_writes_is_listed(self, qapp, tmp_path):
        """The GUI's file list must match the pipeline's names, including the upscale stages."""
        from lithic_editor.gui.main_window import LithicProcessorGUI
        img = np.full((120, 120), 255, dtype=np.uint8)
        img[10:12, 10:110] = img[108:110, 10:110] = 0
        img[10:110, 10:12] = img[10:110, 108:110] = 0
        for row in range(20, 100, 6):
            img[row:row + 2, 20:70] = 0
        process_lithic_drawing(img, output_folder=str(tmp_path), dpi_info=150,
                               upscale_low_dpi=True, save_debug=True, debug_filename="t")
        written = sorted(p.name for p in tmp_path.glob("*.png"))
        window = LithicProcessorGUI()
        window.original_filename = "t"
        window.output_folder = str(tmp_path)
        window.debug_images = []
        window.load_debug_images()
        listed = sorted(os.path.basename(p) for p in window.debug_images)
        assert listed == written


class TestRebuiltPanel:
    def test_dpi_choice_is_hidden_until_a_file_has_no_tag(self, qapp):
        from lithic_editor.gui.main_window import LithicProcessorGUI
        window = LithicProcessorGUI()
        assert not window.dpi_choice_widget.isVisibleTo(window)

    def test_configuration_can_be_loaded_and_reset(self, qapp, tmp_path):
        from lithic_editor.gui.main_window import LithicProcessorGUI
        window = LithicProcessorGUI()
        path = tmp_path / "mine.yaml"
        path.write_text("skeleton:\n  spur_length_px: 7\n")
        window.apply_configuration(str(path))
        assert window.config.skeleton.spur_length_px == 7
        assert window.config_label.text() == "Configuration: mine.yaml"
        window.reset_configuration()
        assert window.config.skeleton.spur_length_px == 5
        assert window.config_label.text() == "Configuration: default"

    def test_bad_configuration_keeps_the_previous_one(self, qapp, tmp_path, monkeypatch):
        from PyQt5.QtWidgets import QMessageBox
        from lithic_editor.gui.main_window import LithicProcessorGUI
        monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)
        window = LithicProcessorGUI()
        path = tmp_path / "bad.yaml"
        path.write_text("typo:\n  x: 1\n")
        window.apply_configuration(str(path))
        assert window.config is None
        assert "not read" in window.config_label.text()


class TestKeepUpscaledWarning:
    def test_panel_warns_when_ticked(self, qapp, no_modal_warnings):
        from lithic_editor.gui.main_window import LithicProcessorGUI
        calls = no_modal_warnings
        window = LithicProcessorGUI()
        assert not window.keep_upscaled_checkbox.isChecked()
        window.keep_upscaled_checkbox.setChecked(True)
        assert len(calls) == 1
        assert calls[0][0] == "Keep the upscaled size"
        assert "do not have the same pixel size" in calls[0][1]
        window.keep_upscaled_checkbox.setChecked(False)
        assert len(calls) == 1, "clearing the option must not warn"

    def test_dialog_warns_with_the_factor(self, qapp, no_modal_warnings):
        from lithic_editor.gui.main_window import UpscalingDialog
        calls = no_modal_warnings
        dialog = UpscalingDialog(LineGeometry(1.8, 9.8, 0.1), 4)
        dialog.keep_checkbox.setChecked(True)
        assert len(calls) == 1
        assert "4x the pixel size" in calls[0][1]

    def test_dialog_choice_copied_to_panel_does_not_warn_twice(self, qapp, tmp_path, monkeypatch, no_modal_warnings):
        from PyQt5.QtWidgets import QDialog
        from lithic_editor.gui.main_window import LithicProcessorGUI, UpscalingDialog
        calls = no_modal_warnings
        img = np.full((60, 60), 255, dtype=np.uint8)
        img[5, 5:55] = img[54, 5:55] = 0
        img[5:55, 5] = img[5:55, 54] = 0
        for row in range(12, 48, 3):
            img[row, 10:40] = 0
        path = tmp_path / "thin.png"
        Image.fromarray(img).save(path, dpi=(150, 150))

        def fake_exec(self):
            self.keep_checkbox.setChecked(True)
            self.confirm_upscale()
            return QDialog.Accepted
        monkeypatch.setattr(UpscalingDialog, "exec_", fake_exec)
        window = LithicProcessorGUI()
        window.input_image_path = str(path)
        window.check_and_prompt_upscaling((150, 150))
        assert window.keep_upscaled_checkbox.isChecked()
        assert len(calls) == 1
