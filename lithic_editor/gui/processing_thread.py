"""
The thread that runs the pipeline while the window stays responsive.

The pipeline reports its progress through the ``lithic_editor.processing`` logger.
While a run lasts, a handler on that logger turns every record into the thread's
``progress_signal``, which the window appends to its log panel.
"""

import logging

from PyQt5.QtCore import QThread, pyqtSignal

from lithic_editor.processing import process_lithic_drawing

PIPELINE_LOGGER = "lithic_editor.processing"


class _SignalHandler(logging.Handler):
    """A logging handler that emits each message through a Qt signal."""

    def __init__(self, signal):
        super().__init__(level=logging.INFO)
        self._signal = signal

    def emit(self, record):
        self._signal.emit(record.getMessage())


class ProcessingThread(QThread):
    """Thread for processing images without freezing the UI"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, object, object)  # image_data, dpi_info, format_info

    def __init__(self, input_path, output_folder, dpi_info=None, format_info=None, output_dpi=None,
                 save_debug=False, upscale_low_dpi=False, default_dpi=None, upscale_model='espcn',
                 debug_filename=None, preserve_cortex=True, keep_upscaled=False, scale_image_path=None,
                 config=None, measure_angles=False):
        super().__init__()
        self.input_path = input_path
        self.config = config
        self.keep_upscaled = keep_upscaled
        self.scale_image_path = scale_image_path
        self.measure_angles = measure_angles
        # One record per scar with enough ripples, on the pixel grid of the result
        self.striking_angles = []
        # Filled in after a run: the factor the returned image is scaled by relative to
        # the input, and the scale image resampled by the same factor, if one was given.
        self.result_scale_factor = 1
        self.processed_scale_image = None
        self.output_folder = output_folder
        self.dpi_info = dpi_info  # Original DPI
        self.format_info = format_info
        self.output_dpi = output_dpi  # User-selected output DPI
        self.save_debug = save_debug
        self.upscale_low_dpi = upscale_low_dpi
        self.default_dpi = default_dpi
        self.upscale_model = upscale_model
        self.debug_filename = debug_filename
        self.preserve_cortex = preserve_cortex

    def run(self):
        logger = logging.getLogger(PIPELINE_LOGGER)
        handler = _SignalHandler(self.progress_signal)
        logger.addHandler(handler)
        previous_level = logger.level
        if logger.getEffectiveLevel() > logging.INFO:
            logger.setLevel(logging.INFO)
        try:
            result = process_lithic_drawing(
                self.input_path, self.output_folder,
                dpi_info=self.dpi_info,
                format_info=self.format_info,
                output_dpi=self.output_dpi,
                save_debug=self.save_debug,
                upscale_low_dpi=self.upscale_low_dpi,
                default_dpi=self.default_dpi,
                upscale_model=self.upscale_model,
                debug_filename=self.debug_filename,
                preserve_cortex=self.preserve_cortex,
                restore_original_size=not self.keep_upscaled,
                scale_image_path=self.scale_image_path,
                return_scale_factor=True,
                config=self.config,
                measure_angles=self.measure_angles,
            )
            result_image = result['processed_image']
            self.result_scale_factor = result['scale_factor']
            self.processed_scale_image = result.get('processed_scale')
            self.striking_angles = result.get('striking_angles', [])
            # When the result is kept at the working size its DPI is the input DPI times
            # the factor, which keeps the physical size truthful.
            dpi_out = self.output_dpi or self.dpi_info
            if self.result_scale_factor > 1 and dpi_out:
                if isinstance(dpi_out, tuple):
                    dpi_out = (dpi_out[0] * self.result_scale_factor, dpi_out[1] * self.result_scale_factor)
                else:
                    dpi_out = dpi_out * self.result_scale_factor
            self.finished_signal.emit(result_image, dpi_out, self.format_info)
        except Exception as e:
            self.progress_signal.emit(f"Error: {str(e)}")
            self.finished_signal.emit(None, None, None)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
