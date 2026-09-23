"""
Help text for the Lithic Editor and Annotator command line.

All text follows Simplified Technical English: short sentences, active voice,
one instruction for each sentence.
"""

import sys

from lithic_editor import __version__


def show_help():
    """Print the full help text."""
    help_text = f"""
LITHIC EDITOR AND ANNOTATOR v{__version__}
========================================

Lithic Editor removes the ripple lines from a scanned lithic drawing. It keeps
the scar contours, the outline and the cortex stipple. You can then add
direction arrows to the result.

INSTALLATION
------------
Install from the repository:
  pip install git+https://github.com/JasonGellis/lithic-editor.git

Python 3.10 or later is necessary.

COMMANDS
--------
Start the graphical interface:
  lithic-editor gui
  lithic-editor --gui

Process one image from the command line:
  lithic-editor process INPUT [options]

Read the documentation:
  lithic-editor docs              Open the documentation in the web browser
  lithic-editor docs --offline    Serve the documentation on this computer
  Online: https://jasongellis.github.io/lithic-editor/

Show the version:
  lithic-editor --version

Show this help:
  lithic-editor help
  lithic-editor help api          Show the Python API help

PROCESS OPTIONS
---------------
  -o, --output DIR      Output directory (default: image_debug). The result is
                        written there as <name>_cleaned.png.
  --debug               Write the debug images for each processing step.
  -q, --quiet           Do not print processing messages.
  --auto-upscale        Upscale the image for processing when the lines are
                        too thin or too near to each other. The factor is
                        measured from the drawing.
  --default-dpi DPI     The DPI to use when the image file has no DPI value.
  --upscale-model MODEL The neural model for upscaling: espcn (default) or
                        fsrcnn.
  --config PATH         A configuration file. It holds every size threshold of
                        the pipeline. Default: the file named by the environment
                        variable LITHIC_EDITOR_CONFIG, or the file shipped with
                        the package (lithic_editor/config/config.yaml).
  --keep-upscaled       Keep the result at the upscaled size. The DPI value
                        increases by the same factor. Without this option the
                        result has the pixel size and DPI of the input.
  --scale-image PATH    The scale bar image scanned with the drawing. With
                        --keep-upscaled it is scaled by the same factor and
                        written as <name>_scale.png.
  --no-preserve-cortex  Process the cortex stipple as lines.

EXAMPLES
--------
  lithic-editor process lithic.png --output results --debug
  lithic-editor process image.jpg --quiet
  lithic-editor process scan_75dpi.png --auto-upscale
  lithic-editor process scan.png --auto-upscale --keep-upscaled --scale-image bar.png
  lithic-editor process drawing.png --auto-upscale --upscale-model fsrcnn
  lithic-editor process drawing.png --no-preserve-cortex

PYTHON API
----------
  from lithic_editor.processing import process_lithic_drawing

  result = process_lithic_drawing("lithic.png", output_folder="results", save_debug=True)

Type `lithic-editor help api` for the full parameter list.

DEVELOPMENT
-----------
  git clone https://github.com/JasonGellis/lithic-editor.git
  cd lithic-editor
  pip install -e ".[dev]"     Package, tests and ruff
  pip install -e ".[docs]"    Documentation tools

  pytest                      Run the tests
  ruff check lithic_editor tests
  mkdocs serve                Serve the documentation on this computer

VERSION
-------
Version: {__version__}
Python: {sys.version}
Platform: {sys.platform}
"""
    print(help_text)


def show_version():
    """Print the version."""
    print(f"Lithic Editor and Annotator v{__version__}")


def show_api_help():
    """Print the Python API help text."""
    api_help = f"""
LITHIC EDITOR API v{__version__}
================================

PROCESSING
----------
from lithic_editor.processing import process_lithic_drawing

process_lithic_drawing(
    image_path,                   # File path, or a grayscale numpy array
    output_folder="image_debug",  # Directory for the debug images
    dpi_info=None,                # DPI of the input: a number or (x, y)
    format_info=None,             # Format of the input file
    output_dpi=None,              # DPI value for the debug images
    save_debug=False,             # Write the debug images
    upscale_low_dpi=False,        # Upscale when lines are too thin or too near
    default_dpi=None,             # DPI to use when the file has no DPI value
    upscale_model='espcn',        # 'espcn' or 'fsrcnn'
    scale_image_path=None,        # Scale bar image scanned with the drawing
    return_scale_factor=False,    # Return a dict with the scale details
    debug_filename=None,          # Base name for the debug images
    preserve_cortex=True,         # Keep the cortex stipple
    max_upscale_factor=4,         # Largest upscale factor: 2, 3 or 4
    restore_original_size=True,   # Return the result at the input size
    smooth_lines=None,            # Smooth before the threshold (default: configuration value)
    config=None,                  # Config object, YAML path, or None for the default file
)

Returns a numpy array: black lines on a white background, uint8.

With return_scale_factor=True or scale_image_path, returns a dict:
    processed_image        The result
    scale_factor           Size of the result relative to the input (1 when restored)
    working_scale_factor   The factor used for processing
    original_dpi           DPI of the input
    final_dpi              DPI of the result
    processed_scale        The scale image, scaled with the result (with scale_image_path)

CONFIGURATION
-------------
from lithic_editor.config import load_config, default_config_path

config = load_config("my_config.yaml")      # a copy of the shipped file with your changes
result = process_lithic_drawing("lithic.png", config=config)
print(default_config_path())                # the shipped file, with comments for every key

MEASUREMENT
-----------
from lithic_editor.processing.resolution import measure_line_geometry, choose_upscale_factor

geometry = measure_line_geometry(gray_array)   # .line_width, .hatch_gap, .ink_fraction (pixels)
factor = choose_upscale_factor(geometry)       # 1, 2, 3 or 4

GUI
---
from PyQt5.QtWidgets import QApplication
from lithic_editor.gui.main_window import LithicProcessorGUI

app = QApplication([])
window = LithicProcessorGUI()
window.show()
app.exec_()

ARROWS
------
from lithic_editor.annotations.arrows import Arrow, ArrowCanvasWidget

arrow = Arrow(position=(100, 200), angle=45, size=30)
canvas = ArrowCanvasWidget()
canvas.set_base_image(pixmap)

EXAMPLE: KEEP THE UPSCALED SIZE WITH A SCALE IMAGE
--------------------------------------------------
result = process_lithic_drawing(
    "scan_75dpi.png",
    dpi_info=75,
    upscale_low_dpi=True,
    restore_original_size=False,
    scale_image_path="bar.png",
)
image = result["processed_image"]        # 4x the input size, at 300 DPI
scale = result["processed_scale"]        # the scale bar, also 4x
"""
    print(api_help)


if __name__ == "__main__":
    show_help()
