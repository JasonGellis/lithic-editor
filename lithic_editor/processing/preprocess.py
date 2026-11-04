"""
From the grayscale scan to a binary image: optional smoothing, then a Sauvola threshold.
"""

import logging

import numpy as np
from skimage.filters import threshold_sauvola

from .resolution import smooth_for_threshold

logger = logging.getLogger(__name__)


def threshold_drawing(image, geometry, scale_factor, working_dpi, cfg, smooth_lines, debug,
                      dpi_info=None, output_dpi=None):
    """
    Turn the grayscale drawing into ink.

    When ``smooth_lines`` is on and the line width is known, the image is blurred in
    proportion to the line width first, so the tapered tips of hatch lines detach
    from the contours they touch. The unsmoothed ink is kept as well: the skeleton
    comes from the smoothed image, but the lines are rebuilt from the unsmoothed ink
    so they keep the pen's weight.

    Returns ``(binary_image, unsmoothed_ink)``: a uint8 image with 255 for ink, and a
    boolean array of the unsmoothed ink or None when nothing was smoothed.
    """
    unsmoothed = None
    if smooth_lines and geometry is not None and image.max() > 1:
        working_line_width = geometry.line_width * scale_factor
        unsmoothed = image
        image = smooth_for_threshold(image, working_line_width, cfg.smoothing)
        logger.info(f"Smoothed with sigma {cfg.smoothing.sigma_in_line_widths * working_line_width:.1f} px "
                    "before thresholding")
        debug.save(image, '1b', 'smoothed', 'Smoothed', dpi_info, output_dpi)

    if image.max() > 1:
        window_size = cfg.threshold.window_for_dpi(working_dpi)
        sauvola_k = cfg.threshold.sauvola_k
        logger.info(f"Applying Sauvola thresholding at {working_dpi} DPI (window={window_size})")
        # Dark pixels (lines) become True
        binary = image < threshold_sauvola(image, window_size=window_size, k=sauvola_k, r=None)
        unsmoothed_ink = None
        if unsmoothed is not None:
            unsmoothed_ink = unsmoothed < threshold_sauvola(unsmoothed, window_size=window_size,
                                                            k=sauvola_k, r=None)
        logger.info("Image thresholded using Sauvola")
    else:
        binary = image > 0
        unsmoothed_ink = None
        logger.info("Image already binary")

    binary_image = binary.astype(np.uint8) * 255
    debug.save(binary_image, '1c', 'binary_thresholded', 'Binary Thresholded', dpi_info, output_dpi)
    return binary_image, unsmoothed_ink
