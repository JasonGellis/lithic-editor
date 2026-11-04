"""
Reading the drawing and cropping it to its content.
"""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def read_drawing(image_path, dpi_info=None, format_info=None):
    """
    Read the drawing as a grayscale array and return it with its DPI and format.

    ``image_path`` is a file path or an array. A file is read with Pillow so the DPI
    tag and the format are kept; OpenCV is the fallback. ``dpi_info`` and
    ``format_info`` given by the caller win over the file's own values.
    """
    if not isinstance(image_path, str):
        logger.info(f"Using provided numpy array. Shape: {image_path.shape}")
        return image_path, dpi_info, format_info
    try:
        pil_image = Image.open(image_path)
        if format_info is None:
            format_info = pil_image.format
            logger.info(f"Image format: {format_info}")
        if dpi_info is None and hasattr(pil_image, 'info') and 'dpi' in pil_image.info:
            dpi_info = pil_image.info['dpi']
            logger.info(f"Image DPI: {dpi_info}")
        image = np.array(pil_image.convert('L'))
        logger.info(f"Image read successfully. Shape: {image.shape}")
    except Exception as e:
        logger.info(f"Error reading with Pillow: {e}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        logger.info(f"Image read with OpenCV. Shape: {image.shape}")
    return image, dpi_info, format_info


def crop_to_content(image, padding=10):
    """
    Crop image to content boundaries with configurable padding.

    Analyzes the input image to determine the minimal bounding rectangle containing
    all non-background content, then crops to this region with additional padding.
    Useful for removing excessive whitespace from processed lithic drawings.

    Parameters
    ----------
    image : numpy.ndarray
        Input image as numpy array. Handles both grayscale and binary images.
    padding : int, optional
        Additional padding pixels around detected content boundaries (default: 10).

    Returns
    -------
    tuple
        cropped_image : numpy.ndarray
            Image cropped to content boundaries with specified padding.
        bbox : tuple
            Bounding box coordinates as (x, y, width, height) describing the crop region.

    Notes
    -----
    Content detection assumes that background pixels have higher intensity values
    than foreground content, which is standard for lithic drawing processing where
    drawings are dark lines on light backgrounds.
    """
    # If image is grayscale, convert to 3 channels for consistent handling
    if len(image.shape) == 2:
        img_for_crop = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_for_crop = image.copy()

    # Convert to grayscale for finding bounds
    gray = cv2.cvtColor(img_for_crop, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding box for all contours
    if contours:
        x_min, y_min = image.shape[1], image.shape[0]
        x_max, y_max = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Add padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)

        # Crop the image
        return image[y_min:y_max, x_min:x_max]

    # If no contours found, return original
    return image
