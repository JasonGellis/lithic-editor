"""
The debug images: one picture for each pipeline step, written only when asked for.
"""

import logging
import os

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class DebugWriter:
    """
    Writes the debug images of one run as ``<step>_<base>_<name>.png`` in ``folder``.

    ``enabled`` False makes every ``save`` a no-op, so the pipeline can call it
    without checking. ``format_info`` is the format of the input, used for every
    image unless a call gives another.
    """

    def __init__(self, enabled, folder, base_name, format_info=None):
        self.enabled = bool(enabled and folder)
        self.folder = folder
        self.base_name = base_name
        self.format_info = format_info
        if self.enabled:
            os.makedirs(folder, exist_ok=True)
            logger.info(f"Debug mode enabled. Output folder: {folder}")

    def save(self, image, step, name, title, dpi_info=None, output_dpi=None, format_info=None):
        """Write ``image`` as the picture of ``step`` with ``title``, when enabled."""
        if not self.enabled:
            return
        path = os.path.join(self.folder, f'{step}_{self.base_name}_{name}.png')
        save_debug_image(image, path, title, dpi_info, format_info or self.format_info, output_dpi)


def debug_image_info(name, img):
    """Log one row of the dimension table for an image, at debug level."""
    if img is None:
        logger.debug(f"{name:<30} {'N/A':>10} {'N/A':>10} {'None':>15} {'N/A':>8} {'N/A':>8}")
        return

    if isinstance(img, bool) or (hasattr(img, 'dtype') and img.dtype == bool):
        img_type = "Boolean"
    elif not hasattr(img, 'dtype'):
        img_type = type(img).__name__
    else:
        img_type = str(img.dtype)

    h, w = "N/A", "N/A"
    min_val, max_val = "N/A", "N/A"
    if hasattr(img, 'shape') and len(img.shape) in (2, 3):
        h, w = img.shape[:2]

    if hasattr(img, 'min') and hasattr(img, 'max'):
        try:
            min_val = f"{img.min():.2f}"
            max_val = f"{img.max():.2f}"
        except (TypeError, ValueError):
            min_val = "Error"
            max_val = "Error"

    logger.debug(f"{name:<30} {w:>10} {h:>10} {img_type:>15} {min_val:>8} {max_val:>8}")
    return img


def save_debug_image(image, output_path, title=None, dpi_info=None, format=None, output_dpi=None):
    """
    Save a debug image with optional title using Pillow to preserve metadata

    Args:
        image: Image to save (grayscale or color)
        output_path: Path to save the image
        title: Optional title to add to the image
        dpi_info: Original DPI information
        format: Original image format to use
        output_dpi: User-selected DPI for output (overrides dpi_info if provided)
    """
    # Debug image info before processing
    debug_image_info(f"Before saving {os.path.basename(output_path)}", image)

    # Ensure image is in proper format for saving
    if image.dtype == bool:
        img = image.astype(np.uint8) * 255
    elif image.max() <= 1:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.copy()

    # If grayscale, keep it as grayscale for PIL
    if len(img.shape) == 2:
        pil_mode = 'L'
    else:
        # Convert BGR to RGB for PIL (OpenCV uses BGR, PIL uses RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_mode = 'RGB'

    # Create PIL Image (Pillow infers the mode from dtype and shape)
    pil_img = Image.fromarray(img)

    # Set DPI if provided (output_dpi takes precedence over dpi_info)
    if output_dpi:
        if isinstance(output_dpi, (int, float)):
            save_kwargs = {'dpi': (output_dpi, output_dpi)}
        else:
            save_kwargs = {'dpi': output_dpi}
    elif dpi_info:
        if isinstance(dpi_info, (int, float)):
            save_kwargs = {'dpi': (dpi_info, dpi_info)}
        else:
            save_kwargs = {'dpi': dpi_info}
    else:
        save_kwargs = {}

    # Determine format
    save_format = format if format else 'PNG'

    # If title is provided, create image with title; otherwise save clean image
    if title:
        # Create space for title
        title_height = 30

        if pil_mode == 'L':
            display_img = Image.new('RGB', (img.shape[1], img.shape[0] + title_height), color=(255, 255, 255))
            display_img.paste(pil_img, (0, title_height))
        else:
            display_img = Image.new('RGB', (img.shape[1], img.shape[0] + title_height), color=(255, 255, 255))
            display_img.paste(pil_img, (0, title_height))

        # Add title using OpenCV (PIL doesn't have good text support)
        display_np = np.array(display_img)
        cv2.putText(display_np, title, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        display_img = Image.fromarray(display_np)

        # Save the titled version
        display_img.save(output_path, format=save_format, **save_kwargs)
    else:
        # Save clean image without title
        pil_img.save(output_path, format=save_format, **save_kwargs)

    logger.info(f"Debug image saved to {output_path}")
