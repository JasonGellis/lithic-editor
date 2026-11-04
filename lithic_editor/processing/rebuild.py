"""
Rebuilding the drawing from the structural skeleton at the measured line weight.
"""

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def structural_mask_from(skeleton, labeled_segments, ripple_segments, structural_pixels,
                         endpoints, junctions):
    """
    The skeleton pixels to keep: structural segments, junctions, and the endpoints
    that touch a structural segment. Returns ``(mask, structural_endpoints)``.
    """
    logger.info("Creating structural mask...")
    height, width = skeleton.shape
    mask = np.zeros_like(skeleton, dtype=bool)
    mask |= structural_pixels
    for x, y in junctions:
        mask[y, x] = True
    structural_endpoints = []
    for x, y in endpoints:
        touches_structure = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if (0 <= ny < height and 0 <= nx < width and labeled_segments[ny, nx] > 0
                        and labeled_segments[ny, nx] not in ripple_segments):
                    touches_structure = True
                    break
            if touches_structure:
                break
        if touches_structure:
            mask[y, x] = True
            structural_endpoints.append((x, y))
    logger.info(f"Filtered endpoints: {len(endpoints)} total -> {len(structural_endpoints)} structural")
    return mask, structural_endpoints


def endpoint_filter_picture(skeleton, structural_pixels, endpoints, structural_endpoints, junctions):
    """Skeleton grey, structure white, junctions green, kept endpoints blue, dropped red (BGR)."""
    height, width = skeleton.shape
    picture = np.zeros((height, width, 3), dtype=np.uint8)
    picture[skeleton] = [100, 100, 100]
    picture[structural_pixels] = [255, 255, 255]
    for x, y in junctions:
        cv2.circle(picture, (x, y), 2, (0, 255, 0), -1)
    for x, y in structural_endpoints:
        cv2.circle(picture, (x, y), 2, (255, 0, 0), -1)
    for x, y in endpoints:
        if (x, y) not in structural_endpoints:
            cv2.circle(picture, (x, y), 2, (0, 0, 255), -1)
    return picture


def line_radius_range(geometry, scale_factor, working_dpi, lines_settings):
    """
    The radius range for rebuilding the lines, from the measured line width.

    The DPI buckets are only a fallback when nothing was measured.
    """
    if geometry is not None and not math.isnan(geometry.line_width):
        return lines_settings.radius_range(geometry.line_width * scale_factor)
    if working_dpi and working_dpi >= 300:
        return 4, 6
    if working_dpi and working_dpi >= 150:
        return 3, 4
    if working_dpi:
        return 2, 3
    return 4, 6


def rebuild_lines(ink, structural_mask, min_thickness, max_thickness, cortex_mask=None):
    """
    Rebuild the structural lines from ``ink`` at the measured weight.

    Returns a uint8 image with 255 for ink, the cortex stipple added back when
    ``cortex_mask`` is given.
    """
    logger.info("Creating final image with hybrid thickness preservation...")
    logger.info(f"Rebuilding lines with radius {min_thickness}-{max_thickness} px")
    kept = create_thickness_aware_mask(
        original_binary=ink, structural_mask=structural_mask,
        min_thickness=min_thickness, max_thickness=max_thickness)
    final = np.zeros(ink.shape, dtype=np.uint8)
    final[kept] = 255
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    if cortex_mask is not None:
        logger.info("Adding cortex stippling back to final result...")
        final[cortex_mask] = 255
        logger.info(f"Cortex pixels added: {np.count_nonzero(cortex_mask)}")
    return final


def create_thickness_aware_mask(original_binary, structural_mask, min_thickness=1, max_thickness=10):
    """
    Generate DPI-adaptive thickness reconstruction mask for structural line preservation.

    Creates a morphologically-aware reconstruction mask that preserves original line
    thickness characteristics while restricting reconstruction to verified structural
    elements. Implements distance transform analysis to determine optimal dilation
    parameters for each skeletal component.

    Parameters
    ----------
    original_binary : numpy.ndarray
        Original binary image as boolean array where True represents foreground pixels.
    structural_mask : numpy.ndarray
        Skeleton-based binary mask identifying verified structural elements as boolean array.
    min_thickness : int, optional
        Minimum line thickness preservation threshold in pixels (default: 1).
    max_thickness : int, optional
        Maximum line thickness preservation threshold in pixels (default: 10).

    Returns
    -------
    numpy.ndarray
        Boolean reconstruction mask preserving original line thickness characteristics
        exclusively for structural elements identified in the input mask.

    Notes
    -----
    The function employs distance transform analysis to determine local line thickness
    and creates adaptive dilation zones that respect original drawing characteristics
    while preventing over-thickening artifacts.
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt, maximum_filter

    # Local line thickness at every pixel: the largest distance-to-background found in a
    # window around it. Equivalent to the per-pixel neighbourhood search this replaced,
    # but computed once for the whole image.
    inside_distance = distance_transform_edt(original_binary)
    window = 2 * (max_thickness + 2) + 1
    local_thickness = maximum_filter(inside_distance, size=window, mode='constant', cval=0.0)

    struct_y, struct_x = np.where(structural_mask)
    thickness_at_skeleton = local_thickness[struct_y, struct_x]
    radius = np.clip((thickness_at_skeleton * 0.2).astype(int), min_thickness, max_thickness)
    has_ink_nearby = thickness_at_skeleton > 0

    # Union of discs around the skeleton pixels, one dilation per distinct radius
    reconstruction_mask = np.zeros_like(original_binary, dtype=bool)
    for r in np.unique(radius[has_ink_nearby]):
        selected = has_ink_nearby & (radius == r)
        points = np.zeros_like(original_binary, dtype=bool)
        points[struct_y[selected], struct_x[selected]] = True
        offsets = np.arange(-r, r + 1)
        disc = (offsets[:, None] ** 2 + offsets[None, :] ** 2) <= r * r
        reconstruction_mask |= binary_dilation(points, structure=disc)

    # Final mask: original content AND within reconstruction zones
    final_mask = original_binary & reconstruction_mask

    return final_mask


def improve_line_quality_antialias(binary_image, line_boost=1.0, preserve_thickness=True):
    """
    Apply anti-aliasing enhancement to binary lithic drawing images.

    Implements advanced anti-aliasing algorithms to improve visual quality of processed
    lithic drawings while maintaining structural integrity and original line characteristics.
    The function employs Gaussian smoothing and edge-preserving filters to reduce pixelation
    artifacts introduced during digital processing.

    Parameters
    ----------
    binary_image : numpy.ndarray
        Input binary image with black lines (0) on white background (255).
        Expected format: 2D array with dtype uint8.
    line_boost : float, optional
        Multiplicative factor for line presence enhancement (default: 1.0).
        Values > 1.0 strengthen lines, < 1.0 weaken them.
    preserve_thickness : bool, optional
        Enable original line thickness preservation during enhancement (default: True).

    Returns
    -------
    numpy.ndarray
        Enhanced binary image with improved line quality and reduced aliasing artifacts.
        Output maintains same dimensions and format as input.

    Notes
    -----
    The anti-aliasing process applies Gaussian blur followed by adaptive thresholding
    to create smooth line edges while preserving structural details essential for
    archaeological analysis.
    """
    # Ensure correct format
    if binary_image.max() <= 1:
        binary_image = binary_image * 255

    # Convert to proper format
    binary_image = binary_image.astype(np.uint8)

    if preserve_thickness:
        # Gentle enhancement that preserves thickness
        scale_factor = 3  # Reduced from 6 for better thickness preservation
        h, w = binary_image.shape
        upscaled = cv2.resize(binary_image, (w * scale_factor, h * scale_factor),
                             interpolation=cv2.INTER_CUBIC)

        # Very light morphological operations
        if line_boost > 1.0:
            kernel = np.ones((2, 2), np.uint8)
            upscaled = cv2.dilate(upscaled, kernel, iterations=1)

        # Gentle Gaussian blur for smoothing
        blurred = cv2.GaussianBlur(upscaled, (3, 3), 0.5)  # Reduced sigma

        # Conservative threshold to maintain line width
        _, smoothed = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)  # Lower threshold

        # Light median filter
        smoothed = cv2.medianBlur(smoothed, 3)

        # Scale back down with high-quality interpolation
        result = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_AREA)

        # Final threshold with preservation bias
        _, result = cv2.threshold(result, 180, 255, cv2.THRESH_BINARY)  # Lower threshold

    else:
        # Original aggressive enhancement for very thin lines
        scale_factor = 6
        h, w = binary_image.shape
        upscaled = cv2.resize(binary_image, (w * scale_factor, h * scale_factor),
                             interpolation=cv2.INTER_CUBIC)

        kernel = np.ones((2, 2), np.uint8)
        if line_boost > 1.0:
            upscaled = cv2.dilate(upscaled, kernel, iterations=1)

        blurred = cv2.GaussianBlur(upscaled, (3, 3), 0.8)
        _, smoothed = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
        smoothed = cv2.medianBlur(smoothed, 3)

        if line_boost > 1.3:
            smoothed = cv2.erode(smoothed, kernel, iterations=1)

        result = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_AREA)
        _, result = cv2.threshold(result, 200, 255, cv2.THRESH_BINARY)

    return result
