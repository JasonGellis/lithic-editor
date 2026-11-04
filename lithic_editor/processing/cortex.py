"""
Separating the cortex stipple from the structural lines.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def separate_cortex_and_structure(binary_image, preserve_cortex=True, cortex_size_threshold=60, cortex_min_threshold=5):
    """
    Separate cortex stippling from structural elements using DPI-adaptive thresholding.

    Implements connected component analysis to distinguish between cortex stippling
    (small, isolated features) and structural elements (continuous line networks) in
    lithic drawings. Uses DPI-scaled size thresholds to maintain consistent separation
    performance across different image resolutions.

    The separation process prevents cortex destruction during skeletonization while
    enabling targeted morphological operations on structural elements only.

    Parameters
    ----------
    binary_image : numpy.ndarray
        Input binary image with 0=background, 255=foreground pixels.
    preserve_cortex : bool, optional
        Enable cortex separation. If False, processes all elements together (default: True).
    cortex_size_threshold : int, optional
        Maximum pixel area for cortex component classification (default: 60).
        Dynamically scales quadratically with DPI (base value at 150 DPI).
    cortex_min_threshold : int, optional
        Minimum pixel area for cortex component preservation (default: 5).
        Components below this threshold are filtered as noise. Scales with DPI.

    Returns
    -------
    tuple
        structural_image : numpy.ndarray
            Binary image containing only structural line elements.
        cortex_mask : numpy.ndarray
            Boolean mask identifying cortex stippling regions for later restoration.

    Notes
    -----
    **DPI Scaling Formula:**

    Both thresholds scale quadratically with image resolution to maintain consistent
    performance: ``threshold_scaled = threshold_base * (dpi/150)²``

    This ensures that cortex separation remains effective regardless of scanning
    resolution while preventing misclassification of structural elements as cortex.
    """
    if not preserve_cortex:
        # Process everything together - no cortex separation
        return binary_image, np.zeros_like(binary_image, dtype=bool)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Create separate images for cortex and structural elements
    structural_image = np.zeros_like(binary_image)
    cortex_mask = np.zeros_like(binary_image, dtype=bool)

    cortex_count = 0
    structural_count = 0

    # Process each component (skip background label 0)
    for component_id in range(1, num_labels):
        component_area = stats[component_id, cv2.CC_STAT_AREA]
        component_mask = labels == component_id

        if component_area < cortex_min_threshold:
            # Too small - filter out as noise (don't add to either cortex or structural)
            continue
        elif component_area <= cortex_size_threshold:
            # Small component - classify as cortex stippling
            cortex_mask[component_mask] = True
            cortex_count += 1
        else:
            # Large component - structural line
            structural_image[component_mask] = 255
            structural_count += 1

    logger.info(f"Separated: {structural_count} structural components, {cortex_count} cortex stipples")
    return structural_image, cortex_mask
