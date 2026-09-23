import cv2
import numpy as np
import networkx as nxgraph
from skimage.morphology import skeletonize
from skimage.filters import threshold_sauvola  # Wolf is a variant of Sauvola
from scipy.ndimage import convolve, label
import math
import os
import sys
import traceback
from PIL import Image
from lithic_editor.config import resolve_config

from .resolution import (
    choose_upscale_factor, measure_line_geometry, restore_to_grid, smooth_for_threshold,
    upscale_by_factor
)


def exception_hook(exc_type, exc_value, exc_traceback):
    print("Exception hook triggered:")
    print("Type:", exc_type)
    print("Value:", exc_value)
    print("Traceback:", "".join(traceback.format_tb(exc_traceback)))
    # Write to a file for inspection
    with open("error_log.txt", "w") as f:
        f.write("Exception hook triggered:\n")
        f.write(f"Type: {exc_type}\n")
        f.write(f"Value: {exc_value}\n")
        f.write(f"Traceback: {''.join(traceback.format_tb(exc_traceback))}")
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

_EIGHT_NEIGHBOURS = np.ones((3, 3), dtype=np.int32)

# Install the exception hook
sys.excepthook = exception_hook


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


def debug_image_info(name, img):
    """Print detailed image information for debugging"""
    if img is None:
        print(f"{name:<30} {'N/A':>10} {'N/A':>10} {'None':>15} {'N/A':>8} {'N/A':>8}")
        return

    if isinstance(img, bool) or (hasattr(img, 'dtype') and img.dtype == bool):
        img_type = "Boolean"
    elif not hasattr(img, 'dtype'):
        img_type = type(img).__name__
    else:
        img_type = str(img.dtype)

    shape_info = "N/A"
    h, w = "N/A", "N/A"
    min_val, max_val = "N/A", "N/A"

    if hasattr(img, 'shape'):
        if len(img.shape) == 2:
            h, w = img.shape
            shape_info = f"2D: {w}x{h}"
        elif len(img.shape) == 3:
            h, w, c = img.shape
            shape_info = f"3D: {w}x{h}x{c}"
        else:
            shape_info = f"Other: {img.shape}"

    if hasattr(img, 'min') and hasattr(img, 'max'):
        try:
            min_val = f"{img.min():.2f}"
            max_val = f"{img.max():.2f}"
        except (TypeError, ValueError):
            min_val = "Error"
            max_val = "Error"

    print(f"{name:<30} {w:>10} {h:>10} {img_type:>15} {min_val:>8} {max_val:>8}")
    return img

def _find_endpoints_and_junctions(skeleton):
    """Interior skeleton pixels with exactly one 8-neighbour, and with three or more."""
    neighbor_count = convolve(skeleton.astype(np.int32), _EIGHT_NEIGHBOURS, mode='constant') - skeleton
    interior = np.zeros_like(skeleton)
    interior[1:-1, 1:-1] = True
    endpoints = [(int(x), int(y)) for y, x in np.argwhere(skeleton & interior & (neighbor_count == 1))]
    junctions = [(int(x), int(y)) for y, x in np.argwhere(skeleton & interior & (neighbor_count >= 3))]
    return endpoints, junctions


def _bridge_close_endpoints(skeleton, endpoints, max_distance):
    """
    Draw a one-pixel line between every pair of free ends within ``max_distance``.

    Modifies ``skeleton`` in place and returns the number of bridges drawn. Each
    endpoint is bridged at most once, to its nearest partner.
    """
    if len(endpoints) < 2 or max_distance <= 0:
        return 0
    points = np.array(endpoints, dtype=np.float64)
    used = set()
    bridges = 0
    canvas = skeleton.view(np.uint8)
    for i in range(len(points)):
        if i in used:
            continue
        distances = np.hypot(*(points - points[i]).T)
        distances[i] = np.inf
        for j in used:
            distances[j] = np.inf
        j = int(np.argmin(distances))
        if distances[j] <= max_distance:
            cv2.line(canvas, tuple(int(v) for v in points[i]), tuple(int(v) for v in points[j]), 1, 1)
            used.update((i, j))
            bridges += 1
    return bridges


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

    print(f"Separated: {structural_count} structural components, {cortex_count} cortex stipples")
    return structural_image, cortex_mask


def process_lithic_drawing(image_path, output_folder="image_debug", dpi_info=None, format_info=None, output_dpi=None, save_debug=False,
                          upscale_low_dpi=False, default_dpi=None, upscale_model=None,
                          scale_image_path=None, return_scale_factor=False, debug_filename=None, preserve_cortex=True,
                          max_upscale_factor=None, restore_original_size=True, smooth_lines=None, config=None):
    """
    Process lithic drawings to remove ripple artifacts while preserving structural elements.

    This function implements a comprehensive digital image processing pipeline specifically
    designed for archaeological lithic illustration analysis. The algorithm employs
    DPI-adaptive parameter scaling to optimize processing for images of varying resolutions,
    ensuring consistent results across different scanning conditions.

    The processing pipeline consists of three main phases:
    1. **Preprocessing & Upscaling**: Image loading, DPI detection, optional neural network
       upscaling for low-resolution images, and binary thresholding
    2. **Structural Analysis**: DPI-adaptive cortex separation, morphological operations,
       skeletonization, Y-tip junction conversion, and ripple identification
    3. **Reconstruction**: DPI-aware thickness reconstruction, cortex restoration, and
       quality enhancement with anti-aliasing

    Key algorithmic innovations:
    - **Y-tip Elimination**: Converts junctions within DPI-scaled threshold (2-8 pixels)
      to endpoints, eliminating common artifacts without removing structural elements
    - **DPI-Adaptive Processing**: All parameters scale automatically based on image
      resolution to maintain consistent performance across scanning conditions
    - **Cortex Preservation**: Separates stippling from structural lines using
      connected component analysis with quadratically-scaled minimum thresholds

    Parameters
    ----------
    image_path : str
        Path to the input lithic drawing image file. Supports PNG, JPEG, TIFF, and BMP formats.
    output_folder : str, optional
        Directory path for saving debug images and intermediate processing steps (default: "image_debug").
    dpi_info : tuple of int or int, optional
        Override DPI metadata as (x_dpi, y_dpi) tuple or single value. If None, extracts from image metadata.
    format_info : str, optional
        Override output format specification. If None, preserves original image format.
    output_dpi : int, optional
        Target DPI for output images. If None, preserves original DPI.
    save_debug : bool, optional
        Enable saving intermediate processing steps for analysis (default: False).
    upscale_low_dpi : bool, optional
        Allow automatic upscaling for processing when the drawing's strokes are too
        thin or too close together for a reliable skeleton (default: False). The
        factor (2x, 3x or 4x) is chosen from the measured line width and hatch
        clearance, not from the DPI. Images are never downscaled.
    default_dpi : int, optional
        DPI to assume for images without metadata. Used only to scale the
        pipeline's size thresholds.
    upscale_model : str, optional
        Neural network model for upscaling: 'espcn' or 'fsrcnn'. Default: the
        configuration value (espcn).
    scale_image_path : str, optional
        Path to scale bar image to be processed with the same upscaling factor as main image.
    return_scale_factor : bool, optional
        Include upscaling metadata in return value (default: False).
    debug_filename : str, optional
        Custom base filename for debug output files. If None, derives from input filename.
    preserve_cortex : bool, optional
        Enable cortex stippling preservation during processing (default: True).
    max_upscale_factor : int, optional
        Largest factor automatic upscaling may use. Default: the configuration
        value (4).
    restore_original_size : bool, optional
        Return the result on the input's pixel grid even when processing was
        upscaled (default: True), so a scale bar measured alongside the drawing
        keeps its meaning. Set False to keep the upscaled result; the returned
        'scale_factor' then reports its size relative to the input.
    smooth_lines : bool, optional
        Blur the grayscale image in proportion to the measured line width before
        thresholding. This detaches the tapered tips of hatch lines from the
        contours they touch, so they are classified as ripples. Default: the
        configuration value (True).
    config : Config or str or path, optional
        The pipeline configuration: a ``lithic_editor.config.Config``, the path
        of a YAML file, or None for the file named by ``LITHIC_EDITOR_CONFIG`` or
        the shipped default. Every size threshold in the pipeline comes from it.

    Returns
    -------
    numpy.ndarray or dict
        **Standard mode**: Returns processed image as numpy.ndarray with ripple artifacts removed.

        **Extended mode** (when return_scale_factor=True or scale_image_path provided): Returns dict containing:

        - 'processed_image' : numpy.ndarray
            The processed lithic drawing with ripple artifacts removed
        - 'scale_factor' : int
            Size of the returned image relative to the input (1 when the result
            was returned on the input's pixel grid, which is the default)
        - 'working_scale_factor' : int
            Upscale factor used internally for processing (1 when none was needed)
        - 'original_dpi' : int
            Input image DPI
        - 'final_dpi' : int
            DPI of the returned image (equals original_dpi when restored)
        - 'processed_scale' : numpy.ndarray, optional
            Processed scale bar image (included when scale_image_path provided)

    Raises
    ------
    FileNotFoundError
        If the input image file cannot be located at the specified path.
    ValueError
        If image format is unsupported or DPI parameters are invalid.
    MemoryError
        If insufficient memory is available for processing large images.

    Notes
    -----
    **DPI-Adaptive Algorithm Parameters:**

    - **Y-tip Removal Thresholds:**
        - ≥600 DPI: 8-pixel threshold
        - 300-599 DPI: 5-pixel threshold
        - 150-299 DPI: 3-pixel threshold
        - <150 DPI: 2-pixel threshold (conservative)

    - **Thickness Reconstruction:**
        - ≥300 DPI: 4-6 pixel thickness range
        - 150-299 DPI: 3-4 pixel thickness range
        - <150 DPI: 1-2 pixel thickness range

    - **Cortex Filtering:**
        - Minimum threshold scales quadratically with DPI to filter noise
        - Maximum threshold prevents over-filtering of legitimate stippling

    **Performance Considerations:**

    For optimal processing performance, images should be scanned at 300+ DPI with good
    contrast. Very large images (>4000×4000 pixels) may require significant memory
    and processing time. Enable debug mode for detailed algorithm analysis on complex images.

    Examples
    --------
    >>> # Basic processing
    >>> result = process_lithic_drawing("artifact.png")
    >>>
    >>> # Full processing with upscaling and debug output
    >>> result = process_lithic_drawing(
    ...     image_path="drawing.png",
    ...     output_folder="results",
    ...     save_debug=True,
    ...     upscale_low_dpi=True,
    ...     preserve_cortex=True
    ... )
    >>>
    >>> # Extended return format with metadata
    >>> result = process_lithic_drawing(
    ...     "low_res_image.png",
    ...     return_scale_factor=True,
    ...     upscale_low_dpi=True
    ... )
    >>> print(f"Upscaled by factor: {result['scale_factor']}")
    >>> print(f"Final DPI: {result['final_dpi']}")

    References
    ----------
    .. [1] Zhang, T.Y. and Suen, C.Y., 1984. A fast parallel algorithm for thinning
           digital patterns. Communications of the ACM, 27(3), pp.236-239.
    .. [2] Otsu, N., 1979. A threshold selection method from gray-level histograms.
           IEEE transactions on systems, man, and cybernetics, 9(1), pp.62-66.
    """
    # Only create output folder if saving debug images
    if save_debug and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Debug mode enabled. Output folder: {output_folder}")

    # Extract base filename for debug images
    if debug_filename:
        base_filename = debug_filename
    elif isinstance(image_path, str):
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
    else:
        base_filename = "image"

    # Add debugging header
    print("\n=== IMAGE DIMENSIONS DEBUGGING ===")
    print(f"{'Step':<30} {'Width':>10} {'Height':>10} {'Type':>15} {'Min':>8} {'Max':>8}")
    print("-" * 80)

    # Step 1: Read the image and preserve metadata
    print("Reading image...")
    if isinstance(image_path, str):
        # Use Pillow to read the image and preserve metadata
        try:
            pil_image = Image.open(image_path)
            # Store metadata if not already provided
            if format_info is None:
                format_info = pil_image.format
                print(f"Image format: {format_info}")

            if dpi_info is None and hasattr(pil_image, 'info') and 'dpi' in pil_image.info:
                dpi_info = pil_image.info['dpi']
                print(f"Image DPI: {dpi_info}")

            # Convert to numpy array for processing (grayscale)
            original_image = np.array(pil_image.convert('L'))
            print(f"Image read successfully. Shape: {original_image.shape}")
        except Exception as e:
            print(f"Error reading with Pillow: {e}")
            # Fallback to OpenCV
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                raise ValueError(f"Could not read image at {image_path}")
            print(f"Image read with OpenCV. Shape: {original_image.shape}")
    else:
        # Assume image_path is already a numpy array
        original_image = image_path
        print(f"Using provided numpy array. Shape: {original_image.shape}")

    # Every size threshold below comes from the configuration; explicit arguments win.
    cfg = resolve_config(config)
    if upscale_model is None:
        upscale_model = cfg.resolution.upscale_model
    if max_upscale_factor is None:
        max_upscale_factor = cfg.resolution.max_upscale_factor
    if smooth_lines is None:
        smooth_lines = cfg.smoothing.enabled
    print(f"Configuration: {cfg.source}")

    # Working resolution. Upscale only when the drawing's strokes are too thin or too
    # close together for a reliable skeleton; never downscale, since shrinking throws
    # away the separation that makes the graph analysis work.
    scale_factor = 1
    processed_scale = None
    input_shape = original_image.shape[:2]
    input_dpi_info = dpi_info

    current_dpi = None
    if dpi_info:
        current_dpi = max(dpi_info[0], dpi_info[1]) if isinstance(dpi_info, tuple) else int(dpi_info)
    elif default_dpi:
        current_dpi = default_dpi
        print(f"Using default DPI: {current_dpi} (no metadata found)")
    input_dpi = current_dpi

    # Line width and hatch clearance drive upscaling, smoothing and the rebuilt line weight
    geometry = measure_line_geometry(original_image, cfg.resolution) if original_image.max() > 1 else None
    if geometry is not None:
        print(f"Measured {geometry.describe()}")
    if upscale_low_dpi:
        scale_factor = choose_upscale_factor(geometry, max_upscale_factor, cfg.resolution)
        print(f"Upscale factor {scale_factor}x")
        if scale_factor > 1:
            if save_debug:
                os.makedirs(output_folder, exist_ok=True)
                save_debug_image(original_image, os.path.join(output_folder, f'0_{base_filename}_input.png'),
                                f'Input ({geometry.describe()})', dpi_info, format_info, dpi_info)
            original_image = upscale_by_factor(original_image, scale_factor, upscale_model)
            if current_dpi:
                current_dpi = int(current_dpi * scale_factor)
                dpi_info = (current_dpi, current_dpi) if isinstance(dpi_info, tuple) else current_dpi
            print(f"Upscaled {scale_factor}x for processing: {input_shape} -> {original_image.shape[:2]}")
            if save_debug:
                save_debug_image(original_image, os.path.join(output_folder, f'0a_{base_filename}_upscaled.png'),
                                f'Upscaled {scale_factor}x', dpi_info, format_info, dpi_info)

    if scale_image_path:
        # The scale bar must stay on the same pixel grid as the drawing. It is only
        # upscaled when the drawing itself is returned upscaled.
        try:
            processed_scale = np.array(Image.open(scale_image_path).convert('L'))
            if scale_factor > 1 and not restore_original_size:
                processed_scale = upscale_by_factor(processed_scale, scale_factor, upscale_model)
                if save_debug:
                    save_debug_image(processed_scale, os.path.join(output_folder, f'0b_{base_filename}_upscaled_scale.png'),
                                    f'Upscaled Scale {scale_factor}x', dpi_info, 'PNG', dpi_info)
        except Exception as e:
            print(f"Error processing scale image: {e}")
            processed_scale = None

    # Print metadata info
    if dpi_info:
        print(f"Original DPI information: {dpi_info}")
    if output_dpi:
        print(f"Output DPI will be set to: {output_dpi}")
    elif dpi_info:
        print(f"Output DPI will match original: {dpi_info}")
    else:
        print("Output DPI will be unset (application defaults will apply)")
    if format_info:
        print(f"Preserving original format: {format_info}")

    # Save the original image (before thresholding)
    if save_debug:
        save_debug_image(original_image, os.path.join(output_folder, f'1_{base_filename}_original_image.png'),
                        'Original Image (Pre-threshold)', dpi_info, format_info, output_dpi)

    # Step 2: Preprocess the image at the working resolution
    print("Preprocessing image...")

    # All size thresholds below are keyed to the working resolution
    current_dpi_value = current_dpi

    # Smooth in proportion to line width so hatch-line tips detach from contours. The
    # unsmoothed image is kept: the skeleton and graph come from the smoothed one, but
    # the final lines are rebuilt from the unsmoothed ink so they keep the pen's weight.
    unsmoothed_image = None
    if smooth_lines and geometry is not None and original_image.max() > 1:
        working_line_width = geometry.line_width * scale_factor
        unsmoothed_image = original_image
        original_image = smooth_for_threshold(original_image, working_line_width, cfg.smoothing)
        print(f"Smoothed with sigma {cfg.smoothing.sigma_in_line_widths * working_line_width:.1f} px before thresholding")
        if save_debug:
            save_debug_image(original_image, os.path.join(output_folder, f'1b_{base_filename}_smoothed.png'),
                            'Smoothed', dpi_info, format_info, output_dpi)

    # Threshold to binary (if necessary)
    if original_image.max() > 1:  # Check if image is not already binary
        # Sauvola local threshold; the window adapts to the working DPI
        window_size = cfg.threshold.window_for_dpi(current_dpi_value)
        sauvola_k = cfg.threshold.sauvola_k

        print(f"Applying Sauvola thresholding at {current_dpi_value} DPI (window={window_size})")
        # Sauvola with k=0.2 (standard for document images)
        sauvola_thresh = threshold_sauvola(original_image, window_size=window_size, k=sauvola_k, r=None)
        # For lithic drawings (dark lines on light background), we need inverted comparison
        binary = original_image < sauvola_thresh  # Dark pixels (lines) become True
        if unsmoothed_image is not None:
            unsmoothed_ink = unsmoothed_image < threshold_sauvola(unsmoothed_image, window_size=window_size, k=sauvola_k, r=None)
        else:
            unsmoothed_ink = None

        print("Image thresholded using Sauvola")
    else:
        binary = original_image > 0
        unsmoothed_ink = None
        print("Image already binary")

    # Convert to uint8 binary image
    binary_image = binary.astype(np.uint8) * 255

    # Save the binary thresholded image
    if save_debug:
        save_debug_image(binary_image, os.path.join(output_folder, f'1c_{base_filename}_binary_thresholded.png'),
                        'Binary Thresholded', dpi_info, format_info, output_dpi)

    # Cortex area limits scale with the square of the working DPI (see config.yaml)
    cortex_min_threshold, cortex_threshold = cfg.cortex.area_limits_for_dpi(current_dpi_value)

    print(f"Using DPI-scaled cortex threshold: {cortex_min_threshold}-{cortex_threshold} pixels (DPI: {current_dpi_value})")

    # Separate cortex stippling from structural lines before morphological operations
    # Preserves cortex dots from being merged or modified during processing
    structural_image, cortex_mask = separate_cortex_and_structure(
        binary_image,
        preserve_cortex=preserve_cortex,
        cortex_size_threshold=cortex_threshold,
        cortex_min_threshold=cortex_min_threshold
    )

    # Skip morphological operations - test without them
    print("Skipping morphological operations for simplified processing")

    # Save debug images for cortex separation
    if save_debug and preserve_cortex:
        # Show separated structural elements
        save_debug_image(structural_image, os.path.join(output_folder, f'2a_{base_filename}_structural_only.png'),
                        'Structural Lines Only', dpi_info, format_info, output_dpi)

        # Show cortex mask
        cortex_debug = cortex_mask.astype(np.uint8) * 255
        save_debug_image(cortex_debug, os.path.join(output_folder, f'2b_{base_filename}_cortex_mask.png'),
                        'Cortex Stippling Mask', dpi_info, format_info, output_dpi)

    # Thin the structural elements (cortex bypasses thinning)
    print("Thinning structural elements using Lee skeletonization...")
    if preserve_cortex:
        # Thin only structural lines
        structural_binary = structural_image > 0
        # Use Lee skeletonization method
        skeleton = skeletonize(structural_binary, method='lee').astype(bool)
        print("Using Lee skeletonization method")
        # Update binary_image to reflect structural-only processing
        binary_image = structural_image
        binary = structural_binary
    else:
        # Process everything together (original behavior)
        binary = binary_image > 0
        # Use Lee skeletonization method
        skeleton = skeletonize(binary, method='lee').astype(bool)
        print("Using Lee skeletonization method")
    skeleton_img = skeleton.astype(np.uint8) * 255
    print(f"Skeleton created. Non-zero pixels: {np.count_nonzero(skeleton)}")

    # Apply morphological operations after skeletonization to smooth skeleton
    print("Applying post-skeleton morphological operations...")

    # Light dilation to slightly thicken skeleton (helps with connectivity)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton_smoothed = cv2.dilate(skeleton_img, dilate_kernel, iterations=1)
    print("Applied dilation to thicken skeleton slightly")

    # Light closing to connect small skeleton gaps
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skeleton_smoothed = cv2.morphologyEx(skeleton_smoothed, cv2.MORPH_CLOSE, close_kernel)
    print("Applied closing to connect skeleton gaps")

    # Re-thin to get back to 1-pixel width but smoother
    skeleton_smoothed_binary = skeleton_smoothed > 0
    # Use Lee method for re-thinning
    skeleton = skeletonize(skeleton_smoothed_binary, method='lee').astype(bool)
    print("Using Lee method for re-thinning")
    skeleton_img = skeleton.astype(np.uint8) * 255

    print(f"Smoothed skeleton created. Non-zero pixels: {np.count_nonzero(skeleton)}")

    # Branch length analysis to remove short spurs
    print("Analyzing branch lengths to remove spurs...")

    def remove_short_branches(skel, min_length=5):
        """Remove branches shorter than min_length that connect to junctions"""
        skel_copy = skel.copy()
        h, w = skel.shape

        # Find endpoints (interior pixels with exactly 1 neighbor), in row-major order
        neighbor_count = convolve(skel.astype(np.int32), _EIGHT_NEIGHBOURS, mode='constant') - skel
        is_endpoint = skel & (neighbor_count == 1)
        is_endpoint[0, :] = is_endpoint[-1, :] = False
        is_endpoint[:, 0] = is_endpoint[:, -1] = False
        endpoints = [(int(x), int(y)) for y, x in np.argwhere(is_endpoint)]

        print(f"Found {len(endpoints)} endpoints to analyze")

        # For each endpoint, trace back to find branch length
        branches_to_remove = []

        for ex, ey in endpoints:
            # Trace from endpoint until we hit a junction or another endpoint
            path = [(ex, ey)]
            current_x, current_y = ex, ey
            visited = set()
            visited.add((current_x, current_y))

            while True:
                # Find next pixel in the path
                next_pixels = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = current_x + dx, current_y + dy
                        if (0 <= nx < w and 0 <= ny < h and
                            skel[ny, nx] and (nx, ny) not in visited):
                            next_pixels.append((nx, ny))

                if len(next_pixels) == 0:
                    # Dead end (shouldn't happen in a connected skeleton)
                    break
                elif len(next_pixels) == 1:
                    # Continue along the branch
                    current_x, current_y = next_pixels[0]
                    path.append((current_x, current_y))
                    visited.add((current_x, current_y))
                else:
                    # We've reached a junction (multiple possible next pixels)
                    break

                # Safety check to prevent infinite loops
                if len(path) > 100:
                    break

            # Check if this branch should be removed
            branch_length = len(path)

            # Only remove if:
            # 1. Branch is shorter than min_length
            # 2. Branch ends at a junction (not isolated)
            if branch_length <= min_length and len(path) > 1:
                # Check if the end of the path is a junction
                end_x, end_y = path[-1]
                if end_x != ex or end_y != ey:  # Make sure we actually traced somewhere
                    end_neighbors = np.sum(skel[end_y-1:end_y+2, end_x-1:end_x+2]) - 1
                    if end_neighbors >= 3:  # It's a junction
                        # Mark this branch for removal (except the junction point)
                        branches_to_remove.extend(path[:-1])
                        print(f"Marking branch of length {branch_length} for removal")

        # Remove the marked branches
        for x, y in branches_to_remove:
            skel_copy[y, x] = False

        return skel_copy

    # Apply branch length analysis
    skeleton_pruned = remove_short_branches(skeleton, min_length=cfg.skeleton.spur_length_px)

    # Count removed pixels
    removed_pixels = np.count_nonzero(skeleton) - np.count_nonzero(skeleton_pruned)
    print(f"Branch length analysis removed {removed_pixels} pixels")

    skeleton = skeleton_pruned
    skeleton_img = skeleton.astype(np.uint8) * 255
    print(f"Final skeleton created. Non-zero pixels: {np.count_nonzero(skeleton)}")

    # Save the skeleton image
    if save_debug:
        save_debug_image(skeleton_img, os.path.join(output_folder, f'3_{base_filename}_skeleton.png'),
                        'Pruned Skeleton', dpi_info, format_info, output_dpi)

    # Step 3: Find endpoints and junctions
    print("Finding endpoints and junctions...")
    height, width = skeleton.shape

    # Initialize arrays to store special points
    endpoints = []
    junctions = []

    print(f"Total skeleton pixels: {np.count_nonzero(skeleton)}")

    # Thinning can leave a break of a few pixels in a contour. An open contour has two
    # free ends and would be classed as a ripple, so free ends closer together than
    # about a line width are bridged before classification. Hatch-line tips are never
    # that close to another free end: they sit a hatch gap apart.
    if geometry is not None and not math.isnan(geometry.line_width):
        working_gap = None if math.isnan(geometry.hatch_gap) else geometry.hatch_gap * scale_factor
        bridge_distance = cfg.skeleton.bridge_distance(geometry.line_width * scale_factor, working_gap)
    else:
        bridge_distance = cfg.skeleton.bridge_distance(None, None)
    endpoints, junctions = _find_endpoints_and_junctions(skeleton)
    bridged = _bridge_close_endpoints(skeleton, endpoints, bridge_distance)
    if bridged:
        print(f"Bridged {bridged} skeleton gaps of up to {bridge_distance:.1f} px")
        endpoints, junctions = _find_endpoints_and_junctions(skeleton)

    print(f"Found {len(endpoints)} endpoints and {len(junctions)} junctions")

    # Convert junctions that are very close to endpoints into endpoints themselves
    # This handles Y-tips by making the junction part of the ripple line
    print("Converting Y-tip junctions to endpoints...")

    # Y-tip threshold optimized for 300 DPI processing
    # Keyed to the working DPI, which adaptive upscaling raises when strokes are thin
    y_tip_threshold = cfg.ripples.y_tip_distance_for_dpi(current_dpi_value)

    print(f"Using Y-tip threshold: {y_tip_threshold} pixels (DPI: {current_dpi_value})")
    junctions_to_convert = []

    for jx, jy in junctions:
        # Check distance to nearest endpoint
        min_distance = float('inf')
        for ex, ey in endpoints:
            distance = ((jx - ex)**2 + (jy - ey)**2)**0.5
            if distance < min_distance:
                min_distance = distance
                if distance <= y_tip_threshold:
                    break  # Found a close endpoint, no need to check others

        # If junction is very close to an endpoint, convert it to an endpoint
        if min_distance <= y_tip_threshold:
            junctions_to_convert.append((jx, jy))
            print(f"  Converting junction at ({jx},{jy}) to endpoint (distance {min_distance:.1f} to nearest endpoint)")

    # Remove converted junctions from junction list and add to endpoints
    for junction in junctions_to_convert:
        junctions.remove(junction)
        endpoints.append(junction)

    print(f"Converted {len(junctions_to_convert)} Y-tip junctions to endpoints")
    print(f"Updated: {len(endpoints)} endpoints and {len(junctions)} junctions")

    # Create visualization of endpoints and junctions
    debug_img = np.zeros((height, width, 3), dtype=np.uint8)
    debug_img[skeleton] = [255, 255, 255]  # White for skeleton

    # Use small markers
    for x, y in endpoints:
        cv2.circle(debug_img, (x, y), 1, (0, 0, 255), -1)  # Red for endpoints

    for x, y in junctions:
        cv2.circle(debug_img, (x, y), 1, (0, 255, 0), -1)  # Green for junctions

    if save_debug:
        save_debug_image(debug_img, os.path.join(output_folder, f'4_{base_filename}_endpoints_junctions.png'),
                        'Endpoints & Junctions', dpi_info, format_info, output_dpi)


    # Step 4: Identify line segments by removing junctions and endpoints
    print("Identifying line segments...")

    # Create a mask for special points (endpoints and junctions)
    special_points_mask = np.zeros_like(skeleton, dtype=bool)
    for x, y in endpoints + junctions:
        special_points_mask[y, x] = True

    # Remove special points from skeleton to get segments
    segments = skeleton.copy()
    segments[special_points_mask] = False

    # Label the connected components (segments)
    structure = np.ones((3, 3), dtype=np.int32)  # 8-connectivity
    labeled_segments, num_segments = label(segments, structure=structure)
    print(f"Found {num_segments} line segments")

    # Create a colored visualization of the labeled segments
    segment_colors = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate random colors for each segment
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(50, 256, size=(num_segments + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    # Apply colors to segments
    segment_colors = colors[labeled_segments]

    if save_debug:
        save_debug_image(segment_colors, os.path.join(output_folder, f'5_{base_filename}_labeled_segments.png'),
                        f'Labeled Segments ({num_segments})', dpi_info, format_info, output_dpi)

    # Step 5: Build graph representation
    print("Building graph representation...")
    G = nxgraph.Graph()
    # Add nodes for endpoints and junctions
    for i, (x, y) in enumerate(endpoints):
        G.add_node(f"e{i}", type="endpoint", pos=(x, y))

    for i, (x, y) in enumerate(junctions):
        G.add_node(f"j{i}", type="junction", pos=(x, y))

    # Create a mapping from coordinates to node IDs
    coord_to_node = {}
    for node, data in G.nodes(data=True):
        coord_to_node[data['pos']] = node

    # For each segment, find which special points it touches. A point touches a segment
    # when a pixel of that segment lies in the point's 3x3 neighbourhood.
    touching = {segment_id: [] for segment_id in range(1, num_segments + 1)}
    for x, y in endpoints + junctions:
        window = labeled_segments[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        for segment_id in np.unique(window):
            if segment_id > 0:
                touching[int(segment_id)].append(coord_to_node[(x, y)])

    segment_connections = {}
    for segment_id in range(1, num_segments + 1):
        connected_points = touching[segment_id]

        # Store the connections for this segment
        if len(connected_points) >= 2:
            segment_connections[segment_id] = connected_points

            # Add edges between all pairs of connected points
            for i in range(len(connected_points)):
                for j in range(i+1, len(connected_points)):
                    G.add_edge(connected_points[i], connected_points[j], segment=segment_id)

    # Step 6: Identify ripple lines
    print("Identifying ripple lines...")

    # A segment is a ripple line if it connects to at least one endpoint
    ripple_segments = set()

    for segment_id, connected_points in segment_connections.items():
        # Check if any connected point is an endpoint
        has_endpoint = any(node.startswith('e') for node in connected_points)

        if has_endpoint:
            ripple_segments.add(segment_id)

    print(f"Identified {len(ripple_segments)} ripple segments")

    # Create visualization of ripple segments
    ripple_viz = np.zeros((height, width, 3), dtype=np.uint8)

    is_ripple = np.zeros(num_segments + 1, dtype=bool)
    is_ripple[list(ripple_segments)] = True
    ripple_pixels = is_ripple[labeled_segments]
    structural_pixels = (labeled_segments > 0) & ~ripple_pixels
    ripple_viz[structural_pixels] = [255, 255, 255]
    ripple_viz[ripple_pixels] = [0, 0, 255]

    # Mark endpoints and junctions
    for x, y in endpoints:
        cv2.circle(ripple_viz, (x, y), 1, (0, 0, 255), -1)  # Red for endpoints

    for x, y in junctions:
        cv2.circle(ripple_viz, (x, y), 1, (0, 255, 0), -1)  # Green for junctions

    if save_debug:
        save_debug_image(ripple_viz, os.path.join(output_folder, f'6_{base_filename}_ripple_identification.png'),
                        'Ripple vs Structural', dpi_info, format_info, output_dpi)

    # Step 7: Create mask of structural elements (excluding ripple endpoints)
    print("Creating structural mask...")

    # Start with clean binary mask
    structural_mask = np.zeros_like(skeleton, dtype=bool)

    # Add non-ripple segments to the mask
    structural_mask |= structural_pixels

    # Add junction points (single pixel) to ensure connectivity
    for x, y in junctions:
        structural_mask[y, x] = True

    # Filter out endpoints that connect to ripple segments to preserve line quality
    # Only add endpoints that connect to structural segments
    structural_endpoints = []
    for x, y in endpoints:
        # Check if this endpoint connects to any structural segments
        # by looking in a small neighborhood
        connects_to_structural = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if (0 <= ny < height and 0 <= nx < width and
                    labeled_segments[ny, nx] > 0 and
                    labeled_segments[ny, nx] not in ripple_segments):
                    connects_to_structural = True
                    break
            if connects_to_structural:
                break

        # Only add endpoint if it connects to structural elements
        if connects_to_structural:
            structural_mask[y, x] = True
            structural_endpoints.append((x, y))

    print(f"Filtered endpoints: {len(endpoints)} total -> {len(structural_endpoints)} structural")

    # Create a skeleton-based cleaned image for comparison
    skeleton_cleaned = np.zeros_like(skeleton_img)
    skeleton_cleaned[structural_mask] = 255

    # INVERT the skeleton-based cleaned image (black lines on white background)
    skeleton_cleaned_inverted = 255 - skeleton_cleaned

    # Save the skeleton-based cleaned image (inverted)
    if save_debug:
        save_debug_image(skeleton_cleaned_inverted, os.path.join(output_folder, f'7_{base_filename}_skeleton_cleaned.png'),
                        'Skeleton Cleaned', dpi_info, format_info, output_dpi)

    # NEW: Create debug image showing filtered endpoints
    filtered_endpoints_viz = np.zeros((height, width, 3), dtype=np.uint8)
    filtered_endpoints_viz[skeleton] = [100, 100, 100]  # Gray for all skeleton

    # Mark structural segments in white
    filtered_endpoints_viz[structural_pixels] = [255, 255, 255]

    # Mark junctions in green
    for x, y in junctions:
        cv2.circle(filtered_endpoints_viz, (x, y), 2, (0, 255, 0), -1)

    # Mark structural endpoints in blue
    for x, y in structural_endpoints:
        cv2.circle(filtered_endpoints_viz, (x, y), 2, (255, 0, 0), -1)

    # Mark filtered-out endpoints in red
    for x, y in endpoints:
        if (x, y) not in structural_endpoints:
            cv2.circle(filtered_endpoints_viz, (x, y), 2, (0, 0, 255), -1)

    if save_debug:
        if save_debug:
            save_debug_image(filtered_endpoints_viz, os.path.join(output_folder, f'7a_{base_filename}_endpoint_filtering.png'),
                            'Endpoint Filtering', dpi_info, format_info, output_dpi)

    # Step 8: Create final image with hybrid thickness preservation
    print("Creating final image with hybrid thickness preservation...")

    # Convert binary_image to boolean for processing
    binary_bool = binary_image > 0

    # Rebuild structural lines at the pen width measured from the drawing, so the output
    # keeps the original line weight at any resolution. The DPI buckets are only a
    # fallback when nothing was measured.
    if geometry is not None and not math.isnan(geometry.line_width):
        min_thickness, max_thickness = cfg.lines.radius_range(geometry.line_width * scale_factor)
    elif current_dpi_value and current_dpi_value >= 300:
        min_thickness, max_thickness = 4, 6
    elif current_dpi_value and current_dpi_value >= 150:
        min_thickness, max_thickness = 3, 4
    elif current_dpi_value:
        min_thickness, max_thickness = 2, 3
    else:
        min_thickness, max_thickness = 4, 6

    print(f"Rebuilding lines with radius {min_thickness}-{max_thickness} px (working DPI: {current_dpi_value})")

    # Use the new thickness-aware reconstruction
    # Rebuild from the unsmoothed ink when available, so lines keep their full weight
    reconstruction_ink = unsmoothed_ink if unsmoothed_ink is not None else binary_bool
    thickness_preserved_mask = create_thickness_aware_mask(
        original_binary=reconstruction_ink,
        structural_mask=structural_mask,
        min_thickness=min_thickness,
        max_thickness=max_thickness
    )

    # Create the final cleaned image
    final_cleaned = np.zeros_like(binary_image)
    final_cleaned[thickness_preserved_mask] = 255

    # Apply slight morphological closing to connect any small gaps
    kernel_close = np.ones((2, 2), np.uint8)
    final_cleaned = cv2.morphologyEx(final_cleaned, cv2.MORPH_CLOSE, kernel_close)

    # Add cortex stippling back to final result if preserving
    if preserve_cortex and 'cortex_mask' in locals():
        print("Adding cortex stippling back to final result...")
        final_cleaned[cortex_mask] = 255
        print(f"Cortex pixels added: {np.count_nonzero(cortex_mask)}")

    # INVERT the final cleaned image (black lines on white background)
    final_cleaned_inverted = 255 - final_cleaned

    # Save the final cleaned image (inverted)
    if save_debug:
        save_debug_image(final_cleaned_inverted, os.path.join(output_folder, f'8_{base_filename}_final_cleaned.png'),
                        'Final Cleaned', dpi_info, format_info, output_dpi if output_dpi else dpi_info)

    # Return to the input's pixel grid so measurements taken against a scale bar still hold
    output_scale = scale_factor
    if scale_factor > 1 and restore_original_size:
        final_cleaned_inverted = restore_to_grid(final_cleaned_inverted, input_shape, cfg.resolution)
        dpi_info = input_dpi_info
        current_dpi = input_dpi
        output_scale = 1
        print(f"Restored result to input size {input_shape} (processed at {scale_factor}x)")
        if save_debug:
            save_debug_image(final_cleaned_inverted, os.path.join(output_folder, f'9_{base_filename}_restored.png'),
                            'Restored to input size', dpi_info, format_info, output_dpi if output_dpi else dpi_info)

    print("Processing complete!")

    # Return results based on requested format
    if return_scale_factor or scale_image_path:
        result = {
            'processed_image': final_cleaned_inverted,
            'scale_factor': output_scale,
            'working_scale_factor': scale_factor,
            'original_dpi': input_dpi,
            'final_dpi': int(input_dpi * output_scale) if input_dpi else None,
        }
        if processed_scale is not None:
            result['processed_scale'] = processed_scale
        return result
    return final_cleaned_inverted

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

    print(f"Debug image saved to {output_path}")

def create_comparison_image(images, titles=None):
    """Create a comparison image grid using pure OpenCV"""
    # Check if all images are grayscale, convert to BGR if needed
    processed_images = []
    for img in images:
        if len(img.shape) == 2:
            # Convert grayscale to BGR
            processed_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            processed_img = img.copy()
        processed_images.append(processed_img)

    # Get the maximum dimensions
    max_height = max(img.shape[0] for img in processed_images)
    max_width = max(img.shape[1] for img in processed_images)

    # Add padding and resize all images to the same size
    padded_images = []
    for img in processed_images:
        # Create a white canvas of the maximum size
        padded = np.ones((max_height, max_width, 3), dtype=np.uint8) * 255
        # Place the image in the center
        h, w = img.shape[:2]
        y_offset = (max_height - h) // 2
        x_offset = (max_width - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = img
        padded_images.append(padded)

    # Define the grid layout (here: 2 rows, 3 columns)
    num_images = len(padded_images)
    rows = 2
    cols = 3

    # Add title space if titles are provided
    title_height = 30 if titles else 0

    # Create the comparison grid
    grid_height = rows * (max_height + title_height)
    grid_width = cols * max_width
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    # Place images in the grid
    for i, img in enumerate(padded_images):
        if i >= rows * cols:
            break

        row = i // cols
        col = i % cols

        y_start = row * (max_height + title_height)
        x_start = col * max_width

        # Add title if provided
        if titles and i < len(titles):
            cv2.putText(grid, titles[i], (x_start + 10, y_start + 20),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Add image
        grid[y_start + title_height:y_start + title_height + max_height,
             x_start:x_start + max_width] = img

    return grid