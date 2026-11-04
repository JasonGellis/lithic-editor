"""
The ripple removal pipeline.

``process_lithic_drawing`` runs the stages in order. Each stage lives in its own
module: reading (``image_io``), thresholding (``preprocess``), cortex separation
(``cortex``), the skeleton and its special points (``skeleton``), the segments and
the ripple classification (``segments``), the striking angles (``striking_angle``)
and the rebuilt lines (``rebuild``). Debug images go through ``debug_images``.
"""

import logging
import math
import os

import numpy as np

from lithic_editor.config import resolve_config

from ..annotations.arrow_shapes import minimum_size_for_dpi
from .cortex import separate_cortex_and_structure
from .debug_images import DebugWriter
from .image_io import read_drawing
from .preprocess import threshold_drawing
from .rebuild import (
    endpoint_filter_picture,
    line_radius_range,
    rebuild_lines,
    structural_mask_from,
)
from .resolution import (
    choose_upscale_factor,
    measure_line_geometry,
    restore_to_grid,
    upscale_by_factor,
)
from .segments import (
    classify_pixels,
    connect_segments,
    identify_ripples,
    label_segments,
    ripples_picture,
    segments_picture,
)
from .skeleton import find_special_points, make_skeleton, special_points_picture
from .striking_angle import draw_arrows, draw_striking_angles, measure_striking_angles

logger = logging.getLogger(__name__)


def _dpi_value(dpi_info):
    """One number for a DPI given as a pair or a single value, or None."""
    if not dpi_info:
        return None
    return max(dpi_info[0], dpi_info[1]) if isinstance(dpi_info, tuple) else int(dpi_info)


def _log_metadata(dpi_info, output_dpi, format_info):
    if dpi_info:
        logger.info(f"Original DPI information: {dpi_info}")
    if output_dpi:
        logger.info(f"Output DPI will be set to: {output_dpi}")
    elif dpi_info:
        logger.info(f"Output DPI will match original: {dpi_info}")
    else:
        logger.info("Output DPI will be unset (application defaults will apply)")
    if format_info:
        logger.info(f"Preserving original format: {format_info}")


def _measure_angles(labeled_segments, ripple_segments, structural_pixels, junctions, cortex_mask,
                    angle_settings):
    """The striking angle of each scar, logged one per line. Returns the records."""
    logger.info("Measuring the striking angle of each scar...")
    structure = structural_pixels.copy()
    for x, y in junctions:
        structure[y, x] = True
    records, skipped = measure_striking_angles(labeled_segments, ripple_segments, structure,
                                               angle_settings, cortex=cortex_mask)
    for record in records:
        note = "" if record.direction_certain else ", direction uncertain"
        logger.info(f"  Scar {record.scar}: {record.compass} degrees from {record.ripples} ripples, "
                    f"spread {record.spread:.2f}{note}")
    for scar, count, reason in skipped:
        if reason == "cortex":
            logger.info(f"  Scar {scar}: skipped, it holds cortex stipple")
        else:
            logger.info(f"  Scar {scar}: skipped, {count} ripples, fewer than {angle_settings.min_ripples}")
    logger.info(f"Measured {len(records)} striking angles")
    return records


def process_lithic_drawing(image_path, output_folder="image_debug", dpi_info=None, format_info=None, output_dpi=None, save_debug=False,
                          upscale_low_dpi=False, default_dpi=None, upscale_model=None,
                          scale_image_path=None, return_scale_factor=False, debug_filename=None, preserve_cortex=True,
                          max_upscale_factor=None, restore_original_size=True, smooth_lines=None, config=None,
                          measure_angles=False):
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
    measure_angles : bool, optional
        Measure the striking angle of each scar from its ripples before they are
        removed (default: False). The result is then a dict with a
        ``striking_angles`` list: one record per scar with at least
        ``config.angles.min_ripples`` ripples, holding ``scar``, ``x``, ``y`` (the
        ripples' centroid on the output grid), ``compass`` (the direction of the force,
        away from the point of percussion, in the PyLithics ``arrow_angle`` frame: down
        0, left 90, up 180, right 270), ``ripples``, ``spread``, ``direction_certain`` and ``extent``.
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
    # Debug images: written only when asked for
    if debug_filename:
        base_filename = debug_filename
    elif isinstance(image_path, str):
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
    else:
        base_filename = "image"

    logger.debug("=== IMAGE DIMENSIONS DEBUGGING ===")
    logger.debug(f"{'Step':<30} {'Width':>10} {'Height':>10} {'Type':>15} {'Min':>8} {'Max':>8}")
    logger.debug("-" * 80)

    # Stage 1: read the image and its metadata
    logger.info("Reading image...")
    original_image, dpi_info, format_info = read_drawing(image_path, dpi_info, format_info)
    debug = DebugWriter(save_debug, output_folder, base_filename, format_info)

    # Every size threshold below comes from the configuration; explicit arguments win.
    cfg = resolve_config(config)
    if upscale_model is None:
        upscale_model = cfg.resolution.upscale_model
    if max_upscale_factor is None:
        max_upscale_factor = cfg.resolution.max_upscale_factor
    if smooth_lines is None:
        smooth_lines = cfg.smoothing.enabled
    logger.info(f"Configuration: {cfg.source}")

    # Working resolution. Upscale only when the drawing's strokes are too thin or too
    # close together for a reliable skeleton; never downscale, since shrinking throws
    # away the separation that makes the graph analysis work.
    scale_factor = 1
    processed_scale = None
    input_shape = original_image.shape[:2]
    input_dpi_info = dpi_info
    current_dpi = _dpi_value(dpi_info)
    if current_dpi is None and default_dpi:
        current_dpi = default_dpi
        logger.info(f"Using default DPI: {current_dpi} (no metadata found)")
    input_dpi = current_dpi

    # Line width and hatch clearance drive upscaling, smoothing and the rebuilt line weight
    geometry = measure_line_geometry(original_image, cfg.resolution) if original_image.max() > 1 else None
    if geometry is not None:
        logger.info(f"Measured {geometry.describe()}")
    if upscale_low_dpi:
        scale_factor = choose_upscale_factor(geometry, max_upscale_factor, cfg.resolution)
        logger.info(f"Upscale factor {scale_factor}x")
        if scale_factor > 1:
            debug.save(original_image, '0', 'input', f'Input ({geometry.describe()})', dpi_info, dpi_info)
            original_image = upscale_by_factor(original_image, scale_factor, upscale_model)
            if current_dpi:
                current_dpi = int(current_dpi * scale_factor)
                dpi_info = (current_dpi, current_dpi) if isinstance(dpi_info, tuple) else current_dpi
            logger.info(f"Upscaled {scale_factor}x for processing: {input_shape} -> {original_image.shape[:2]}")
            debug.save(original_image, '0a', 'upscaled', f'Upscaled {scale_factor}x', dpi_info, dpi_info)

    if scale_image_path:
        # The scale bar must stay on the same pixel grid as the drawing. It is only
        # upscaled when the drawing itself is returned upscaled.
        try:
            processed_scale, _, _ = read_drawing(scale_image_path)
            if scale_factor > 1 and not restore_original_size:
                processed_scale = upscale_by_factor(processed_scale, scale_factor, upscale_model)
                debug.save(processed_scale, '0b', 'upscaled_scale', f'Upscaled Scale {scale_factor}x',
                           dpi_info, dpi_info, format_info='PNG')
        except Exception as e:
            logger.info(f"Error processing scale image: {e}")
            processed_scale = None

    _log_metadata(dpi_info, output_dpi, format_info)
    debug.save(original_image, '1', 'original_image', 'Original Image (Pre-threshold)', dpi_info, output_dpi)

    # Stage 2: threshold at the working resolution
    logger.info("Preprocessing image...")
    binary_image, unsmoothed_ink = threshold_drawing(
        original_image, geometry, scale_factor, current_dpi, cfg, smooth_lines, debug, dpi_info, output_dpi)

    # Stage 3: separate the cortex stipple from the lines before anything thins them
    cortex_min, cortex_max = cfg.cortex.area_limits_for_dpi(current_dpi)
    logger.info(f"Using DPI-scaled cortex threshold: {cortex_min}-{cortex_max} pixels (DPI: {current_dpi})")
    structural_image, cortex_mask = separate_cortex_and_structure(
        binary_image, preserve_cortex=preserve_cortex,
        cortex_size_threshold=cortex_max, cortex_min_threshold=cortex_min)
    logger.info("Skipping morphological operations for simplified processing")
    if preserve_cortex:
        debug.save(structural_image, '2a', 'structural_only', 'Structural Lines Only', dpi_info, output_dpi)
        debug.save(cortex_mask.astype(np.uint8) * 255, '2b', 'cortex_mask', 'Cortex Stippling Mask',
                   dpi_info, output_dpi)
        binary_image = structural_image

    # Stage 4: the skeleton, its endpoints and junctions
    skeleton = make_skeleton(binary_image, cfg.skeleton.spur_length_px)
    debug.save(skeleton.astype(np.uint8) * 255, '3', 'skeleton', 'Pruned Skeleton', dpi_info, output_dpi)

    if geometry is not None and not math.isnan(geometry.line_width):
        working_gap = None if math.isnan(geometry.hatch_gap) else geometry.hatch_gap * scale_factor
        bridge_distance = cfg.skeleton.bridge_distance(geometry.line_width * scale_factor, working_gap)
    else:
        bridge_distance = cfg.skeleton.bridge_distance(None, None)
    y_tip_threshold = cfg.ripples.y_tip_distance_for_dpi(current_dpi)
    endpoints, junctions = find_special_points(skeleton, bridge_distance, y_tip_threshold)
    debug.save(special_points_picture(skeleton, endpoints, junctions), '4', 'endpoints_junctions',
               'Endpoints & Junctions', dpi_info, output_dpi)

    # Stage 5: the segments between the special points, and which are ripples
    labeled_segments, num_segments = label_segments(skeleton, endpoints, junctions)
    debug.save(segments_picture(labeled_segments, num_segments), '5', 'labeled_segments',
               f'Labeled Segments ({num_segments})', dpi_info, output_dpi)
    logger.info("Building graph representation...")
    segment_connections = connect_segments(labeled_segments, num_segments, endpoints, junctions)
    ripple_segments = identify_ripples(segment_connections)
    ripple_pixels, structural_pixels = classify_pixels(labeled_segments, num_segments, ripple_segments)
    ripple_viz = ripples_picture(ripple_pixels, structural_pixels, endpoints, junctions)
    debug.save(ripple_viz, '6', 'ripple_identification', 'Ripple vs Structural', dpi_info, output_dpi)

    # Stage 6: the striking angle of each scar from its ripples, before they go
    striking_angles = []
    if measure_angles:
        striking_angles = _measure_angles(labeled_segments, ripple_segments, structural_pixels, junctions,
                                          cortex_mask if preserve_cortex else None, cfg.angles)
        if debug.enabled:
            line_length = 6 * (geometry.hatch_gap * scale_factor if geometry is not None
                               and not math.isnan(geometry.hatch_gap) else 20)
            debug.save(draw_striking_angles(ripple_viz, striking_angles, line_length), '6a', 'striking_angles',
                       'Striking Angles', dpi_info, output_dpi)

    # Stage 7: the structural skeleton without the ripples
    structural_mask, structural_endpoints = structural_mask_from(
        skeleton, labeled_segments, ripple_segments, structural_pixels, endpoints, junctions)
    skeleton_cleaned = np.zeros(skeleton.shape, dtype=np.uint8)
    skeleton_cleaned[structural_mask] = 255
    debug.save(255 - skeleton_cleaned, '7', 'skeleton_cleaned', 'Skeleton Cleaned', dpi_info, output_dpi)
    debug.save(endpoint_filter_picture(skeleton, structural_pixels, endpoints, structural_endpoints, junctions),
               '7a', 'endpoint_filtering', 'Endpoint Filtering', dpi_info, output_dpi)

    # Stage 8: rebuild the lines at the measured weight, from the unsmoothed ink when available
    min_thickness, max_thickness = line_radius_range(geometry, scale_factor, current_dpi, cfg.lines)
    ink = unsmoothed_ink if unsmoothed_ink is not None else binary_image > 0
    final_cleaned = rebuild_lines(ink, structural_mask, min_thickness, max_thickness,
                                  cortex_mask if preserve_cortex else None)
    final_cleaned_inverted = 255 - final_cleaned
    debug.save(final_cleaned_inverted, '8', 'final_cleaned', 'Final Cleaned', dpi_info,
               output_dpi if output_dpi else dpi_info)

    # Stage 9: back to the input's pixel grid so measurements against a scale bar still hold
    output_scale = scale_factor
    if scale_factor > 1 and restore_original_size:
        final_cleaned_inverted = restore_to_grid(final_cleaned_inverted, input_shape, cfg.resolution)
        dpi_info = input_dpi_info
        current_dpi = input_dpi
        output_scale = 1
        logger.info(f"Restored result to input size {input_shape} (processed at {scale_factor}x)")
        debug.save(final_cleaned_inverted, '9', 'restored', 'Restored to input size', dpi_info,
                   output_dpi if output_dpi else dpi_info)

    if measure_angles:
        striking_angles = [r.on_grid(scale_factor / output_scale) for r in striking_angles]
        if debug.enabled and striking_angles:
            # The arrows as the GUI places them: the minimum detectable size for the DPI
            arrow_size = minimum_size_for_dpi(current_dpi)
            debug.save(draw_arrows(final_cleaned_inverted, striking_angles, arrow_size), '10', 'striking_arrows',
                       'Striking Arrows', dpi_info, output_dpi if output_dpi else dpi_info)

    logger.info("Processing complete!")

    if return_scale_factor or scale_image_path or measure_angles:
        result = {
            'processed_image': final_cleaned_inverted,
            'scale_factor': output_scale,
            'working_scale_factor': scale_factor,
            'original_dpi': input_dpi,
            'final_dpi': int(input_dpi * output_scale) if input_dpi else None,
        }
        if processed_scale is not None:
            result['processed_scale'] = processed_scale
        if measure_angles:
            result['striking_angles'] = [r.to_dict() for r in striking_angles]
        return result
    return final_cleaned_inverted
