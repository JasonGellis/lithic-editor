"""
Content-adaptive working resolution.

The ripple classifier works on a skeleton graph, and that graph is only right
when lines are a few pixels wide and neighbouring hatch lines have clear
background between them. Below that, thin contours break and adjacent hatch
lines merge, and the classifier does the right thing on the wrong graph.

This module measures those two quantities in pixels and chooses the integer
upscale factor that brings both above their floors. It never downscales:
shrinking a drawing throws away the separation that made it work. After
processing, ``restore_to_grid`` returns the result to the input's pixel grid,
so a scale bar measured alongside the drawing keeps its meaning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, maximum_filter
from skimage.morphology import skeletonize

from lithic_editor.config import ResolutionSettings, SmoothingSettings

from .upscaling import upscale_with_model

# All numbers come from the configuration (lithic_editor/config/config.yaml). The
# dataclass defaults are used when a caller passes no settings.
_RESOLUTION = ResolutionSettings()
_SMOOTHING = SmoothingSettings()
# Factors the bundled ESPCN and FSRCNN models provide.
SUPPORTED_FACTORS = (1, 2, 3, 4)


@dataclass(frozen=True)
class LineGeometry:
    """Stroke width and hatch clearance of a drawing, in pixels."""

    line_width: float
    hatch_gap: float
    ink_fraction: float

    def describe(self) -> str:
        width = "unknown" if math.isnan(self.line_width) else f"{self.line_width:.1f} px"
        gap = "unknown" if math.isnan(self.hatch_gap) else f"{self.hatch_gap:.1f} px"
        return f"line width {width}, hatch gap {gap}"


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink mask of a grayscale drawing using Otsu's threshold (inclusive, so a binary image works)."""
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray <= threshold


def measure_line_geometry(
    gray: np.ndarray, settings: ResolutionSettings = _RESOLUTION
) -> LineGeometry:
    """
    Measure typical stroke width and the typical clearance between strokes.

    Width is twice the median distance-to-edge along the skeleton. Clearance is
    measured on the background enclosed by the drawing: the ridge of the
    background's distance-to-ink field runs midway between neighbouring
    strokes, and its median is the typical clearance. Dot-like components are
    left out of that measurement. Either value is NaN when it cannot be measured.
    """
    ink = binarize(gray)
    if not ink.any():
        return LineGeometry(math.nan, math.nan, 0.0)

    skeleton = skeletonize(ink, method="lee").astype(bool)
    if skeleton.any():
        half_widths = distance_transform_edt(ink)[skeleton]
        line_width = max(1.0, 2.0 * float(np.median(half_widths)) - 1.0)
    else:
        line_width = math.nan

    # Clearance is measured between stroke-like components only. Cortex stipple
    # and specks sit close together and would otherwise dominate the ridge.
    strokes = _stroke_components(ink, line_width, settings.dot_extent_in_line_widths)
    enclosed = binary_fill_holes(strokes) & ~strokes
    hatch_gap = math.nan
    if enclosed.any():
        clearance = distance_transform_edt(~strokes)
        ridge = enclosed & (clearance > 0) & (clearance >= maximum_filter(clearance, size=3))
        if ridge.any():
            hatch_gap = max(1.0, 2.0 * float(np.median(clearance[ridge])) - 1.0)

    return LineGeometry(line_width, hatch_gap, float(ink.mean()))


def _stroke_components(ink: np.ndarray, line_width: float, dot_extent: float) -> np.ndarray:
    """Ink with components shorter than ``dot_extent`` line widths removed."""
    if math.isnan(line_width) or dot_extent <= 0:
        return ink
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = extent >= dot_extent * line_width
    keep[0] = False
    return keep[labels]


def choose_upscale_factor(
    geometry: LineGeometry,
    max_factor: int | None = None,
    settings: ResolutionSettings = _RESOLUTION,
) -> int:
    """
    Smallest supported integer factor that lifts width and clearance above their floors.

    The floors are ``settings.min_line_width_px`` and ``settings.min_hatch_gap_px``.
    ``max_factor`` overrides ``settings.max_upscale_factor``. Returns 1 when no
    upscaling is needed or nothing could be measured.
    """
    if max_factor is None:
        max_factor = settings.max_upscale_factor
    needed = 1.0
    if not math.isnan(geometry.line_width) and geometry.line_width > 0:
        needed = max(needed, settings.min_line_width_px / geometry.line_width)
    if not math.isnan(geometry.hatch_gap) and geometry.hatch_gap > 0:
        needed = max(needed, settings.min_hatch_gap_px / geometry.hatch_gap)
    factor = math.ceil(needed - 1e-9)
    factor = min(factor, max_factor, max(SUPPORTED_FACTORS))
    if factor not in SUPPORTED_FACTORS:
        factor = min(f for f in SUPPORTED_FACTORS if f >= factor)
    return factor


def upscale_by_factor(gray: np.ndarray, factor: int, model: str = "espcn") -> np.ndarray:
    """Enlarge a grayscale drawing by an integer factor with the chosen neural model."""
    if factor == 1:
        return gray
    return upscale_with_model(gray, float(factor), model)


def smooth_for_threshold(
    gray: np.ndarray, line_width: float, settings: SmoothingSettings = _SMOOTHING
) -> np.ndarray:
    """
    Lightly blur a grayscale drawing in proportion to its line width.

    Hatch lines taper where they meet a contour. Blurring by a fraction of the
    line width (``settings.sigma_in_line_widths``) pushes those tapered tips
    below the threshold, which detaches the ripple and gives it the free
    endpoint the classifier looks for. The same blur leaves contours, which are
    drawn at full width, intact.
    """
    if not settings.enabled or math.isnan(line_width):
        return gray
    sigma = settings.sigma_in_line_widths * line_width
    if sigma < 0.5:
        return gray
    return cv2.GaussianBlur(gray, (0, 0), sigma)


def restore_to_grid(
    black_on_white: np.ndarray, shape: tuple[int, int], settings: ResolutionSettings = _RESOLUTION
) -> np.ndarray:
    """
    Resample a processed black-on-white image back onto the input's pixel grid.

    Ink coverage is area-averaged so a line keeps its footprint, then
    thresholded at ``settings.restore_ink_coverage`` so the result stays binary.
    A slightly low coverage threshold keeps thin lines continuous after a large
    reduction.
    """
    if black_on_white.shape[:2] == tuple(shape):
        return black_on_white
    coverage = (black_on_white < 128).astype(np.float32)
    coverage = cv2.resize(coverage, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return np.where(coverage >= settings.restore_ink_coverage, 0, 255).astype(np.uint8)
