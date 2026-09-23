"""
Metrics for comparing ripple-removal outputs on a common grid.

All thresholds are stated in pixels at the source resolution (600 DPI). Inputs
are boolean ink masks (True where there is ink) on the same grid, except where
a function says otherwise.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt, label
from skimage.morphology import skeletonize

from lithic_editor.processing.ripple_removal import separate_cortex_and_structure

from .sources import SOURCE_DPI

TOLERANCE_PX = 4
# Longest skeleton segment still counted as a hatch line. About 8.5 mm at 600 DPI.
MAX_RIPPLE_LENGTH_PX = 200
_EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


@dataclass(frozen=True)
class ReferenceScores:
    """How an output compares with a reference on the common grid."""

    precision: float
    recall: float
    f1: float
    structural_loss: float
    cortex_ratio: float


def to_common_grid(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Resample a black-on-white uint8 image to ``shape`` and return an ink mask.

    The image is binarized with Otsu's threshold first, so light anti-aliased
    lines in a low-resolution scan count as ink. Coverage is then averaged with
    area interpolation and thresholded at 50%, so a line keeps its footprint
    whether the image is scaled up or down.
    """
    threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu returns 0 for an already binary image, so the comparison must be inclusive.
    ink = (image <= threshold).astype(np.float32)
    if ink.shape != tuple(shape):
        ink = cv2.resize(ink, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return ink >= 0.5


def skeleton(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide skeleton of an ink mask, as bool."""
    return skeletonize(mask, method="lee").astype(bool)


def tolerance_scores(
    predicted: np.ndarray, reference: np.ndarray, tolerance: float = TOLERANCE_PX
) -> tuple[float, float, float]:
    """
    Precision, recall and F1 of one skeleton against another with a distance tolerance.

    A predicted pixel is correct when a reference pixel lies within ``tolerance``,
    and a reference pixel is found when a predicted pixel lies within it. This is
    the boundary-matching score used by edge-detection benchmarks.
    """
    predicted_count = int(predicted.sum())
    reference_count = int(reference.sum())
    if predicted_count == 0 and reference_count == 0:
        return 1.0, 1.0, 1.0
    if predicted_count == 0 or reference_count == 0:
        return 0.0, 0.0, 0.0
    distance_to_reference = distance_transform_edt(~reference)
    distance_to_predicted = distance_transform_edt(~predicted)
    precision = float((distance_to_reference[predicted] <= tolerance).mean())
    recall = float((distance_to_predicted[reference] <= tolerance).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def residual_ripples(skel: np.ndarray, max_length: int = MAX_RIPPLE_LENGTH_PX) -> int:
    """
    Count skeleton segments that end in a free endpoint and are short.

    Junction pixels are removed to split the skeleton into segments. A segment
    counts when it contains at least one endpoint and has at most ``max_length``
    pixels. This mirrors the pipeline's own rule, that a segment touching an
    endpoint is a ripple, and is a proxy for hatch lines that survived.
    """
    if not skel.any():
        return 0
    neighbours = convolve(skel.astype(int), _EIGHT_CONNECTED, mode="constant") - skel
    endpoints = skel & (neighbours == 1)
    junctions = skel & (neighbours >= 3)
    labels, count = label(skel & ~junctions, structure=_EIGHT_CONNECTED)
    if count == 0:
        return 0
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    with_endpoint = np.unique(labels[endpoints])
    with_endpoint = with_endpoint[with_endpoint > 0]
    return int((sizes[with_endpoint] <= max_length).sum())


def structural_loss(
    predicted_ink: np.ndarray, reference_ink: np.ndarray, tolerance: float = TOLERANCE_PX
) -> float:
    """
    Fraction of reference ink with no predicted ink within ``tolerance``.

    Measured on the thick masks rather than the skeletons, so it reports lost
    outline area and breaks in continuity rather than centreline jitter.
    """
    if not reference_ink.any():
        return float("nan")
    distance_to_predicted = distance_transform_edt(~predicted_ink)
    return float(1.0 - (distance_to_predicted[reference_ink] <= tolerance).mean())


def cortex_area(ink: np.ndarray, dpi: int = SOURCE_DPI) -> int:
    """
    Cortex stippling area in pixels, using the pipeline's own separation rule.

    The area thresholds are the pipeline's, scaled quadratically from 150 DPI.
    """
    scale = dpi / 150
    size_threshold = max(30, int(60 * scale * scale))
    min_threshold = max(2, int(3 * scale * scale))
    binary = ink.astype(np.uint8) * 255
    with contextlib.redirect_stdout(io.StringIO()):  # the pipeline prints its component counts
        _, cortex_mask = separate_cortex_and_structure(
            binary,
            preserve_cortex=True,
            cortex_size_threshold=size_threshold,
            cortex_min_threshold=min_threshold,
        )
    return int(cortex_mask.sum())


def fit_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pad with background or crop so a mask has exactly ``shape``, anchored top-left."""
    out = np.zeros(shape, dtype=bool)
    h = min(shape[0], mask.shape[0])
    w = min(shape[1], mask.shape[1])
    out[:h, :w] = mask[:h, :w]
    return out


def estimate_registration(
    mask: np.ndarray, reference: np.ndarray, blur_sigma: float = 6.0
) -> tuple[np.ndarray, float]:
    """
    Estimate the rigid transform that aligns ``mask`` to ``reference`` on the same grid.

    Meant for two scans of the same drawing, whose content is identical up to a
    shift and a small rotation. The centroid offset is the initial guess and ECC
    refinement fits a Euclidean transform on blurred masks. Returns the warp in
    the ``WARP_INVERSE_MAP`` convention and the ECC correlation, NaN when
    refinement failed and only the centroid shift is used.
    """
    template = cv2.GaussianBlur(reference.astype(np.float32), (0, 0), blur_sigma)
    moving = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), blur_sigma)
    ref_moments = cv2.moments(reference.astype(np.uint8), binaryImage=True)
    src_moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    warp = np.eye(2, 3, dtype=np.float32)
    if ref_moments["m00"] > 0 and src_moments["m00"] > 0:
        warp[0, 2] = src_moments["m10"] / src_moments["m00"] - ref_moments["m10"] / ref_moments["m00"]
        warp[1, 2] = src_moments["m01"] / src_moments["m00"] - ref_moments["m01"] / ref_moments["m00"]
    correlation = float("nan")
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        correlation, warp = cv2.findTransformECC(
            template, moving, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
        )
    except cv2.error:
        pass
    return warp, correlation


def apply_registration(mask: np.ndarray, warp: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Warp a mask with a transform from ``estimate_registration`` onto ``shape``."""
    aligned = cv2.warpAffine(
        mask.astype(np.uint8), warp, (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP, borderValue=0,
    )
    return aligned > 0


def score_against_reference(
    predicted_ink: np.ndarray, reference_ink: np.ndarray, tolerance: float = TOLERANCE_PX
) -> ReferenceScores:
    """All reference-based scores for one output on the common grid."""
    precision, recall, f1 = tolerance_scores(
        skeleton(predicted_ink), skeleton(reference_ink), tolerance
    )
    reference_cortex = cortex_area(reference_ink)
    cortex_ratio = (
        float("nan") if reference_cortex == 0 else cortex_area(predicted_ink) / reference_cortex
    )
    return ReferenceScores(
        precision=precision,
        recall=recall,
        f1=f1,
        structural_loss=structural_loss(predicted_ink, reference_ink, tolerance),
        cortex_ratio=cortex_ratio,
    )
