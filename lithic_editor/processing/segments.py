"""
The segments of the skeleton between its endpoints and junctions, and which of them
are ripples.
"""

import logging

import cv2
import numpy as np
from scipy.ndimage import label

logger = logging.getLogger(__name__)


def label_segments(skeleton, endpoints, junctions):
    """
    Split the skeleton at its endpoints and junctions and label the pieces.

    Returns ``(labeled_segments, num_segments)``: an int array with 0 for
    background and 1..n for the segments, and n.
    """
    logger.info("Identifying line segments...")
    special = np.zeros_like(skeleton, dtype=bool)
    for x, y in endpoints + junctions:
        special[y, x] = True
    segments = skeleton.copy()
    segments[special] = False
    labeled, count = label(segments, structure=np.ones((3, 3), dtype=np.int32))
    logger.info(f"Found {count} line segments")
    return labeled, count


def segments_picture(labeled_segments, num_segments):
    """Each segment in its own colour, the same colours on every run."""
    np.random.seed(42)
    colors = np.random.randint(50, 256, size=(num_segments + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    return colors[labeled_segments]


def connect_segments(labeled_segments, num_segments, endpoints, junctions):
    """
    For each segment, the endpoints and junctions it touches.

    A point touches a segment when a pixel of that segment lies in the point's 3x3
    neighbourhood. Points are named ``e<i>`` (endpoint) and ``j<i>`` (junction).
    Only segments that touch two or more points are returned.
    """
    names = {}
    for i, point in enumerate(endpoints):
        names[point] = f"e{i}"
    for i, point in enumerate(junctions):
        names[point] = f"j{i}"
    touching = {segment_id: [] for segment_id in range(1, num_segments + 1)}
    for x, y in endpoints + junctions:
        window = labeled_segments[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        for segment_id in np.unique(window):
            if segment_id > 0:
                touching[int(segment_id)].append(names[(x, y)])
    return {segment_id: points for segment_id, points in touching.items() if len(points) >= 2}


def identify_ripples(segment_connections):
    """A segment is a ripple when it touches at least one endpoint."""
    logger.info("Identifying ripple lines...")
    ripples = {segment_id for segment_id, points in segment_connections.items()
               if any(name.startswith('e') for name in points)}
    logger.info(f"Identified {len(ripples)} ripple segments")
    return ripples


def classify_pixels(labeled_segments, num_segments, ripple_segments):
    """Boolean masks of the ripple pixels and of the structural pixels."""
    is_ripple = np.zeros(num_segments + 1, dtype=bool)
    is_ripple[list(ripple_segments)] = True
    ripple_pixels = is_ripple[labeled_segments]
    structural_pixels = (labeled_segments > 0) & ~ripple_pixels
    return ripple_pixels, structural_pixels


def ripples_picture(ripple_pixels, structural_pixels, endpoints, junctions):
    """Structure in white, ripples in red, endpoints red, junctions green (BGR)."""
    height, width = ripple_pixels.shape
    picture = np.zeros((height, width, 3), dtype=np.uint8)
    picture[structural_pixels] = [255, 255, 255]
    picture[ripple_pixels] = [0, 0, 255]
    for x, y in endpoints:
        cv2.circle(picture, (x, y), 1, (0, 0, 255), -1)
    for x, y in junctions:
        cv2.circle(picture, (x, y), 1, (0, 255, 0), -1)
    return picture
