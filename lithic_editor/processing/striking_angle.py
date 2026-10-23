"""
Striking angle of each scar, measured from its ripples.

The ripples of a scar are arcs around the point of impact. The pipeline finds them
as skeleton segments with a free end, just before it removes them. This module
groups those segments by the scar they lie in, takes the line through their
centroid that is perpendicular to their mean orientation as the striking line, and
uses the curvature of the arcs to point that line the way the force travelled: the
ripples are arcs around the point of percussion and curve away from it.

Angles are reported in the frame PyLithics uses for ``arrow_angle``: the clockwise
bearing of the arrowhead, with down at 0 degrees, left 90, up 180 and right 270. An
arrow at the top of a drawing that points down into the piece is at 0.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from lithic_editor.config import AngleSettings


@dataclass(frozen=True)
class StrikingAngle:
    """The striking direction of one scar.

    ``x`` and ``y`` are the centroid of the scar's ripples in pixels. ``compass`` is
    the direction of the force, away from the point of percussion, in the PyLithics
    frame. ``spread`` is
    0 when every ripple has the same orientation and approaches 1 when they are
    spread evenly over all orientations. ``direction_certain`` is False when the
    ripples are too straight to tell which way they curve. ``extent`` is the square
    root of the area of the ripples' bounding box, in pixels: a measure of the size
    of the scar, used to size its arrow.
    """

    scar: int
    x: float
    y: float
    compass: float
    ripples: int
    spread: float
    direction_certain: bool
    extent: float = 0.0

    def on_grid(self, divisor: float) -> StrikingAngle:
        """The same record with the centroid and the extent divided by ``divisor``."""
        return dataclasses.replace(self, x=self.x / divisor, y=self.y / divisor,
                                   extent=self.extent / divisor)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --- The PyLithics angle frame -------------------------------------------------

def compass_from_vector(dx: float, dy: float) -> float:
    """PyLithics ``arrow_angle`` of an arrow whose head points along ``(dx, dy)``.

    Image coordinates, y down. Down gives 0, left 90, up 180, right 270. The value
    is rounded to one decimal, as PyLithics rounds it. This is PyLithics' own
    formula, ``(270 + degrees(atan2(dy, dx))) % 360``, applied to the head direction.
    """
    return round((270.0 + math.degrees(math.atan2(dy, dx))) % 360.0, 1)


def vector_from_compass(compass: float) -> tuple[float, float]:
    """Unit vector ``(dx, dy)`` in image coordinates for a compass angle."""
    rad = math.radians(compass + 90.0)
    return math.cos(rad), math.sin(rad)


def widget_angle_from_compass(compass: float) -> float:
    """The ``Arrow.angle`` value that draws an arrowhead pointing at ``compass``.

    The arrow widget's 0 points right on screen and Qt rotates clockwise, so the
    widget angle is the compass angle plus 90.
    """
    return (compass + 90.0) % 360.0


# --- Geometry of one ripple ----------------------------------------------------

def principal_axis(points: np.ndarray) -> np.ndarray:
    """Unit vector of the longest axis of an ``(n, 2)`` array of ``(x, y)`` points."""
    centred = points - points.mean(axis=0)
    if len(points) < 2:
        return np.array([1.0, 0.0])
    cov = centred.T @ centred
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))]
    return axis / (np.linalg.norm(axis) or 1.0)


def ripple_orientation(points: np.ndarray) -> float:
    """Orientation of a ripple in radians, modulo pi, from its principal axis."""
    ax, ay = principal_axis(points)
    return math.atan2(ay, ax) % math.pi


def ripple_arc_depth(points: np.ndarray) -> np.ndarray:
    """Vector from the chord of an arc to its apex, perpendicular to the chord, in pixels.

    The chord joins the two points farthest apart along the principal axis. The
    apex is the point farthest from that chord. A straight line gives a zero
    vector.
    """
    if len(points) < 3:
        return np.zeros(2)
    axis = principal_axis(points)
    along = (points - points.mean(axis=0)) @ axis
    start, end = points[int(np.argmin(along))], points[int(np.argmax(along))]
    chord = end - start
    length = float(np.linalg.norm(chord))
    if length == 0.0:
        return np.zeros(2)
    normal = np.array([-chord[1], chord[0]]) / length
    distance = (points - start) @ normal
    return normal * float(distance[int(np.argmax(np.abs(distance)))])


# --- Geometry of a group of ripples --------------------------------------------

def mean_orientation(thetas, weights) -> tuple[float, float]:
    """Doubled-angle circular mean of orientations.

    Returns ``(mean, resultant)``: the mean orientation in radians, modulo pi,
    and the resultant length in [0, 1], which is 1 when every orientation is
    the same and 0 when they are spread evenly.
    """
    t = np.asarray(thetas, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total == 0.0:
        return 0.0, 0.0
    c = float(np.sum(w * np.cos(2.0 * t)))
    s = float(np.sum(w * np.sin(2.0 * t)))
    return (0.5 * math.atan2(s, c)) % math.pi, math.hypot(c, s) / total


def measure_scar(scar: int, ripples: list[np.ndarray], settings: AngleSettings) -> StrikingAngle:
    """The striking angle of one scar from its ripples, each an ``(n, 2)`` point array."""
    thetas = [ripple_orientation(p) for p in ripples]
    weights = [float(len(p)) for p in ripples]
    depths = np.array([ripple_arc_depth(p) for p in ripples])

    mean, resultant = mean_orientation(thetas, weights)
    line = np.array([math.cos(mean + math.pi / 2), math.sin(mean + math.pi / 2)])

    # Ripples are arcs around the point of percussion and curve away from it, in the
    # direction the force travelled. The arrow points the way the ripples curve.
    mean_depth = float((depths @ line * np.asarray(weights)).sum() / sum(weights))
    direction = line if mean_depth >= 0 else -line
    certain = abs(mean_depth) >= settings.min_arc_depth_px

    points = np.concatenate(ripples)
    centroid = points.mean(axis=0)
    span = points.max(axis=0) - points.min(axis=0) + 1
    return StrikingAngle(
        scar=scar,
        x=float(centroid[0]),
        y=float(centroid[1]),
        compass=compass_from_vector(float(direction[0]), float(direction[1])),
        ripples=len(ripples),
        spread=round(1.0 - resultant, 3),
        direction_certain=certain,
        extent=float(math.sqrt(span[0] * span[1])),
    )


# --- Grouping ripples by scar --------------------------------------------------

def enclosed_regions(structure: np.ndarray) -> np.ndarray:
    """Label the faces the structural lines enclose.

    ``structure`` is a boolean mask of the structural skeleton. It is closed by one
    pixel of dilation, then the background is labelled with 4-connectivity. Faces
    that touch the image border are the outside and get label 0.
    """
    closed = ndimage.binary_dilation(structure, structure=np.ones((3, 3), dtype=bool))
    regions, _ = ndimage.label(~closed)
    border = np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]])
    outside = np.unique(border[border > 0])
    if outside.size:
        regions[np.isin(regions, outside)] = 0
    return regions


def group_ripples_by_scar(labeled_segments: np.ndarray, ripple_ids, regions: np.ndarray) -> dict[int, list[np.ndarray]]:
    """Map each enclosed region to the point arrays of the ripples that lie in it.

    A ripple belongs to the region that holds most of its pixels. Pixels on the
    structure or outside every region do not count. A ripple with no pixel in any
    region is left out.
    """
    ripple_ids = np.fromiter(ripple_ids, dtype=np.int64)
    is_ripple = np.zeros(int(labeled_segments.max()) + 1, dtype=bool)
    is_ripple[ripple_ids] = True
    ys, xs = np.nonzero(is_ripple[labeled_segments])
    segment_of_pixel = labeled_segments[ys, xs]
    region_of_pixel = regions[ys, xs]

    pairs, counts = np.unique(
        np.stack([segment_of_pixel, region_of_pixel]), axis=1, return_counts=True)
    home: dict[int, tuple[int, int]] = {}
    for (segment, region), count in zip(pairs.T, counts):
        if region == 0:
            continue
        if segment not in home or count > home[segment][1]:
            home[int(segment)] = (int(region), int(count))

    order = np.argsort(segment_of_pixel, kind="stable")
    sorted_segments = segment_of_pixel[order]
    points = np.column_stack([xs, ys]).astype(float)[order]
    groups: dict[int, list[np.ndarray]] = {}
    for segment, start, stop in _runs(sorted_segments):
        if segment in home:
            groups.setdefault(home[segment][0], []).append(points[start:stop])
    return groups


def _runs(sorted_values: np.ndarray):
    """Yield ``(value, start, stop)`` for each run of equal values in a sorted array."""
    if sorted_values.size == 0:
        return
    change = np.nonzero(np.diff(sorted_values))[0] + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [sorted_values.size]])
    for start, stop in zip(starts, stops):
        yield int(sorted_values[start]), int(start), int(stop)


def cortex_dots_per_region(cortex: np.ndarray, regions: np.ndarray) -> dict[int, int]:
    """Count the cortex stipple components in each enclosed region."""
    labels, count = ndimage.label(cortex, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return {}
    ys, xs = np.nonzero(labels)
    pairs = np.unique(np.stack([labels[ys, xs], regions[ys, xs]]), axis=1)
    dots: dict[int, int] = {}
    for _, region in pairs.T:
        if region > 0:
            dots[int(region)] = dots.get(int(region), 0) + 1
    return dots


def measure_striking_angles(labeled_segments: np.ndarray, ripple_ids, structure: np.ndarray,
                            settings: AngleSettings, cortex: np.ndarray | None = None,
                            ) -> tuple[list[StrikingAngle], list[tuple[int, int, str]]]:
    """Measure the striking angle of every scar with enough ripples.

    Returns ``(records, skipped)``. ``skipped`` lists ``(scar, ripple_count, reason)``
    for each face that gets no angle: reason ``"ripples"`` when it has fewer than
    ``settings.min_ripples`` ripples, ``"cortex"`` when ``cortex`` (a boolean mask of
    the cortex stipple) has more than ``settings.max_cortex_dots`` dots in it.
    Scars are numbered from 1 in the order of their region label; centroids are in
    pixels of ``labeled_segments``.
    """
    regions = enclosed_regions(structure)
    groups = group_ripples_by_scar(labeled_segments, ripple_ids, regions)
    dots = cortex_dots_per_region(cortex, regions) if cortex is not None else {}
    records, skipped = [], []
    for scar, region in enumerate(sorted(groups), start=1):
        ripples = groups[region]
        if dots.get(region, 0) > settings.max_cortex_dots:
            skipped.append((scar, len(ripples), "cortex"))
        elif len(ripples) < settings.min_ripples:
            skipped.append((scar, len(ripples), "ripples"))
        else:
            records.append(measure_scar(scar, ripples, settings))
    return records, skipped


# --- Debug drawing -------------------------------------------------------------

def draw_striking_angles(image: np.ndarray, records: list[StrikingAngle], length: float) -> np.ndarray:
    """Draw each scar's centroid, striking line and direction on a copy of a BGR image.

    ``length`` is the length of the striking line in pixels. The line weight, the
    arrowhead and the label scale with the image size so they stay readable.
    """
    out = image.copy()
    unit = max(1.0, max(out.shape[:2]) / 1000.0)
    thin, thick = max(1, int(round(unit))), max(2, int(round(3 * unit)))
    font_scale = 1.2 * unit
    for record in records:
        cx, cy = int(round(record.x)), int(round(record.y))
        dx, dy = vector_from_compass(record.compass)
        half = max(1, int(round(length / 2)))
        p1 = (int(round(cx - dx * half)), int(round(cy - dy * half)))
        p2 = (int(round(cx + dx * half)), int(round(cy + dy * half)))
        colour = (0, 255, 255) if record.direction_certain else (0, 165, 255)
        cv2.line(out, p1, p2, colour, thin)
        cv2.arrowedLine(out, (cx, cy), p2, colour, thick, tipLength=0.3)
        cv2.circle(out, (cx, cy), int(round(4 * unit)), (255, 255, 0), -1)
        label = f"{record.scar}: {record.compass:.0f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick + 2)
        x = cx + int(8 * unit)
        if x + text_w > out.shape[1]:
            x = max(0, cx - int(8 * unit) - text_w)
        y = max(text_h, cy - int(8 * unit))
        origin = (x, y)
        cv2.putText(out, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(out, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thin + 1, cv2.LINE_AA)
    return out


ARROW_SIZE_FRACTION = 0.2    # arrow size as a fraction of the scar's extent
ARROW_SIZE_MAX_FACTOR = 4.0  # never more than this many times the minimum size


def arrow_size_for(extent: float, minimum: float) -> float:
    """The size of the arrow for a scar: a fraction of its extent, within limits."""
    return float(min(max(ARROW_SIZE_FRACTION * extent, minimum), ARROW_SIZE_MAX_FACTOR * minimum))


def arrow_polygon(x: float, y: float, compass: float, size: float) -> np.ndarray:
    """The filled arrow the GUI draws, as an ``(7, 2)`` int32 polygon centred on ``(x, y)``.

    Same proportions as ``Arrow.draw``: shaft 1.3 x size, head 0.8 x size long and
    0.6 x size wide, shaft width 5% of size (at least 2 px).
    """
    shaft_length, head_length, head_width = 1.3 * size, 0.8 * size, 0.6 * size
    shaft_width = max(2.0, 0.05 * size)
    local = np.array([
        [-shaft_length / 2, -shaft_width / 2],
        [shaft_length / 2 - head_length, -shaft_width / 2],
        [shaft_length / 2 - head_length, -head_width / 2],
        [shaft_length / 2, 0.0],
        [shaft_length / 2 - head_length, head_width / 2],
        [shaft_length / 2 - head_length, shaft_width / 2],
        [-shaft_length / 2, shaft_width / 2],
    ])
    dx, dy = vector_from_compass(compass)
    rotation = np.array([[dx, -dy], [dy, dx]])
    return np.rint(local @ rotation.T + np.array([x, y])).astype(np.int32)


def draw_arrows(image: np.ndarray, records: list[StrikingAngle], minimum_size: float) -> np.ndarray:
    """Draw a filled black arrow for each record on a copy of a grayscale image.

    Each arrow is sized by ``arrow_size_for`` from the scar's extent, as the GUI
    sizes them. ``minimum_size`` is the minimum detectable size for the image DPI.
    """
    out = image.copy()
    for record in records:
        size = arrow_size_for(record.extent, minimum_size)
        cv2.fillPoly(out, [arrow_polygon(record.x, record.y, record.compass, size)], 0)
    return out


__all__ = [
    "StrikingAngle", "compass_from_vector", "vector_from_compass", "widget_angle_from_compass",
    "principal_axis", "ripple_orientation", "ripple_arc_depth", "mean_orientation", "measure_scar",
    "enclosed_regions", "group_ripples_by_scar", "cortex_dots_per_region", "measure_striking_angles",
    "draw_striking_angles",
    "arrow_polygon", "draw_arrows", "arrow_size_for",
]
