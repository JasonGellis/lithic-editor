"""
The shapes of the arrow styles.

One module gives the geometry of every arrow style, so the GUI canvas and the CLI
debug image draw the same arrows. The shapes are in local coordinates: the arrow
is centred on (0, 0), its shaft lies along the x axis and its head points to +x.
All lengths are fractions of the arrow size. No Qt and no OpenCV here.

The styles follow the conventions in Inizan et al. 1999 (fig. 52), Raczynski-Henk
2017 (fig. 7) and Martingell & Saville 1988 (fig. 27):

- ``filled``: a filled head on a shaft. The direction of the blow. The default.
- ``open``: an outline head on a shaft. The bulb imprint is not visible in the scar.
- ``dot``: a filled arrow with a dot at the tail. A bulb or negative bulb is present.
- ``crossed``: a filled arrow with a bar across the shaft. The orientation and
  direction of debitage.
- ``dashed``: a filled head on a dashed shaft. An earlier burin blow.
"""

from dataclasses import dataclass

# (name, label) in the order the GUI shows them. The name is what the code uses.
STYLES = (
    ("filled", "Filled"),
    ("open", "Open"),
    ("dot", "Dot"),
    ("crossed", "Crossed"),
    ("dashed", "Dashed"),
)
STYLE_NAMES = tuple(name for name, _ in STYLES)
DEFAULT_STYLE = "filled"

# The minimum arrow size: the size at the reference DPI, scaled down with the DPI so the
# arrow keeps its physical size (about 4 mm), with a floor so it keeps its shape. PyLithics
# scales its detection thresholds the same way.
BASE_SIZE = 50
REFERENCE_DPI = 300
FLOOR_SIZE = 20


def minimum_size_for_dpi(dpi=None):
    """The minimum arrow size in pixels for an image at ``dpi``.

    ``dpi`` is a number, a ``(x, y)`` tuple or None. None means the reference DPI.
    At the reference DPI or more the minimum is ``BASE_SIZE``. Below it the minimum
    shrinks in proportion, never below ``FLOOR_SIZE``.
    """
    if isinstance(dpi, (tuple, list)):
        dpi = min(dpi) if dpi else None
    if not dpi or dpi >= REFERENCE_DPI:
        return BASE_SIZE
    return max(FLOOR_SIZE, int(round(BASE_SIZE * dpi / REFERENCE_DPI)))

# Fractions of the arrow size
SHAFT_LENGTH = 1.3
HEAD_LENGTH = 0.8
HEAD_WIDTH = 0.6
SHAFT_WIDTH = 0.05
MIN_SHAFT_WIDTH_PX = 2.0
DOT_RADIUS = 0.12
BAR_LENGTH = 0.4
BAR_OFFSET = 0.25  # from the tail
DASHES = 4

Point = tuple[float, float]
Polygon = tuple[Point, ...]


@dataclass(frozen=True)
class ArrowParts:
    """The parts of one arrow, in local coordinates."""

    head: Polygon                     # the triangle, tip at +x
    shafts: tuple[Polygon, ...]       # one rectangle, or several dashes
    filled_head: bool                 # False draws the head as an outline
    dot: tuple[Point, float] | None   # (centre, radius) at the tail, or None
    bar: Polygon | None               # the bar across the shaft, or None


def check_style(style: str) -> str:
    """Return ``style`` if it is a known style name, else raise ``ValueError``."""
    if style not in STYLE_NAMES:
        names = ", ".join(STYLE_NAMES)
        raise ValueError(f"Unknown arrow style '{style}'. The styles are: {names}.")
    return style


def shaft_width(size: float) -> float:
    """The shaft width in pixels: 5 percent of the size, at least 2 px."""
    return max(MIN_SHAFT_WIDTH_PX, SHAFT_WIDTH * size)


def filled_outline(size: float) -> Polygon:
    """The filled arrow as one seven-point polygon, tail first, clockwise from the top."""
    tail, tip = -SHAFT_LENGTH * size / 2, SHAFT_LENGTH * size / 2
    base = tip - HEAD_LENGTH * size
    half_head, half_shaft = HEAD_WIDTH * size / 2, shaft_width(size) / 2
    return (
        (tail, -half_shaft),
        (base, -half_shaft),
        (base, -half_head),
        (tip, 0.0),
        (base, half_head),
        (base, half_shaft),
        (tail, half_shaft),
    )


def _rectangle(x0: float, x1: float, half_height: float) -> Polygon:
    return ((x0, -half_height), (x1, -half_height), (x1, half_height), (x0, half_height))


def arrow_parts(style: str, size: float) -> ArrowParts:
    """The parts of an arrow of ``style`` and ``size`` pixels."""
    check_style(style)
    tail, tip = -SHAFT_LENGTH * size / 2, SHAFT_LENGTH * size / 2
    base = tip - HEAD_LENGTH * size
    half_head, half_shaft = HEAD_WIDTH * size / 2, shaft_width(size) / 2
    head = ((base, -half_head), (tip, 0.0), (base, half_head))

    if style == "dashed":
        unit = (base - tail) / (2 * DASHES - 1)  # dashes and gaps of equal length
        shafts = tuple(_rectangle(tail + 2 * i * unit, tail + (2 * i + 1) * unit, half_shaft)
                       for i in range(DASHES))
    else:
        shafts = (_rectangle(tail, base, half_shaft),)

    dot = None
    if style == "dot":
        radius = DOT_RADIUS * size
        dot = ((tail + radius, 0.0), radius)

    bar = None
    if style == "crossed":
        x = tail + BAR_OFFSET * size
        bar = _rectangle(x - half_shaft, x + half_shaft, BAR_LENGTH * size / 2)

    return ArrowParts(head=head, shafts=shafts, filled_head=(style != "open"), dot=dot, bar=bar)


__all__ = [
    "STYLES", "STYLE_NAMES", "DEFAULT_STYLE", "ArrowParts",
    "check_style", "shaft_width", "filled_outline", "arrow_parts",
]
