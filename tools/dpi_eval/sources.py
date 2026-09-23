"""
Source images, aligned DPI variants and ground-truth lookup.

Every primary source is a real 600 DPI scan. Lower-DPI variants are made from it
by area interpolation, so all variants of one source share the same pixels and
their outputs can be compared on the source's grid.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SOURCE_DPI = 600
DPIS = (600, 300, 150, 75)

# Plate crops 365 to 374 plus the full-page scan. All tagged and trusted at 600 DPI.
PRIMARY_SOURCES = tuple(str(n) for n in range(365, 375)) + ("lithic_600dpi",)
QUICK_SOURCES = ("369", "371")

# One drawing scanned four times at different resolutions. These are real scans, not
# resamples, so they differ slightly in crop and rotation and must be registered to the
# 600 DPI scan before they can be compared.
REAL_SCANS = (("lithic_75dpi", 75), ("lithic_150", 150), ("lithic_300dpi", 300), ("lithic_600dpi", 600))
REAL_SCAN_REFERENCE = "lithic_600dpi"

GROUND_TRUTH_DIRNAME = "ground_truth"


def load_grayscale(path: Path) -> np.ndarray:
    """Load an image as an 8-bit grayscale array. The file's DPI tag is ignored."""
    with Image.open(path) as image:
        return np.array(image.convert("L"))


def source_path(name: str, example_dir: Path) -> Path:
    """Path of a source image by its bare name."""
    return example_dir / f"{name}.png"


def ground_truth_path(name: str, example_dir: Path) -> Path | None:
    """Path of the hand-cleaned reference for a source, or None when there is none."""
    path = example_dir / GROUND_TRUTH_DIRNAME / f"{name}.png"
    return path if path.exists() else None


def make_variant(image: np.ndarray, source_dpi: int, target_dpi: int) -> np.ndarray:
    """
    Downsample a source to a lower DPI with area interpolation.

    Area interpolation averages the covered source pixels, which is closer to
    what a scanner does at a lower resolution than nearest or Lanczos sampling.
    """
    if target_dpi == source_dpi:
        return image.copy()
    if target_dpi > source_dpi:
        raise ValueError(f"Cannot make a {target_dpi} DPI variant from a {source_dpi} DPI source")
    factor = target_dpi / source_dpi
    width = max(1, round(image.shape[1] * factor))
    height = max(1, round(image.shape[0] * factor))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def save_with_dpi(image: np.ndarray, path: Path, dpi: int) -> None:
    """Write a grayscale array as PNG with a truthful DPI tag."""
    Image.fromarray(image).save(path, dpi=(dpi, dpi))
