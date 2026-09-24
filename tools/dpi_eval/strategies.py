"""
The strategies under comparison.

Every strategy has the same signature::

    run(variant_path: Path, dpi: int, workdir: Path) -> np.ndarray

It receives a grayscale PNG of one DPI variant, the DPI to assume for it, and a
scratch directory, and returns the processed image as black-on-white uint8 at
whatever resolution the strategy produces. Bringing outputs onto a common grid
is the job of ``metrics``, not of the strategy.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from lithic_editor.processing import process_lithic_drawing

from .sources import load_grayscale

WORKING_DPI = 300
LEGACY_WORKTREE_ENV = "LITHIC_LEGACY_WORKTREE"
LEGACY_WORKTREE_DIRNAME = "lithic_editor_legacy"
# The last commit of the previous pipeline, with per-DPI parameter buckets
LEGACY_COMMIT = "b9a9d75"


class StrategyUnavailable(RuntimeError):
    """The strategy cannot run in this checkout, for example a missing worktree."""


def _run_pipeline(image: np.ndarray, dpi: int, workdir: Path, **kwargs) -> np.ndarray:
    """Call this branch's pipeline with its console output captured to a log."""
    log_path = workdir / "pipeline.log"
    with open(log_path, "a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        return process_lithic_drawing(
            image,
            output_folder=str(workdir),
            dpi_info=dpi,
            save_debug=False,
            preserve_cortex=True,
            **kwargs,
        )


def native(variant_path: Path, dpi: int, workdir: Path) -> np.ndarray:
    """This branch with no resolution change at all."""
    image = load_grayscale(variant_path)
    return _run_pipeline(image, dpi, workdir, upscale_low_dpi=False)


def unsmoothed(variant_path: Path, dpi: int, workdir: Path) -> np.ndarray:
    """``native`` without the line-width smoothing step, to isolate its effect."""
    image = load_grayscale(variant_path)
    return _run_pipeline(image, dpi, workdir, upscale_low_dpi=False, smooth_lines=False)


def resample(variant_path: Path, dpi: int, workdir: Path) -> np.ndarray:
    """Normalize to the working DPI with plain interpolation, then process."""
    image = load_grayscale(variant_path)
    if dpi != WORKING_DPI:
        factor = WORKING_DPI / dpi
        interpolation = cv2.INTER_LANCZOS4 if factor > 1 else cv2.INTER_AREA
        size = (round(image.shape[1] * factor), round(image.shape[0] * factor))
        image = cv2.resize(image, size, interpolation=interpolation)
    return _run_pipeline(image, WORKING_DPI, workdir, upscale_low_dpi=False)


def adaptive(variant_path: Path, dpi: int, workdir: Path) -> np.ndarray:
    """
    This branch's intended default: upscale only when measured line width or hatch
    clearance is too small, by the smallest sufficient factor, then return the result
    on the input's pixel grid.
    """
    image = load_grayscale(variant_path)
    return _run_pipeline(
        image, dpi, workdir, upscale_low_dpi=True, upscale_model="espcn", restore_original_size=True
    )


def adaptive_keep(variant_path: Path, dpi: int, workdir: Path) -> np.ndarray:
    """``adaptive`` but the result stays at the working resolution instead of the input grid."""
    image = load_grayscale(variant_path)
    return _run_pipeline(
        image, dpi, workdir, upscale_low_dpi=True, upscale_model="espcn", restore_original_size=False
    )


def find_legacy_worktree() -> Path | None:
    """
    Locate a checkout of the previous pipeline (commit ``LEGACY_COMMIT``).

    Looks at the ``LITHIC_LEGACY_WORKTREE`` environment variable first, then at
    ``../lithic_editor_legacy`` beside this repository.
    """
    candidates = []
    from_env = os.environ.get(LEGACY_WORKTREE_ENV)
    if from_env:
        candidates.append(Path(from_env))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root.parent / LEGACY_WORKTREE_DIRNAME)
    for candidate in candidates:
        if (candidate / "lithic_editor" / "processing" / "ripple_removal.py").is_file():
            return candidate
    return None


def legacy(variant_path: Path, dpi: int, workdir: Path) -> np.ndarray:
    """
    The previous pipeline, with its per-DPI parameter buckets and no resolution handling.

    Runs in a subprocess so the two versions of ``lithic_editor`` never share a
    process. Raises ``StrategyUnavailable`` when no worktree is found.
    """
    worktree = find_legacy_worktree()
    if worktree is None:
        raise StrategyUnavailable(
            "No legacy worktree found. Create one with "
            f"'git worktree add ../{LEGACY_WORKTREE_DIRNAME} {LEGACY_COMMIT}' "
            f"or set {LEGACY_WORKTREE_ENV}."
        )
    output_path = workdir / "legacy_output.png"
    runner = Path(__file__).with_name("_legacy_runner.py")
    command = [
        sys.executable, str(runner),
        str(variant_path), str(dpi), str(workdir), str(output_path), str(worktree),
    ]
    completed = subprocess.run(
        command, cwd=str(workdir), capture_output=True, text=True, check=False
    )
    (workdir / "legacy.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"legacy pipeline failed:\n{completed.stderr[-2000:]}")
    return load_grayscale(output_path)


STRATEGIES = {
    "legacy": legacy,
    "native": native,
    "unsmoothed": unsmoothed,
    "resample": resample,
    "adaptive": adaptive,
    "adaptive_keep": adaptive_keep,
}
