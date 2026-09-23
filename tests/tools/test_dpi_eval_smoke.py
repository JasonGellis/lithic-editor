"""End-to-end smoke test: one strategy on one tiny synthetic drawing."""

import numpy as np

from tools.dpi_eval import metrics, sources
from tools.dpi_eval.strategies import STRATEGIES


def _synthetic_drawing(size=160):
    """A thick closed outline with a few hatch lines inside, black on white."""
    image = np.full((size, size), 255, dtype=np.uint8)
    image[20:24, 20:140] = 0
    image[136:140, 20:140] = 0
    image[20:140, 20:24] = 0
    image[20:140, 136:140] = 0
    for row in range(50, 110, 15):
        image[row:row + 2, 40:90] = 0
    return image


def test_native_strategy_runs_and_scores(tmp_path):
    image = _synthetic_drawing()
    variant_path = tmp_path / "input_600.png"
    sources.save_with_dpi(image, variant_path, 600)

    result = STRATEGIES["native"](variant_path, 600, tmp_path)

    assert result.dtype == np.uint8
    assert result.shape == image.shape
    ink = metrics.to_common_grid(result, image.shape)
    reference = metrics.to_common_grid(image, image.shape)
    scores = metrics.score_against_reference(ink, reference)
    assert 0.0 <= scores.f1 <= 1.0
    assert metrics.residual_ripples(metrics.skeleton(ink)) >= 0
