# Testing Guide

This page tells you how to run the test suite, what it covers, and how to run the evaluation harness.

## Install

```bash
pip install -e ".[test]"   # pytest, pytest-qt, pytest-cov
pip install -e ".[dev]"    # the same, plus ruff
```

## Run the tests

```bash
pytest                                   # all tests
pytest tests/test_processing.py          # one file
pytest tests/test_cli.py::TestCLIParser  # one class
pytest -k "arrow"                        # tests whose name contains "arrow"
pytest -x                                # stop at the first failure
```

The pytest configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`. It sets `testpaths = ["tests"]`, verbose output (`-v`) and `qt_api = "pyqt5"`.

On a Linux machine without a display, set `QT_QPA_PLATFORM=offscreen` before you run `pytest`.

### Coverage

```bash
pytest --cov=lithic_editor --cov-report=term-missing --cov-report=html
```

The HTML report is written to `htmlcov/index.html`. No coverage threshold is enforced.

The script `tests/run_tests.py` runs `pytest -v --tb=short` and adds these coverage options when `pytest-cov` is installed.

## Test layout

```
tests/
├── conftest.py                    # Shared fixtures
├── run_tests.py                   # Convenience runner
├── test_processing.py             # Processing pipeline
├── test_resolution.py             # Line geometry and upscale factor
├── test_upscaling.py              # DPI detection and neural upscaling
├── test_annotations.py            # Arrows and the arrow canvas
├── test_cli.py                    # Command-line interface
├── test_docs_server.py            # Offline documentation server
├── test_gui.py                    # Main window
├── test_integration.py            # Combined features
└── tools/
    ├── test_dpi_eval_metrics.py   # Harness metrics
    └── test_dpi_eval_smoke.py     # Harness end to end
```

## Fixtures

`tests/conftest.py` defines these fixtures.

| Fixture | Scope | Content |
|---|---|---|
| `qapp` | session | The `QApplication` instance. Use it in each test that creates a widget. |
| `temp_dir` | function | An empty temporary directory as a `Path`. It is removed after the test. |
| `sample_image` | function | `test_image.png` in `temp_dir`: 100 x 100 px, horizontal lines every 5 px and one vertical line. No DPI tag. |
| `sample_image_with_dpi` | function | `test_image_dpi.png` in `temp_dir`: 300 x 300 px, horizontal lines, 300 DPI tag. |
| `sample_numpy_array` | function | A 100 x 100 `uint8` array with the same pattern as `sample_image`. |
| `mock_processed_image` | function | A dictionary with a blank `processed_image`, `dpi_info`, `format_info` and three `debug_stages`. |
| `sample_pixmap` | function | A white 100 x 100 `QPixmap`. |

## What each file covers

### `test_processing.py`

- `process_lithic_drawing` with a file path and with a NumPy array.
- Debug images written when `save_debug=True`, and not written when it is `False`.
- The DPI tag of the input is kept.
- An invalid path raises an error. A missing output folder is created.
- An empty image and an already binary image are processed.
- Cortex preservation on and off, its debug images, and cortex next to structural lines.
- Input formats PNG, JPEG, TIFF and BMP.

### `test_resolution.py`

- `measure_line_geometry` on wide and thin strokes, on a blank image, and with dots present.
- `choose_upscale_factor`: no factor for good geometry, factor driven by thin lines or close hatching, the cap, and unmeasurable geometry.
- `restore_to_grid`: same shape unchanged, and a thin line that survives a four times reduction.

### `test_upscaling.py`

- DPI detection from image metadata, with and without a tag, and with an invalid file.
- Upscale factor calculation and the `needs_upscaling` check.
- Model path lookup, model loading, missing model file, and the model cache.
- Interpolation upscaling, model upscaling, and fallback to interpolation when the model fails.
- Input validation and the model scale chosen for each factor.

### `test_annotations.py`

- `Arrow`: creation, custom parameters, detection status, `make_detectable`, and DPI-aware sizing.
- `ArrowCanvasWidget`: base image, add, select, delete and clear arrows, coordinate mapping, image DPI, final image, and minimum arrow size.

### `test_cli.py`

- Parser: `--version`, `gui`, `--gui`, `process` with its options, `help`, `help api`, `docs`, `docs --offline`.
- Input validation: missing file, directory, unsupported format, and the supported extensions `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`.
- `process`: success, quiet mode, missing file, cortex parameter.
- `--keep-upscaled`: by default the result has the input size and DPI. With the option, the result and the scale image are scaled by the same factor.

### `test_docs_server.py`

- Server start, port in use, documentation directory check, the `docs` command, and shutdown.

### `test_gui.py`

- Window creation, widgets, and initial button states.
- The processing thread and its signals, with and without the cortex parameter.
- The canvas: set image and clear.
- The brush size slider, the debug images checkbox, the cortex checkbox, and the clear annotations button.
- The load and save dialogs, with the dialogs mocked.

Two workflow tests are marked `skip` because they need full GUI interaction.

### `test_integration.py`

- A thin, tightly hatched drawing is upscaled for processing and returned at the input size, or kept at the working size on request.
- A drawing with thick lines is not upscaled.
- Cortex preservation together with upscaling.
- The CLI and the GUI call the processing function correctly.
- All debug images are written with consistent names.
- Combinations of `preserve_cortex` and `save_debug`; model fallback; invalid model name; missing output directory.

### `tests/tools/`

- `test_dpi_eval_metrics.py`: `tolerance_scores` on identical, shifted, partial and empty skeletons. `residual_ripples` on closed outlines, short and long free lines, and branches. `to_common_grid`, `structural_loss`, and `make_variant` with an upsampling request.
- `test_dpi_eval_smoke.py`: the `native` strategy runs on a synthetic drawing and returns valid scores.

## Continuous integration

The workflow `.github/workflows/tests.yml` runs on each push and pull request to `main` and `develop`, and on manual dispatch.

The `test` job runs the suite on this matrix:

| OS | Python |
|---|---|
| Ubuntu | 3.10, 3.11, 3.12, 3.13 |
| Windows | 3.10, 3.11, 3.12, 3.13 |
| macOS | 3.10, 3.11, 3.12, 3.13 |

Each job installs `.[test]` and runs `pytest --cov=lithic_editor --cov-report=xml --cov-report=term-missing` with `QT_QPA_PLATFORM=offscreen`. On Ubuntu, the job installs the Qt system libraries and runs pytest under `xvfb-run`. The Ubuntu job with Python 3.12 uploads `coverage.xml` to Codecov. An upload error does not fail the job.

The `lint` job runs `ruff check lithic_editor tests` on Ubuntu with Python 3.12.

## Evaluation harness

The harness in `tools/dpi_eval/` measures how the pipeline output changes with input DPI and compares processing strategies. It measures. It does not change any pipeline parameter.

Run it from the repository root.

```bash
python -m tools.dpi_eval.run --quick         # two crops: 369 and 371
python -m tools.dpi_eval.run                 # crops 365 to 374 plus lithic_600dpi
python -m tools.dpi_eval.run --real-scans    # four real scans of one drawing
```

### Options

| Option | Default | Effect |
|---|---|---|
| `--sources NAME ...` | all primary sources | Source images in `example_images/`, without extension. |
| `--quick` | off | Only the quick subset, `369` and `371`. |
| `--real-scans` | off | Evaluate `lithic_75dpi`, `lithic_150`, `lithic_300dpi` and `lithic_600dpi`, registered to the 600 DPI scan. |
| `--strategies NAME ...` | all | One or more of `native`, `unsmoothed`, `resample`, `adaptive`, `adaptive_keep`, `develop`. |
| `--dpis N ...` | `600 300 150 75` | DPI variants to evaluate. The list must include 600. |
| `--reference-strategy NAME` | `native` | The strategy whose 600 DPI output is the reference for every cell. |
| `--tolerance PX` | `4` | Skeleton match tolerance in pixels at 600 DPI. |
| `--example-dir DIR` | `example_images` | Where the sources are. |
| `--out-dir DIR` | `results/dpi_eval` | Where the runs are written. |

### Strategies

| Name | What it does |
|---|---|
| `native` | This branch's pipeline with no resolution change. |
| `unsmoothed` | `native` without the Gaussian smoothing step. |
| `adaptive` | The default pipeline: upscale when the measured lines are too thin or too close, then return the result at the input size. |
| `adaptive_keep` | `adaptive`, but the result stays at the working size. |
| `resample` | Resize to 300 DPI with plain interpolation, then process. |
| `develop` | The `develop` branch's pipeline, run in a subprocess. |

The `develop` strategy needs a checkout of the `develop` branch. Create one with `git worktree add ../lithic_editor_develop develop`, or set the `LITHIC_DEVELOP_WORKTREE` environment variable. Without a checkout, the harness skips the strategy and continues.

### Sources and variants

Each primary source is a real 600 DPI scan. The harness makes each lower DPI variant from it by area interpolation. All variants of one source share the same pixel grid, so their outputs can be compared on that grid.

With `--real-scans`, the inputs are four separate scans of one drawing. They differ in crop and rotation. The harness scales each scan by its DPI ratio and estimates a rigid transform to the 600 DPI scan. It applies the same transform to each output of that scan. The console shows the registration correlation and a ceiling F1 for each scan.

### Output layout

Each run writes `results/dpi_eval/<timestamp>/`:

```
<timestamp>/
├── summary.csv                  # one row per (source, strategy, dpi)
├── report.md                    # tables of the metrics
└── <source>/
    ├── variants/input_<dpi>.png # the DPI variant given to each strategy
    ├── <strategy>_<dpi>.png     # the output of each cell
    ├── montage_<dpi>.png        # input, ground truth if any, and every output side by side
    └── work/<strategy>_<dpi>/   # the strategy's scratch directory and pipeline.log
```

With `--real-scans`, the per-source directory is `real_scans/`.

The report contains the mean of each metric per strategy and DPI. It lists the best three strategies per DPI and the cells that did not run. When an earlier run exists under the same `--out-dir`, it also shows the change in reference F1. That comparison uses only the sources that both runs evaluated.

### Metrics

Before scoring, the harness puts each output on the source grid. It binarizes with Otsu's threshold, resamples with area interpolation, and keeps pixels with at least 50% ink coverage.

| Column | Meaning |
|---|---|
| `ref_precision` | Fraction of the output skeleton within the tolerance of the reference skeleton. |
| `ref_recall` | Fraction of the reference skeleton within the tolerance of the output skeleton. |
| `ref_f1` | Harmonic mean of `ref_precision` and `ref_recall`. |
| `ref_structural_loss` | Fraction of reference ink with no output ink within the tolerance. Measured on the thick masks, not the skeletons. |
| `ref_cortex_ratio` | Cortex area in the output divided by cortex area in the reference. It uses the pipeline's own separation rule. |
| `gt_*` | The same five scores against `example_images/ground_truth/<source>.png`, when that file exists. |
| `residual_ripples` | Count of skeleton segments with a free endpoint and at most 200 px. This is a reference-free proxy for hatch lines that survived. |
| `runtime_s` | Wall-clock time of the strategy for that cell. |
| `registration_cc` | ECC correlation of the scan registration. Real scans only. |

`residual_ripples` also counts the stubs left where hatching was cut. Compare it within one strategy across DPI, not between strategies.

!!! warning "What the reference is"
    The reference for each source is the output of the reference strategy (`native` by default) at 600 DPI. It is a pipeline output, not a hand-cleaned drawing. The `ref_*` scores say how close an output is to that output. If the reference has a defect, a strategy that reproduces the defect scores higher. Only the `gt_*` scores measure agreement with a hand-cleaned drawing.

    The tolerance is 4 px at 600 DPI. A pixel that is at most 4 px from its match counts as correct. The scores do not see a centreline shift smaller than that. Change it with `--tolerance`.

    With `--real-scans`, the registration is estimated on the input scans, not on the outputs. A registration error lowers every score of that scan by the same amount.

The harness reports these numbers. It does not choose or change pipeline parameters from them.
